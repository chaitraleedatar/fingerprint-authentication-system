from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from tqdm import tqdm

from .enrollment import TemplateDatabase
from .features import Minutiae

@dataclass
class MatchResult:
    label: str
    score: float
    index: int

class MinutiaeMatcher:
    """Matches fingerprints using minutiae alignment with RANSAC and optional weighted scoring."""
    
    def __init__(self, database: TemplateDatabase, threshold: float = 0.3, ransac_iterations: int = 100, distance_threshold: float = 15.0):
        self.database = database
        self.threshold = threshold
        self.ransac_iterations = ransac_iterations
        self.distance_threshold = distance_threshold

    def match(self, query_minutiae: List[Minutiae]) -> MatchResult:
        if not query_minutiae:
            return MatchResult(label="unknown", score=0.0, index=-1)
        
        query_points = np.array([[m.x, m.y, m.orientation] for m in query_minutiae])
        best_score = 0.0
        best_label = "unknown"
        best_idx = -1

        templates_by_person = defaultdict(list)
        for idx, label in enumerate(self.database.labels):
            templates_by_person[label].append(idx)

        min_minutiae_threshold = max(5, len(query_minutiae) // 4)

        for person_label, template_indices in templates_by_person.items():
            template_scores = []
            template_indices_used = []

            for idx in template_indices[:3]:
                template_minutiae = self.database.minutiae_list[idx]
                if len(template_minutiae) < min_minutiae_threshold:
                    continue
                template_points = np.array([[m.x, m.y, m.orientation] for m in template_minutiae])
                score = self._compute_match_score(query_points, template_points)
                template_scores.append(score)
                template_indices_used.append(idx)
                if score > 0.6:
                    break

            if not template_scores:
                continue

            top_scores = sorted(template_scores, reverse=True)[:2]
            person_weighted_score = sum(top_scores) / len(top_scores)
            person_best_idx = template_indices_used[np.argmax(template_scores)]

            if person_weighted_score > best_score:
                best_score = person_weighted_score
                best_label = person_label
                best_idx = person_best_idx
                if best_score > 0.75:
                    break

        if best_score < self.threshold:
            best_label = "unknown"

        return MatchResult(label=best_label, score=best_score, index=best_idx)

    def _compute_match_score(self, query: np.ndarray, template: np.ndarray) -> float:
        if len(query) == 0 or len(template) == 0:
            return 0.0
        if len(query) < 3 or len(template) < 3:
            return self._simple_match_score(query, template)

        best_matches = 0
        iterations = min(self.ransac_iterations, 30)

        for _ in range(iterations):
            if len(query) >= 2 and len(template) >= 2:
                q_idx = np.random.choice(len(query), 2, replace=False)
                t_idx = np.random.choice(len(template), 2, replace=False)
            else:
                return self._simple_match_score(query, template)

            try:
                transform = self._estimate_transform(query[q_idx], template[t_idx])
                transformed_query = self._apply_transform(query, transform)
                matches = self._count_matches(transformed_query, template)
                best_matches = max(best_matches, matches)
                if best_matches / max(min(len(query), len(template)), 1) > 0.6:
                    break
            except:
                continue

        if best_matches == 0:
            best_matches = self._simple_match_score(query, template)

        return float(min(best_matches / min(len(query), len(template)), 1.0))

    def _estimate_transform(self, source: np.ndarray, target: np.ndarray) -> dict:
        translation = np.mean(target[:, :2], axis=0) - np.mean(source[:, :2], axis=0)
        rotation = np.mean(target[:, 2]) - np.mean(source[:, 2])
        return {'translation': translation, 'rotation': rotation}

    def _apply_transform(self, points: np.ndarray, transform: dict) -> np.ndarray:
        translated = points[:, :2] + transform['translation']
        cos_r, sin_r = np.cos(transform['rotation']), np.sin(transform['rotation'])
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        rotated = (rotation_matrix @ translated.T).T
        transformed = np.zeros_like(points)
        transformed[:, :2] = rotated
        transformed[:, 2] = points[:, 2] + transform['rotation']
        return transformed

    def _count_matches(self, query: np.ndarray, template: np.ndarray) -> int:
        if len(query) == 0 or len(template) == 0:
            return 0
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(template[:, :2])
        distances, indices = nbrs.kneighbors(query[:, :2])
        matches = 0
        orientation_threshold = np.pi / 6
        for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
            if dist < self.distance_threshold:
                orient_diff = abs(query[i, 2] - template[idx, 2])
                orient_diff = min(orient_diff, 2 * np.pi - orient_diff)
                if orient_diff < orientation_threshold:
                    matches += 1
        return matches

    def _simple_match_score(self, query: np.ndarray, template: np.ndarray) -> int:
        return self._count_matches(query, template)

    def predict_batch(self, minutiae_list: Sequence[List[Minutiae]]) -> List[MatchResult]:
        results = []
        for m in tqdm(minutiae_list, desc="Matching", unit="img"):
            results.append(self.match(m))
        return results
