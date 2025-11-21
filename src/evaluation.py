from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from .data_utils import FingerprintSample
from .matching import MatchResult


@dataclass
class EvaluationSummary:
    closed_set_accuracy: float
    open_set_accuracy: float
    unknown_detection_rate: float
    false_accept_rate: float
    average_match_score_genuine: float
    average_match_score_impostor: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _is_known(sample: FingerprintSample, enrolled_ids: Iterable[str]) -> bool:
    return sample.label in enrolled_ids


def evaluate_predictions(
    samples: Sequence[FingerprintSample],
    predictions: Sequence[MatchResult],
    enrolled_ids: Iterable[str],
) -> EvaluationSummary:
    enrolled_ids = set(enrolled_ids)
    known_total = 0
    known_correct = 0
    all_correct = 0
    unknown_total = 0
    unknown_detected = 0
    unknown_false_accept = 0
    genuine_scores = []
    impostor_scores = []

    for sample, prediction in zip(samples, predictions):
        is_known = _is_known(sample, enrolled_ids)
        predicted_label = prediction.label
        if is_known:
            known_total += 1
            if predicted_label == sample.label:
                known_correct += 1
                genuine_scores.append(prediction.score)
            else:
                impostor_scores.append(prediction.score)
        else:
            unknown_total += 1
            if predicted_label == "unknown":
                unknown_detected += 1
            else:
                unknown_false_accept += 1
                impostor_scores.append(prediction.score)

        if (is_known and predicted_label == sample.label) or (
            not is_known and predicted_label == "unknown"
        ):
            all_correct += 1

    closed_acc = known_correct / max(known_total, 1)
    open_acc = all_correct / max(len(samples), 1)
    unknown_detection = unknown_detected / max(unknown_total, 1)
    far = unknown_false_accept / max(unknown_total, 1)
    avg_genuine = float(np.mean(genuine_scores)) if genuine_scores else 0.0
    avg_impostor = float(np.mean(impostor_scores)) if impostor_scores else 0.0

    return EvaluationSummary(
        closed_set_accuracy=closed_acc,
        open_set_accuracy=open_acc,
        unknown_detection_rate=unknown_detection,
        false_accept_rate=far,
        average_match_score_genuine=avg_genuine,
        average_match_score_impostor=avg_impostor,
    )


def save_predictions_csv(
    path: Path,
    samples: Sequence[FingerprintSample],
    predictions: Sequence[MatchResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("filename,label,prediction,score\n")
        for sample, prediction in zip(samples, predictions):
            handle.write(
                f"{sample.path.name},{sample.label},{prediction.label},{prediction.score:.5f}\n"
            )


def save_metrics_json(path: Path, summary: EvaluationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2)

