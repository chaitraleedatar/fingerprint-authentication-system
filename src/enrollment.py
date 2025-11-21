from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from tqdm import tqdm

from .config import PipelineConfig
from .data_utils import FingerprintSample
from .features import MinutiaeExtractor, Minutiae
from .preprocessing import load_and_preprocess


@dataclass
class TemplateDatabase:
    labels: List[str] = field(default_factory=list)
    minutiae_list: List[List[Minutiae]] = field(default_factory=list)
    
    def add(self, label: str, minutiae: List[Minutiae]) -> None:
        self.labels.append(label)
        self.minutiae_list.append(minutiae)
    
    @property
    def unique_labels(self) -> List[str]:
        return sorted(set(self.labels))
    
    def get_person_minutiae(self, label: str) -> List[List[Minutiae]]:
        """Get all minutiae templates for a specific person."""
        return [
            minutiae for l, minutiae in zip(self.labels, self.minutiae_list)
            if l == label
        ]


def enroll_templates(
    samples: Sequence[FingerprintSample],
    config: PipelineConfig,
    extractor: MinutiaeExtractor,
    show_progress: bool = True,
) -> TemplateDatabase:
    """Enroll fingerprints by extracting and storing minutiae."""
    database = TemplateDatabase()
    iterator = tqdm(samples, desc="Enrolling", unit="img") if show_progress else samples
    
    for sample in iterator:
        # Preprocess and extract minutiae
        thinned = load_and_preprocess(sample.path, config)
        minutiae = extractor.extract(thinned)
        database.add(sample.label, minutiae)
    
    return database
