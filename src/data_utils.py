from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class FingerprintSample:
    path: Path
    label: str


def parse_subject_id(filename: str) -> str:
    return filename.split("_", maxsplit=1)[0]


def collect_samples(root: Path) -> List[FingerprintSample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory {root} not found.")

    samples: List[FingerprintSample] = []
    for path in sorted(root.glob("*.bmp")):
        samples.append(FingerprintSample(path=path, label=parse_subject_id(path.name)))
    return samples


def split_known_unknown(
    samples: Sequence[FingerprintSample], enrolled_ids: Iterable[str]
):
    enrolled_set = set(enrolled_ids)
    known, unknown = [], []
    for sample in samples:
        (known if sample.label in enrolled_set else unknown).append(sample)
    return known, unknown

