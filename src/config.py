from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class PipelineConfig:
    """Hyper-parameters and file paths used across the fingerprint pipeline."""

    train_dir: Path
    validate_dir: Path
    test_dir: Path
    # Minutiae matching parameters
    ransac_iterations: int = 100
    distance_threshold: float = 15.0
    unknown_threshold: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("train_dir", "validate_dir", "test_dir"):
            value = data[key]
            if isinstance(value, Path):
                data[key] = str(value)
        return data

    @classmethod
    def from_args(
        cls,
        train_dir: str,
        validate_dir: str,
        test_dir: str,
        **overrides: Any,
    ) -> "PipelineConfig":
        base = cls(
            train_dir=Path(train_dir),
            validate_dir=Path(validate_dir),
            test_dir=Path(test_dir),
        )
        for key, value in overrides.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base

