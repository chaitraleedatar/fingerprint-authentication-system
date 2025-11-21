from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from src.config import PipelineConfig
from src.data_utils import collect_samples
from src.enrollment import enroll_templates
from src.evaluation import (
    EvaluationSummary,
    evaluate_predictions,
    save_metrics_json,
    save_predictions_csv,
)
from src.features import build_feature_extractor, Minutiae
from src.matching import MinutiaeMatcher
from src.preprocessing import load_and_preprocess

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------- Helper Functions --------------------
def extract_minutiae(
    samples, config: PipelineConfig, extractor, desc: str
) -> List[List[Minutiae]]:
    minutiae_list: List[List[Minutiae]] = []
    for sample in tqdm(samples, desc=desc, unit="img"):
        thinned = load_and_preprocess(sample.path, config)
        minutiae = extractor.extract(thinned)
        minutiae_list.append(minutiae)
    return minutiae_list

def save_config(config: PipelineConfig, output_dir: Path) -> None:
    path = output_dir / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

def run_stage(
    samples,
    minutiae_list: List[List[Minutiae]],
    matcher: MinutiaeMatcher,
    enrolled_ids,
    output_dir: Path,
    split_name: str,
) -> EvaluationSummary:
    # ------------------ Speed tweak: use batch predictions in chunks ------------------
    predictions: List = []
    chunk_size = 50  # adjust depending on memory
    for i in tqdm(range(0, len(minutiae_list), chunk_size), desc=f"{split_name} batch predict"):
        batch = minutiae_list[i:i+chunk_size]
        predictions.extend(matcher.predict_batch(batch))
    # ----------------------------------------------------------------------------------
    
    save_predictions_csv(
        output_dir / f"{split_name}_predictions.csv", samples, predictions
    )
    summary = evaluate_predictions(samples, predictions, enrolled_ids)
    save_metrics_json(output_dir / f"{split_name}_metrics.json", summary)
    return summary

def parse_args():
    parser = argparse.ArgumentParser(description="Minutiae-based Fingerprint Authentication")
    parser.add_argument("--train-dir", type=str, default="Project-Data/Project-Data/train")
    parser.add_argument("--validate-dir", type=str, default="Project-Data/Project-Data/validate")
    parser.add_argument("--test-dir", type=str, default="Project-Data/Project-Data/test")
    parser.add_argument("--output-dir", type=str, default="artifacts/minutiae_baseline")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--ransac-iterations", type=int, default=20)
    parser.add_argument("--distance-threshold", type=float, default=15.0)
    return parser.parse_args()

# -------------------- Main Pipeline --------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = PipelineConfig.from_args(
        train_dir=args.train_dir,
        validate_dir=args.validate_dir,
        test_dir=args.test_dir,
    )
    config.ransac_iterations = args.ransac_iterations
    config.distance_threshold = args.distance_threshold
    if args.threshold is not None:
        config.unknown_threshold = args.threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    logger.info("Starting minutiae-based fingerprint authentication pipeline")
    logger.info(f"Output directory: {output_dir}")

    extractor = build_feature_extractor(config)

    train_samples = collect_samples(config.train_dir)
    val_samples = collect_samples(config.validate_dir)
    test_samples = collect_samples(config.test_dir)

    logger.info(f"Loaded {len(train_samples)} training images")
    logger.info(f"Loaded {len(val_samples)} validation images")
    logger.info(f"Loaded {len(test_samples)} test images")

    logger.info("Enrolling training fingerprints...")
    database = enroll_templates(train_samples, config, extractor)
    logger.info(f"Enrolled {len(database.unique_labels)} unique subjects")

    matcher = MinutiaeMatcher(
        database,
        threshold=config.unknown_threshold,
        ransac_iterations=config.ransac_iterations,
        distance_threshold=config.distance_threshold,
    )

    logger.info("Extracting validation minutiae...")
    val_minutiae = extract_minutiae(val_samples, config, extractor, desc="Validate")

    # Use fixed threshold for speed
    if args.threshold is None:
        logger.info("Using default threshold 0.35 (skip tuning for speed)")
        matcher.threshold = 0.35
    else:
        logger.info(f"Using fixed threshold: {args.threshold}")
        matcher.threshold = args.threshold

    logger.info("Evaluating on validation set...")
    val_summary = run_stage(
        val_samples, val_minutiae, matcher, database.unique_labels, output_dir, "val"
    )
    logger.info(f"Validation closed-set accuracy: {val_summary.closed_set_accuracy:.3f}")
    logger.info(f"Validation open-set accuracy: {val_summary.open_set_accuracy:.3f}")
    logger.info(f"Validation FAR: {val_summary.false_accept_rate:.3f}")

    logger.info("Extracting test minutiae...")
    test_minutiae = extract_minutiae(test_samples, config, extractor, desc="Test")
    logger.info("Evaluating on test set...")
    test_summary = run_stage(
        test_samples, test_minutiae, matcher, database.unique_labels, output_dir, "test"
    )
    logger.info(f"Test closed-set accuracy: {test_summary.closed_set_accuracy:.3f}")
    logger.info(f"Test open-set accuracy: {test_summary.open_set_accuracy:.3f}")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
