#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from evaluate_all_methods import (
    CSV_3CLS,
    PRIMARY_CHECKPOINT_PATH,
    PRIMARY_REFERENCE_RUN,
    FineTunedModelEvaluator,
    evaluate_method,
    load_test_data,
)
from project_config import REPO_ROOT


OUTPUT_PATH = Path(
    os.getenv(
        "OUTCOME_PRIMARY_CHECKPOINT_METRICS_PATH",
        str(
            REPO_ROOT
            / "private_outputs"
            / "primary_checkpoint"
            / "primary_checkpoint_metrics.json"
        ),
    )
)
EXPECTED_MANUSCRIPT_TEST_N = 4503


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the retained primary Bioformer checkpoint. Outputs are "
            "private by default and never overwrite curated paper results."
        )
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--allow-non-manuscript-test-size",
        action="store_true",
        help="Permit a smoke run whose test split does not contain 4,503 outcomes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Dataset CSV: {CSV_3CLS}")
    print(f"Reference checkpoint: {PRIMARY_CHECKPOINT_PATH}")

    texts, labels = load_test_data(CSV_3CLS, model_path=PRIMARY_CHECKPOINT_PATH)
    if len(labels) != EXPECTED_MANUSCRIPT_TEST_N and not args.allow_non_manuscript_test_size:
        raise RuntimeError(
            f"Refusing to label a {len(labels)}-item test split as the manuscript "
            f"result; expected {EXPECTED_MANUSCRIPT_TEST_N}. Supply the restricted "
            "full dataset, or pass --allow-non-manuscript-test-size for a smoke run."
        )
    evaluator = FineTunedModelEvaluator(PRIMARY_CHECKPOINT_PATH)
    start = time.time()
    preds, _ = evaluator.predict_batch(texts, batch_size=32, return_probs=True)
    inference_time = time.time() - start
    results = evaluate_method(labels, preds, "Fine-Tuned Bioformer-8L (exact primary checkpoint inference)")

    payload = {
        "method": "Fine-Tuned Bioformer-8L (exact primary checkpoint inference)",
        "checkpoint_directory_name": Path(PRIMARY_CHECKPOINT_PATH).name,
        "dataset_csv": Path(CSV_3CLS).name,
        "dataset_note": (
            "Restricted full dataset used for the manuscript."
            if len(labels) == EXPECTED_MANUSCRIPT_TEST_N
            else "Non-manuscript smoke run on a smaller dataset."
        ),
        "n_test_samples": int(len(labels)),
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "kappa": results["kappa"],
        "per_class_f1": results["per_class_f1"],
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"],
        "inference_time": inference_time,
        "time_per_sample": inference_time / max(len(labels), 1),
        "artifact_id": PRIMARY_REFERENCE_RUN["artifact_id"],
        "run_family": PRIMARY_REFERENCE_RUN["run_family"],
        "paper_role": PRIMARY_REFERENCE_RUN["paper_role"],
        "nominal_hparams": PRIMARY_REFERENCE_RUN["nominal_hparams"],
        "provenance_note": PRIMARY_REFERENCE_RUN["provenance_note"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("accuracy", "macro_f1", "kappa", "n_test_samples")}, indent=2))
    print(f"Saved exact-checkpoint metrics to: {args.output}")


if __name__ == "__main__":
    main()
