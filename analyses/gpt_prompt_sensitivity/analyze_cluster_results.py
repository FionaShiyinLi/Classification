#!/usr/bin/env python3
"""Cluster-aware analysis for the 10- and 18-example GPT sensitivities.

The script reproduces test-set confidence intervals and paired differences by
resampling whole Cochrane reviews. It writes aggregate results only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import binomtest


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
load_dotenv(REPO_ROOT / ".env")
DEFAULT_TEST_MANIFEST = Path(
    os.getenv(
        "OUTCOME_TEST_MANIFEST",
        REPO_ROOT / "restricted_data" / "leakage_similarity_test_items.csv",
    )
)
DEFAULT_PRIMARY = Path(
    os.getenv(
        "OUTCOME_PRIMARY_PREDICTIONS",
        REPO_ROOT
        / "results/outputs_hparam_search_base16/"
        "lr3e-05_wu0.04_uf8_rd1.0/best_model/test_predictions.csv",
    )
)
DEFAULT_GPT10 = Path(
    os.getenv(
        "OUTCOME_GPT10_PREDICTIONS",
        REPO_ROOT
        / "private_outputs/gpt_prompt_sensitivity/"
        "openrouter_non_zdr_10_examples/test/reasoning_low/predictions.csv",
    )
)
DEFAULT_GPT18 = Path(
    os.getenv(
        "OUTCOME_GPT18_PREDICTIONS",
        REPO_ROOT
        / "private_outputs/gpt_prompt_sensitivity/"
        "openrouter_non_zdr_18_examples/test/reasoning_low/predictions.csv",
    )
)
DEFAULT_OUTPUT = REPO_ROOT / "private_outputs" / "gpt_prompt_sensitivity" / "cluster_analysis"

LABELS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--primary-predictions", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--gpt10-predictions", type=Path, default=DEFAULT_GPT10)
    parser.add_argument("--gpt18-predictions", type=Path, default=DEFAULT_GPT18)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def prediction_vector(path: Path, manifest: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(manifest):
        raise ValueError(f"{path.name}: expected {len(manifest)} rows, found {len(frame)}")
    text_column = "text" if "text" in frame.columns else "outcome"
    if text_column in frame.columns:
        observed = frame[text_column].astype(str).str.strip().tolist()
        expected = manifest["outcome"].astype(str).str.strip().tolist()
        if observed != expected:
            raise ValueError(f"{path.name}: text/order does not match the test manifest")
    if "true" in frame.columns and not np.array_equal(
        frame["true"].to_numpy(dtype=int),
        manifest["outcome.class"].to_numpy(dtype=int),
    ):
        raise ValueError(f"{path.name}: true labels do not match the test manifest")
    columns = [
        column
        for column in ("pred", "pred_Bioformer-8L", "prediction")
        if column in frame.columns
    ]
    if len(columns) != 1:
        raise ValueError(f"{path.name}: could not identify one prediction column")
    values = frame[columns[0]].to_numpy(dtype=int)
    if not set(np.unique(values)).issubset(set(LABELS)):
        raise ValueError(f"{path.name}: predictions outside labels 0, 1, and 2")
    return values


def confusion_by_review(
    truth: np.ndarray, predictions: np.ndarray, review_codes: np.ndarray, n_reviews: int
) -> np.ndarray:
    matrices = np.zeros((n_reviews, 3, 3), dtype=np.int64)
    np.add.at(matrices, (review_codes, truth, predictions), 1)
    return matrices


def metrics_from_confusion(matrix: np.ndarray) -> tuple[float, float]:
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total)
    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    denominator = 2 * true_positive + (predicted_support - true_positive) + (
        true_support - true_positive
    )
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(3, dtype=float),
        where=denominator != 0,
    )
    return accuracy, float(f1.mean())


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(
        args.test_manifest,
        usecols=["CDSR.id", "outcome", "outcome.class"],
    )
    if len(manifest) != 4_503 or manifest["CDSR.id"].nunique() != 2_090:
        raise ValueError("Expected 4,503 outcomes from 2,090 Cochrane reviews")
    truth = manifest["outcome.class"].to_numpy(dtype=int)
    predictions = {
        "Bioformer-8L": prediction_vector(args.primary_predictions, manifest),
        "GPT-5.2 taxonomy-aligned 10 examples": prediction_vector(
            args.gpt10_predictions, manifest
        ),
        "GPT-5.2 taxonomy-aligned 18 examples": prediction_vector(
            args.gpt18_predictions, manifest
        ),
    }
    review_codes, review_ids = pd.factorize(manifest["CDSR.id"], sort=True)
    matrices = {
        name: confusion_by_review(truth, pred, review_codes, len(review_ids))
        for name, pred in predictions.items()
    }
    point = {
        name: metrics_from_confusion(matrix.sum(axis=0))
        for name, matrix in matrices.items()
    }

    rng = np.random.default_rng(args.seed)
    bootstrap = {
        name: {
            "accuracy": np.empty(args.bootstrap_reps, dtype=float),
            "macro_f1": np.empty(args.bootstrap_reps, dtype=float),
        }
        for name in matrices
    }
    for index in range(args.bootstrap_reps):
        selected = rng.integers(0, len(review_ids), size=len(review_ids))
        weights = np.bincount(selected, minlength=len(review_ids))
        for name, matrix in matrices.items():
            sampled = np.tensordot(weights, matrix, axes=(0, 0))
            accuracy, macro_f1 = metrics_from_confusion(sampled)
            bootstrap[name]["accuracy"][index] = accuracy
            bootstrap[name]["macro_f1"][index] = macro_f1

    ten_name = "GPT-5.2 taxonomy-aligned 10 examples"
    eighteen_name = "GPT-5.2 taxonomy-aligned 18 examples"
    primary_name = "Bioformer-8L"
    rows = []
    for name in predictions:
        rows.append(
            {
                "model": name,
                "accuracy": point[name][0],
                "accuracy_ci_lower": interval(bootstrap[name]["accuracy"])[0],
                "accuracy_ci_upper": interval(bootstrap[name]["accuracy"])[1],
                "macro_f1": point[name][1],
                "macro_f1_ci_lower": interval(bootstrap[name]["macro_f1"])[0],
                "macro_f1_ci_upper": interval(bootstrap[name]["macro_f1"])[1],
            }
        )

    differences = {
        "18_examples_minus_10_examples": {
            metric: {
                "point_difference": point[eighteen_name][metric_index]
                - point[ten_name][metric_index],
                "paired_cluster_bootstrap_95_ci": interval(
                    bootstrap[eighteen_name][metric] - bootstrap[ten_name][metric]
                ),
            }
            for metric_index, metric in enumerate(("accuracy", "macro_f1"))
        },
        "Bioformer_minus_10_examples": {
            metric: {
                "point_difference": point[primary_name][metric_index]
                - point[ten_name][metric_index],
                "paired_cluster_bootstrap_95_ci": interval(
                    bootstrap[primary_name][metric] - bootstrap[ten_name][metric]
                ),
            }
            for metric_index, metric in enumerate(("accuracy", "macro_f1"))
        },
    }
    ten = predictions[ten_name]
    eighteen = predictions[eighteen_name]
    ten_only = int(np.sum((ten == truth) & (eighteen != truth)))
    eighteen_only = int(np.sum((ten != truth) & (eighteen == truth)))
    discordant = ten_only + eighteen_only
    mcnemar_p = (
        float(binomtest(min(ten_only, eighteen_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )

    result = {
        "analysis": "cluster-aware GPT prompt-sensitivity comparison",
        "test_outcomes_n": len(manifest),
        "test_reviews_n": len(review_ids),
        "bootstrap_unit": "Cochrane review (CDSR.id)",
        "bootstrap_replicates": args.bootstrap_reps,
        "bootstrap_seed": args.seed,
        "paired_resamples": True,
        "differences": differences,
        "mcnemar_10_vs_18": {
            "ten_only_correct": ten_only,
            "eighteen_only_correct": eighteen_only,
            "discordant_n": discordant,
            "exact_two_sided_p": mcnemar_p,
        },
        "outputs_contain_source_text": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "gpt_prompt_sensitivity_metrics.csv", index=False
    )
    (args.output_dir / "gpt_prompt_sensitivity_cluster_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
