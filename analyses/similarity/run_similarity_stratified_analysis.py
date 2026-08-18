#!/usr/bin/env python3
"""Report Bioformer test performance by nearest-training-text similarity.

The analysis joins the already calculated character 5-gram Jaccard
similarities to the exact Bioformer-8L predictions used for the manuscript.
Confidence intervals use a percentile bootstrap that resamples whole Cochrane
reviews within each stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


LABELS = [0, 1, 2]
LABEL_NAMES = ["Objective", "Semi-objective", "Subjective"]
REVIEW_COL = "CDSR.id"
KEY_COLUMNS = [REVIEW_COL, "outcome.id", "outcome"]
SIMILARITY_COL = "nearest_train_jaccard_5gram"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
load_dotenv(PROJECT_DIR / ".env")
DEFAULT_SIMILARITY_FILE = Path(
    os.getenv(
        "OUTCOME_TEST_MANIFEST",
        PROJECT_DIR / "restricted_data" / "leakage_similarity_test_items.csv",
    )
)
DEFAULT_PREDICTION_FILE = (
    PROJECT_DIR
    / "private_outputs/uncertainty/aligned_test_predictions.csv"
)

EXPECTED_TEST_N = 4_503
EXPECTED_REVIEW_N = 2_090
EXPECTED_ACCURACY = 0.9153897401732178
EXPECTED_MACRO_F1 = 0.9151846159003062

BIN_LABELS = ["<0.50", "0.50–<0.75", "0.75–<0.90", "≥0.90"]
_DASH_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--similarity-file", type=Path, default=DEFAULT_SIMILARITY_FILE
    )
    parser.add_argument(
        "--prediction-file", type=Path, default=DEFAULT_PREDICTION_FILE
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "private_outputs" / "similarity",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(data: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(data.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def normalize_text(value: object) -> str:
    """Use the same normalization as leakage_risk_audit.py."""
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = _DASH_RE.sub("-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, object]:
    """Calculate accuracy and macro F1 across the three prespecified classes."""
    matrix = matrix.astype(float)
    total = matrix.sum()
    if total == 0:
        raise ValueError("Cannot calculate metrics for an empty stratum")

    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    correct = np.diag(matrix)
    precision = safe_divide(correct, predicted_counts)
    recall = safe_divide(correct, true_counts)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": float(correct.sum() / total),
        "macro_f1": float(f1.mean()),
        "true_counts": true_counts.astype(int),
    }


def review_confusion_matrices(data: pd.DataFrame) -> np.ndarray:
    """Create one 3-by-3 confusion matrix per Cochrane review."""
    review_codes, reviews = pd.factorize(data[REVIEW_COL].astype(str), sort=True)
    matrices = np.zeros((len(reviews), 3, 3), dtype=np.int64)
    y_true = data["y_true"].to_numpy(dtype=int)
    y_pred = data["y_pred"].to_numpy(dtype=int)
    np.add.at(matrices, (review_codes, y_true, y_pred), 1)
    return matrices


def cluster_bootstrap(
    review_matrices: np.ndarray, repetitions: int, seed: int
) -> dict[str, tuple[float, float]]:
    """Resample whole reviews and return percentile 95% intervals."""
    review_n = len(review_matrices)
    if review_n == 0:
        raise ValueError("A stratum contains no Cochrane reviews")

    accuracy = np.empty(repetitions, dtype=float)
    macro_f1 = np.empty(repetitions, dtype=float)
    rng = np.random.default_rng(seed)

    for repetition in range(repetitions):
        selected = rng.integers(0, review_n, size=review_n)
        weights = np.bincount(selected, minlength=review_n)
        matrix = np.einsum("g,gij->ij", weights, review_matrices)
        metrics = metrics_from_confusion(matrix)
        accuracy[repetition] = metrics["accuracy"]
        macro_f1[repetition] = metrics["macro_f1"]

    return {
        "accuracy": tuple(float(x) for x in np.percentile(accuracy, [2.5, 97.5])),
        "macro_f1": tuple(float(x) for x in np.percentile(macro_f1, [2.5, 97.5])),
    }


def load_and_align(similarity_file: Path, prediction_file: Path) -> pd.DataFrame:
    similarity = pd.read_csv(similarity_file)
    predictions = pd.read_csv(prediction_file)

    require_columns(
        similarity,
        set(KEY_COLUMNS)
        | {
            "outcome.class",
            SIMILARITY_COL,
            "nearest_train_outcome",
        },
        "Similarity file",
    )
    require_columns(
        predictions,
        set(KEY_COLUMNS) | {"true", "pred_Bioformer-8L"},
        "Prediction file",
    )

    if similarity.duplicated(KEY_COLUMNS).any():
        raise ValueError("Similarity file has duplicated test-item keys")
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError("Prediction file has duplicated test-item keys")

    data = similarity.merge(
        predictions[KEY_COLUMNS + ["true", "pred_Bioformer-8L"]],
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not data["_merge"].eq("both").all():
        counts = data["_merge"].value_counts().to_dict()
        raise ValueError(f"Similarity/prediction alignment failed: {counts}")
    data = data.drop(columns="_merge")

    data["outcome.class"] = data["outcome.class"].astype(int)
    data["true"] = data["true"].astype(int)
    data["pred_Bioformer-8L"] = data["pred_Bioformer-8L"].astype(int)
    if not np.array_equal(data["outcome.class"], data["true"]):
        raise ValueError("True labels disagree between the two input files")

    data = data.rename(columns={"true": "y_true", "pred_Bioformer-8L": "y_pred"})
    if not data["y_true"].isin(LABELS).all() or not data["y_pred"].isin(LABELS).all():
        raise ValueError("Labels and predictions must be integers in {0, 1, 2}")
    if data[[REVIEW_COL, "y_true", "y_pred", SIMILARITY_COL]].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")
    if not data[SIMILARITY_COL].between(0, 1, inclusive="both").all():
        raise ValueError("Jaccard similarities must lie between 0 and 1")

    data["similarity_stratum"] = pd.cut(
        data[SIMILARITY_COL],
        bins=[-np.inf, 0.50, 0.75, 0.90, np.inf],
        labels=BIN_LABELS,
        right=False,
    ).astype(str)

    test_normalized = data["outcome"].map(normalize_text)
    nearest_normalized = data["nearest_train_outcome"].map(normalize_text)
    data["normalized_exact_match"] = (
        data["nearest_train_outcome"].notna()
        & test_normalized.ne("")
        & test_normalized.eq(nearest_normalized)
    )

    if len(data) != EXPECTED_TEST_N:
        raise ValueError(f"Expected {EXPECTED_TEST_N:,} test outcomes, found {len(data):,}")
    if data[REVIEW_COL].nunique() != EXPECTED_REVIEW_N:
        raise ValueError(
            f"Expected {EXPECTED_REVIEW_N:,} reviews, found {data[REVIEW_COL].nunique():,}"
        )

    overall = metrics_from_confusion(review_confusion_matrices(data).sum(axis=0))
    if not np.isclose(overall["accuracy"], EXPECTED_ACCURACY, atol=1e-12):
        raise ValueError("Predictions do not reproduce the manuscript Bioformer accuracy")
    if not np.isclose(overall["macro_f1"], EXPECTED_MACRO_F1, atol=1e-12):
        raise ValueError("Predictions do not reproduce the manuscript Bioformer macro F1")

    return data


def analyze_stratum(
    name: str,
    data: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
    mutually_exclusive: bool,
) -> dict[str, object]:
    matrices = review_confusion_matrices(data)
    point = metrics_from_confusion(matrices.sum(axis=0))
    intervals = cluster_bootstrap(matrices, bootstrap_reps, seed)
    class_counts = point["true_counts"]

    result: dict[str, object] = {
        "stratum": name,
        "mutually_exclusive": mutually_exclusive,
        "n": int(len(data)),
        "reviews_n": int(data[REVIEW_COL].nunique()),
        "accuracy": point["accuracy"],
        "accuracy_ci_low": intervals["accuracy"][0],
        "accuracy_ci_high": intervals["accuracy"][1],
        "macro_f1": point["macro_f1"],
        "macro_f1_ci_low": intervals["macro_f1"][0],
        "macro_f1_ci_high": intervals["macro_f1"][1],
        "bootstrap_seed": seed,
    }
    for class_index, class_name in enumerate(LABEL_NAMES):
        key = class_name.lower().replace("-", "_")
        count = int(class_counts[class_index])
        result[f"{key}_n"] = count
        result[f"{key}_percent"] = float(100 * count / len(data))
    return result


def format_percent_metric(point: float, low: float, high: float) -> str:
    return f"{100 * point:.2f}% ({100 * low:.2f}%–{100 * high:.2f}%)"


def format_decimal_metric(point: float, low: float, high: float) -> str:
    return f"{point:.4f} ({low:.4f}–{high:.4f})"


def format_count_percent(count: int, percent: float) -> str:
    return f"{count:,} ({percent:.1f}%)"


def formatted_table(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        label = str(result["stratum"])
        if not result["mutually_exclusive"]:
            label += "*"
        rows.append(
            {
                "Nearest-training similarity": label,
                "Test outcomes, n": f"{result['n']:,}",
                "Reviews, n": f"{result['reviews_n']:,}",
                "Accuracy (95% CI)": format_percent_metric(
                    result["accuracy"],
                    result["accuracy_ci_low"],
                    result["accuracy_ci_high"],
                ),
                "Macro F1 (95% CI)": format_decimal_metric(
                    result["macro_f1"],
                    result["macro_f1_ci_low"],
                    result["macro_f1_ci_high"],
                ),
                "Objective, n (%)": format_count_percent(
                    result["objective_n"], result["objective_percent"]
                ),
                "Semi-objective, n (%)": format_count_percent(
                    result["semi_objective_n"], result["semi_objective_percent"]
                ),
                "Subjective, n (%)": format_count_percent(
                    result["subjective_n"], result["subjective_percent"]
                ),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame, output_path: Path, settings: str) -> None:
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.extend(
        [
            "",
            "*Normalized exact-text match is an overlapping subset of the ≥0.90 stratum, not a fifth mutually exclusive bin.",
            "",
            settings,
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_align(args.similarity_file, args.prediction_file)
    results: list[dict[str, object]] = []

    for index, bin_label in enumerate(BIN_LABELS):
        subset = data[data["similarity_stratum"] == bin_label].copy()
        results.append(
            analyze_stratum(
                bin_label,
                subset,
                bootstrap_reps=args.bootstrap_reps,
                seed=args.seed + index,
                mutually_exclusive=True,
            )
        )

    exact_subset = data[data["normalized_exact_match"]].copy()
    if not exact_subset[SIMILARITY_COL].ge(0.90).all():
        raise RuntimeError("A normalized exact match fell outside the ≥0.90 stratum")
    results.append(
        analyze_stratum(
            "Normalized exact-text match",
            exact_subset,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed + len(BIN_LABELS),
            mutually_exclusive=False,
        )
    )

    if sum(result["n"] for result in results[:4]) != len(data):
        raise RuntimeError("The four mutually exclusive similarity bins do not sum to the test set")

    numeric = pd.DataFrame(results)
    table = formatted_table(results)
    numeric.to_csv(args.output_dir / "similarity_stratified_metrics.csv", index=False)
    table.to_csv(args.output_dir / "similarity_stratified_table.csv", index=False)

    settings_note = (
        f"Note: 95% CIs are 2.5th–97.5th percentile intervals from "
        f"{args.bootstrap_reps:,} bootstrap replicates that resampled Cochrane "
        f"reviews as clusters within each stratum (base seed {args.seed}). "
        f"Stratum-specific seeds were {args.seed}–{args.seed + 4}. "
        f"Macro F1 averages the three prespecified classes."
    )
    write_markdown(
        table,
        args.output_dir / "similarity_stratified_table.md",
        settings_note,
    )

    metadata = {
        "analysis": "Bioformer-8L performance stratified by maximum character 5-gram Jaccard similarity to a training outcome",
        "similarity_file": args.similarity_file.name,
        "similarity_file_sha256": sha256(args.similarity_file),
        "prediction_file": args.prediction_file.name,
        "prediction_file_sha256": sha256(args.prediction_file),
        "test_outcomes_n": int(len(data)),
        "test_reviews_n": int(data[REVIEW_COL].nunique()),
        "bootstrap_unit": "CDSR.id (Cochrane review)",
        "bootstrap_replicates": args.bootstrap_reps,
        "base_seed": args.seed,
        "confidence_interval": "2.5th and 97.5th percentiles",
        "macro_f1_labels": LABEL_NAMES,
        "normalized_exact_match_note": "Overlapping subset; all normalized exact matches are included within the >=0.90 bin.",
        "results": results,
    }
    with (args.output_dir / "similarity_stratified_analysis.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(table.to_string(index=False))
    print(f"\nWrote aggregate results to {args.output_dir}")


if __name__ == "__main__":
    main()
