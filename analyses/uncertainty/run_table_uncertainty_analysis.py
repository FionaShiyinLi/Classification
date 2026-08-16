#!/usr/bin/env python3
"""Uncertainty analysis for manuscript Tables 1 and 2.

This script does not train models or change checkpoints. It:
1. loads the exact 4,503-outcome primary test set;
2. reads the saved primary Bioformer predictions;
3. generates predictions from the three saved Table 2 checkpoints;
4. calculates review-clustered bootstrap confidence intervals; and
5. performs paired review-cluster permutation tests versus Bioformer.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = [0, 1, 2]
LABEL_NAMES = ["Objective", "Semi-objective", "Subjective"]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
load_dotenv(PROJECT_DIR / ".env")
CLASSIFICATION_DIR = PROJECT_DIR

TEST_MANIFEST = Path(
    os.getenv(
        "OUTCOME_TEST_MANIFEST",
        PROJECT_DIR / "restricted_data" / "leakage_similarity_test_items.csv",
    )
)
PRIMARY_PREDICTIONS = Path(
    os.getenv(
        "OUTCOME_PRIMARY_PREDICTIONS",
        CLASSIFICATION_DIR
        / "results/outputs_hparam_search_base16/lr3e-05_wu0.04_uf8_rd1.0/"
        "best_model/test_predictions.csv",
    )
)

MODEL_PATHS = {
    "PubMedBERT": Path(
        os.getenv(
            "OUTCOME_PUBMEDBERT_CHECKPOINT",
            CLASSIFICATION_DIR
            / "outputs_model_comparison/"
            "microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_best",
        )
    ),
    "BioBERT": Path(
        os.getenv(
            "OUTCOME_BIOBERT_CHECKPOINT",
            CLASSIFICATION_DIR
            / "outputs_model_comparison/dmis-lab_biobert-base-cased-v1.2_best",
        )
    ),
    "SciBERT": Path(
        os.getenv(
            "OUTCOME_SCIBERT_CHECKPOINT",
            CLASSIFICATION_DIR
            / "outputs_model_comparison/allenai_scibert_scivocab_uncased_best",
        )
    ),
}

# Point estimates reported in manuscript Table 2. The script stops if the
# regenerated predictions do not reproduce these values.
EXPECTED_METRICS = {
    "Bioformer-8L": (0.9153897401732178, 0.9151846159003062),
    "PubMedBERT": (0.9222740395292027, 0.9248589877064076),
    "BioBERT": (0.9207195203197868, 0.9228132621877174),
    "SciBERT": (0.9173884077281812, 0.9235162666798287),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "private_outputs" / "uncertainty",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--permutation-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=43)
    parser.add_argument("--permutation-seed", type=int, default=44)
    parser.add_argument(
        "--force-predict",
        action="store_true",
        help="Regenerate comparator predictions even when cached files exist.",
    )
    return parser.parse_args()


def load_test_manifest() -> pd.DataFrame:
    test = pd.read_csv(TEST_MANIFEST)
    required = {"CDSR.id", "outcome.id", "outcome", "outcome.class"}
    missing = required.difference(test.columns)
    if missing:
        raise ValueError(f"Test manifest is missing columns: {sorted(missing)}")
    if len(test) != 4503:
        raise ValueError(f"Expected 4,503 test outcomes, found {len(test):,}")

    test = test[["CDSR.id", "outcome.id", "outcome", "outcome.class"]].copy()
    test["outcome"] = test["outcome"].astype(str)
    test["outcome.class"] = test["outcome.class"].astype(int)
    return test


def load_primary_predictions(test: pd.DataFrame) -> np.ndarray:
    saved = pd.read_csv(PRIMARY_PREDICTIONS)
    if len(saved) != len(test):
        raise ValueError("Primary prediction count does not match the test manifest")
    if saved["text"].astype(str).tolist() != test["outcome"].tolist():
        raise ValueError("Primary prediction text order does not match the test manifest")
    if saved["true"].astype(int).tolist() != test["outcome.class"].tolist():
        raise ValueError("Primary prediction labels do not match the test manifest")
    return saved["pred"].astype(int).to_numpy()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_checkpoint(
    model_name: str,
    model_path: Path,
    test: pd.DataFrame,
    cache_path: Path,
    batch_size: int,
    max_length: int,
    force_predict: bool,
) -> np.ndarray:
    if cache_path.exists() and not force_predict:
        cached = pd.read_csv(cache_path)
        if (
            len(cached) == len(test)
            and cached["outcome"].astype(str).tolist() == test["outcome"].tolist()
            and cached["true"].astype(int).tolist()
            == test["outcome.class"].tolist()
        ):
            print(f"[{model_name}] Using cached predictions: {cache_path}")
            return cached["pred"].astype(int).to_numpy()

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = choose_device()
    print(f"[{model_name}] Loading {model_path}")
    print(f"[{model_name}] Device: {device}; batch size: {batch_size}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True
    )
    model.to(device)
    model.eval()

    texts = test["outcome"].tolist()
    predictions: list[int] = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            logits = model(**encoded).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            if start % (20 * batch_size) == 0:
                print(f"[{model_name}] Predicted {min(start + batch_size, len(texts)):,}/{len(texts):,}")

    pred = np.asarray(predictions, dtype=int)
    cached = test[["CDSR.id", "outcome.id", "outcome"]].copy()
    cached["true"] = test["outcome.class"].to_numpy()
    cached["pred"] = pred
    cached.to_csv(cache_path, index=False)
    print(f"[{model_name}] Finished in {time.time() - started:.1f} seconds")

    del model, tokenizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return pred


def confusion_by_review(
    y_true: np.ndarray, y_pred: np.ndarray, review_codes: np.ndarray, n_reviews: int
) -> np.ndarray:
    matrices = np.zeros((n_reviews, 3, 3), dtype=np.int64)
    np.add.at(matrices, (review_codes, y_true, y_pred), 1)
    return matrices


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    matrix = matrix.astype(float)
    total = matrix.sum()
    true_counts = matrix.sum(axis=1)
    pred_counts = matrix.sum(axis=0)
    diagonal = np.diag(matrix)

    precision = safe_divide(diagonal, pred_counts)
    recall = safe_divide(diagonal, true_counts)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = diagonal.sum() / total
    expected_agreement = (true_counts * pred_counts).sum() / (total * total)
    kappa = (accuracy - expected_agreement) / (1 - expected_agreement)

    result = {
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * true_counts).sum() / total),
        "kappa": float(kappa),
    }
    for index, class_name in enumerate(LABEL_NAMES):
        result[f"{class_name}_precision"] = float(precision[index])
        result[f"{class_name}_recall"] = float(recall[index])
        result[f"{class_name}_f1"] = float(f1[index])
    return result


def cluster_bootstrap(
    model_matrices: dict[str, np.ndarray], n_reps: int, seed: int
) -> dict[str, dict[str, np.ndarray]]:
    """Use identical review-resampling draws for every model."""
    model_names = list(model_matrices)
    stacked = np.stack([model_matrices[name] for name in model_names])
    n_reviews = stacked.shape[1]
    metric_names = list(metrics_from_confusion(stacked[0].sum(axis=0)))
    samples = {
        name: {metric: np.empty(n_reps, dtype=float) for metric in metric_names}
        for name in model_names
    }

    rng = np.random.default_rng(seed)
    for rep in range(n_reps):
        selected = rng.integers(0, n_reviews, size=n_reviews)
        weights = np.bincount(selected, minlength=n_reviews)
        replicate_matrices = np.einsum("g,mgij->mij", weights, stacked)
        for model_index, model_name in enumerate(model_names):
            values = metrics_from_confusion(replicate_matrices[model_index])
            for metric_name, value in values.items():
                samples[model_name][metric_name][rep] = value
        if (rep + 1) % 1000 == 0:
            print(f"[Bootstrap] Completed {rep + 1:,}/{n_reps:,} replicates")
    return samples


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def paired_cluster_permutation(
    primary_matrices: np.ndarray,
    comparator_matrices: np.ndarray,
    n_reps: int,
    seed: int,
) -> dict[str, float]:
    """Swap both models' predictions within randomly selected whole reviews."""
    primary_total = primary_matrices.sum(axis=0)
    comparator_total = comparator_matrices.sum(axis=0)
    review_difference = comparator_matrices - primary_matrices

    primary_point = metrics_from_confusion(primary_total)
    comparator_point = metrics_from_confusion(comparator_total)
    observed = {
        metric: comparator_point[metric] - primary_point[metric]
        for metric in ("accuracy", "macro_f1")
    }
    exceedances = {"accuracy": 0, "macro_f1": 0}

    rng = np.random.default_rng(seed)
    n_reviews = len(primary_matrices)
    for _ in range(n_reps):
        swap = rng.integers(0, 2, size=n_reviews).astype(bool)
        shift = review_difference[swap].sum(axis=0)
        permuted_primary = primary_total + shift
        permuted_comparator = comparator_total - shift
        primary_values = metrics_from_confusion(permuted_primary)
        comparator_values = metrics_from_confusion(permuted_comparator)
        for metric in exceedances:
            difference = comparator_values[metric] - primary_values[metric]
            if abs(difference) >= abs(observed[metric]) - 1e-15:
                exceedances[metric] += 1

    return {
        metric: (exceedances[metric] + 1) / (n_reps + 1)
        for metric in exceedances
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted.tolist()


def verify_point_estimates(point_metrics: dict[str, dict[str, float]]) -> None:
    for model_name, (expected_accuracy, expected_macro_f1) in EXPECTED_METRICS.items():
        actual = point_metrics[model_name]
        if not np.isclose(actual["accuracy"], expected_accuracy, atol=1e-12):
            raise ValueError(
                f"{model_name} accuracy mismatch: {actual['accuracy']} vs {expected_accuracy}"
            )
        if not np.isclose(actual["macro_f1"], expected_macro_f1, atol=1e-12):
            raise ValueError(
                f"{model_name} macro F1 mismatch: {actual['macro_f1']} vs {expected_macro_f1}"
            )
    print("[Check] All four point estimates exactly reproduce Tables 1 and 2")


def write_results(
    output_dir: Path,
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    point_metrics: dict[str, dict[str, float]],
    bootstrap: dict[str, dict[str, np.ndarray]],
    model_matrices: dict[str, np.ndarray],
    permutation_reps: int,
    permutation_seed: int,
    settings: dict,
) -> None:
    aligned = test[["CDSR.id", "outcome.id", "outcome", "outcome.class"]].copy()
    aligned = aligned.rename(columns={"outcome.class": "true"})
    for model_name, pred in predictions.items():
        aligned[f"pred_{model_name}"] = pred
    aligned.to_csv(output_dir / "aligned_test_predictions.csv", index=False)

    table1_rows = []
    primary = point_metrics["Bioformer-8L"]
    table1_metrics = ["accuracy", "macro_f1", "weighted_f1", "kappa"]
    table1_metrics += [
        f"{class_name}_{metric}"
        for class_name in LABEL_NAMES
        for metric in ("precision", "recall", "f1")
    ]
    for metric in table1_metrics:
        lower, upper = percentile_interval(bootstrap["Bioformer-8L"][metric])
        table1_rows.append(
            {"metric": metric, "estimate": primary[metric], "ci_lower": lower, "ci_upper": upper}
        )
    pd.DataFrame(table1_rows).to_csv(output_dir / "table1_uncertainty.csv", index=False)

    table2_rows = []
    for model_name in predictions:
        accuracy_lower, accuracy_upper = percentile_interval(
            bootstrap[model_name]["accuracy"]
        )
        f1_lower, f1_upper = percentile_interval(bootstrap[model_name]["macro_f1"])
        table2_rows.append(
            {
                "model": model_name,
                "accuracy": point_metrics[model_name]["accuracy"],
                "accuracy_ci_lower": accuracy_lower,
                "accuracy_ci_upper": accuracy_upper,
                "macro_f1": point_metrics[model_name]["macro_f1"],
                "macro_f1_ci_lower": f1_lower,
                "macro_f1_ci_upper": f1_upper,
            }
        )
    pd.DataFrame(table2_rows).to_csv(output_dir / "table2_uncertainty.csv", index=False)

    comparison_rows = []
    for index, comparator in enumerate(["PubMedBERT", "BioBERT", "SciBERT"]):
        comparison_seed = permutation_seed + index
        p_values = paired_cluster_permutation(
            model_matrices["Bioformer-8L"],
            model_matrices[comparator],
            n_reps=permutation_reps,
            seed=comparison_seed,
        )
        for metric in ("accuracy", "macro_f1"):
            bootstrap_difference = (
                bootstrap[comparator][metric] - bootstrap["Bioformer-8L"][metric]
            )
            lower, upper = percentile_interval(bootstrap_difference)
            comparison_rows.append(
                {
                    "comparison": f"{comparator} minus Bioformer-8L",
                    "metric": metric,
                    "difference": point_metrics[comparator][metric]
                    - point_metrics["Bioformer-8L"][metric],
                    "difference_ci_lower": lower,
                    "difference_ci_upper": upper,
                    "permutation_p_raw": p_values[metric],
                    "permutation_replicates": permutation_reps,
                    "permutation_seed": comparison_seed,
                }
            )

    adjusted = holm_adjust([row["permutation_p_raw"] for row in comparison_rows])
    for row, adjusted_p in zip(comparison_rows, adjusted):
        row["permutation_p_holm"] = adjusted_p
        row["holm_significant_0.05"] = adjusted_p < 0.05
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "table2_pairwise_tests.csv", index=False
    )

    summary = {
        "design": (
            "Fixed-checkpoint analysis using whole-review clustered bootstrap and "
            "paired whole-review permutation tests."
        ),
        "n_test_outcomes": len(test),
        "n_test_reviews": int(test["CDSR.id"].nunique()),
        "settings": settings,
        "point_metrics": point_metrics,
        "table1_uncertainty": table1_rows,
        "table2_uncertainty": table2_rows,
        "planned_comparisons": comparison_rows,
        "interpretation_limit": (
            "Confidence intervals and tests are conditional on the fitted single-seed "
            "checkpoints and fixed test split. They do not quantify retraining, alternate-split, "
            "hyperparameter-selection, or leakage uncertainty and do not establish universal "
            "architecture-level superiority."
        ),
    }
    with (output_dir / "uncertainty_analysis.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test = load_test_manifest()
    predictions = {"Bioformer-8L": load_primary_predictions(test)}
    for model_name, model_path in MODEL_PATHS.items():
        cache_path = args.output_dir / f"predictions_{model_name}.csv"
        predictions[model_name] = predict_checkpoint(
            model_name=model_name,
            model_path=model_path,
            test=test,
            cache_path=cache_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            force_predict=args.force_predict,
        )

    y_true = test["outcome.class"].to_numpy(dtype=int)
    review_codes, review_ids = pd.factorize(test["CDSR.id"], sort=True)
    model_matrices = {
        model_name: confusion_by_review(y_true, pred, review_codes, len(review_ids))
        for model_name, pred in predictions.items()
    }
    point_metrics = {
        model_name: metrics_from_confusion(matrices.sum(axis=0))
        for model_name, matrices in model_matrices.items()
    }
    verify_point_estimates(point_metrics)

    bootstrap = cluster_bootstrap(
        model_matrices, n_reps=args.bootstrap_reps, seed=args.bootstrap_seed
    )
    settings = {
        "bootstrap_replicates": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "permutation_replicates": args.permutation_reps,
        "permutation_seed_base": args.permutation_seed,
        "permutation_seeds": {
            "PubMedBERT_vs_Bioformer-8L": args.permutation_seed,
            "BioBERT_vs_Bioformer-8L": args.permutation_seed + 1,
            "SciBERT_vs_Bioformer-8L": args.permutation_seed + 2,
        },
        "confidence_interval": "2.5th and 97.5th percentiles",
        "bootstrap_unit": "CDSR.id (entire review)",
        "multiple_testing": "Holm correction across 6 planned tests",
        "difference_direction": "comparator minus Bioformer-8L",
        "difference_ci_note": (
            "Paired difference intervals are nominal 95% intervals and are not "
            "multiplicity-adjusted; formal inference uses the Holm-adjusted permutation p-values."
        ),
    }
    write_results(
        output_dir=args.output_dir,
        test=test,
        predictions=predictions,
        point_metrics=point_metrics,
        bootstrap=bootstrap,
        model_matrices=model_matrices,
        permutation_reps=args.permutation_reps,
        permutation_seed=args.permutation_seed,
        settings=settings,
    )
    print(f"[Done] Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
