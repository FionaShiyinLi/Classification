#!/usr/bin/env python3
"""Train a general-domain ModernBERT baseline under the Table 2 protocol.

The existing ``train_other_models.py`` file is imported, not changed. This script
uses its data cleaning, fixed seed-42 split, R-Drop trainer, class weighting,
early stopping, and optimization settings. Only ModernBERT-specific layer
unfreezing and result reporting are added here.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
load_dotenv(PROJECT_DIR / ".env")
sys.path.insert(0, str(PROJECT_DIR))
from project_config import MODEL_REVISIONS  # noqa: E402

BASELINE_SCRIPT = PROJECT_DIR / "train_other_models.py"
UNCERTAINTY_SCRIPT = (
    PROJECT_DIR / "analyses/uncertainty/run_table_uncertainty_analysis.py"
)
DATA_FILE = Path(
    os.getenv(
        "OUTCOME_DATASET_CSV",
        PROJECT_DIR / "restricted_data" / "outcome_3cls.csv",
    )
)
TEST_MANIFEST = Path(
    os.getenv(
        "OUTCOME_TEST_MANIFEST",
        PROJECT_DIR / "restricted_data" / "leakage_similarity_test_items.csv",
    )
)
EXISTING_PREDICTIONS = (
    PROJECT_DIR
    / "private_outputs/uncertainty/aligned_test_predictions.csv"
)

MODEL_ID = "answerdotai/ModernBERT-base"
MODEL_NAME = "ModernBERT-base"
COMPARATOR_NAMES = ["Bioformer-8L", "PubMedBERT", "BioBERT", "SciBERT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "private_outputs" / "modernbert",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--permutation-reps", type=int, default=10_000)
    return parser.parse_args()


def import_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unlock_modernbert(model, n_last_blocks: int = 8) -> None:
    """Match the original protocol: classifier plus the final eight blocks."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module_name in ("head", "classifier"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True

    backbone = model.model
    n_last = max(1, min(n_last_blocks, len(backbone.layers)))
    for layer in backbone.layers[-n_last:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in backbone.final_norm.parameters():
        parameter.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"[MODEL] Trainable parameters: {trainable:,}/{total:,} "
        f"({100 * trainable / total:.1f}%)"
    )


def prepare_exact_split(baseline, output_dir: Path):
    baseline.OUTPUT_DIR = str(output_dir / "training")
    baseline.RESULTS_DIR = str(output_dir / "results")
    baseline._rng = np.random.default_rng(baseline.SEED)
    set_seed(baseline.SEED)

    frame = baseline.load_and_clean(str(DATA_FILE), keep_duplicates=False)
    split = baseline.stratified_split_70_10_20(frame)

    manifest = pd.read_csv(TEST_MANIFEST)
    split_text = list(split["test"][baseline.TEXT_COL])
    split_labels = [int(value) for value in split["test"]["labels"]]
    if split_text != manifest["outcome"].astype(str).tolist():
        raise ValueError("Recreated test text/order does not match the saved manifest")
    if split_labels != manifest["outcome.class"].astype(int).tolist():
        raise ValueError("Recreated test labels do not match the saved manifest")
    print("[CHECK] Recreated the exact 4,503-item manuscript test split")
    return split, manifest


def tokenize_split(baseline, split, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        revision=MODEL_REVISIONS[MODEL_ID],
        local_files_only=local_files_only,
    )

    def encode(batch):
        return tokenizer(
            batch[baseline.TEXT_COL],
            padding=True,
            truncation=True,
            max_length=baseline.MAX_LENGTH,
        )

    tokenized = {
        name: dataset.map(
            encode, batched=True, remove_columns=[baseline.TEXT_COL]
        )
        for name, dataset in split.items()
    }
    return tokenized


def train_if_needed(baseline, split, output_dir: Path, force_train: bool):
    model_dir = output_dir / "training/answerdotai_ModernBERT-base_best"
    if model_dir.exists() and not force_train:
        print(f"[TRAIN] Reusing completed isolated checkpoint: {model_dir}")
        return model_dir

    tokenized = tokenize_split(baseline, split, baseline.LOCAL_FILES_ONLY)
    device, device_type = baseline.check_gpu_availability()
    result = baseline.train_model(
        MODEL_ID,
        tokenized["train"],
        tokenized["validation"],
        tokenized["test"],
        device,
        device_type,
    )
    if result is None:
        raise RuntimeError("ModernBERT could not be loaded or trained")

    model_dir = Path(result["model_path"])
    with (output_dir / "results/training_result.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)
    gc.collect()
    if device_type == "mps":
        torch.mps.empty_cache()
    return model_dir


def predict(model_dir: Path, manifest: pd.DataFrame, batch_size: int) -> np.ndarray:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[PREDICT] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    ).to(device)
    model.eval()

    texts = manifest["outcome"].astype(str).tolist()
    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            predictions.extend(model(**encoded).logits.argmax(dim=-1).cpu().tolist())
    return np.asarray(predictions, dtype=int)


def analyze_results(
    uncertainty,
    manifest: pd.DataFrame,
    modernbert_predictions: np.ndarray,
    results_dir: Path,
    bootstrap_reps: int,
    permutation_reps: int,
) -> None:
    existing = pd.read_csv(EXISTING_PREDICTIONS)
    if existing["outcome"].astype(str).tolist() != manifest["outcome"].astype(str).tolist():
        raise ValueError("Existing Table 2 predictions do not match the test manifest")
    if existing["true"].astype(int).tolist() != manifest["outcome.class"].astype(int).tolist():
        raise ValueError("Existing Table 2 labels do not match the test manifest")

    prediction_columns = {
        name: existing[f"pred_{name}"].astype(int).to_numpy()
        for name in COMPARATOR_NAMES
    }
    prediction_columns[MODEL_NAME] = modernbert_predictions

    output_predictions = manifest[
        ["CDSR.id", "outcome.id", "outcome", "outcome.class"]
    ].copy()
    output_predictions = output_predictions.rename(columns={"outcome.class": "true"})
    output_predictions[f"pred_{MODEL_NAME}"] = modernbert_predictions
    output_predictions.to_csv(
        results_dir / "predictions_ModernBERT.csv", index=False
    )

    y_true = manifest["outcome.class"].astype(int).to_numpy()
    review_codes, review_ids = pd.factorize(manifest["CDSR.id"], sort=True)
    matrices = {
        name: uncertainty.confusion_by_review(
            y_true, values, review_codes, len(review_ids)
        )
        for name, values in prediction_columns.items()
    }
    metrics = {
        name: uncertainty.metrics_from_confusion(values.sum(axis=0))
        for name, values in matrices.items()
    }
    bootstrap = uncertainty.cluster_bootstrap(
        matrices, n_reps=bootstrap_reps, seed=43
    )

    table_rows = []
    for name in prediction_columns:
        accuracy_ci = uncertainty.percentile_interval(
            bootstrap[name]["accuracy"]
        )
        f1_ci = uncertainty.percentile_interval(bootstrap[name]["macro_f1"])
        table_rows.append(
            {
                "model": name,
                "accuracy": metrics[name]["accuracy"],
                "accuracy_ci_lower": accuracy_ci[0],
                "accuracy_ci_upper": accuracy_ci[1],
                "macro_f1": metrics[name]["macro_f1"],
                "macro_f1_ci_lower": f1_ci[0],
                "macro_f1_ci_upper": f1_ci[1],
            }
        )
    pd.DataFrame(table_rows).to_csv(
        results_dir / "table2_with_modernbert_uncertainty.csv", index=False
    )

    comparison_rows = []
    for index, comparator in enumerate(COMPARATOR_NAMES):
        permutation_seed = 47 + index
        p_values = uncertainty.paired_cluster_permutation(
            matrices[comparator],
            matrices[MODEL_NAME],
            n_reps=permutation_reps,
            seed=permutation_seed,
        )
        for metric_name in ("accuracy", "macro_f1"):
            difference_samples = (
                bootstrap[MODEL_NAME][metric_name]
                - bootstrap[comparator][metric_name]
            )
            difference_ci = uncertainty.percentile_interval(difference_samples)
            comparison_rows.append(
                {
                    "comparison": f"{MODEL_NAME} minus {comparator}",
                    "metric": metric_name,
                    "difference": (
                        metrics[MODEL_NAME][metric_name]
                        - metrics[comparator][metric_name]
                    ),
                    "difference_ci_lower": difference_ci[0],
                    "difference_ci_upper": difference_ci[1],
                    "permutation_p_raw": p_values[metric_name],
                    "permutation_seed": permutation_seed,
                }
            )

    adjusted = uncertainty.holm_adjust(
        [row["permutation_p_raw"] for row in comparison_rows]
    )
    for row, adjusted_p in zip(comparison_rows, adjusted):
        row["permutation_p_holm"] = adjusted_p
        row["holm_significant_0.05"] = adjusted_p < 0.05
    pd.DataFrame(comparison_rows).to_csv(
        results_dir / "modernbert_pairwise_tests.csv", index=False
    )

    summary = {
        "design": (
            "ModernBERT-base was fine-tuned on the exact fixed Table 2 split and "
            "under the same supervised training protocol as the biomedical encoders."
        ),
        "model_id": MODEL_ID,
        "n_test_outcomes": len(manifest),
        "n_test_reviews": int(manifest["CDSR.id"].nunique()),
        "point_metrics": metrics,
        "table2_with_uncertainty": table_rows,
        "modernbert_pairwise_tests": comparison_rows,
        "settings": {
            "seed": 42,
            "bootstrap_replicates": bootstrap_reps,
            "bootstrap_seed": 43,
            "permutation_replicates": permutation_reps,
            "permutation_seeds": list(range(47, 47 + len(COMPARATOR_NAMES))),
            "bootstrap_unit": "CDSR.id (entire Cochrane review)",
            "multiple_testing": "Holm correction across 8 tests",
        },
        "interpretation_limit": (
            "ModernBERT differs from the biomedical encoders in architecture, "
            "pretraining corpus, pretraining scale, and model generation. This is a "
            "strong practical general-domain baseline, but it does not by itself "
            "identify a pure causal effect of biomedical pretraining. Intervals are "
            "conditional on the fitted single-seed checkpoints and fixed test split."
        ),
    }
    with (results_dir / "modernbert_analysis.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    modernbert = metrics[MODEL_NAME]
    print(
        f"[RESULT] {MODEL_NAME}: accuracy={modernbert['accuracy']:.4f}; "
        f"macro F1={modernbert['macro_f1']:.4f}"
    )


def main() -> None:
    args = parse_args()
    results_dir = args.output_dir / "results"
    (args.output_dir / "training").mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline = import_script(BASELINE_SCRIPT, "table2_baseline")
    uncertainty = import_script(UNCERTAINTY_SCRIPT, "table2_uncertainty")
    baseline.unlock_last_blocks_and_layernorms = unlock_modernbert
    baseline.LOCAL_FILES_ONLY = args.local_files_only

    split, manifest = prepare_exact_split(baseline, args.output_dir)
    baseline.LOCAL_FILES_ONLY = args.local_files_only
    model_dir = train_if_needed(baseline, split, args.output_dir, args.force_train)
    predictions = predict(model_dir, manifest, args.inference_batch_size)

    y_true = manifest["outcome.class"].astype(int).to_numpy()
    training_result_path = results_dir / "training_result.json"
    if training_result_path.exists():
        saved_training_result = json.loads(training_result_path.read_text())
        reproduced_accuracy = accuracy_score(y_true, predictions)
        reproduced_f1 = f1_score(y_true, predictions, average="macro")
        if not np.isclose(
            reproduced_accuracy, saved_training_result["test_accuracy"], atol=1e-12
        ) or not np.isclose(
            reproduced_f1, saved_training_result["test_macro_f1"], atol=1e-12
        ):
            raise ValueError("Reloaded checkpoint does not reproduce its test metrics")
        print("[CHECK] Reloaded best checkpoint exactly reproduces its test metrics")

    analyze_results(
        uncertainty,
        manifest,
        predictions,
        results_dir,
        args.bootstrap_reps,
        args.permutation_reps,
    )
    print(f"[DONE] Outputs written to {results_dir}")


if __name__ == "__main__":
    main()
