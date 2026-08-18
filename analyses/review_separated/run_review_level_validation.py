#!/usr/bin/env python3
"""Train and evaluate Bioformer-8L with a review-disjoint split.

This is a separate sensitivity analysis. It never reads from or writes to the
manuscript's existing checkpoint directories. Whole Cochrane reviews are
assigned to exactly one of train, validation, or test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
from datasets import Dataset, DatasetDict, Value
from torch.utils.data import SequentialSampler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


# -----------------------------------------------------------------------------
# Paths and manuscript-matched settings
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from project_config import (  # noqa: E402
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    MAX_GRAD_NORM,
    MODEL_REVISIONS,
    OPTIMIZER_NAME,
    PRIMARY_SCHEDULER,
)

TEXT_COL = "outcome"
LABEL_COL = "outcome.class"
REVIEW_COL = "CDSR.id"
OUTCOME_ID_COL = "outcome.id"

MODEL_NAME = "bioformers/bioformer-8L"
NUM_LABELS = 3
ID2LABEL = {0: "Objective", 1: "Semi-objective", 2: "Subjective"}
LABEL2ID = {value: key for key, value in ID2LABEL.items()}

SEED = 42
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.04
UNFREEZE_BLOCKS = 8
RDROP_ALPHA = 1.0
WEIGHT_DECAY = 0.02
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 1
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2

N_GROUP_FOLDS = 10
TARGET_TRAIN = 0.70
TARGET_VALIDATION = 0.10
TARGET_TEST = 0.20

_DASH_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")


# -----------------------------------------------------------------------------
# Data preparation and group-disjoint split
# -----------------------------------------------------------------------------


def normalize_text(value: str) -> str:
    """Normalize text only for the post-split overlap audit."""
    text = str(value).lower().strip()
    text = _DASH_RE.sub("-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    """Apply the same validity checks and exact text-label deduplication."""
    raw = pd.read_csv(path)
    required = [REVIEW_COL, OUTCOME_ID_COL, TEXT_COL, LABEL_COL]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = raw[required].copy()
    frame[LABEL_COL] = pd.to_numeric(frame[LABEL_COL], errors="coerce")
    valid = (
        frame[REVIEW_COL].notna()
        & frame[OUTCOME_ID_COL].notna()
        & frame[TEXT_COL].notna()
        & frame[LABEL_COL].isin([0, 1, 2])
    )
    frame = frame.loc[valid].copy()
    frame[REVIEW_COL] = frame[REVIEW_COL].astype(str).str.strip()
    frame[OUTCOME_ID_COL] = frame[OUTCOME_ID_COL].astype(str).str.strip()
    frame[TEXT_COL] = frame[TEXT_COL].astype(str).str.strip()
    frame = frame[(frame[REVIEW_COL].str.len() > 0) & (frame[TEXT_COL].str.len() > 0)].copy()
    frame[LABEL_COL] = frame[LABEL_COL].astype(int)

    valid_rows = len(frame)
    frame = frame.drop_duplicates(subset=[TEXT_COL, LABEL_COL], keep="first").reset_index(drop=True)

    cleaning = {
        "source_rows": int(len(raw)),
        "valid_rows_before_deduplication": int(valid_rows),
        "rows_after_exact_text_label_deduplication": int(len(frame)),
        "exact_text_label_duplicates_removed": int(valid_rows - len(frame)),
        "unique_reviews_after_deduplication": int(frame[REVIEW_COL].nunique()),
    }
    return frame, cleaning


def _class_proportions(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(float)
    return counts / counts.sum()


def make_review_level_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Create approximately 70/10/20 splits with no shared review IDs.

    First, StratifiedGroupKFold creates ten whole-review folds. We then choose
    two test folds and one validation fold that best match the requested sizes
    and the overall class proportions. Model predictions are never considered.
    """
    labels = frame[LABEL_COL].to_numpy(dtype=int)
    groups = frame[REVIEW_COL].to_numpy()
    fold_ids = np.full(len(frame), -1, dtype=int)

    splitter = StratifiedGroupKFold(
        n_splits=N_GROUP_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    for fold, (_, held_out_indices) in enumerate(splitter.split(frame, labels, groups)):
        fold_ids[held_out_indices] = fold

    if (fold_ids < 0).any():
        raise RuntimeError("Some outcomes were not assigned to a review-level fold.")

    overall_proportions = _class_proportions(labels)
    best_choice = None

    for test_folds in itertools.combinations(range(N_GROUP_FOLDS), 2):
        remaining = [fold for fold in range(N_GROUP_FOLDS) if fold not in test_folds]
        for validation_fold in remaining:
            test_mask = np.isin(fold_ids, test_folds)
            validation_mask = fold_ids == validation_fold

            size_error = (
                abs(test_mask.mean() - TARGET_TEST)
                + abs(validation_mask.mean() - TARGET_VALIDATION)
            )
            balance_error = (
                np.abs(_class_proportions(labels[test_mask]) - overall_proportions).sum()
                + np.abs(_class_proportions(labels[validation_mask]) - overall_proportions).sum()
            )
            score = float(size_error + balance_error)
            choice = (score, tuple(test_folds), int(validation_fold))
            if best_choice is None or choice < best_choice:
                best_choice = choice

    _, test_folds, validation_fold = best_choice
    split = np.full(len(frame), "train", dtype=object)
    split[fold_ids == validation_fold] = "validation"
    split[np.isin(fold_ids, test_folds)] = "test"

    assigned = frame.copy()
    assigned["review_fold"] = fold_ids
    assigned["split"] = split

    review_sets = {
        name: set(assigned.loc[assigned["split"] == name, REVIEW_COL])
        for name in ["train", "validation", "test"]
    }
    overlaps = {
        "train_validation": len(review_sets["train"] & review_sets["validation"]),
        "train_test": len(review_sets["train"] & review_sets["test"]),
        "validation_test": len(review_sets["validation"] & review_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Review leakage detected: {overlaps}")

    split_details = {
        "method": "10-fold StratifiedGroupKFold; 7 train folds, 1 validation fold, 2 test folds",
        "seed": int(seed),
        "selected_test_folds": list(test_folds),
        "selected_validation_fold": int(validation_fold),
        "selection_rule": (
            "Minimize absolute split-size error plus absolute class-proportion error; "
            "model predictions are not used"
        ),
        "selection_score": float(best_choice[0]),
        "review_overlap_counts": overlaps,
    }
    return assigned, split_details


def summarize_split(assigned: pd.DataFrame, cleaning: dict, split_details: dict) -> dict:
    summary = {
        "cleaning": cleaning,
        "split_design": split_details,
        "total_outcomes": int(len(assigned)),
        "total_reviews": int(assigned[REVIEW_COL].nunique()),
        "splits": {},
    }
    for split_name in ["train", "validation", "test"]:
        subset = assigned[assigned["split"] == split_name]
        counts = subset[LABEL_COL].value_counts().sort_index()
        summary["splits"][split_name] = {
            "outcomes": int(len(subset)),
            "outcome_percent": float(100 * len(subset) / len(assigned)),
            "reviews": int(subset[REVIEW_COL].nunique()),
            "class_counts": {ID2LABEL[i]: int(counts.get(i, 0)) for i in range(NUM_LABELS)},
            "class_percent": {
                ID2LABEL[i]: float(100 * counts.get(i, 0) / len(subset))
                for i in range(NUM_LABELS)
            },
        }

    train = assigned[assigned["split"] == "train"]
    test = assigned[assigned["split"] == "test"]
    train_normalized = set(train[TEXT_COL].map(normalize_text))
    train_normalized_label = set(zip(train[TEXT_COL].map(normalize_text), train[LABEL_COL]))
    test_normalized = test[TEXT_COL].map(normalize_text)
    summary["cross_split_text_audit"] = {
        "test_normalized_text_seen_in_train_any_label": int(test_normalized.isin(train_normalized).sum()),
        "test_normalized_text_and_label_seen_in_train": int(
            sum(
                (text, label) in train_normalized_label
                for text, label in zip(test_normalized, test[LABEL_COL])
            )
        ),
    }
    return summary


def to_dataset_dict(assigned: pd.DataFrame) -> DatasetDict:
    datasets = {}
    for split_name in ["train", "validation", "test"]:
        subset = assigned.loc[assigned["split"] == split_name, [TEXT_COL, LABEL_COL]].copy()
        subset = subset.rename(columns={LABEL_COL: "labels"}).reset_index(drop=True)
        dataset = Dataset.from_pandas(subset, preserve_index=False)
        datasets[split_name] = dataset.cast_column("labels", Value("int64"))
    return DatasetDict(datasets)


def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    def encode(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=MAX_LENGTH, padding=False)

    tokenized = DatasetDict()
    for split_name in dataset:
        columns = dataset[split_name].column_names
        part = dataset[split_name].map(
            encode,
            batched=True,
            remove_columns=[column for column in columns if column != "labels"],
        )
        part.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized[split_name] = part
    return tokenized


# -----------------------------------------------------------------------------
# Model and training utilities (same structure as the primary sweep code)
# -----------------------------------------------------------------------------


def get_device(requested: str) -> tuple[torch.device, str]:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda"), "cuda"
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps"), "mps"
    if requested == "cpu":
        return torch.device("cpu"), "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def unlock_last_blocks(model, number_of_blocks: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if hasattr(model, "classifier"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif hasattr(model, "score"):
        for parameter in model.score.parameters():
            parameter.requires_grad = True

    backbone = getattr(model, "bert", None) or getattr(model, "roberta", None)
    if backbone is not None and hasattr(backbone, "encoder"):
        for layer in backbone.encoder.layer[-number_of_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    for module in model.modules():
        if "LayerNorm" in module.__class__.__name__:
            for parameter in module.parameters():
                parameter.requires_grad = True


def compute_class_weights(labels) -> torch.Tensor:
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(np.float32)
    if (counts == 0).any():
        raise ValueError(f"Every training class must be represented; observed counts: {counts.tolist()}")
    weights = len(labels) / (NUM_LABELS * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(evaluation):
    logits, labels = (
        evaluation if isinstance(evaluation, tuple) else (evaluation.predictions, evaluation.label_ids)
    )
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


class RDropTrainer(Trainer):
    def __init__(self, rdrop_alpha=0.0, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = float(rdrop_alpha)
        self.class_weights = class_weights

    def _get_eval_sampler(self, eval_dataset):
        """Keep validation labels and predictions in a stable one-pass order."""
        return SequentialSampler(eval_dataset)

    @staticmethod
    def _symmetric_kl(first_logits, second_logits):
        import torch.nn.functional as functional

        first_log = functional.log_softmax(first_logits, dim=-1)
        second_log = functional.log_softmax(second_logits, dim=-1)
        first_probability = first_log.exp()
        second_probability = second_log.exp()
        first_to_second = torch.sum(first_probability * (first_log - second_log), dim=-1)
        second_to_first = torch.sum(second_probability * (second_log - first_log), dim=-1)
        return (first_to_second + second_to_first).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch.nn.functional as functional

        labels = inputs.pop("labels")
        weights = self.class_weights.to(model.device) if self.class_weights is not None else None
        first_output = model(**inputs)
        first_logits = first_output.logits

        if self.rdrop_alpha > 0 and model.training:
            second_output = model(**inputs)
            second_logits = second_output.logits
            first_loss = functional.cross_entropy(first_logits, labels, weight=weights)
            second_loss = functional.cross_entropy(second_logits, labels, weight=weights)
            loss = 0.5 * (first_loss + second_loss)
            loss = loss + self.rdrop_alpha * self._symmetric_kl(first_logits, second_logits)
            outputs = (first_output, second_output)
        else:
            loss = functional.cross_entropy(first_logits, labels, weight=weights)
            outputs = first_output

        return (loss, outputs) if return_outputs else loss


def make_training_arguments(output_dir: Path, seed: int, epochs: int, device_type: str):
    common = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=PRIMARY_SCHEDULER,
        optim=OPTIMIZER_NAME,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        adam_epsilon=ADAM_EPSILON,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=epochs,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        seed=seed,
        fp16=device_type == "cuda",
        dataloader_num_workers=0,
        dataloader_pin_memory=device_type != "mps",
        report_to="none",
        group_by_length=True,
        max_grad_norm=MAX_GRAD_NORM,
    )
    try:
        return TrainingArguments(evaluation_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(eval_strategy="epoch", **common)


# -----------------------------------------------------------------------------
# Evaluation and saved outputs
# -----------------------------------------------------------------------------


def predict_in_order(model, tokenizer, test_frame: pd.DataFrame, batch_size: int = 64):
    predictions = []
    confidences = []
    model.eval()
    texts = test_frame[TEXT_COL].astype(str).tolist()

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())

    return np.asarray(predictions, dtype=int), np.asarray(confidences, dtype=float)


def clustered_bootstrap_intervals(
    prediction_frame: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict:
    """Percentile CIs obtained by resampling whole test reviews."""
    reviews = prediction_frame[REVIEW_COL].unique()
    indices_by_review = {
        review: prediction_frame.index[prediction_frame[REVIEW_COL] == review].to_numpy()
        for review in reviews
    }
    rng = np.random.default_rng(seed)
    accuracy_values = []
    macro_f1_values = []

    for _ in range(repetitions):
        sampled_reviews = rng.choice(reviews, size=len(reviews), replace=True)
        sampled_indices = np.concatenate([indices_by_review[review] for review in sampled_reviews])
        sample = prediction_frame.loc[sampled_indices]
        accuracy_values.append(accuracy_score(sample["true"], sample["pred"]))
        macro_f1_values.append(
            f1_score(
                sample["true"],
                sample["pred"],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        )

    return {
        "method": "percentile bootstrap resampling whole held-out reviews",
        "repetitions": int(repetitions),
        "seed": int(seed),
        "accuracy_95_ci": [
            float(np.percentile(accuracy_values, 2.5)),
            float(np.percentile(accuracy_values, 97.5)),
        ],
        "macro_f1_95_ci": [
            float(np.percentile(macro_f1_values, 2.5)),
            float(np.percentile(macro_f1_values, 97.5)),
        ],
    }


def full_test_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)],
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_test_outcomes": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "per_class_f1": {
            ID2LABEL[i]: float(f1_score(labels, predictions, labels=[i], average="macro"))
            for i in range(NUM_LABELS)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": report,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def run_analysis(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Review-level holdout validation")
    print("=" * 72)
    print(f"Data: {args.data}")
    print(f"New output directory: {args.output_dir}")
    print("Existing manuscript checkpoints are not modified.")

    frame, cleaning = load_and_clean(args.data)
    assigned, split_details = make_review_level_split(frame, args.seed)
    split_summary = summarize_split(assigned, cleaning, split_details)

    assignment_path = args.results_dir / "review_level_split_assignments.csv"
    summary_path = args.results_dir / "review_level_split_summary.json"
    assigned.to_csv(assignment_path, index=False)
    write_json(summary_path, split_summary)

    print(json.dumps(split_summary, indent=2))
    print(f"Split assignments: {assignment_path}")
    print(f"Split summary: {summary_path}")

    if args.split_only:
        print("Split-only mode requested; model training was not run.")
        return

    device, device_type = get_device(args.device)
    print(f"Training device: {device} ({device_type})")

    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        revision=MODEL_REVISIONS[MODEL_NAME],
        local_files_only=local_files_only,
    )
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        revision=MODEL_REVISIONS[MODEL_NAME],
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        revision=MODEL_REVISIONS[MODEL_NAME],
        local_files_only=local_files_only,
    ).to(device)
    unlock_last_blocks(model, UNFREEZE_BLOCKS)

    dataset = to_dataset_dict(assigned)
    tokenized = tokenize_dataset(dataset, tokenizer)
    class_weights = compute_class_weights(tokenized["train"]["labels"])
    print(f"Train-only class weights: {class_weights.tolist()}")

    checkpoint_dir = args.output_dir / "training_checkpoints"
    training_arguments = make_training_arguments(
        checkpoint_dir,
        args.seed,
        args.epochs,
        device_type,
    )
    trainer = RDropTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        rdrop_alpha=RDROP_ALPHA,
        class_weights=class_weights,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))
    trainer.train()

    validation_metrics = trainer.evaluate(tokenized["validation"])
    test_frame = assigned[assigned["split"] == "test"].reset_index(drop=True)
    test_labels = test_frame[LABEL_COL].to_numpy(dtype=int)
    test_predictions, confidences = predict_in_order(model, tokenizer, test_frame)

    predictions_frame = test_frame[[REVIEW_COL, OUTCOME_ID_COL, TEXT_COL, LABEL_COL]].copy()
    predictions_frame = predictions_frame.rename(columns={LABEL_COL: "true"})
    predictions_frame["pred"] = test_predictions
    predictions_frame["correct"] = predictions_frame["true"] == predictions_frame["pred"]
    predictions_frame["max_softmax_probability"] = confidences

    test_metrics = full_test_metrics(test_labels, test_predictions)
    test_metrics["clustered_bootstrap"] = clustered_bootstrap_intervals(
        predictions_frame,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed + 1,
    )

    baseline_reference = None
    if args.baseline_metrics.exists():
        with args.baseline_metrics.open(encoding="utf-8") as handle:
            baseline = json.load(handle)
        baseline_reference = {
            "source": args.baseline_metrics.name,
            "split": "manuscript item-level test split",
            "accuracy": float(baseline["accuracy"]),
            "macro_f1": float(baseline["macro_f1"]),
            "descriptive_accuracy_difference": float(test_metrics["accuracy"] - baseline["accuracy"]),
            "descriptive_macro_f1_difference": float(test_metrics["macro_f1"] - baseline["macro_f1"]),
            "note": "Descriptive only: the test sets differ and this is not a paired comparison.",
        }

    best_model_dir = args.output_dir / "best_model"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    predictions_path = args.results_dir / "review_level_test_predictions.csv"
    metrics_path = args.results_dir / "review_level_metrics.json"
    predictions_frame.to_csv(predictions_path, index=False)

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "review-level holdout validation",
        "interpretation": (
            "Group-disjoint internal validation on unseen Cochrane reviews; "
            "not validation on an independent external data source"
        ),
        "source_data": args.data.name,
        "base_model": MODEL_NAME,
        "initialization": "fresh base pretrained model; manuscript checkpoint was not used",
        "device": device_type,
        "seed": int(args.seed),
        "hyperparameters": {
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "unfreeze_blocks": UNFREEZE_BLOCKS,
            "rdrop_alpha": RDROP_ALPHA,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "maximum_epochs": int(args.epochs),
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "checkpoint_selection": "validation macro F1",
            "train_only_class_weights": class_weights.tolist(),
        },
        "split_summary": split_summary,
        "best_training_checkpoint": (
            Path(trainer.state.best_model_checkpoint).name
            if trainer.state.best_model_checkpoint
            else None
        ),
        "best_validation_macro_f1": trainer.state.best_metric,
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
        "test_metrics": test_metrics,
        "item_level_baseline_reference": baseline_reference,
        "saved_best_model": best_model_dir.name,
        "saved_test_predictions": predictions_path.name,
    }
    write_json(metrics_path, result)

    print("=" * 72)
    print("Review-level test result")
    print("=" * 72)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Best model: {best_model_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            os.getenv(
                "OUTCOME_DATASET_CSV",
                PROJECT_ROOT / "restricted_data" / "outcome_3cls.csv",
            )
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "private_outputs" / "review_separated" / "training",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "private_outputs" / "review_separated" / "results",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=PROJECT_ROOT / "results" / "primary_checkpoint_metrics.json",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading the base model if it is not already cached.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
