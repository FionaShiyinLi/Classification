

from __future__ import annotations

import itertools
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Value
from sklearn.metrics import accuracy_score, f1_score
import torch
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
# ------------------------------
# Global configuration
# ------------------------------

SEED = 42
CSV_3CLS = os.getenv("OUTCOME_DATASET_CSV", "outcome_3cls.csv")
TEXT_COL = "outcome"
LABEL_COL = "outcome.class"
NUM_LABELS = 3
ID2LABEL = {0: "Objective", 1: "Semi-objective", 2: "Subjective"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
OUTPUT_DIR = "./results"
MODEL_NAME = "bioformers/bioformer-8L"
PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_CHECKPOINT_PATH = str(
    Path(
        os.getenv(
            "OUTCOME_REFERENCE_CHECKPOINT",
            str(PROJECT_ROOT / "results/outputs_hparam_search_base16/lr3e-05_wu0.04_uf8_rd1.0/best_model"),
        )
    )
)
LOCAL_FILES_ONLY = os.getenv("OUTCOME_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no"}
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 8
PATIENCE = 2
WEIGHT_DECAY = 0.02
LABEL_SMOOTH = 0.0
GRAD_ACCUM = 1
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "results/outputs_hparam_search"))
RESULTS_DIR = Path("./results")
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "0"))

# Order hyperparameters so that the previously best-performing configuration appears first.
LEARNING_RATES = [3e-5, 5e-5]
WARMUP_RATIOS = [0.06, 0.04]
UNFREEZE_BLOCKS = [4, 8]
RDROP_ALPHAS = [1.0, 0.5]


_DASHES = "".join(["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"])
_DASH_RE = re.compile(f"[{_DASHES}]")
_rng = np.random.default_rng(SEED)


def _canon_label_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _DASH_RE.sub("-", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_label_column(df: pd.DataFrame, label_col: str, report_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = df[label_col].copy()
    norm_int, reasons, norm_str_seen = [], [], []
    for v in raw:
        num = None
        try:
            f = float(str(v).strip())
            if f in (0.0, 1.0, 2.0):
                num = int(f)
        except Exception:
            pass
        if num is not None:
            norm_int.append(num)
            reasons.append("as-is numeric")
            norm_str_seen.append(str(v).strip())
            continue
        s = _canon_label_str(v)
        direct = {
            "0": 0,
            "1": 1,
            "2": 2,
            "objective": 0,
            "obj": 0,
            "semi": 1,
            "semiobjective": 1,
            "semi objective": 1,
            "subjective": 2,
            "subj": 2,
        }
        if s in direct:
            mapped = direct[s]
            norm_int.append(mapped)
            reasons.append(
                {
                    0: "from string objective",
                    1: "from string semi-objective",
                    2: "from string subjective",
                }[mapped]
            )
            norm_str_seen.append(s)
            continue
        if re.search(r"\bobj(ective)?\b", s):
            norm_int.append(0)
            reasons.append("regex objective")
            norm_str_seen.append(s)
            continue
        if re.search(r"\bsemi(\s*objective)?\b", s):
            norm_int.append(1)
            reasons.append("regex semi-objective")
            norm_str_seen.append(s)
            continue
        if re.search(r"\bsubj(ective)?\b", s):
            norm_int.append(2)
            reasons.append("regex subjective")
            norm_str_seen.append(s)
            continue
        norm_int.append(np.nan)
        reasons.append("unmapped")
        norm_str_seen.append(s)

    out = df.copy()
    out[label_col + "_raw"] = raw
    out[label_col + "_norm_str"] = norm_str_seen
    out[label_col + "_norm_reason"] = reasons
    out[label_col] = pd.Series(norm_int, index=df.index, dtype="float").astype("Int64")

    report = out[[TEXT_COL, label_col + "_raw", label_col + "_norm_str", label_col + "_norm_reason", label_col]].copy()
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "label_normalization_report.csv")
    report.to_csv(report_path, index=False, encoding="utf-8")
    print(f"[LABEL] Report -> {os.path.abspath(report_path)}")
    return out, report


def load_and_clean(csv_path: str, keep_duplicates: bool = False, save_report: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    removed = []

    def mark_removed(df_sub: pd.DataFrame, reason: str):
        if df_sub is None or len(df_sub) == 0:
            return
        tmp = df_sub.copy()
        tmp["__reason__"] = reason
        removed.append(tmp)

    if TEXT_COL not in raw.columns or LABEL_COL not in raw.columns:
        raise ValueError(f"CSV 必须包含列：'{TEXT_COL}', '{LABEL_COL}'")

    df = raw[[TEXT_COL, LABEL_COL]].copy()

    mask_na = df[TEXT_COL].isna() | df[LABEL_COL].isna()
    mark_removed(df[mask_na], "NaN in text or label")
    df = df[~mask_na].copy()

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    mask_empty = df[TEXT_COL].str.len() == 0
    mark_removed(df[mask_empty], "Empty text after strip")
    df = df[~mask_empty].copy()

    df, _ = normalize_label_column(df, LABEL_COL, report_dir=OUTPUT_DIR)

    bad_mask = ~df[LABEL_COL].isin([0, 1, 2])
    mark_removed(df[bad_mask], "Label unmapped")
    df = df[~bad_mask].copy()

    if not keep_duplicates:
        dup = df.duplicated(subset=[TEXT_COL, LABEL_COL], keep="first")
        mark_removed(df[dup], "Duplicate (text+label)")
        df = df[~dup].copy()

    if save_report and len(removed) > 0:
        rem = pd.concat(removed, ignore_index=True)
        rem.to_csv(os.path.join(OUTPUT_DIR, "data_clean_removed.csv"), index=False, encoding="utf-8")

    print("[CLEAN] Final counts:")
    print(df[LABEL_COL].value_counts().sort_index())
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def stratified_split_70_10_20(df: pd.DataFrame) -> DatasetDict:
    train_idx, val_idx, test_idx = [], [], []
    for _, grp in df.groupby(LABEL_COL):
        # Ensure writable ndarray for in-place shuffle on NumPy 2.x / newer pandas.
        idx = grp.index.to_numpy(copy=True)
        _rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(0.70 * n))
        n_va = int(round(0.10 * n))
        n_te = n - n_tr - n_va
        if n >= 1 and n_tr == 0:
            n_tr = 1
            n_te = n - n_tr - n_va
        if n >= 10 and n_va == 0:
            n_va = 1
            n_te = n - n_tr - n_va
        if n_te < 0:
            give = min(n_va, -n_te)
            n_va -= give
            n_te += give
        if n_te < 0:
            give = min(max(n_tr - 1, 0), -n_te)
            n_tr -= give
            n_te += give
        train_idx += idx[:n_tr].tolist()
        val_idx += idx[n_tr:n_tr + n_va].tolist()
        test_idx += idx[n_tr + n_va:n_tr + n_va + n_te].tolist()

    tr = df.loc[train_idx, [TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "labels"})
    va = df.loc[val_idx, [TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "labels"})
    te = df.loc[test_idx, [TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "labels"})

    print(f"Actual sizes -> train {len(tr)}, val {len(va)}, test {len(te)}")

    def to_ds(pdf):
        ds = Dataset.from_pandas(pdf.reset_index(drop=True), preserve_index=False)
        return ds.cast_column("labels", Value("int64"))

    return DatasetDict(train=to_ds(tr), validation=to_ds(va), test=to_ds(te))


# ------------------------------
# Utility dataclasses
# ------------------------------


@dataclass(frozen=True)
class HyperparamSetting:
    learning_rate: float
    warmup_ratio: float
    unfreeze_blocks: int
    rdrop_alpha: float

    def slug(self) -> str:
        return (
            f"lr{self.learning_rate:.0e}_wu{self.warmup_ratio:.2f}"
            f"_uf{self.unfreeze_blocks}_rd{self.rdrop_alpha:.1f}"
        )


@dataclass
class RunResult:
    config: HyperparamSetting
    train_accuracy: float
    train_macro_f1: float
    val_accuracy: float
    val_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
    per_class_f1: Dict[str, float]
    output_dir: str
    best_model_dir: str


def evaluate_primary_reference_checkpoint(
    raw_ds: DatasetDict,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> Dict:
    reference_slug = HyperparamSetting(3e-5, 0.06, 4, 1.0).slug()
    model = AutoModelForSequenceClassification.from_pretrained(
        REFERENCE_CHECKPOINT_PATH, local_files_only=LOCAL_FILES_ONLY
    ).to(device)
    model.eval()

    texts = list(raw_ds["test"][TEXT_COL])
    labels = np.asarray(raw_ds["test"]["labels"], dtype=np.int64)
    preds: List[int] = []
    start = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=-1).tolist())

    elapsed = time.time() - start
    preds_arr = np.asarray(preds, dtype=np.int64)

    per_class = {
        ID2LABEL[i]: f1_score(labels, preds_arr, labels=[i], average="macro", zero_division=0)
        for i in range(NUM_LABELS)
    }

    return {
        "learning_rate": 3e-5,
        "warmup_ratio": 0.06,
        "unfreeze_blocks": 4,
        "rdrop_alpha": 1.0,
        "train_accuracy": None,
        "train_macro_f1": None,
        "val_accuracy": None,
        "val_macro_f1": None,
        "test_accuracy": float(accuracy_score(labels, preds_arr)),
        "test_macro_f1": float(f1_score(labels, preds_arr, average="macro")),
        "per_class_f1": per_class,
        "output_dir": str(Path(REFERENCE_CHECKPOINT_PATH).resolve().parent),
        "best_model_dir": str(Path(REFERENCE_CHECKPOINT_PATH).resolve()),
        "artifact_id": f"search_{reference_slug}",
        "run_family": "hyperparameter_search",
        "paper_role": (
            "Canonical sweep row for the exact primary Bioformer-8L reference checkpoint."
        ),
        "provenance_note": (
            "This is the exact primary Bioformer-8L reference checkpoint used in the main paper. "
            "Within the reported sweep summary, it serves as the canonical row for the "
            "lr=3e-5, warmup=0.06, unfreeze_blocks=4, rdrop_alpha=1.0 configuration."
        ),
        "inference_time": elapsed,
        "time_per_sample": elapsed / max(len(texts), 1),
        "is_primary_reference": True,
    }


# ------------------------------
# Data preparation helpers
# ------------------------------


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    def preprocess(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)

    column_names = {split: dataset[split].column_names for split in dataset}
    tokenized = DatasetDict()
    for split in dataset:
        tokenized_split = dataset[split].map(
            preprocess,
            batched=True,
            remove_columns=[col for col in column_names[split] if col != "labels"],
        )
        tokenized_split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized[split] = tokenized_split
    return tokenized


# ------------------------------
# Model utilities
# ------------------------------


def compute_metrics(eval_pred):
    logits, labels = (eval_pred if isinstance(eval_pred, tuple) else (eval_pred.predictions, eval_pred.label_ids))
    preds = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }
    for i in range(NUM_LABELS):
        metrics[f"f1_class_{i}"] = f1_score(labels, preds, labels=[i], average="macro", zero_division=0)
    return metrics


def unlock_last_blocks(model: AutoModelForSequenceClassification, n_last_blocks: int) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "score"):
        for param in model.score.parameters():
            param.requires_grad = True

    backbone = getattr(model, "bert", None) or getattr(model, "roberta", None) or getattr(model, "deberta", None)
    if backbone and hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        for layer in backbone.encoder.layer[-n_last_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    for module in model.modules():
        if "LayerNorm" in module.__class__.__name__:
            for param in module.parameters():
                param.requires_grad = True


def compute_class_weights(labels: Iterable[int]) -> torch.Tensor:
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(labels) / (NUM_LABELS * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class RDropTrainer(Trainer):
    def __init__(self, label_smoothing=0.0, rdrop_alpha=0.0, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = float(label_smoothing)
        self.rdrop_alpha = float(rdrop_alpha)
        self.class_weights = class_weights

    def _ce_loss(self, logits, labels, weight, smoothing):
        import torch.nn.functional as F
        return F.cross_entropy(logits, labels, weight=weight, label_smoothing=smoothing)

    def _kl_loss(self, p_logit, q_logit):
        import torch
        import torch.nn.functional as F

        p = F.log_softmax(p_logit, dim=-1)
        q = F.log_softmax(q_logit, dim=-1)
        p_soft = p.exp()
        q_soft = q.exp()
        kl_pq = torch.sum(p_soft * (p - q), dim=-1)
        kl_qp = torch.sum(q_soft * (q - p), dim=-1)
        return (kl_pq + kl_qp).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch

        labels = inputs.pop("labels")
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(model.device)

        outputs1 = model(**inputs)
        logits1 = outputs1.logits

        if self.rdrop_alpha > 0.0 and model.training:
            outputs2 = model(**inputs)
            logits2 = outputs2.logits
            ce1 = self._ce_loss(logits1, labels, weight, self.label_smoothing)
            ce2 = self._ce_loss(logits2, labels, weight, self.label_smoothing)
            kl = self._kl_loss(logits1, logits2)
            loss = 0.5 * (ce1 + ce2) + self.rdrop_alpha * kl
            outputs = (outputs1, outputs2)
        else:
            loss = self._ce_loss(logits1, labels, weight, self.label_smoothing)
            outputs = outputs1

        return (loss, outputs) if return_outputs else loss


# ------------------------------
# Training loop
# ------------------------------


def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def create_training_arguments(output_dir: Path, config: HyperparamSetting, device_type: str) -> TrainingArguments:
    fp16 = device_type == "cuda"
    try:
        args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=config.warmup_ratio,
            logging_steps=50,
            seed=SEED,
            fp16=fp16,
            dataloader_num_workers=0,
            report_to="none",
            group_by_length=True,
            max_grad_norm=1.0,
        )
    except TypeError:
        args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=config.warmup_ratio,
            logging_steps=50,
            seed=SEED,
            fp16=fp16,
            dataloader_num_workers=0,
            report_to="none",
            group_by_length=True,
            max_grad_norm=1.0,
        )
    return args


def run_single_setting(
    tokenized_ds: DatasetDict,
    tokenizer: AutoTokenizer,
    class_weights: torch.Tensor,
    device: torch.device,
    device_type: str,
    setting: HyperparamSetting,
    run_dir: Path,
    raw_test_texts: List[str],
) -> RunResult:
    set_seed(SEED)

    output_dir = run_dir / setting.slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config, local_files_only=LOCAL_FILES_ONLY
    )
    model.to(device)
    unlock_last_blocks(model, setting.unfreeze_blocks)

    trainer_args = create_training_arguments(output_dir, setting, device_type)

    trainer = RDropTrainer(
        model=model,
        args=trainer_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        label_smoothing=LABEL_SMOOTH,
        rdrop_alpha=setting.rdrop_alpha,
        class_weights=class_weights,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=PATIENCE))

    trainer.train()

    train_metrics = trainer.evaluate(tokenized_ds["train"])
    val_metrics = trainer.evaluate(tokenized_ds["validation"])
    test_labels = np.array(tokenized_ds["test"]["labels"])

    # Build CSV predictions from raw_test_texts in their original order to
    # guarantee text/label/pred row alignment.
    model.eval()
    ordered_test_preds: List[int] = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(raw_test_texts), batch_size):
            batch_texts = raw_test_texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            batch_preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            ordered_test_preds.extend(int(p) for p in batch_preds)
    test_preds = np.array(ordered_test_preds, dtype=np.int64)

    test_accuracy = float(accuracy_score(test_labels, test_preds))
    test_macro_f1 = float(f1_score(test_labels, test_preds, average="macro"))
    per_class = {
        ID2LABEL[i]: float(f1_score(test_labels, test_preds, labels=[i], average="macro", zero_division=0))
        for i in range(NUM_LABELS)
    }

    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Persist canonical test predictions for downstream evaluation reuse
    test_pred_df = pd.DataFrame(
        {
            "text": raw_test_texts,
            "true": test_labels,
            "pred": test_preds,
        }
    )
    test_pred_df.to_csv(best_dir / "test_predictions.csv", index=False)

    result = RunResult(
        config=setting,
        train_accuracy=train_metrics.get("eval_accuracy", float("nan")),
        train_macro_f1=train_metrics.get("eval_macro_f1", float("nan")),
        val_accuracy=val_metrics.get("eval_accuracy", float("nan")),
        val_macro_f1=val_metrics.get("eval_macro_f1", float("nan")),
        test_accuracy=test_accuracy,
        test_macro_f1=test_macro_f1,
        per_class_f1=per_class,
        output_dir=str(output_dir.resolve()),
        best_model_dir=str(best_dir.resolve()),
    )

    del trainer
    del model
    if device_type == "cuda":
        torch.cuda.empty_cache()

    return result


# ------------------------------
# Search orchestration
# ------------------------------


def build_search_space() -> List[HyperparamSetting]:
    combos = list(
        itertools.product(
            LEARNING_RATES,
            WARMUP_RATIOS,
            UNFREEZE_BLOCKS,
            RDROP_ALPHAS,
        )
    )
    if SEARCH_LIMIT > 0:
        combos = combos[:SEARCH_LIMIT]
    settings = [
        HyperparamSetting(
            learning_rate=lr,
            warmup_ratio=wr,
            unfreeze_blocks=uf,
            rdrop_alpha=rd,
        )
        for (lr, wr, uf, rd) in combos
    ]
    return settings


def main():
    set_seed(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Hyperparameter Search for Fine-Tuning")
    print("=" * 70)
    print(f"Dataset CSV: {CSV_3CLS}")
    print(f"Reference checkpoint for comparison only: {REFERENCE_CHECKPOINT_PATH}")
    print(f"Base model initialization source: {MODEL_NAME}")
    print(f"Local-files-only model loading: {LOCAL_FILES_ONLY}")

    device, device_type = get_device()
    print(f"Device: {device} ({device_type})")

    print("Loading dataset with canonical preprocessing and split...")
    df = load_and_clean(CSV_3CLS, keep_duplicates=False, save_report=False)
    hf_ds = stratified_split_70_10_20(df)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=True, local_files_only=LOCAL_FILES_ONLY
        )
        print(f"Tokenizer loaded from local cache for model id: {MODEL_NAME}")
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                REFERENCE_CHECKPOINT_PATH, use_fast=True, local_files_only=LOCAL_FILES_ONLY
            )
            print(f"Tokenizer loaded from reference checkpoint: {REFERENCE_CHECKPOINT_PATH}")
        except Exception as cache_err:
            raise RuntimeError(
                "Failed to load tokenizer locally from both cached base model id "
                f"({MODEL_NAME}) and reference checkpoint ({REFERENCE_CHECKPOINT_PATH}). "
                "Ensure the checkpoint directory exists and contains tokenizer files."
            ) from cache_err
    tokenized_ds = tokenize_dataset(hf_ds, tokenizer, MAX_LENGTH)
    class_weights = compute_class_weights(tokenized_ds["train"]["labels"])
    raw_test_texts = list(hf_ds["test"][TEXT_COL])

    search_space = build_search_space()
    print(f"Total configurations: {len(search_space)}")

    results: List[RunResult] = []
    best_result: RunResult | None = None

    for idx, setting in enumerate(search_space, start=1):
        print(f"\n[{idx}/{len(search_space)}] Running {setting.slug()}")
        result = run_single_setting(
            tokenized_ds=tokenized_ds,
            tokenizer=tokenizer,
            class_weights=class_weights,
            device=device,
            device_type=device_type,
            setting=setting,
            run_dir=OUTPUT_ROOT,
            raw_test_texts=raw_test_texts,
        )
        results.append(result)

        print(
            f"Validation macro F1: {result.val_macro_f1:.4f} | "
            f"Test macro F1: {result.test_macro_f1:.4f}"
        )

        if best_result is None or result.val_macro_f1 > best_result.val_macro_f1:
            best_result = result

    summary_path = OUTPUT_ROOT / "search_results.json"
    search_summary_rows = [
        {
            **asdict(res.config),
            "train_accuracy": res.train_accuracy,
            "train_macro_f1": res.train_macro_f1,
            "val_accuracy": res.val_accuracy,
            "val_macro_f1": res.val_macro_f1,
            "test_accuracy": res.test_accuracy,
            "test_macro_f1": res.test_macro_f1,
            "per_class_f1": res.per_class_f1,
            "output_dir": res.output_dir,
            "best_model_dir": res.best_model_dir,
            "artifact_id": f"search_{res.config.slug()}",
            "run_family": "hyperparameter_search",
            "paper_role": (
                "Single run from the hyperparameter search grid; reported for sweep comparison, "
                "not the fixed main-paper reference checkpoint."
            ),
            "provenance_note": (
                "This result comes from the hyperparameter-search workflow. Even if its nominal "
                "hyperparameters match the main reference checkpoint, it should be interpreted as "
                "a distinct run artifact."
            ),
            "is_primary_reference": False,
        }
        for res in results
    ]
    summary_data = sorted(
        search_summary_rows,
        key=lambda row: (-row["val_macro_f1"], -row["test_macro_f1"], -row["test_accuracy"]),
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    flat_summary_path = RESULTS_DIR / "search_results.json"
    with flat_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    if best_result is None:
        print("No successful runs recorded.")
        return

    print("\nBest configuration (by validation macro F1):")
    print(json.dumps(
        {
            **asdict(best_result.config),
            "val_macro_f1": best_result.val_macro_f1,
            "test_macro_f1": best_result.test_macro_f1,
            "output_dir": best_result.output_dir,
        },
        indent=2,
    ))
    print(f"\nFull results saved to: {summary_path.resolve()}")
    print(f"Flat results saved to: {flat_summary_path.resolve()}")


if __name__ == "__main__":
    main()
