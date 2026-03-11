# filename: train_other_models.py
# Train additional pre-trained models for comparison
# Models: PubMedBERT, BioBERT, SciBERT

import os, json, re
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
)
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict, Value

# Configuration
SEED = 42
CSV_3CLS = "outcome_3cls.csv"
TEXT_COL = "outcome"
LABEL3 = "outcome.class"
NUM_LABELS = 3
ID2LABEL = {0: "Objective", 1: "Semi-objective", 2: "Subjective"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Models to compare
MODELS_TO_TEST = [
    "bioformers/bioformer-8L",  # Baseline
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",  # PubMedBERT
    "dmis-lab/biobert-base-cased-v1.2",  # BioBERT (if available)
    "allenai/scibert_scivocab_uncased",  # SciBERT
]

# Hyperparameters (same as finetune_high_accuracy.py)
MAX_LENGTH = 128
EPOCHS = 8
LR = 3e-5
BATCH_SIZE = 16
GRAD_ACCUM = 1
WARMUP_R = 0.06
WEIGHT_DECAY = 0.02
LABEL_SMOOTH = 0.0
PATIENCE = 2
RDROP_ALPHA = 1.0
UNFREEZE_BLOCKS = 4

OUTPUT_DIR = "./outputs_model_comparison"
RESULTS_DIR = "./results"

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

    if TEXT_COL not in raw.columns or LABEL3 not in raw.columns:
        raise ValueError(f"CSV must contain columns: '{TEXT_COL}', '{LABEL3}'")

    df = raw[[TEXT_COL, LABEL3]].copy()

    mask_na = df[TEXT_COL].isna() | df[LABEL3].isna()
    mark_removed(df[mask_na], "NaN in text or label")
    df = df[~mask_na].copy()

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    mask_empty = df[TEXT_COL].str.len() == 0
    mark_removed(df[mask_empty], "Empty text after strip")
    df = df[~mask_empty].copy()

    df, _ = normalize_label_column(df, LABEL3, report_dir=OUTPUT_DIR)

    bad_mask = ~df[LABEL3].isin([0, 1, 2])
    mark_removed(df[bad_mask], "Label unmapped")
    df = df[~bad_mask].copy()

    if not keep_duplicates:
        dup = df.duplicated(subset=[TEXT_COL, LABEL3], keep="first")
        mark_removed(df[dup], "Duplicate (text+label)")
        df = df[~dup].copy()

    if save_report and len(removed) > 0:
        rem = pd.concat(removed, ignore_index=True)
        rem.to_csv(os.path.join(OUTPUT_DIR, "data_clean_removed.csv"), index=False, encoding="utf-8")

    print("[CLEAN] Final counts:")
    print(df[LABEL3].value_counts().sort_index())
    df[LABEL3] = df[LABEL3].astype(int)
    return df

def stratified_split_70_10_20(df: pd.DataFrame) -> DatasetDict:
    train_idx, val_idx, test_idx = [], [], []
    for _, grp in df.groupby(LABEL3):
        idx = grp.index.to_numpy()
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

    tr = df.loc[train_idx, [TEXT_COL, LABEL3]].rename(columns={LABEL3: "labels"})
    va = df.loc[val_idx, [TEXT_COL, LABEL3]].rename(columns={LABEL3: "labels"})
    te = df.loc[test_idx, [TEXT_COL, LABEL3]].rename(columns={LABEL3: "labels"})

    print(f"Actual sizes -> train {len(tr)}, val {len(va)}, test {len(te)}")

    def to_ds(pdf):
        ds = Dataset.from_pandas(pdf.reset_index(drop=True), preserve_index=False)
        return ds.cast_column("labels", Value("int64"))

    return DatasetDict(train=to_ds(tr), validation=to_ds(va), test=to_ds(te))

def check_gpu_availability() -> Tuple[torch.device, str]:
    print(f"CWD: {os.getcwd()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return device, "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS backend.")
        return device, "mps"
    print("\n" + "=" * 60)
    print("WARNING: No CUDA or MPS device available. Falling back to CPU.")
    print("Training will be significantly slower but should remain correct.")
    print("=" * 60 + "\n")
    return torch.device("cpu"), "cpu"

def compute_class_weights(y_train: np.ndarray, num_labels: int) -> np.ndarray:
    y = np.asarray(y_train, dtype=np.int64)
    counts = np.bincount(y, minlength=num_labels).astype(np.float64)
    total = counts.sum()
    weights = np.zeros(num_labels, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = total / (num_labels * counts[nonzero])
    return weights

def unlock_last_blocks_and_layernorms(model, n_last_blocks: int = 4):
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    backbone = None
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        backbone = model.bert
    elif hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        backbone = model.roberta
    elif hasattr(model, "deberta") and hasattr(model.deberta, "encoder"):
        backbone = model.deberta
    elif hasattr(model, "electra") and hasattr(model.electra, "encoder"):
        backbone = model.electra

    if backbone is None:
        return

    layers = backbone.encoder.layer
    n_last = max(1, min(n_last_blocks, len(layers)))
    for layer in layers[-n_last:]:
        for p in layer.parameters():
            p.requires_grad = True

    if hasattr(backbone, "pooler"):
        for p in backbone.pooler.parameters():
            p.requires_grad = True

def train_model(model_name: str, train_tok, val_tok, test_tok, device, device_type):
    """Train a single model"""
    # Define RDropTrainer (extracted from finetune_high_accuracy.py)
    class RDropTrainer(Trainer):
        """R-Drop Trainer for consistency regularization"""
        def __init__(self, label_smoothing=0.0, rdrop_alpha=0.0, class_weights=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label_smoothing = float(label_smoothing)
            self.rdrop_alpha = float(rdrop_alpha)
            self.class_weights = None
            if class_weights is not None:
                self.class_weights = class_weights

        def _ce_loss(self, logits, labels, weight, smoothing):
            import torch.nn.functional as F
            return F.cross_entropy(logits, labels, weight=weight, label_smoothing=smoothing)

        def _kl_loss(self, p_logit, q_logit):
            """Symmetric KL divergence: KL(p||q) + KL(q||p)"""
            import torch.nn.functional as F
            import torch
            p = F.log_softmax(p_logit, dim=-1)
            q = F.log_softmax(q_logit, dim=-1)
            p_soft = p.exp()
            q_soft = q.exp()
            kl_pq = torch.sum(p_soft * (p - q), dim=-1)
            kl_qp = torch.sum(q_soft * (q - p), dim=-1)
            return (kl_pq + kl_qp).mean()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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
                loss = (ce1 + ce2) * 0.5 + self.rdrop_alpha * kl
                outputs = (outputs1, outputs2)
            else:
                loss = self._ce_loss(logits1, labels, weight, self.label_smoothing)
                outputs = outputs1

            return (loss, outputs) if return_outputs else loss
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except:
        print(f"  ⚠️  Skipping {model_name} - tokenizer not available")
        return None
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    try:
        config = AutoConfig.from_pretrained(model_name, num_labels=NUM_LABELS, 
                                           id2label=ID2LABEL, label2id=LABEL2ID)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    except Exception as e:
        print(f"  ⚠️  Skipping {model_name} - {e}")
        return None
    
    model = model.to(device)
    unlock_last_blocks_and_layernorms(model, n_last_blocks=UNFREEZE_BLOCKS)
    
    use_fp16 = False if device_type == "mps" else True
    
    y_train = np.array(train_tok["labels"])
    class_weights = torch.tensor(compute_class_weights(y_train, NUM_LABELS), dtype=torch.float32)
    
    model_safe_name = model_name.replace('/', '_')
    common_kwargs = dict(
        output_dir=os.path.join(OUTPUT_DIR, model_safe_name),
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        learning_rate=LR, lr_scheduler_type="cosine",
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_R,
        logging_steps=50, seed=SEED, fp16=use_fp16,
        max_grad_norm=1.0, report_to="none", 
        dataloader_num_workers=0,
        group_by_length=True,
    )
    
    try:
        args = TrainingArguments(evaluation_strategy="epoch", **common_kwargs)
    except TypeError:
        args = TrainingArguments(eval_strategy="epoch", **common_kwargs)
    
    def compute_metrics(eval_pred):
        logits, labels = (eval_pred if isinstance(eval_pred, tuple) 
                         else (eval_pred.predictions, eval_pred.label_ids))
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }
    
    trainer = RDropTrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=collator, compute_metrics=compute_metrics,
        label_smoothing=LABEL_SMOOTH, rdrop_alpha=RDROP_ALPHA, class_weights=class_weights
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=PATIENCE))
    
    trainer.train()
    
    test_metrics = trainer.evaluate(test_tok)
    
    model_dir = os.path.join(OUTPUT_DIR, f"{model_safe_name}_best")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    return {
        "model_name": model_name,
        "test_accuracy": float(test_metrics.get("eval_accuracy", 0)),
        "test_macro_f1": float(test_metrics.get("eval_macro_f1", 0)),
        "model_path": model_dir
    }

def main():
    device, device_type = check_gpu_availability()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)
    
    # Load data
    df = load_and_clean(CSV_3CLS, keep_duplicates=False)
    ds = stratified_split_70_10_20(df)
    
    # Tokenize (will use appropriate tokenizer for each model)
    # We'll tokenize separately for each model
    
    results = []
    
    for model_name in MODELS_TO_TEST:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            def enc(b): 
                return tokenizer(b[TEXT_COL], padding=True, truncation=True, max_length=MAX_LENGTH)
            
            train_tok = ds["train"].map(enc, batched=True, remove_columns=[TEXT_COL])
            val_tok = ds["validation"].map(enc, batched=True, remove_columns=[TEXT_COL])
            test_tok = ds["test"].map(enc, batched=True, remove_columns=[TEXT_COL])
            
            result = train_model(model_name, train_tok, val_tok, test_tok, device, device_type)
            if result:
                results.append(result)
                print(f"\n✅ {model_name}: Accuracy={result['test_accuracy']:.4f}, F1={result['test_macro_f1']:.4f}")
        except Exception as e:
            print(f"\n❌ Failed to train {model_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['model_name']:50s} Acc: {r['test_accuracy']:.4f}  F1: {r['test_macro_f1']:.4f}")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "model_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    flat_output_path = os.path.join(RESULTS_DIR, "model_comparison_results.json")
    with open(flat_output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Flat results saved to: {flat_output_path}")

if __name__ == "__main__":
    main()

