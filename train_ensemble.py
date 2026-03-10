# filename: train_ensemble.py
# Train multiple models with different seeds for ensemble
# Updated to use tuned hyperparameters (lr=3e-5, warmup=0.06, 4 unfrozen blocks, R-Drop α=1.0)

# Configuration
SEED_LIST = [42, 123, 456, 789, 1011]  # 5 different seeds
CSV_3CLS = "outcome_3cls.csv"
TEXT_COL = "outcome"
LABEL3 = "outcome.class"
NUM_LABELS = 3
MODEL_NAME = "bioformers/bioformer-8L"
OUTPUT_DIR = "./outputs_ensemble"
DATA_PREP_OUTPUT_DIR = "./outputs_outcome_3cls_high_acc"
RESULTS_DIR = "./results"
ID2LABEL = {0: "Objective", 1: "Semi-objective", 2: "Subjective"}

# Tuned hyperparameters (Section 3.1.3)
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

_DASHES = "".join(["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"])
_DASH_RE = re.compile(f"[{_DASHES}]")
_rng = np.random.default_rng(SEED_LIST[0])


def _canon_label_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _DASH_RE.sub("-", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_label_column(df: pd.DataFrame, label_col: str, report_dir: str):
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
        raise ValueError(f"CSV 必须包含列：'{TEXT_COL}', '{LABEL3}'")

    df = raw[[TEXT_COL, LABEL3]].copy()

    mask_na = df[TEXT_COL].isna() | df[LABEL3].isna()
    mark_removed(df[mask_na], "NaN in text or label")
    df = df[~mask_na].copy()

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    mask_empty = df[TEXT_COL].str.len() == 0
    mark_removed(df[mask_empty], "Empty text after strip")
    df = df[~mask_empty].copy()

    df, _ = normalize_label_column(df, LABEL3, report_dir=DATA_PREP_OUTPUT_DIR)

    bad_mask = ~df[LABEL3].isin([0, 1, 2])
    mark_removed(df[bad_mask], "Label unmapped")
    df = df[~bad_mask].copy()

    if not keep_duplicates:
        dup = df.duplicated(subset=[TEXT_COL, LABEL3], keep="first")
        mark_removed(df[dup], "Duplicate (text+label)")
        df = df[~dup].copy()

    if save_report and len(removed) > 0:
        rem = pd.concat(removed, ignore_index=True)
        rem.to_csv(os.path.join(DATA_PREP_OUTPUT_DIR, "data_clean_removed.csv"), index=False, encoding="utf-8")

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


def unlock_last_blocks_and_layernorms(model, n_last_blocks=2):
    """Unfreeze last n blocks + all LayerNorm + classifier head (as in working code)."""
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "score"):
        for p in model.score.parameters():
            p.requires_grad = True
    backbone = getattr(model, "bert", None) or getattr(model, "roberta", None) or getattr(model, "deberta", None)
    if backbone and hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        layers = backbone.encoder.layer
        for layer in layers[-n_last_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True
    for _, module in model.named_modules():
        if "LayerNorm" in module.__class__.__name__:
            for p in module.parameters():
                p.requires_grad = True


def compute_class_weights(y: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Standard inverse frequency weights.
    w_c = N / (K * n_c), then normalized.
    """
    n_samples = len(y)
    n_classes = num_labels
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = n_samples / (n_classes * counts)
    w = w / w.mean()
    return w


def check_gpu_availability(allow_cpu_fallback: bool = True):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"\n{'='*60}")
        print(f"GPU Setup (NVIDIA CUDA):")
        print(f"  Device: {device_name}")
        print(f"  Device Count: {device_count}")
        print(f"{'='*60}\n")
        return torch.device("cuda:0"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"\n{'='*60}")
        print(f"GPU Setup (Apple Silicon MPS):")
        print(f"  Device: Apple GPU (MPS)")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"{'='*60}\n")
        return torch.device("mps"), "mps"
    if allow_cpu_fallback:
        print(f"\n{'='*60}")
        print("WARNING: No CUDA or MPS device available. Falling back to CPU.")
        print("Training will be significantly slower but should remain correct.")
        print(f"{'='*60}\n")
        return torch.device("cpu"), "cpu"
    raise RuntimeError(
        "No GPU acceleration available. Set ALLOW_CPU_FALLBACK=1 (default) "
        "if CPU execution is acceptable for your run."
    )

def train_single_model(seed: int, train_tok, val_tok, test_tok, device, device_type):
    """Train a single model with given seed"""
    from transformers import (
        AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
        DataCollatorWithPadding, TrainingArguments, Trainer, 
        EarlyStoppingCallback, set_seed
    )
    
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
    
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Training Model {seed} (Seed: {seed})")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, 
                                       id2label=ID2LABEL, label2id={v:k for k,v in ID2LABEL.items()})
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model = model.to(device)
    unlock_last_blocks_and_layernorms(model, n_last_blocks=UNFREEZE_BLOCKS)
    
    import torch
    use_fp16 = False if device_type == "mps" else True
    
    y_train = np.array(train_tok["labels"])
    class_weights = torch.tensor(compute_class_weights(y_train, NUM_LABELS), dtype=torch.float32)
    
    # Hyperparameters matched to tuned configuration (Section 3.1.3)
    
    common_kwargs = dict(
        output_dir=os.path.join(OUTPUT_DIR, f"model_seed_{seed}"),
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        learning_rate=LR, lr_scheduler_type="cosine",
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_R,
        logging_steps=50, seed=seed, fp16=use_fp16,
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
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_tok)
    preds = trainer.predict(test_tok)
    y_pred = np.argmax(preds.predictions, axis=-1)
    
    model_dir = os.path.join(OUTPUT_DIR, f"model_seed_{seed}_best")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    return {
        "seed": seed,
        "test_accuracy": test_metrics.get("eval_accuracy", 0),
        "test_macro_f1": test_metrics.get("eval_macro_f1", 0),
        "model_path": model_dir,
        "predictions": y_pred.tolist()
    }

def ensemble_predict(model_paths: List[str], texts: List[str], max_length: int = MAX_LENGTH, batch_size: int = 32):
    """Get predictions from multiple models and ensemble using batched inference."""
    all_predictions = []

    device = torch.device("cuda" if torch.cuda.is_available() else
                         ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    for model_path in model_paths:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        predictions = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                                   max_length=max_length, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                batch_preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
                predictions.extend(int(p) for p in batch_preds)

        all_predictions.append(predictions)

    # Majority voting
    all_predictions = np.array(all_predictions)
    ensemble_preds = []
    for i in range(len(texts)):
        votes = all_predictions[:, i]
        ensemble_preds.append(int(np.bincount(votes).argmax()))

    return np.array(ensemble_preds)

def main():
    device, device_type = check_gpu_availability()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load and prepare data (same for all models)
    df = load_and_clean(CSV_3CLS, keep_duplicates=False)
    ds = stratified_split_70_10_20(df)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    def enc(b): 
        return tokenizer(b[TEXT_COL], padding=True, truncation=True, max_length=MAX_LENGTH)
    
    train_tok = ds["train"].map(enc, batched=True, remove_columns=[TEXT_COL])
    val_tok = ds["validation"].map(enc, batched=True, remove_columns=[TEXT_COL])
    test_tok = ds["test"].map(enc, batched=True, remove_columns=[TEXT_COL])
    
    # Train multiple models
    results = []
    model_paths = []
    
    for seed in SEED_LIST:
        result = train_single_model(seed, train_tok, val_tok, test_tok, device, device_type)
        results.append(result)
        model_paths.append(result["model_path"])
        print(f"\nModel {seed}: Accuracy={result['test_accuracy']:.4f}, F1={result['test_macro_f1']:.4f}")
    
    # Ensemble evaluation
    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*60}")
    
    test_texts = [ds["test"][i][TEXT_COL] for i in range(len(ds["test"]))]
    test_labels = np.array(test_tok["labels"])
    
    ensemble_preds = ensemble_predict(model_paths, test_texts)
    
    ensemble_accuracy = accuracy_score(test_labels, ensemble_preds)
    ensemble_f1 = f1_score(test_labels, ensemble_preds, average="macro")
    
    print(f"\nEnsemble Results:")
    print(f"  Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    print(f"  Macro F1: {ensemble_f1:.4f}")
    
    # Compare with individual models
    individual_accs = [r["test_accuracy"] for r in results]
    print(f"\nIndividual Model Accuracies: {[f'{acc:.4f}' for acc in individual_accs]}")
    print(f"  Mean: {np.mean(individual_accs):.4f}")
    print(f"  Std: {np.std(individual_accs):.4f}")
    print(f"  Ensemble: {ensemble_accuracy:.4f}")
    print(f"  Improvement: {ensemble_accuracy - np.mean(individual_accs):.4f} ({100*(ensemble_accuracy - np.mean(individual_accs)):.2f} percentage points)")
    
    # Save results
    ensemble_results = {
        "individual_models": results,
        "ensemble": {
            "accuracy": float(ensemble_accuracy),
            "macro_f1": float(ensemble_f1),
            "improvement_over_mean": float(ensemble_accuracy - np.mean(individual_accs))
        }
    }
    
    output_path = os.path.join(OUTPUT_DIR, "ensemble_results.json")
    with open(output_path, "w") as f:
        json.dump(ensemble_results, f, indent=2)

    flat_output_path = os.path.join(RESULTS_DIR, "ensemble_results.json")
    with open(flat_output_path, "w") as f:
        json.dump(ensemble_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Flat results saved to: {flat_output_path}")

if __name__ == "__main__":
    main()

