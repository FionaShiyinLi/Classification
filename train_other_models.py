# filename: train_other_models.py
# Train additional pre-trained models for comparison
# Models: PubMedBERT, BioBERT, SciBERT

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

# Import from finetune_high_accuracy.py
from finetune_high_accuracy import (
    load_and_clean, stratified_split_70_10_20,
    unlock_last_blocks_and_layernorms, compute_class_weights,
    check_gpu_availability
)

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

