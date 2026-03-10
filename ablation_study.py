# filename: ablation_study.py
# Ablation study: Test which components matter most
# Tests: R-Drop, class weights, unfreezing strategy, learning rate

import os, json
import numpy as np
import pandas as pd
from typing import Dict, List
import torch
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
)
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict, Value

SEED = 42
CSV_3CLS = "outcome_3cls.csv"
TEXT_COL = "outcome"
LABEL3 = "outcome.class"
NUM_LABELS = 3
MODEL_NAME = "bioformers/bioformer-8L"
ID2LABEL = {0: "Objective", 1: "Semi-objective", 2: "Subjective"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

OUTPUT_DIR = "./outputs_ablation"
RESULTS_DIR = "./results"

# Baseline config (from finetune_high_accuracy.py)
BASELINE_CONFIG = {
    "max_length": 128,
    "epochs": 8,
    "lr": 3e-5,
    "batch_size": 16,
    "grad_accum": 1,
    "warmup_r": 0.06,
    "weight_decay": 0.02,
    "label_smooth": 0.0,
    "patience": 2,
    "rdrop_alpha": 1.0,
    "use_class_weights": True,
    "unfreeze_blocks": 2,
}

# Ablation experiments
ABLATION_EXPERIMENTS = [
    {
        "name": "Baseline (R-Drop + Class Weights + 2 blocks)",
        "rdrop_alpha": 1.0,
        "use_class_weights": True,
        "unfreeze_blocks": 2,
    },
    {
        "name": "No R-Drop",
        "rdrop_alpha": 0.0,
        "use_class_weights": True,
        "unfreeze_blocks": 2,
    },
    {
        "name": "No Class Weights",
        "rdrop_alpha": 1.0,
        "use_class_weights": False,
        "unfreeze_blocks": 2,
    },
    {
        "name": "Unfreeze 1 Block",
        "rdrop_alpha": 1.0,
        "use_class_weights": True,
        "unfreeze_blocks": 1,
    },
    {
        "name": "Unfreeze 4 Blocks",
        "rdrop_alpha": 1.0,
        "use_class_weights": True,
        "unfreeze_blocks": 4,
    },
    {
        "name": "Unfreeze 8 Blocks",
        "rdrop_alpha": 1.0,
        "use_class_weights": True,
        "unfreeze_blocks": 8,
    },
    {
        "name": "No R-Drop + No Class Weights",
        "rdrop_alpha": 0.0,
        "use_class_weights": False,
        "unfreeze_blocks": 2,
    },
]

from finetune_high_accuracy import (
    load_and_clean, stratified_split_70_10_20,
    unlock_last_blocks_and_layernorms, compute_class_weights,
    check_gpu_availability
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
        # Move class weights to device
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(model.device)
        
        outputs1 = model(**inputs)
        logits1 = outputs1.logits

        if self.rdrop_alpha > 0.0 and model.training:
            # Second forward pass (depends on dropout differences)
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

def train_ablation(config: Dict, train_tok, val_tok, test_tok, device, device_type):
    """Train with specific ablation configuration"""
    print(f"\n{'='*60}")
    print(f"Experiment: {config['name']}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS,
                                             id2label=ID2LABEL, label2id=LABEL2ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model = model.to(device)
    unlock_last_blocks_and_layernorms(model, n_last_blocks=config["unfreeze_blocks"])
    
    use_fp16 = False if device_type == "mps" else True
    
    class_weights = None
    if config["use_class_weights"]:
        y_train = np.array(train_tok["labels"])
        class_weights = torch.tensor(compute_class_weights(y_train, NUM_LABELS), dtype=torch.float32)
        print(f"  Class weights: {class_weights.numpy()}")
    else:
        print(f"  No class weights")
    
    exp_name = config["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
    common_kwargs = dict(
        output_dir=os.path.join(OUTPUT_DIR, exp_name),
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        learning_rate=BASELINE_CONFIG["lr"], lr_scheduler_type="cosine",
        per_device_train_batch_size=BASELINE_CONFIG["batch_size"],
        per_device_eval_batch_size=BASELINE_CONFIG["batch_size"],
        gradient_accumulation_steps=BASELINE_CONFIG["grad_accum"],
        num_train_epochs=BASELINE_CONFIG["epochs"],
        weight_decay=BASELINE_CONFIG["weight_decay"],
        warmup_ratio=BASELINE_CONFIG["warmup_r"],
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
        label_smoothing=BASELINE_CONFIG["label_smooth"],
        rdrop_alpha=config["rdrop_alpha"],
        class_weights=class_weights
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=BASELINE_CONFIG["patience"]))
    
    trainer.train()
    
    test_metrics = trainer.evaluate(test_tok)
    
    return {
        "experiment": config["name"],
        "config": config,
        "test_accuracy": float(test_metrics.get("eval_accuracy", 0)),
        "test_macro_f1": float(test_metrics.get("eval_macro_f1", 0)),
    }

def main():
    device, device_type = check_gpu_availability()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)
    
    # Load data
    df = load_and_clean(CSV_3CLS, keep_duplicates=False)
    ds = stratified_split_70_10_20(df)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    def enc(b):
        return tokenizer(b[TEXT_COL], padding=True, truncation=True, max_length=BASELINE_CONFIG["max_length"])
    
    train_tok = ds["train"].map(enc, batched=True, remove_columns=[TEXT_COL])
    val_tok = ds["validation"].map(enc, batched=True, remove_columns=[TEXT_COL])
    test_tok = ds["test"].map(enc, batched=True, remove_columns=[TEXT_COL])
    
    results = []
    baseline_result = None
    
    for exp_config in ABLATION_EXPERIMENTS:
        result = train_ablation(exp_config, train_tok, val_tok, test_tok, device, device_type)
        results.append(result)
        
        if "Baseline" in exp_config["name"]:
            baseline_result = result
        
        print(f"\n✅ {exp_config['name']}: Accuracy={result['test_accuracy']:.4f}, F1={result['test_macro_f1']:.4f}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    
    if baseline_result:
        baseline_acc = baseline_result["test_accuracy"]
        print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
        print(f"\nImpact of Removing Components:")
        for r in results:
            if r["experiment"] != baseline_result["experiment"]:
                diff = r["test_accuracy"] - baseline_acc
                print(f"  {r['experiment']:40s} Acc: {r['test_accuracy']:.4f}  Change: {diff:+.4f} ({diff*100:+.2f}pp)")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    flat_output_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(flat_output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Flat results saved to: {flat_output_path}")

if __name__ == "__main__":
    main()

