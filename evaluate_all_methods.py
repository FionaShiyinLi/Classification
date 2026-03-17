
import os, json, time, re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Value
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, cohen_kappa_score, roc_auc_score
)
from scipy import stats
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration
SEED = 42
CSV_3CLS = os.getenv("OUTCOME_DATASET_CSV", "outcome_3cls.csv")
TEXT_COL = "outcome"
LABEL3 = "outcome.class"
NUM_LABELS = 3
ID2LABEL = {0: "Objective", 1: "Semi-objective", 2: "Subjective"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
OUTPUT_DIR = "./outputs_outcome_3cls_high_acc"
CANONICAL_RESULTS_PATH = os.getenv("OUTCOME_EVALUATION_RESULTS_PATH", "results/evaluation_results.json")

# Model path conventions:
# - Primary reference checkpoint: validation-selected Bioformer-8L sweep winner
#   used for the main paper evaluation.
TUNED_4BLOCK_MODEL_PATH = os.getenv(
    "OUTCOME_REFERENCE_CHECKPOINT",
    "./results/outputs_hparam_search_base16/lr3e-05_wu0.04_uf8_rd1.0/best_model",
)
LOCAL_FILES_ONLY = os.getenv("OUTCOME_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no"}
PRIMARY_REFERENCE_RUN = {
    "artifact_id": "table1_primary_reference_bioformer8l",
    "run_family": "primary_evaluation",
    "paper_role": "Primary Bioformer-8L reference checkpoint selected by validation macro F1 in the reduced sweep",
    "nominal_hparams": {
        "learning_rate": 3e-5,
        "warmup_ratio": 0.04,
        "unfreeze_blocks": 8,
        "rdrop_alpha": 1.0,
    },
    "provenance_note": (
        "This is the validation-selected primary reference checkpoint used for the main evaluation. "
        "It was chosen by validation macro F1 within the reduced 16-run sweep and then evaluated once "
        "on the held-out test set."
    ),
}
MODEL_NAME = "bioformers/bioformer-8L"
MAX_LENGTH = 128  # Match finetune_high_accuracy.py

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OUTCOME_LLM_MODEL", "openai/gpt-5.2")

# Hybrid approach thresholds
CONFIDENCE_THRESHOLD = 0.7  # Use LLM for predictions below this confidence
USE_SAVED_TEST_PREDICTIONS = False  # Force canonical split regeneration

np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== Helper Functions ==========

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
        raise ValueError(f"CSV 必须包含列：'{TEXT_COL}', '{LABEL3}'")

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
        # Ensure indices are writable for in-place shuffling across NumPy/Pandas versions.
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

    tr = df.loc[train_idx, [TEXT_COL, LABEL3]].rename(columns={LABEL3: "labels"})
    va = df.loc[val_idx, [TEXT_COL, LABEL3]].rename(columns={LABEL3: "labels"})
    te = df.loc[test_idx, [TEXT_COL, LABEL3]].rename(columns={LABEL3: "labels"})

    print(f"Actual sizes -> train {len(tr)}, val {len(va)}, test {len(te)}")

    def to_ds(pdf):
        ds = Dataset.from_pandas(pdf.reset_index(drop=True), preserve_index=False)
        return ds.cast_column("labels", Value("int64"))

    return DatasetDict(train=to_ds(tr), validation=to_ds(va), test=to_ds(te))

def load_test_data(csv_path: str, model_path: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
    """Load test data using the exact preprocessing pipeline as training.
    
    Args:
        csv_path: Path to the CSV file
        model_path: Path to the model directory (optional). If provided, will try to load
                   test_predictions.csv from that directory first.
    
    Returns:
        Tuple of (texts, labels)
    """
    # Try to load test_predictions.csv from the model directory if enabled
    if USE_SAVED_TEST_PREDICTIONS and model_path is not None:
        preferred_path = os.path.join(model_path, "test_predictions.csv")
        legacy_path = os.path.join(os.path.dirname(model_path), "test_predictions.csv")
        candidate_paths = [preferred_path]
        if legacy_path != preferred_path:
            candidate_paths.append(legacy_path)

        for test_pred_path in candidate_paths:
            if os.path.exists(test_pred_path):
                print(f"  Loading test set from: {test_pred_path}")
                df_test = pd.read_csv(test_pred_path)
                texts = df_test["text"].tolist()
                labels = df_test["true"].values
                print(f"  Loaded {len(texts)} samples from saved test predictions")
                return texts, labels

        print(
            "  Note: test_predictions.csv not found in expected model directories "
            f"({preferred_path} or {legacy_path}). Falling back to canonical split."
        )
    elif model_path is not None:
        print("  Skipping saved test_predictions.csv and forcing canonical split regeneration.")

    # Fallback: regenerate test split using canonical pipeline
    print(f"  Regenerating test set from CSV with canonical cleaning/split (seed={SEED})...")
    df = load_and_clean(csv_path, keep_duplicates=False, save_report=False)
    ds = stratified_split_70_10_20(df)
    test_ds = ds["test"]
    texts = list(test_ds[TEXT_COL])
    labels = np.array(test_ds["labels"])
    print(f"  Generated {len(texts)} samples from canonical split")
    return texts, labels

def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval"""
    n = len(y_true)
    metrics = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        m = metric_fn(y_true[indices], y_pred[indices])
        metrics.append(m)
    alpha = 1 - confidence
    lower = np.percentile(metrics, 100 * alpha / 2)
    upper = np.percentile(metrics, 100 * (1 - alpha / 2))
    return lower, upper, np.mean(metrics)

# ========== Method 1: Fine-Tuned Model ==========

class FineTunedModelEvaluator:
    def __init__(self, model_path: str, max_length: int = 128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=LOCAL_FILES_ONLY)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=LOCAL_FILES_ONLY)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        
    def predict(self, texts: List[str], return_probs: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict with fine-tuned model"""
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.max_length,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = np.argmax(probs)
                
                predictions.append(pred)
                probabilities.append(probs)
        
        preds = np.array(predictions)
        probs = np.array(probabilities) if return_probs else None
        return preds, probs
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, return_probs: bool = False):
        """Batch prediction for efficiency"""
        all_preds = []
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)
                
                all_preds.extend(preds)
                if return_probs:
                    all_probs.extend(probs)
        
        preds = np.array(all_preds)
        probs = np.array(all_probs) if return_probs else None
        return preds, probs

# ========== Method 2: LLM API (OpenRouter) ==========

class LLMAPIEvaluator:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-5.2",
        max_retries: int = 6,
        backoff_factor: float = 1.7,
        initial_delay: float = 0.5,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Please install: pip install openai")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.max_retries = max(1, max_retries)
        self.backoff_factor = max(1.0, backoff_factor)
        self.initial_delay = max(0.1, initial_delay)
        self.few_shot_examples = [
            # Objective (mortality/fatality endpoints in this dataset)
            ("All-cause mortality", 0),
            ("Perinatal mortality", 0),
            ("Neonatal death", 0),
            ("Stillbirth", 0),
            # Semi-objective
            ("Induction of labour (all indications)", 1),
            ("Low birthweight", 1),
            ("Wound gaping - up to 10 days", 1),
            ("Analgesia use - up to day 10", 1),
            ("Cesarean section rate", 1),
            ("Hospital admission", 1),
            ("Withdrawal from study", 1),
            ("Treatment discontinuation due to adverse events", 1),
            # Subjective (increase to reflect prevalence)
            ("Pain score at day 10", 2),
            ("Maternal satisfaction", 2),
            ("Quality of life", 2),
            ("Nausea severity", 2),
            ("Patient-reported adverse events", 2),
            ("Depressive symptoms", 2),
            ("Fatigue severity", 2),
            ("Sleep quality score", 2),
        ]
    
    def build_prompt(self, texts: List[str]) -> str:

        """Build few-shot prompt"""
        system_prompt = """You are a careful classifier helping a clinical trial outcomes study. You will label trial outcome text into exactly one of three outcome types.

        Background:
        - Outcomes come from Cochrane reviews of randomized trials and are short phrases (not full sentences).
        - The goal is to classify the measurement type (objective vs semi-objective vs subjective), not whether the outcome is good or bad.
        - Focus on how objectively the outcome can be measured from a trial report, not on treatment effects.
        - This is Turner et al.'s three-part taxonomy used for outcome-stratified meta-analysis.

        Classes:
        0 = Objective | mortality/fatality endpoints with minimal interpretation in this dataset
            (e.g., all-cause mortality, perinatal mortality, neonatal death, stillbirth, case fatality)
        1 = Semi-objective | clinical events or measurements with definition/measurement variability (e.g., complications, procedures, hospitalizations, resource use, clinician-assessed findings, lab values or biomarker thresholds such as low birthweight)
        2 = Subjective | patient-reported symptoms or scales (e.g., pain, QoL, mental health, functional status) or outcomes requiring subjective interpretation.

        Decision rules:
        - If it is a patient-reported scale or symptom, choose 2.
        - If it is a clinical event or clinician assessment but has variability in diagnosis/definition, choose 1.
        - If it is a mortality/fatality endpoint (including all-cause and perinatal/neonatal forms), choose 0.
        - If you are unsure between 1 and 2, check who reports the outcome: patient-reported -> 2; clinician-reported -> 1.
        - Class prevalence in this dataset is approximately: Objective 7%, Semi-objective 44%, Subjective 49%.
        Tie-breakers for common ambiguous phrases:
        - Mortality, death, case fatality, stillbirth, or survival endpoints -> 0.
        - Hospital admission, rehospitalization, length of stay, or resource use -> 1.
        - Withdrawal from study, discontinuation due to AEs, treatment stop -> 1 unless explicitly patient-reported.
        - Patient satisfaction, pain, fatigue, nausea, or other symptoms -> 2.

        Output rules:
        - Return ONLY a JSON array of integers in {0,1,2}, one per input, same order.
        - No explanations or extra keys.
        - If ambiguous, pick the best fit.
        """

        few_shot = "\n".join([f'"{ex[0]}" -> {ex[1]}' for ex in self.few_shot_examples])

        items = "\n".join([f'{i + 1}. "{t}"' for i, t in enumerate(texts)])

        user_prompt = f"""Here are labeled examples for calibration:

        {few_shot}

        Task:
        You will be given {len(texts)} outcome texts. Each item is a short outcome phrase from a clinical trial report.
        For each item, output one label id in {{0,1,2}} that best matches the outcome type.
        Remember: classify the measurement type, not the direction or magnitude of the effect.

        Items:
        {items}

        Return exactly this JSON:
        [<label_id for item 1>, <label_id for item 2>, ..., <label_id for item {len(texts)}>]
        """
        return system_prompt, user_prompt
    
    def predict(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """Predict using LLM API - reduced batch size to avoid token limits"""
        all_predictions = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            system_prompt, user_prompt = self.build_prompt(batch)

            response = None
            attempts = 0
            for attempt in range(1, self.max_retries + 1):
                attempts = attempt
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=2000  # Reduced batch size means shorter responses
                    )
                    break
                except Exception as e:
                    if attempt == self.max_retries:
                        print(f"Error in batch {i} after {attempt} attempts: {e}")
                        response = None
                    else:
                        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
                        print(
                            f"  Batch {i//batch_size + 1}/{total_batches} attempt "
                            f"{attempt}/{self.max_retries} failed ({e}). Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
            if response is None:
                all_predictions.extend([1] * len(batch))
                continue

            content = response.choices[0].message.content.strip()
            print(
                f"  Batch {i//batch_size + 1}/{total_batches} succeeded after {attempts} attempt(s)"
            )

            # Extract JSON array - handle potential truncation
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                try:
                    preds = json.loads(json_match.group())
                    # Ensure we have the right number of predictions
                    if len(preds) == len(batch):
                        all_predictions.extend(preds)
                    else:
                        print(f"Warning: Batch {i} got {len(preds)} predictions for {len(batch)} samples")
                        # Pad or truncate to match batch size
                        if len(preds) < len(batch):
                            preds.extend([1] * (len(batch) - len(preds)))  # Pad with most common class
                        all_predictions.extend(preds[:len(batch)])
                except json.JSONDecodeError as je:
                    print(f"JSON decode error in batch {i}: {je}")
                    # Fallback: predict class 1 (most common)
                    all_predictions.extend([1] * len(batch))
            else:
                # Try to parse as-is
                try:
                    preds = json.loads(content)
                    if isinstance(preds, list) and len(preds) == len(batch):
                        all_predictions.extend(preds)
                    else:
                        raise ValueError("Invalid prediction format")
                except:
                    print(f"Could not parse response in batch {i}, using fallback")
                    all_predictions.extend([1] * len(batch))
            
            # Progress update
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(texts)} samples...")
            
            time.sleep(0.2)  # Rate limiting - slightly longer delay
        
        return np.array(all_predictions)

# ========== Method 3: Hybrid Approach ==========

class HybridEvaluator:
    def __init__(self, fine_tuned_model: FineTunedModelEvaluator, llm_evaluator: LLMAPIEvaluator, 
                 confidence_threshold: float = 0.7):
        self.fine_tuned = fine_tuned_model
        self.llm = llm_evaluator
        self.threshold = confidence_threshold
        self.stats = {"fine_tuned_only": 0, "llm_fallback": 0}
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """Hybrid prediction: use LLM for low-confidence cases"""
        # Get fine-tuned predictions with probabilities
        preds_ft, probs_ft = self.fine_tuned.predict_batch(texts, return_probs=True)
        
        # Calculate confidence (max probability)
        confidences = np.max(probs_ft, axis=1)
        low_confidence_mask = confidences < self.threshold
        
        # Use LLM for low-confidence cases
        final_preds = preds_ft.copy()
        if low_confidence_mask.sum() > 0:
            low_conf_texts = [texts[i] for i in range(len(texts)) if low_confidence_mask[i]]
            llm_preds = self.llm.predict(low_conf_texts)
            
            llm_idx = 0
            for i in range(len(texts)):
                if low_confidence_mask[i]:
                    final_preds[i] = llm_preds[llm_idx]
                    llm_idx += 1
            
            self.stats["llm_fallback"] = low_confidence_mask.sum()
        
        self.stats["fine_tuned_only"] = len(texts) - self.stats["llm_fallback"]
        
        return final_preds, self.stats

# ========== Evaluation Function ==========

def evaluate_method(y_true: np.ndarray, y_pred: np.ndarray, method_name: str) -> Dict:
    """Comprehensive evaluation"""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    
    # Bootstrap CIs
    acc_ci_lower, acc_ci_upper, _ = bootstrap_ci(accuracy_score, y_true, y_pred)
    f1_ci_lower, f1_ci_upper, _ = bootstrap_ci(
        lambda yt, yp: f1_score(yt, yp, average="macro"), y_true, y_pred
    )
    
    results = {
        "method": method_name,
        "accuracy": accuracy,
        "accuracy_ci": (acc_ci_lower, acc_ci_upper),
        "macro_f1": macro_f1,
        "macro_f1_ci": (f1_ci_lower, f1_ci_upper),
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "per_class_f1": {
            ID2LABEL[i]: f1 for i, f1 in enumerate(per_class_f1)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, 
                                                      target_names=[ID2LABEL[i] for i in range(3)],
                                                      output_dict=True)
    }
    
    return results

# ========== Main Evaluation ==========

def main():
    print("=" * 60)
    print("Comprehensive Evaluation: All Methods")
    print("=" * 60)
    print(f"Dataset CSV: {CSV_3CLS}")
    print(f"Reference checkpoint: {TUNED_4BLOCK_MODEL_PATH}")
    print(f"Local-files-only model loading: {LOCAL_FILES_ONLY}")
    
    # Load test data
    print("\n[1/5] Loading test data...")
    texts, labels = load_test_data(CSV_3CLS, TUNED_4BLOCK_MODEL_PATH)
    print(f"Test samples: {len(texts)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    all_results = []
    
    # Method 1: Fine-tuned model
    print("\n[2/5] Evaluating fine-tuned model...")
    if os.path.exists(TUNED_4BLOCK_MODEL_PATH):
        ft_evaluator = FineTunedModelEvaluator(TUNED_4BLOCK_MODEL_PATH)
        start_time = time.time()
        preds_ft, _ = ft_evaluator.predict_batch(texts, batch_size=32)
        ft_time = time.time() - start_time
        
        results_ft = evaluate_method(
            labels, preds_ft, "Fine-Tuned Bioformer-8L (validation-selected primary checkpoint)"
        )
        results_ft["inference_time"] = ft_time
        results_ft["time_per_sample"] = ft_time / len(texts)
        results_ft["model_path"] = TUNED_4BLOCK_MODEL_PATH
        results_ft["artifact_id"] = PRIMARY_REFERENCE_RUN["artifact_id"]
        results_ft["run_family"] = PRIMARY_REFERENCE_RUN["run_family"]
        results_ft["paper_role"] = PRIMARY_REFERENCE_RUN["paper_role"]
        results_ft["nominal_hparams"] = PRIMARY_REFERENCE_RUN["nominal_hparams"]
        results_ft["provenance_note"] = PRIMARY_REFERENCE_RUN["provenance_note"]
        all_results.append(results_ft)
        print(f"  Accuracy: {results_ft['accuracy']:.4f}")
        print(f"  Macro F1: {results_ft['macro_f1']:.4f}")
        print(f"  Time: {ft_time:.2f}s ({ft_time/len(texts)*1000:.2f}ms per sample)")
        
        # Validation check: warn if accuracy is suspiciously low
        if results_ft['accuracy'] < 0.85: # Lowered threshold to be more realistic
            print(f"\n  ⚠️  WARNING: Accuracy ({results_ft['accuracy']:.4f}) is lower than expected.")
            print(f"  ⚠️  Expected accuracy for this validation-selected primary checkpoint is ~91.54%.")
            print(f"  ⚠️  This may indicate:")
            print(f"      - A mismatch in test data (wrong split or preprocessing).")
            print(f"      - The wrong model checkpoint is being loaded.")
            print(f"      - A bug in the model loading or evaluation logic.")
            print(f"  ⚠️  Model path being evaluated: {TUNED_4BLOCK_MODEL_PATH}")
    else:
        print(f"  Warning: Fine-tuned model not found at {TUNED_4BLOCK_MODEL_PATH}")
        ft_evaluator = None
    
    # Method 2: LLM API
    if OPENROUTER_API_KEY:
        print("\n[3/5] Evaluating LLM API...")
        llm_evaluator = LLMAPIEvaluator(OPENROUTER_API_KEY, OPENROUTER_MODEL)
        start_time = time.time()
        preds_llm = llm_evaluator.predict(texts, batch_size=50)  # Reduced to avoid token limits
        llm_time = time.time() - start_time
        
        results_llm = evaluate_method(labels, preds_llm, f"LLM API ({OPENROUTER_MODEL})")
        results_llm["inference_time"] = llm_time
        results_llm["time_per_sample"] = llm_time / len(texts)
        all_results.append(results_llm)
        print(f"  Accuracy: {results_llm['accuracy']:.4f}")
        print(f"  Macro F1: {results_llm['macro_f1']:.4f}")
        print(f"  Time: {llm_time:.2f}s ({llm_time/len(texts)*1000:.2f}ms per sample)")
    else:
        print("\n[3/5] Skipping LLM API (no API key)")
        llm_evaluator = None
    
    # Method 3: Hybrid approach
    if ft_evaluator and llm_evaluator:
        print("\n[4/5] Evaluating hybrid approach...")
        hybrid_evaluator = HybridEvaluator(ft_evaluator, llm_evaluator, CONFIDENCE_THRESHOLD)
        start_time = time.time()
        preds_hybrid, hybrid_stats = hybrid_evaluator.predict(texts)
        hybrid_time = time.time() - start_time
        
        results_hybrid = evaluate_method(labels, preds_hybrid, "Hybrid (Fine-tuned + LLM fallback)")
        results_hybrid["inference_time"] = hybrid_time
        results_hybrid["time_per_sample"] = hybrid_time / len(texts)
        results_hybrid["hybrid_stats"] = hybrid_stats
        all_results.append(results_hybrid)
        print(f"  Accuracy: {results_hybrid['accuracy']:.4f}")
        print(f"  Macro F1: {results_hybrid['macro_f1']:.4f}")
        print(f"  Fine-tuned only: {hybrid_stats['fine_tuned_only']}")
        print(f"  LLM fallback: {hybrid_stats['llm_fallback']}")
        print(f"  Time: {hybrid_time:.2f}s")
    
    # Save results
    print("\n[5/5] Saving results...")
    output_file = CANONICAL_RESULTS_PATH
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy types to native Python types"""
        # Handle numpy integers
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        # Handle numpy floats (np.float_ removed in NumPy 2.0)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in all_results:
        print(f"\n{result['method']}:")
        print(f"  Accuracy: {result['accuracy']:.4f} (95% CI: {result['accuracy_ci'][0]:.4f}-{result['accuracy_ci'][1]:.4f})")
        print(f"  Macro F1: {result['macro_f1']:.4f} (95% CI: {result['macro_f1_ci'][0]:.4f}-{result['macro_f1_ci'][1]:.4f})")
        print(f"  Kappa: {result['kappa']:.4f}")
        if 'time_per_sample' in result:
            print(f"  Time: {result['time_per_sample']*1000:.2f}ms per sample")
        if 'model_path' in result:
            print(f"  Model: {result['model_path']}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Final validation summary
    print("\n" + "=" * 60)
    print("VALIDATION NOTES")
    print("=" * 60)
    print("Expected results (current local artifacts):")
    print("  - Primary evaluation reference checkpoint (Bioformer-8L, 8 blocks, lr=3e-5, warmup=0.04, R-Drop=1.0): ~91.54% accuracy")
    print("    Note: this is the validation-selected checkpoint for the main paper.")
    print("  - LLM API (GPT-5.2, improved prompt): ~59.29% accuracy")
    print("  - Hybrid (confidence < 0.7): ~91.23% accuracy")
    print("\nIf fine-tuned model accuracy is significantly different:")
    print("  1. Check that the correct model checkpoint is loaded (see TUNED_4BLOCK_MODEL_PATH).")
    print("  2. Verify test data preprocessing matches the canonical training split (seed=42).")
    print("  3. Ensure predictions are being read from the intended checkpoint directory.")
    print("\nAuthoritative result files for verification:")
    print(f"  - {CANONICAL_RESULTS_PATH} (canonical results for all methods)")
    print("  - results/search_results.json (hyperparameter search results)")
    print("  - results/ablation_results.json (ablation study results)")
    print("  - results/model_comparison_results.json (cross-model results)")

if __name__ == "__main__":
    main()
