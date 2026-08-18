#!/usr/bin/env python3
"""Select a hybrid-routing threshold on validation data, then lock it for test.

Selection rule (the default and recommended simple analysis):
  1. Sweep prespecified confidence thresholds on the validation split.
  2. Route Bioformer predictions with max-softmax confidence < threshold to GPT.
  3. Select the threshold with the highest validation hybrid accuracy.
  4. Break ties by macro-F1, then by the lower routing rate, then lower threshold.
  5. Save the selected threshold in a lock file before the separate test command.

The output caches contain labels and predictions, but never outcome text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SEED = 42
TEXT_COLUMN = "outcome"
LABEL_COLUMN = "outcome.class"
EXPECTED_SPLIT_COUNTS = {"train": 15763, "validation": 2252, "test": 4503}
DEFAULT_THRESHOLDS = "0.00,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_DATASET = Path(
    os.getenv(
        "OUTCOME_DATASET_CSV",
        PROJECT_ROOT / "restricted_data" / "outcome_3cls.csv",
    )
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "private_outputs" / "hybrid_threshold"
CHECKPOINT_RELATIVE = Path(
    "results/outputs_hparam_search_base16/"
    "lr3e-05_wu0.04_uf8_rd1.0/best_model"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def resolve_checkpoint(value: Optional[str]) -> Path:
    if value:
        checkpoint = Path(value).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
        return checkpoint

    candidates = [PROJECT_ROOT / CHECKPOINT_RELATIVE]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Primary checkpoint not found. Supply it with --checkpoint or set "
        "OUTCOME_REFERENCE_CHECKPOINT."
    )


def checkpoint_signature(checkpoint: Path) -> Dict[str, str]:
    weights = checkpoint / "model.safetensors"
    config = checkpoint / "config.json"
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError(
            f"Expected config.json and model.safetensors in {checkpoint}"
        )
    return {
        "config_sha256": sha256_file(config),
        "weights_sha256": sha256_file(weights),
    }


def load_canonical_splits(dataset_path: Path) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Reproduce the project's fixed stratified 70/10/20 item-level split."""
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, usecols=[TEXT_COLUMN, LABEL_COLUMN])
    frame = frame.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()
    frame = frame[frame[TEXT_COLUMN].str.len() > 0].copy()
    frame[LABEL_COLUMN] = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    frame = frame[frame[LABEL_COLUMN].isin([0, 1, 2])].copy()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(int)
    frame = frame.drop_duplicates([TEXT_COLUMN, LABEL_COLUMN], keep="first")
    frame["source_row"] = frame.index.astype(int)

    rng = np.random.default_rng(SEED)
    partition_indices: Dict[str, List[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for _, group in frame.groupby(LABEL_COLUMN):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(0.70 * n))
        n_validation = int(round(0.10 * n))
        partition_indices["train"].extend(indices[:n_train].tolist())
        partition_indices["validation"].extend(
            indices[n_train : n_train + n_validation].tolist()
        )
        partition_indices["test"].extend(indices[n_train + n_validation :].tolist())

    splits: Dict[str, pd.DataFrame] = {}
    for name, indices in partition_indices.items():
        split = frame.loc[indices, ["source_row", TEXT_COLUMN, LABEL_COLUMN]].copy()
        split = split.rename(columns={LABEL_COLUMN: "true_label"}).reset_index(drop=True)
        splits[name] = split

    observed = {name: len(split) for name, split in splits.items()}
    if observed != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "The supplied file does not reproduce the canonical full-data split. "
            f"Expected {EXPECTED_SPLIT_COUNTS}; observed {observed}."
        )
    return splits, sha256_file(dataset_path)


def parse_thresholds(value: str) -> List[float]:
    try:
        thresholds = sorted({round(float(item.strip()), 10) for item in value.split(",")})
    except ValueError as error:
        raise ValueError("--thresholds must be comma-separated numbers") from error
    if not thresholds or any(item < 0.0 or item > 1.0 for item in thresholds):
        raise ValueError("Every threshold must be between 0 and 1")
    if 0.0 not in thresholds:
        thresholds.insert(0, 0.0)  # Required no-fallback comparator.
    return thresholds


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_bioformer(
    texts: Sequence[str], checkpoint: Path, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    device = choose_device()
    print(f"Running Bioformer inference on {len(texts)} items ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    ).to(device)
    model.eval()

    probabilities: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())

    probs = np.concatenate(probabilities, axis=0)
    predictions = np.argmax(probs, axis=1).astype(int)
    return predictions, probs


def load_or_create_bioformer_predictions(
    split_name: str,
    split: pd.DataFrame,
    output_dir: Path,
    checkpoint: Path,
    dataset_sha256: str,
    signature: Dict[str, str],
    batch_size: int,
) -> pd.DataFrame:
    predictions_path = output_dir / f"{split_name}_bioformer_predictions.csv"
    metadata_path = output_dir / f"{split_name}_bioformer_metadata.json"
    expected_metadata = {
        "split": split_name,
        "n_items": len(split),
        "dataset_sha256": dataset_sha256,
        **signature,
    }

    if predictions_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = pd.read_csv(predictions_path)
        identifiers_match = cached["source_row"].tolist() == split["source_row"].tolist()
        labels_match = cached["true_label"].tolist() == split["true_label"].tolist()
        if metadata == expected_metadata and identifiers_match and labels_match:
            print(f"Reusing {predictions_path.name}")
            return cached
        raise RuntimeError(
            f"Stale or incompatible cache: {predictions_path}. "
            "Use a new --output-dir rather than overwriting an audit artifact."
        )
    if predictions_path.exists() or metadata_path.exists():
        raise RuntimeError(
            f"Incomplete cache for {split_name}; use a new --output-dir."
        )

    predictions, probabilities = predict_bioformer(
        split[TEXT_COLUMN].tolist(), checkpoint, batch_size
    )
    cached = pd.DataFrame(
        {
            "source_row": split["source_row"].to_numpy(),
            "true_label": split["true_label"].to_numpy(),
            "bioformer_pred": predictions,
            "prob_objective": probabilities[:, 0],
            "prob_semi_objective": probabilities[:, 1],
            "prob_subjective": probabilities[:, 2],
            "confidence": probabilities.max(axis=1),
        }
    )
    atomic_write_csv(predictions_path, cached)
    atomic_write_json(metadata_path, expected_metadata)
    return cached


def parse_llm_response(content: str, expected_length: int) -> List[int]:
    match = re.search(r"\[.*?\]", content.strip(), re.DOTALL)
    candidate = match.group(0) if match else content.strip()
    values = json.loads(candidate)
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} LLM labels; received "
            f"{len(values) if isinstance(values, list) else 'non-list output'}"
        )
    if any(type(value) is not int or value not in (0, 1, 2) for value in values):
        raise ValueError("LLM output must contain only integer labels 0, 1, or 2")
    return values


def strict_llm_batch(helper, texts: Sequence[str], max_retries: int = 6) -> List[int]:
    """Use the existing study prompt, but never replace API failures with class 1."""
    system_prompt, user_prompt = helper.build_prompt(list(texts))
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = helper.client.chat.completions.create(
                model=helper.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            return parse_llm_response(response.choices[0].message.content, len(texts))
        except Exception as error:  # API and strict parsing errors are both retried.
            last_error = error
            if attempt < max_retries:
                time.sleep(0.5 * (1.7 ** (attempt - 1)))
    raise RuntimeError(
        "LLM batch failed after all retries; no fallback labels were inserted."
    ) from last_error


def load_or_create_llm_predictions(
    split_name: str,
    split: pd.DataFrame,
    bioformer: pd.DataFrame,
    route_threshold: float,
    output_dir: Path,
    llm_model: str,
    llm_batch_size: int,
) -> Dict[int, int]:
    needed_ids = bioformer.loc[
        bioformer["confidence"] < route_threshold, "source_row"
    ].astype(int).tolist()
    if not needed_ids:
        return {}

    prompt_file = PROJECT_ROOT / "evaluate_all_methods.py"
    prompt_sha256 = sha256_file(prompt_file)
    cache_path = output_dir / f"{split_name}_llm_predictions.csv"
    columns = ["source_row", "llm_pred", "llm_model", "prompt_sha256"]
    if cache_path.is_file():
        cache = pd.read_csv(cache_path)
        if not set(columns).issubset(cache.columns):
            raise RuntimeError(f"Invalid LLM cache format: {cache_path}")
    else:
        cache = pd.DataFrame(columns=columns)

    compatible = cache[
        (cache["llm_model"] == llm_model)
        & (cache["prompt_sha256"] == prompt_sha256)
    ].drop_duplicates("source_row", keep="last")
    prediction_map = dict(
        zip(compatible["source_row"].astype(int), compatible["llm_pred"].astype(int))
    )
    missing_ids = [source_row for source_row in needed_ids if source_row not in prediction_map]

    if missing_ids:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                f"{len(missing_ids)} {split_name} LLM predictions are missing. "
                "Set OPENROUTER_API_KEY privately and rerun; the key is never saved."
            )
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluate_all_methods import LLMAPIEvaluator

        helper = LLMAPIEvaluator(api_key=api_key, model=llm_model)
        text_by_id = split.set_index("source_row")[TEXT_COLUMN]
        print(
            f"Requesting {len(missing_ids)} {split_name} GPT predictions "
            f"in batches of {llm_batch_size}..."
        )
        for start in range(0, len(missing_ids), llm_batch_size):
            batch_ids = missing_ids[start : start + llm_batch_size]
            batch_texts = text_by_id.loc[batch_ids].tolist()
            batch_predictions = strict_llm_batch(helper, batch_texts)
            additions = pd.DataFrame(
                {
                    "source_row": batch_ids,
                    "llm_pred": batch_predictions,
                    "llm_model": llm_model,
                    "prompt_sha256": prompt_sha256,
                }
            )
            cache = pd.concat([cache, additions], ignore_index=True)
            cache = cache.drop_duplicates(
                ["source_row", "llm_model", "prompt_sha256"], keep="last"
            )
            atomic_write_csv(cache_path, cache)
            prediction_map.update(dict(zip(batch_ids, batch_predictions)))
            print(f"  completed {min(start + llm_batch_size, len(missing_ids))}/{len(missing_ids)}")
            time.sleep(0.2)
    else:
        print(f"Reusing all required {split_name} GPT predictions")

    return {source_row: prediction_map[source_row] for source_row in needed_ids}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def optional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    return accuracy(y_true, y_pred) if len(y_true) else None


def evaluate_thresholds(
    bioformer: pd.DataFrame,
    llm_predictions: Dict[int, int],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    y_true = bioformer["true_label"].to_numpy(dtype=int)
    bio_pred = bioformer["bioformer_pred"].to_numpy(dtype=int)
    confidence = bioformer["confidence"].to_numpy(dtype=float)
    source_rows = bioformer["source_row"].to_numpy(dtype=int)
    baseline_accuracy = accuracy(y_true, bio_pred)

    rows: List[Dict] = []
    for threshold in thresholds:
        routed = confidence < threshold
        retained = ~routed
        hybrid_pred = bio_pred.copy()
        if routed.any():
            missing = [int(row) for row in source_rows[routed] if int(row) not in llm_predictions]
            if missing:
                raise RuntimeError(
                    f"Missing {len(missing)} LLM predictions for threshold {threshold:.2f}"
                )
            routed_llm = np.array(
                [llm_predictions[int(row)] for row in source_rows[routed]], dtype=int
            )
            hybrid_pred[routed] = routed_llm
        else:
            routed_llm = np.array([], dtype=int)

        routed_n = int(routed.sum())
        routing_rate = routed_n / len(y_true)
        hybrid_accuracy = accuracy(y_true, hybrid_pred)
        rows.append(
            {
                "threshold": threshold,
                "coverage": 1.0 - routing_rate,
                "retained_n": int(retained.sum()),
                "routed_n": routed_n,
                "routing_rate": routing_rate,
                "retained_bioformer_accuracy": optional_accuracy(
                    y_true[retained], bio_pred[retained]
                ),
                "selective_error": (
                    1.0 - accuracy(y_true[retained], bio_pred[retained])
                    if retained.any()
                    else None
                ),
                "routed_bioformer_accuracy": optional_accuracy(
                    y_true[routed], bio_pred[routed]
                ),
                "routed_llm_accuracy": optional_accuracy(y_true[routed], routed_llm),
                "hybrid_accuracy": hybrid_accuracy,
                "hybrid_macro_f1": float(
                    f1_score(
                        y_true,
                        hybrid_pred,
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "accuracy_delta_vs_bioformer": hybrid_accuracy - baseline_accuracy,
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(
    table: pd.DataFrame, max_routing_rate: Optional[float]
) -> pd.Series:
    eligible = table.copy()
    if max_routing_rate is not None:
        if max_routing_rate < 0.0 or max_routing_rate > 1.0:
            raise ValueError("--max-routing-rate must be between 0 and 1")
        eligible = eligible[eligible["routing_rate"] <= max_routing_rate + 1e-12]
    if eligible.empty:
        raise ValueError("No candidate threshold satisfies the routing-rate constraint")
    ranked = eligible.sort_values(
        ["hybrid_accuracy", "hybrid_macro_f1", "routing_rate", "threshold"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def run_selection(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    lock_path = output_dir / "selected_threshold.json"
    test_result_path = output_dir / "locked_test_results.json"
    if test_result_path.exists():
        raise RuntimeError("The locked test result already exists; selection cannot be changed.")
    if lock_path.exists() and not args.overwrite_selection:
        raise RuntimeError(
            f"Selection is already locked at {lock_path}. Use the separate test command."
        )

    dataset_path = Path(args.dataset)
    checkpoint = resolve_checkpoint(args.checkpoint)
    splits, dataset_sha256 = load_canonical_splits(dataset_path)
    signature = checkpoint_signature(checkpoint)
    thresholds = parse_thresholds(args.thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_bio = load_or_create_bioformer_predictions(
        "validation",
        splits["validation"],
        output_dir,
        checkpoint,
        dataset_sha256,
        signature,
        args.batch_size,
    )
    llm_predictions = load_or_create_llm_predictions(
        "validation",
        splits["validation"],
        validation_bio,
        max(thresholds),
        output_dir,
        args.llm_model,
        args.llm_batch_size,
    )
    table = evaluate_thresholds(validation_bio, llm_predictions, thresholds)
    best = select_best_threshold(table, args.max_routing_rate)
    table["eligible"] = (
        True
        if args.max_routing_rate is None
        else table["routing_rate"] <= args.max_routing_rate + 1e-12
    )
    table["selected"] = np.isclose(table["threshold"], float(best["threshold"]))
    table_path = output_dir / "validation_threshold_table.csv"
    atomic_write_csv(table_path, table)

    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_threshold": float(best["threshold"]),
        "selection_rule": (
            "highest validation hybrid accuracy; tie-break by higher validation macro-F1, "
            "then lower routing rate, then lower threshold"
        ),
        "max_routing_rate": args.max_routing_rate,
        "candidate_thresholds": thresholds,
        "validation_selected_row": {
            key: (None if pd.isna(value) else float(value))
            for key, value in best.items()
            if key not in {"retained_n", "routed_n"}
        }
        | {
            "retained_n": int(best["retained_n"]),
            "routed_n": int(best["routed_n"]),
        },
        "validation_n": len(splits["validation"]),
        "dataset_sha256": dataset_sha256,
        "checkpoint_signature": signature,
        "llm_model": args.llm_model,
        "seed": SEED,
        "routing_rule": "route when maximum softmax probability is strictly below threshold",
        "threshold_table": table_path.name,
    }
    atomic_write_json(lock_path, lock)

    print(f"Validation threshold table: {table_path}")
    print(f"Locked threshold: {float(best['threshold']):.2f}")
    print(f"Validation hybrid accuracy: {float(best['hybrid_accuracy']):.4f}")
    print(f"Validation routing rate: {float(best['routing_rate']):.2%}")
    if not np.isclose(float(best["threshold"]), 0.70):
        print("IMPORTANT: 0.70 was not selected; do not describe it as validation-selected.")


def run_locked_test(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    lock_path = output_dir / "selected_threshold.json"
    result_path = output_dir / "locked_test_results.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Run the select command first; no threshold lock was found.")
    if result_path.exists():
        raise RuntimeError(
            f"Locked test result already exists at {result_path}; it will not be overwritten."
        )

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checkpoint = resolve_checkpoint(args.checkpoint)
    splits, dataset_sha256 = load_canonical_splits(Path(args.dataset))
    signature = checkpoint_signature(checkpoint)
    if dataset_sha256 != lock["dataset_sha256"]:
        raise RuntimeError("Dataset differs from the one used to lock the threshold")
    if signature != lock["checkpoint_signature"]:
        raise RuntimeError("Checkpoint differs from the one used to lock the threshold")

    threshold = float(lock["selected_threshold"])
    test_bio = load_or_create_bioformer_predictions(
        "test",
        splits["test"],
        output_dir,
        checkpoint,
        dataset_sha256,
        signature,
        args.batch_size,
    )
    llm_predictions = load_or_create_llm_predictions(
        "test",
        splits["test"],
        test_bio,
        threshold,
        output_dir,
        lock["llm_model"],
        args.llm_batch_size,
    )
    test_table = evaluate_thresholds(test_bio, llm_predictions, [threshold])
    test_row = test_table.iloc[0]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_lock": lock_path.name,
        "selected_threshold": threshold,
        "test_n": len(splits["test"]),
        "test_metrics": {
            key: (None if pd.isna(value) else float(value))
            for key, value in test_row.items()
            if key not in {"retained_n", "routed_n"}
        }
        | {
            "retained_n": int(test_row["retained_n"]),
            "routed_n": int(test_row["routed_n"]),
        },
        "dataset_sha256": dataset_sha256,
        "checkpoint_signature": signature,
        "llm_model": lock["llm_model"],
    }
    atomic_write_json(result_path, result)
    print(f"Locked test result: {result_path}")
    print(f"Test hybrid accuracy: {float(test_row['hybrid_accuracy']):.4f}")
    print(f"Test routing rate: {float(test_row['routing_rate']):.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", default=str(DEFAULT_DATASET))
        subparser.add_argument(
            "--checkpoint",
            default=os.getenv("OUTCOME_REFERENCE_CHECKPOINT"),
            help="Primary Bioformer checkpoint (auto-detected locally if omitted)",
        )
        subparser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
        subparser.add_argument("--batch-size", type=int, default=64)
        subparser.add_argument("--llm-batch-size", type=int, default=50)

    select = subparsers.add_parser("select", help="Select and lock on validation only")
    add_common(select)
    select.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    select.add_argument(
        "--max-routing-rate",
        type=float,
        default=None,
        help=(
            "Optional prespecified routing cap, e.g. 0.05. Do not add one after "
            "seeing results merely to make a preferred threshold win."
        ),
    )
    select.add_argument(
        "--llm-model", default=os.getenv("OUTCOME_LLM_MODEL", "openai/gpt-5.2")
    )
    select.add_argument(
        "--overwrite-selection",
        action="store_true",
        help="Replace a validation lock only if no test result exists",
    )
    select.set_defaults(function=run_selection)

    test = subparsers.add_parser("test", help="Evaluate the already locked threshold once")
    add_common(test)
    test.set_defaults(function=run_locked_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
