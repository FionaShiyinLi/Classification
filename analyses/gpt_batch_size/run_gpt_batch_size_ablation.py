#!/usr/bin/env python3
"""GPT-5.2 prompt batch-size sensitivity analysis.

This script leaves the manuscript baseline code, results, and checkpoints
unchanged. It reuses the exact GPT prompt from ``evaluate_all_methods.py`` and
compares several numbers of outcomes per API prompt on the same test items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
load_dotenv(PROJECT_DIR / ".env")
BASELINE_SCRIPT = PROJECT_DIR / "evaluate_all_methods.py"
DEFAULT_TEST_MANIFEST = Path(
    os.getenv(
        "OUTCOME_TEST_MANIFEST",
        PROJECT_DIR / "restricted_data" / "leakage_similarity_test_items.csv",
    )
)
LOCAL_ENV_FILE = PROJECT_DIR / ".env"

LABELS = [0, 1, 2]
LABEL_NAMES = ["Objective", "Semi-objective", "Subjective"]
EXPECTED_CLASS_COUNTS = [313, 1967, 2223]
PROTOCOL_VERSION = "1.1-baseline-parser-interleaved"

# Import only the existing prompt builder and OpenRouter client configuration.
# The baseline script's main() function is not called.
sys.path.insert(0, str(PROJECT_DIR))
from evaluate_all_methods import LLMAPIEvaluator  # noqa: E402


def load_local_environment() -> None:
    """Load only this analysis's supported settings from its private .env file."""
    if not LOCAL_ENV_FILE.exists():
        return
    supported = {"OPENROUTER_API_KEY", "OUTCOME_LLM_MODEL"}
    for raw_line in LOCAL_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in supported:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[10, 25, 50],
        help="Numbers of outcomes placed in each prompt.",
    )
    parser.add_argument(
        "--reference-batch-size",
        type=int,
        default=50,
        help="Batch size used as the paired reference.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OUTCOME_LLM_MODEL", "openai/gpt-5.2"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--order-seed", type=int, default=43)
    parser.add_argument(
        "--execution-seed",
        type=int,
        default=45,
        help="Seed used to interleave requests from the batch-size conditions.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=44)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2_000)
    parser.add_argument(
        "--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "private_outputs" / "gpt_batch_size",
    )
    parser.add_argument(
        "--keep-original-order",
        action="store_true",
        help="Use the manuscript split order instead of one fixed random order.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore successful cached API responses and request them again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the design and print request counts without using the API.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_test_items(
    test_manifest: Path, order_seed: int, keep_original_order: bool
) -> pd.DataFrame:
    test = pd.read_csv(test_manifest)
    required = {"CDSR.id", "outcome.id", "outcome", "outcome.class"}
    missing = required.difference(test.columns)
    if missing:
        raise ValueError(f"Test manifest is missing columns: {sorted(missing)}")
    if len(test) != 4_503:
        raise ValueError(f"Expected 4,503 test outcomes, found {len(test):,}")

    test = test[["CDSR.id", "outcome.id", "outcome", "outcome.class"]].copy()
    test["outcome"] = test["outcome"].astype(str)
    test["outcome.class"] = test["outcome.class"].astype(int)
    counts = test["outcome.class"].value_counts().sort_index().tolist()
    if counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"Unexpected class counts: {counts}")

    test["canonical_test_index"] = np.arange(len(test))
    if not keep_original_order:
        test = test.sample(frac=1.0, random_state=order_seed).reset_index(drop=True)
    return test


def parse_predictions(
    content: str, expected: int
) -> tuple[list[Any], str, int, list[bool], int]:
    """Apply the baseline parser exactly and record where fallbacks were used."""
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if match:
        try:
            values = json.loads(match.group())
        except json.JSONDecodeError:
            predictions = [1] * expected
            return predictions, "parse_fallback", 0, [True] * expected, 0
        if not isinstance(values, list):
            predictions = [1] * expected
            return predictions, "parse_fallback", 0, [True] * expected, 0

        returned_count = len(values)
        predictions = list(values)
        if returned_count < expected:
            missing = expected - returned_count
            predictions.extend([1] * missing)
            fallback_mask = [False] * returned_count + [True] * missing
            status = "length_padded"
        elif returned_count > expected:
            predictions = predictions[:expected]
            fallback_mask = [False] * expected
            status = "length_truncated"
        else:
            fallback_mask = [False] * expected
            status = "ok"
    else:
        try:
            values = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            predictions = [1] * expected
            return predictions, "parse_fallback", 0, [True] * expected, 0
        returned_count = len(values) if isinstance(values, list) else 0
        if isinstance(values, list) and returned_count == expected:
            predictions = list(values)
            fallback_mask = [False] * expected
            status = "ok"
        else:
            # This is the baseline's no-bracket, wrong-format behavior: replace
            # the entire batch rather than padding or truncating it.
            predictions = [1] * expected
            fallback_mask = [True] * expected
            status = "format_fallback"

    invalid_count = sum(
        not isinstance(value, int) or isinstance(value, bool) or value not in LABELS
        for value in predictions
    )
    return predictions, status, returned_count, fallback_mask, invalid_count


def cached_result(
    cache_path: Path, input_hash: str, model: str, max_tokens: int
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            saved = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    expected_settings = {
        "protocol_version": PROTOCOL_VERSION,
        "input_hash": input_hash,
        "requested_model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if any(saved.get(key) != value for key, value in expected_settings.items()):
        return None
    # Transport failures should be retried on the next run.
    if saved.get("status") == "api_fallback":
        return None
    saved["from_cache"] = True
    return saved


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def request_one_batch(
    evaluator: LLMAPIEvaluator,
    texts: list[str],
    batch_size: int,
    batch_number: int,
    start_index: int,
    cache_path: Path,
    model: str,
    max_tokens: int,
    force: bool,
) -> dict[str, Any]:
    system_prompt, user_prompt = evaluator.build_prompt(texts)
    input_hash = sha256_text(system_prompt + "\n" + user_prompt)
    if not force:
        saved = cached_result(cache_path, input_hash, model, max_tokens)
        if saved is not None:
            return saved

    started_at_utc = datetime.now(timezone.utc).isoformat()
    started = time.time()
    response = None
    error_message = None
    attempts = 0
    for attempt in range(1, evaluator.max_retries + 1):
        attempts = attempt
        try:
            response = evaluator.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            break
        except Exception as error:  # API/network errors are retried as in baseline.
            error_message = str(error)
            if attempt < evaluator.max_retries:
                delay = evaluator.initial_delay * (evaluator.backoff_factor ** (attempt - 1))
                time.sleep(delay)

    elapsed = time.time() - started
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    if response is None:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "batch_size": batch_size,
            "batch_number": batch_number,
            "start_index": start_index,
            "n_items": len(texts),
            "input_hash": input_hash,
            "requested_model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "returned_model": None,
            "response_id": None,
            "finish_reason": None,
            "status": "api_fallback",
            "returned_prediction_count": 0,
            "predictions": [1] * len(texts),
            "fallback_mask": [True] * len(texts),
            "fallback_item_count": len(texts),
            "invalid_prediction_count": 0,
            "attempts": attempts,
            "latency_seconds": elapsed,
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "response_text": None,
            "error": error_message,
            "from_cache": False,
        }
        write_json_atomic(cache_path, payload)
        return payload

    content = (response.choices[0].message.content or "").strip()
    predictions, status, returned_count, fallback_mask, invalid_count = (
        parse_predictions(content, len(texts))
    )
    usage = getattr(response, "usage", None)
    choice = response.choices[0]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "batch_size": batch_size,
        "batch_number": batch_number,
        "start_index": start_index,
        "n_items": len(texts),
        "input_hash": input_hash,
        "requested_model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "returned_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "status": status,
        "returned_prediction_count": returned_count,
        "predictions": predictions,
        "fallback_mask": fallback_mask,
        "fallback_item_count": int(sum(fallback_mask)),
        "invalid_prediction_count": invalid_count,
        "attempts": attempts,
        "latency_seconds": elapsed,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "response_text": content,
        "error": None,
        "from_cache": False,
    }
    write_json_atomic(cache_path, payload)
    return payload


def assemble_batch_size_results(
    test: pd.DataFrame,
    batch_size: int,
    results: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results.sort(key=lambda item: item["batch_number"])
    predictions: list[Any] = []
    batch_numbers: list[int] = []
    positions: list[int] = []
    statuses: list[str] = []
    fallback_flags: list[bool] = []
    for result in results:
        predictions.extend(result["predictions"])
        batch_numbers.extend([result["batch_number"]] * result["n_items"])
        positions.extend(range(1, result["n_items"] + 1))
        statuses.extend([result["status"]] * result["n_items"])
        fallback_flags.extend(result["fallback_mask"])
    if len(predictions) != len(test):
        raise ValueError(f"Prediction count mismatch for batch size {batch_size}")

    per_item = test.copy()
    per_item["prediction"] = predictions
    per_item["prompt_batch_size"] = batch_size
    per_item["prompt_batch_number"] = batch_numbers
    per_item["position_in_prompt"] = positions
    per_item["request_status"] = statuses
    per_item["used_fallback"] = fallback_flags
    request_log = pd.DataFrame(results).drop(
        columns=["predictions", "response_text", "fallback_mask"]
    )
    return per_item, request_log


def run_all_batch_sizes(
    evaluator: LLMAPIEvaluator,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], float]:
    """Randomly interleave requests so no condition is tied to run time."""
    tasks: list[tuple[int, int, int, list[str], Path]] = []
    texts = test["outcome"].tolist()
    for batch_size in args.batch_sizes:
        cache_dir = args.output_dir / "cache" / f"batch_size_{batch_size}"
        for batch_number, start in enumerate(
            range(0, len(test), batch_size), start=1
        ):
            tasks.append(
                (
                    batch_size,
                    batch_number,
                    start,
                    texts[start : start + batch_size],
                    cache_dir / f"batch_{batch_number:04d}.json",
                )
            )

    rng = np.random.default_rng(args.execution_seed)
    rng.shuffle(tasks)
    counts = {size: sum(task[0] == size for task in tasks) for size in args.batch_sizes}
    print(f"Interleaved request counts: {counts} ({len(tasks):,} total)")
    results_by_size: dict[int, list[dict[str, Any]]] = {
        size: [] for size in args.batch_sizes
    }
    run_started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                request_one_batch,
                evaluator,
                batch_texts,
                batch_size,
                batch_number,
                start,
                cache_path,
                args.model,
                args.max_tokens,
                args.force,
            ): (batch_size, batch_number)
            for batch_size, batch_number, start, batch_texts, cache_path in tasks
        }
        completed = 0
        for future in as_completed(future_map):
            result = future.result()
            results_by_size[result["batch_size"]].append(result)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"  completed {completed:,}/{len(tasks):,} requests")
    run_elapsed = time.time() - run_started

    per_size_predictions: dict[int, pd.DataFrame] = {}
    request_logs: dict[int, pd.DataFrame] = {}
    for batch_size in args.batch_sizes:
        per_item, request_log = assemble_batch_size_results(
            test, batch_size, results_by_size[batch_size]
        )
        per_size_predictions[batch_size] = per_item
        request_logs[batch_size] = request_log
    return per_size_predictions, request_logs, run_elapsed


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_f1": {
            label: float(report[label]["f1-score"]) for label in LABEL_NAMES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def confusion_by_review(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    review_codes: np.ndarray,
    n_reviews: int,
) -> np.ndarray:
    matrices = np.zeros((n_reviews, 3, 3), dtype=np.int64)
    np.add.at(matrices, (review_codes, y_true, y_pred), 1)
    return matrices


def metrics_from_confusion(matrix: np.ndarray) -> tuple[float, float]:
    matrix = matrix.astype(float)
    diagonal = np.diag(matrix)
    true_counts = matrix.sum(axis=1)
    pred_counts = matrix.sum(axis=0)
    precision = np.divide(
        diagonal, pred_counts, out=np.zeros(3, dtype=float), where=pred_counts != 0
    )
    recall = np.divide(
        diagonal, true_counts, out=np.zeros(3, dtype=float), where=true_counts != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(3, dtype=float),
        where=(precision + recall) != 0,
    )
    return float(diagonal.sum() / matrix.sum()), float(f1.mean())


def clustered_bootstrap(
    matrices: dict[int, np.ndarray], reps: int, seed: int
) -> dict[int, dict[str, np.ndarray]]:
    sizes = list(matrices)
    stacked = np.stack([matrices[size] for size in sizes])
    n_reviews = stacked.shape[1]
    samples = {
        size: {
            "accuracy": np.empty(reps, dtype=float),
            "macro_f1": np.empty(reps, dtype=float),
        }
        for size in sizes
    }
    rng = np.random.default_rng(seed)
    for rep in range(reps):
        selected = rng.integers(0, n_reviews, size=n_reviews)
        weights = np.bincount(selected, minlength=n_reviews)
        replicate_matrices = np.einsum("g,mgij->mij", weights, stacked)
        for model_index, size in enumerate(sizes):
            accuracy, macro_f1 = metrics_from_confusion(replicate_matrices[model_index])
            samples[size]["accuracy"][rep] = accuracy
            samples[size]["macro_f1"][rep] = macro_f1
    return samples


def percentile_interval(values: np.ndarray) -> list[float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return [float(lower), float(upper)]


def dry_run_summary(
    evaluator: LLMAPIEvaluator, test: pd.DataFrame, args: argparse.Namespace
) -> None:
    design = {
        "n_test_outcomes": len(test),
        "n_test_reviews": int(test["CDSR.id"].nunique()),
        "class_counts": test["outcome.class"].value_counts().sort_index().to_dict(),
        "model": args.model,
        "batch_sizes": args.batch_sizes,
        "fixed_random_order": not args.keep_original_order,
        "order_seed": None if args.keep_original_order else args.order_seed,
        "requests_interleaved": True,
        "execution_seed": args.execution_seed,
        "requests": {},
    }
    for batch_size in args.batch_sizes:
        prompt_lengths = []
        for start in range(0, len(test), batch_size):
            texts = test["outcome"].iloc[start : start + batch_size].tolist()
            system_prompt, user_prompt = evaluator.build_prompt(texts)
            prompt_lengths.append(len(system_prompt) + len(user_prompt))
        design["requests"][str(batch_size)] = {
            "count": len(prompt_lengths),
            "prompt_characters_total": int(sum(prompt_lengths)),
            "prompt_characters_mean": float(np.mean(prompt_lengths)),
            "prompt_characters_max": int(max(prompt_lengths)),
        }
    print(json.dumps(design, indent=2))


def main() -> None:
    load_local_environment()
    args = parse_args()
    args.batch_sizes = sorted(set(args.batch_sizes))
    if any(size < 1 for size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.reference_batch_size not in args.batch_sizes:
        raise ValueError("The reference batch size must be included in --batch-sizes")

    test = load_test_items(
        args.test_manifest, args.order_seed, args.keep_original_order
    )
    placeholder_key = os.getenv("OPENROUTER_API_KEY", "") or "dry-run-placeholder"
    evaluator = LLMAPIEvaluator(
        placeholder_key,
        args.model,
        max_retries=args.max_retries,
    )
    if args.dry_run:
        dry_run_summary(evaluator, test, args)
        return
    if not os.getenv("OPENROUTER_API_KEY", ""):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Set it securely in the environment, "
            "then rerun this script. Do not paste the key into source code."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_size_predictions, request_logs, run_wall_seconds = run_all_batch_sizes(
        evaluator, test, args
    )
    for batch_size in args.batch_sizes:
        per_size_predictions[batch_size].to_csv(
            args.output_dir / f"predictions_batch_size_{batch_size}.csv", index=False
        )
        request_logs[batch_size].to_csv(
            args.output_dir / f"request_log_batch_size_{batch_size}.csv", index=False
        )

    # The baseline parser does not repair exact-length invalid labels. Preserve
    # that behavior, save the evidence above, and stop rather than silently
    # changing any invalid values for metric calculation.
    invalid_by_size = {
        size: int(request_logs[size]["invalid_prediction_count"].sum())
        for size in args.batch_sizes
    }
    if any(invalid_by_size.values()):
        raise RuntimeError(
            f"The API returned invalid labels: {invalid_by_size}. Raw responses and "
            "item predictions were saved; inspect them before calculating metrics."
        )
    api_failures_by_size = {
        size: int((request_logs[size]["status"] == "api_fallback").sum())
        for size in args.batch_sizes
    }
    if any(api_failures_by_size.values()):
        raise RuntimeError(
            f"API requests failed after all retries: {api_failures_by_size}. Raw logs "
            "were saved, but no summary metrics were produced. Rerun the script; "
            "successful request caches will be reused."
        )

    y_true = test["outcome.class"].to_numpy(dtype=int)
    review_codes, review_ids = pd.factorize(test["CDSR.id"], sort=True)
    matrices = {
        size: confusion_by_review(
            y_true,
            frame["prediction"].to_numpy(dtype=int),
            review_codes,
            len(review_ids),
        )
        for size, frame in per_size_predictions.items()
    }
    bootstrap = clustered_bootstrap(matrices, args.bootstrap_reps, args.bootstrap_seed)

    summary_rows = []
    point_results: dict[str, Any] = {}
    for size, frame in per_size_predictions.items():
        y_pred = frame["prediction"].to_numpy(dtype=int)
        metrics = point_metrics(y_true, y_pred)
        metrics["accuracy_95_ci"] = percentile_interval(bootstrap[size]["accuracy"])
        metrics["macro_f1_95_ci"] = percentile_interval(bootstrap[size]["macro_f1"])
        log = request_logs[size]
        status_counts = log["status"].value_counts().to_dict()
        metrics["request_status_counts"] = {
            str(key): int(value) for key, value in status_counts.items()
        }
        metrics["api_fallback_requests"] = int((log["status"] == "api_fallback").sum())
        metrics["non_ok_requests"] = int((log["status"] != "ok").sum())
        metrics["non_ok_request_rate"] = metrics["non_ok_requests"] / len(log)
        metrics["fallback_items"] = int(log["fallback_item_count"].sum())
        metrics["fallback_item_rate"] = metrics["fallback_items"] / len(frame)
        metrics["api_fallback_items"] = int(
            log.loc[log["status"] == "api_fallback", "n_items"].sum()
        )
        metrics["non_ok_items"] = int((frame["request_status"] != "ok").sum())
        metrics["prompt_tokens"] = int(log["prompt_tokens"].fillna(0).sum())
        metrics["completion_tokens"] = int(log["completion_tokens"].fillna(0).sum())
        metrics["missing_token_usage_requests"] = int(log["total_tokens"].isna().sum())
        metrics["request_latency_seconds_sum"] = float(log["latency_seconds"].sum())
        metrics["returned_model_counts"] = {
            str(key): int(value)
            for key, value in log["returned_model"].fillna("missing").value_counts().items()
        }
        metrics["finish_reason_counts"] = {
            str(key): int(value)
            for key, value in log["finish_reason"].fillna("missing").value_counts().items()
        }
        point_results[str(size)] = metrics
        summary_rows.append(
            {
                "batch_size": size,
                "accuracy": metrics["accuracy"],
                "accuracy_ci_lower": metrics["accuracy_95_ci"][0],
                "accuracy_ci_upper": metrics["accuracy_95_ci"][1],
                "macro_f1": metrics["macro_f1"],
                "macro_f1_ci_lower": metrics["macro_f1_95_ci"][0],
                "macro_f1_ci_upper": metrics["macro_f1_95_ci"][1],
                "weighted_f1": metrics["weighted_f1"],
                "kappa": metrics["kappa"],
                "non_ok_requests": metrics["non_ok_requests"],
                "non_ok_request_rate": metrics["non_ok_request_rate"],
                "api_fallback_requests": metrics["api_fallback_requests"],
                "non_ok_items": metrics["non_ok_items"],
                "fallback_items": metrics["fallback_items"],
                "fallback_item_rate": metrics["fallback_item_rate"],
                "api_fallback_items": metrics["api_fallback_items"],
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "missing_token_usage_requests": metrics[
                    "missing_token_usage_requests"
                ],
                "request_latency_seconds_sum": metrics["request_latency_seconds_sum"],
            }
        )

    reference = args.reference_batch_size
    pairwise_rows = []
    reference_pred = per_size_predictions[reference]["prediction"].to_numpy(dtype=int)
    reference_metrics = point_results[str(reference)]
    for size in args.batch_sizes:
        if size == reference:
            continue
        prediction = per_size_predictions[size]["prediction"].to_numpy(dtype=int)
        for metric in ("accuracy", "macro_f1"):
            difference_samples = bootstrap[size][metric] - bootstrap[reference][metric]
            pairwise_rows.append(
                {
                    "comparison": f"batch {size} minus batch {reference}",
                    "metric": metric,
                    "difference": point_results[str(size)][metric]
                    - reference_metrics[metric],
                    "difference_ci_lower": percentile_interval(difference_samples)[0],
                    "difference_ci_upper": percentile_interval(difference_samples)[1],
                    "prediction_agreement": float(np.mean(prediction == reference_pred)),
                    "prediction_changes": int(np.sum(prediction != reference_pred)),
                }
            )

    summary_frame = pd.DataFrame(summary_rows).sort_values("batch_size")
    pairwise_frame = pd.DataFrame(pairwise_rows)
    summary_frame.to_csv(args.output_dir / "batch_size_summary.csv", index=False)
    pairwise_frame.to_csv(
        args.output_dir / f"pairwise_vs_batch_{reference}.csv", index=False
    )

    analysis = {
        "analysis": "GPT-5.2 prompt batch-size sensitivity analysis",
        "protocol_version": PROTOCOL_VERSION,
        "interpretation": (
            "Point estimates compare the same outcomes in one fixed order. "
            "Confidence intervals resample Cochrane reviews as clusters. "
            "Pairwise comparisons are exploratory and are not multiplicity-adjusted."
        ),
        "n_test_outcomes": len(test),
        "n_test_reviews": int(test["CDSR.id"].nunique()),
        "class_counts": test["outcome.class"].value_counts().sort_index().to_dict(),
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "batch_sizes": args.batch_sizes,
        "reference_batch_size": reference,
        "workers": args.workers,
        "requests_interleaved": True,
        "execution_seed": args.execution_seed,
        "current_invocation_wall_seconds": run_wall_seconds,
        "fixed_random_order": not args.keep_original_order,
        "order_seed": None if args.keep_original_order else args.order_seed,
        "bootstrap_replicates": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "baseline_script": BASELINE_SCRIPT.name,
        "baseline_script_sha256": sha256_file(BASELINE_SCRIPT),
        "test_manifest": args.test_manifest.name,
        "test_manifest_sha256": sha256_file(args.test_manifest),
        "point_results": point_results,
        "pairwise_vs_reference": pairwise_rows,
        "limitations": (
            "Temperature zero does not guarantee deterministic API output. The original "
            "item-level GPT predictions were not saved, so batch 50 is rerun contemporaneously. "
            "The fixed shuffled order differs from the class-grouped historical run. Findings "
            "apply to this model alias, prompt, dataset, item order, and run date."
        ),
    }
    write_json_atomic(args.output_dir / "batch_size_ablation.json", analysis)

    print("\nBatch-size summary")
    print(summary_frame.to_string(index=False))
    print(f"\nPaired differences versus batch {reference}")
    print(pairwise_frame.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
