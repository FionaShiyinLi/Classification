#!/usr/bin/env python3
"""Run an isolated GPT-5.2 taxonomy-prompt experiment.

This module never reads or writes the historical LLM result file. Validation
is used to compare reasoning settings; the test phase requires a frozen
validation-selected configuration and an explicit confirmation flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
REPO_ROOT = SCRIPT_DIR.parents[1]
load_dotenv(REPO_ROOT / ".env")
DEFAULT_PROMPT = SCRIPT_DIR / "prompt.txt"
DEFAULT_EXAMPLES = SCRIPT_DIR / "few_shot_examples.json"
DEFAULT_DATASET = Path(
    os.getenv(
        "OUTCOME_DATASET_CSV",
        REPO_ROOT / "restricted_data" / "outcome_3cls.csv",
    )
)
PRIVATE_OUTPUT_ROOT = REPO_ROOT / "private_outputs" / "gpt_prompt_sensitivity"
DEFAULT_OPENAI_OUTPUT_DIR = PRIVATE_OUTPUT_ROOT / "openai_10_examples"
DEFAULT_OPENROUTER_OUTPUT_DIR = Path(
    PRIVATE_OUTPUT_ROOT / "openrouter_10_examples"
)
DEFAULT_OPENROUTER_NON_ZDR_OUTPUT_DIR = Path(
    PRIVATE_OUTPUT_ROOT / "openrouter_non_zdr_10_examples"
)
DEFAULT_SAVED_TEST = Path(
    REPO_ROOT / "results/outputs_hparam_search_base16/"
    "lr3e-05_wu0.04_uf8_rd1.0/best_model/test_predictions.csv"
)
DEFAULT_MODEL = "gpt-5.2-2025-12-11"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.2"
OPENROUTER_CANONICAL_SLUG = "openai/gpt-5.2-20251211"
DIRECT_OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEXT_COL = "outcome"
LABEL_COL = "outcome.class"
LABELS = [0, 1, 2]
LABEL_NAMES = ["Objective", "Semi-objective", "Subjective"]
SPLIT_SEED = 42
BOOTSTRAP_SEED = 20260812
EXPECTED_COUNTS = {
    "cleaned": 22518,
    "train": 15763,
    "validation": 2252,
    "test": 4503,
}
SMOKE_ITEMS = [
    {"id": "outcome_000001", "text": "All-cause mortality", "label": 0},
    {
        "id": "outcome_000002",
        "text": "Withdrawal due to adverse effects",
        "label": 1,
    },
    {
        "id": "outcome_000003",
        "text": "Percentage achieving HbA1c ≤7%",
        "label": 2,
    },
]
NORMALIZE_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ExperimentError(RuntimeError):
    """Raised when an experiment invariant is not satisfied."""


class BatchValidationError(ExperimentError):
    """Raised when an API response fails strict post-response validation."""


def provider_settings(provider: str) -> dict[str, Any]:
    if provider == "openai":
        return {
            "name": "OpenAI",
            "base_url": DIRECT_OPENAI_BASE_URL,
            "default_model": DEFAULT_MODEL,
            "default_output_dir": DEFAULT_OPENAI_OUTPUT_DIR,
            "model_pin": "dated OpenAI model snapshot",
        }
    if provider == "openrouter":
        return {
            "name": "OpenRouter",
            "base_url": OPENROUTER_BASE_URL,
            "default_model": DEFAULT_OPENROUTER_MODEL,
            "default_output_dir": DEFAULT_OPENROUTER_OUTPUT_DIR,
            "model_pin": (
                "OpenRouter model slug; live catalog canonical slug recorded as "
                f"{OPENROUTER_CANONICAL_SLUG}"
            ),
        }
    raise ExperimentError(f"Unsupported API provider: {provider}")


def request_routing_policy(
    api_provider: str, openrouter_allow_non_zdr: bool
) -> dict[str, Any] | None:
    if api_provider != "openrouter":
        return None
    policy: dict[str, Any] = {
        "require_parameters": True,
        "data_collection": "deny",
    }
    if not openrouter_allow_non_zdr:
        policy["zdr"] = True
    return policy


@dataclass(frozen=True)
class CanonicalData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    dataset_sha256: str
    split_sha256: dict[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def normalize_text(value: Any) -> str:
    return " ".join(NORMALIZE_TOKEN_RE.findall(str(value).lower()))


def normalize_label(value: Any) -> int | None:
    try:
        numeric = float(str(value).strip())
        if numeric in (0.0, 1.0, 2.0):
            return int(numeric)
    except (TypeError, ValueError):
        pass

    normalized = normalize_text(value)
    direct = {
        "objective": 0,
        "obj": 0,
        "semi": 1,
        "semi objective": 1,
        "semiobjective": 1,
        "subjective": 2,
        "subj": 2,
    }
    return direct.get(normalized)


def hash_split(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for text, label in zip(frame[TEXT_COL], frame["labels"]):
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_canonical_data(dataset_path: Path) -> CanonicalData:
    raw = pd.read_csv(dataset_path)
    missing = {TEXT_COL, LABEL_COL}.difference(raw.columns)
    if missing:
        raise ExperimentError(f"Dataset is missing columns: {sorted(missing)}")

    frame = raw[[TEXT_COL, LABEL_COL]].copy()
    frame = frame[frame[TEXT_COL].notna() & frame[LABEL_COL].notna()].copy()
    frame[TEXT_COL] = frame[TEXT_COL].astype(str).str.strip()
    frame = frame[frame[TEXT_COL].str.len() > 0].copy()
    frame[LABEL_COL] = frame[LABEL_COL].map(normalize_label)
    frame = frame[frame[LABEL_COL].isin(LABELS)].copy()
    frame[LABEL_COL] = frame[LABEL_COL].astype(int)
    frame = frame.drop_duplicates(subset=[TEXT_COL, LABEL_COL], keep="first")

    rng = np.random.default_rng(SPLIT_SEED)
    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for _, group in frame.groupby(LABEL_COL):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        count = len(indices)
        train_count = int(round(0.70 * count))
        validation_count = int(round(0.10 * count))
        test_count = count - train_count - validation_count
        if count >= 1 and train_count == 0:
            train_count = 1
            test_count = count - train_count - validation_count
        if count >= 10 and validation_count == 0:
            validation_count = 1
            test_count = count - train_count - validation_count
        if test_count < 0:
            give = min(validation_count, -test_count)
            validation_count -= give
            test_count += give
        if test_count < 0:
            give = min(max(train_count - 1, 0), -test_count)
            train_count -= give
            test_count += give

        split_indices["train"].extend(indices[:train_count].tolist())
        split_indices["validation"].extend(
            indices[train_count : train_count + validation_count].tolist()
        )
        split_indices["test"].extend(
            indices[
                train_count
                + validation_count : train_count
                + validation_count
                + test_count
            ].tolist()
        )

    def make_split(name: str) -> pd.DataFrame:
        result = frame.loc[split_indices[name], [TEXT_COL, LABEL_COL]].copy()
        result = result.rename(columns={LABEL_COL: "labels"}).reset_index(drop=True)
        result["labels"] = result["labels"].astype(int)
        return result

    splits = {
        "train": make_split("train"),
        "validation": make_split("validation"),
        "test": make_split("test"),
    }
    observed = {
        "cleaned": len(frame),
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
    }
    if observed != EXPECTED_COUNTS:
        raise ExperimentError(
            "Canonical dataset counts differ from the prespecified experiment: "
            f"observed={observed}, expected={EXPECTED_COUNTS}."
        )

    return CanonicalData(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        dataset_sha256=sha256_file(dataset_path),
        split_sha256={name: hash_split(value) for name, value in splits.items()},
    )


def verify_saved_primary_test_split(data: CanonicalData, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    saved = pd.read_csv(path)
    missing = {"text", "true"}.difference(saved.columns)
    if missing:
        raise ExperimentError(
            f"Saved primary predictions are missing columns: {sorted(missing)}"
        )
    texts_match = saved["text"].astype(str).tolist() == data.test[TEXT_COL].tolist()
    labels_match = np.array_equal(
        saved["true"].to_numpy(dtype=int), data.test["labels"].to_numpy(dtype=int)
    )
    if not texts_match or not labels_match:
        raise ExperimentError(
            "Rebuilt test split does not match the saved primary-checkpoint rows."
        )
    return {
        "available": True,
        "path": str(path),
        "rows": int(len(saved)),
        "texts_match": True,
        "labels_match": True,
    }


def load_examples(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ExperimentError("Few-shot example file must contain a non-empty list.")
    validate_items(examples, require_labels=True)
    return payload, examples


def validate_items(items: Sequence[dict[str, Any]], require_labels: bool) -> None:
    ids: list[str] = []
    for index, item in enumerate(items):
        required = {"id", "text"} | ({"label"} if require_labels else set())
        missing = required.difference(item)
        if missing:
            raise ExperimentError(f"Item {index} is missing fields: {sorted(missing)}")
        if not isinstance(item["id"], str) or not item["id"]:
            raise ExperimentError(f"Item {index} has an invalid ID.")
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ExperimentError(f"Item {index} has invalid text.")
        if require_labels and type(item["label"]) is not int:
            raise ExperimentError(f"Item {index} label must be an integer.")
        if require_labels and item["label"] not in LABELS:
            raise ExperimentError(f"Item {index} label is outside {LABELS}.")
        ids.append(item["id"])
    if len(ids) != len(set(ids)):
        raise ExperimentError("Item IDs must be unique.")


def audit_examples(
    examples: Sequence[dict[str, Any]], data: CanonicalData
) -> list[dict[str, Any]]:
    normalized_by_split: dict[str, dict[str, list[int]]] = {}
    for split_name in ("train", "validation", "test"):
        lookup: dict[str, list[int]] = {}
        frame = getattr(data, split_name)
        for text, label in zip(frame[TEXT_COL], frame["labels"]):
            lookup.setdefault(normalize_text(text), []).append(int(label))
        normalized_by_split[split_name] = lookup

    audit: list[dict[str, Any]] = []
    for example in examples:
        normalized = normalize_text(example["text"])
        exact_matches = {
            split: normalized_by_split[split].get(normalized, [])
            for split in ("train", "validation", "test")
        }
        if exact_matches["validation"] or exact_matches["test"]:
            raise ExperimentError(
                f"Calibration example {example['id']} exactly matches a validation "
                "or test outcome after normalization."
            )
        if exact_matches["train"] and any(
            label != example["label"] for label in exact_matches["train"]
        ):
            raise ExperimentError(
                f"Calibration example {example['id']} conflicts with its exact "
                "training-data match."
            )
        audit.append(
            {
                "id": example["id"],
                "normalized_text_sha256": sha256_bytes(normalized.encode("utf-8")),
                "exact_match_splits": [
                    split for split, labels in exact_matches.items() if labels
                ],
                "exact_match_labels": exact_matches,
            }
        )
    return audit


def make_split_items(frame: pd.DataFrame, split_name: str) -> list[dict[str, Any]]:
    width = max(6, len(str(len(frame))))
    return [
        {
            "id": f"{split_name}_{index:0{width}d}",
            "text": str(text),
            "label": int(label),
        }
        for index, (text, label) in enumerate(
            zip(frame[TEXT_COL], frame["labels"]), start=1
        )
    ]


def batch_user_message(items: Sequence[dict[str, Any]]) -> str:
    request_items = [{"id": item["id"], "text": item["text"]} for item in items]
    return (
        "Classify the following outcome descriptions.\n\n"
        "<outcomes>\n"
        + json.dumps(request_items, ensure_ascii=False, indent=2)
        + "\n</outcomes>"
    )


def expected_assistant_message(examples: Sequence[dict[str, Any]]) -> str:
    payload = {
        "predictions": [
            {"id": item["id"], "label": int(item["label"])} for item in examples
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def response_schema(ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": list(ids)},
                        "label": {"type": "integer", "enum": LABELS},
                    },
                    "required": ["id", "label"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }


def validate_prediction_payload(
    payload: Any, expected_ids: Sequence[str]
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"predictions"}:
        raise BatchValidationError(
            "Response must be an object containing only the predictions key."
        )
    predictions = payload["predictions"]
    if not isinstance(predictions, list) or len(predictions) != len(expected_ids):
        raise BatchValidationError(
            f"Expected {len(expected_ids)} predictions, received "
            f"{len(predictions) if isinstance(predictions, list) else 'non-list'}."
        )

    prediction_by_id: dict[str, int] = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or set(prediction) != {"id", "label"}:
            raise BatchValidationError(
                "Each prediction must contain exactly id and label."
            )
        prediction_id = prediction["id"]
        label = prediction["label"]
        if not isinstance(prediction_id, str):
            raise BatchValidationError("Prediction IDs must be strings.")
        if prediction_id in prediction_by_id:
            raise BatchValidationError(f"Duplicate prediction ID: {prediction_id}")
        if type(label) is not int or label not in LABELS:
            raise BatchValidationError(
                f"Prediction {prediction_id} has invalid label {label!r}."
            )
        prediction_by_id[prediction_id] = label

    expected_set = set(expected_ids)
    observed_set = set(prediction_by_id)
    if observed_set != expected_set:
        raise BatchValidationError(
            "Prediction IDs did not exactly match input IDs. "
            f"missing={sorted(expected_set - observed_set)}, "
            f"unexpected={sorted(observed_set - expected_set)}"
        )
    return [
        {"id": prediction_id, "label": prediction_by_id[prediction_id]}
        for prediction_id in expected_ids
    ]


def dump_sdk_object(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: dump_sdk_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump_sdk_object(item) for item in value]
    return value


def build_manifest(
    *,
    phase: str,
    reasoning_effort: str,
    api_provider: str,
    openrouter_allow_non_zdr: bool,
    model: str,
    batch_size: int,
    max_output_tokens: int,
    prompt_path: Path,
    examples_path: Path,
    data: CanonicalData,
) -> dict[str, Any]:
    settings = provider_settings(api_provider)
    try:
        import openai

        openai_version = openai.__version__
    except ImportError:
        openai_version = None
    return {
        "experiment_role": "exploratory; separate from the historical GPT experiment",
        "phase": phase,
        "provider": settings["name"],
        "endpoint": "Responses API",
        "base_url": settings["base_url"],
        "model_requested": model,
        "model_pin": settings["model_pin"],
        "openrouter_canonical_slug_at_implementation": (
            OPENROUTER_CANONICAL_SLUG if api_provider == "openrouter" else None
        ),
        "reasoning_effort": reasoning_effort,
        "batch_size": batch_size,
        "max_output_tokens": max_output_tokens,
        "temperature": "not set",
        "store": False,
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "few_shot_path": str(examples_path),
        "few_shot_sha256": sha256_file(examples_path),
        "few_shot_delivery": "separate user/assistant calibration exchange",
        "target_batch_delivery": "separate final user message",
        "structured_output": "strict JSON Schema plus exact post-response ID validation",
        "routing_policy": request_routing_policy(
            api_provider, openrouter_allow_non_zdr
        ),
        "openrouter_non_zdr_acknowledged": (
            openrouter_allow_non_zdr if api_provider == "openrouter" else None
        ),
        "split_seed": SPLIT_SEED,
        "dataset_sha256": data.dataset_sha256,
        "split_sha256": data.split_sha256,
        "python_version": platform.python_version(),
        "openai_sdk_version": openai_version,
    }


def manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    identity_keys = [
        "phase",
        "provider",
        "endpoint",
        "base_url",
        "model_requested",
        "reasoning_effort",
        "batch_size",
        "max_output_tokens",
        "prompt_sha256",
        "few_shot_sha256",
        "split_seed",
        "dataset_sha256",
        "split_sha256",
        "python_version",
        "openai_sdk_version",
        "routing_policy",
        "openrouter_non_zdr_acknowledged",
    ]
    return {key: manifest[key] for key in identity_keys}


def ensure_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if manifest_identity(existing) != manifest_identity(manifest):
            raise ExperimentError(
                f"Existing run manifest conflicts with requested configuration: {path}"
            )
        return
    saved = dict(manifest)
    saved["created_at_utc"] = utc_now()
    atomic_write_json(path, saved)


def resolve_api_key(api_provider: str) -> str:
    if api_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            # Backward compatible with the local file created for this experiment.
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
        expected_variables = "OPENROUTER_API_KEY or OPENAI_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        expected_variables = "OPENAI_API_KEY"

    if not api_key:
        raise ExperimentError(
            f"{expected_variables} is not set. No API request was made. Set the key "
            "in the environment, then rerun the same command."
        )
    if api_provider == "openai" and api_key.startswith("sk-or-"):
        raise ExperimentError(
            "The configured credential is an OpenRouter key. Rerun with "
            "--provider openrouter; no direct OpenAI request was made."
        )
    return api_key


def create_client(api_provider: str):
    settings = provider_settings(api_provider)
    api_key = resolve_api_key(api_provider)
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ExperimentError(
            "The OpenAI Python SDK is not installed. Install project requirements first."
        ) from error
    return OpenAI(
        api_key=api_key,
        base_url=settings["base_url"],
        max_retries=0,
        timeout=180.0,
    )


def call_batch(
    *,
    client: Any,
    api_provider: str,
    openrouter_allow_non_zdr: bool,
    model: str,
    reasoning_effort: str,
    prompt: str,
    prompt_sha256: str,
    examples: Sequence[dict[str, Any]],
    items: Sequence[dict[str, Any]],
    max_output_tokens: int,
    max_retries: int,
    batch_path: Path,
    batch_index: int,
) -> dict[str, Any]:
    expected_ids = [item["id"] for item in items]
    request_input = [
        {"role": "user", "content": batch_user_message(examples)},
        {"role": "assistant", "content": expected_assistant_message(examples)},
        {"role": "user", "content": batch_user_message(items)},
    ]
    schema = response_schema(expected_ids)
    attempt_log: list[dict[str, Any]] = []

    for attempt in range(1, max_retries + 1):
        started_at = utc_now()
        started = time.monotonic()
        try:
            request_arguments: dict[str, Any] = {
                "model": model,
                "instructions": prompt,
                "input": request_input,
                "reasoning": {"effort": reasoning_effort},
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "outcome_predictions",
                        "description": (
                            "One study-specific primary label for every supplied "
                            "outcome ID."
                        ),
                        "strict": True,
                        "schema": schema,
                    }
                },
                "max_output_tokens": max_output_tokens,
                "store": False,
            }
            if api_provider == "openai":
                request_arguments["prompt_cache_key"] = (
                    f"outcome-taxonomy-{prompt_sha256[:32]}"
                )
            else:
                request_arguments["extra_body"] = {
                    "provider": request_routing_policy(
                        api_provider, openrouter_allow_non_zdr
                    )
                }

            response = client.responses.create(
                **request_arguments
            )
            elapsed = time.monotonic() - started
            output_text = response.output_text
            if not isinstance(output_text, str) or not output_text.strip():
                raise BatchValidationError("API response did not contain output text.")
            parsed = json.loads(output_text)
            predictions = validate_prediction_payload(parsed, expected_ids)
            attempt_log.append(
                {
                    "attempt": attempt,
                    "started_at_utc": started_at,
                    "elapsed_seconds": elapsed,
                    "status": "success",
                }
            )
            artifact = {
                "status": "success",
                "batch_index": batch_index,
                "completed_at_utc": utc_now(),
                "request": {
                    "provider": provider_settings(api_provider)["name"],
                    "base_url": provider_settings(api_provider)["base_url"],
                    "routing_policy": request_routing_policy(
                        api_provider, openrouter_allow_non_zdr
                    ),
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "prompt_sha256": prompt_sha256,
                    "items": list(items),
                    "expected_ids": expected_ids,
                    "structured_output_schema": schema,
                },
                "predictions": predictions,
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "response_provider": getattr(response, "provider", None),
                "response_status": getattr(response, "status", None),
                "usage": dump_sdk_object(getattr(response, "usage", None)),
                "raw_response": dump_sdk_object(response),
                "attempts": attempt_log,
            }
            atomic_write_json(batch_path, artifact)
            return artifact
        except Exception as error:
            elapsed = time.monotonic() - started
            attempt_log.append(
                {
                    "attempt": attempt,
                    "started_at_utc": started_at,
                    "elapsed_seconds": elapsed,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if attempt < max_retries:
                delay = min(30.0, 1.0 * (2 ** (attempt - 1)))
                print(
                    f"    attempt {attempt}/{max_retries} failed: {error}; "
                    f"retrying in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
                continue

            failure = {
                "status": "failed",
                "batch_index": batch_index,
                "failed_at_utc": utc_now(),
                "request": {
                    "provider": provider_settings(api_provider)["name"],
                    "base_url": provider_settings(api_provider)["base_url"],
                    "routing_policy": request_routing_policy(
                        api_provider, openrouter_allow_non_zdr
                    ),
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "prompt_sha256": prompt_sha256,
                    "items": list(items),
                    "expected_ids": expected_ids,
                    "structured_output_schema": schema,
                },
                "attempts": attempt_log,
            }
            atomic_write_json(batch_path, failure)
            raise ExperimentError(
                f"Batch {batch_index} failed after {max_retries} attempts. "
                f"Failure details were saved to {batch_path}."
            ) from error

    raise AssertionError("Retry loop exited unexpectedly")


def validate_success_artifact(
    artifact: dict[str, Any],
    *,
    expected_items: Sequence[dict[str, Any]],
    api_provider: str,
    openrouter_allow_non_zdr: bool,
    model: str,
    reasoning_effort: str,
    prompt_sha256: str,
) -> list[dict[str, Any]]:
    if artifact.get("status") != "success":
        raise BatchValidationError("Saved artifact is not successful.")
    request = artifact.get("request", {})
    expected_ids = [item["id"] for item in expected_items]
    settings = provider_settings(api_provider)
    if request.get("provider") != settings["name"]:
        raise BatchValidationError("Saved batch provider does not match this run.")
    if request.get("base_url") != settings["base_url"]:
        raise BatchValidationError("Saved batch base URL does not match this run.")
    expected_routing = request_routing_policy(
        api_provider, openrouter_allow_non_zdr
    )
    if request.get("routing_policy") != expected_routing:
        raise BatchValidationError("Saved batch routing policy does not match this run.")
    if request.get("model") != model:
        raise BatchValidationError("Saved batch model does not match this run.")
    if request.get("reasoning_effort") != reasoning_effort:
        raise BatchValidationError("Saved batch reasoning effort does not match.")
    if request.get("prompt_sha256") != prompt_sha256:
        raise BatchValidationError("Saved batch prompt hash does not match.")
    if request.get("expected_ids") != expected_ids:
        raise BatchValidationError("Saved batch IDs do not match this split.")
    return validate_prediction_payload(
        {"predictions": artifact.get("predictions")}, expected_ids
    )


def iter_batches(items: Sequence[dict[str, Any]], batch_size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    for start in range(0, len(items), batch_size):
        yield start // batch_size + 1, list(items[start : start + batch_size])


def compute_metrics(items: Sequence[dict[str, Any]], predictions: Sequence[int]) -> dict[str, Any]:
    true = np.asarray([item["label"] for item in items], dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    if len(true) != len(predicted):
        raise ExperimentError("Metric inputs have different lengths.")
    return {
        "n": int(len(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(true, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                true, predicted, labels=LABELS, average="weighted", zero_division=0
            )
        ),
        "cohen_kappa": float(cohen_kappa_score(true, predicted)),
        "confusion_matrix": confusion_matrix(true, predicted, labels=LABELS).tolist(),
        "classification_report": classification_report(
            true,
            predicted,
            labels=LABELS,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def sum_usage(artifacts: Sequence[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for artifact in artifacts:
        usage = artifact.get("usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
        details = usage.get("output_tokens_details") or {}
        reasoning = details.get("reasoning_tokens")
        if isinstance(reasoning, int):
            totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + reasoning
    return totals


def run_items(
    *,
    phase: str,
    reasoning_effort: str,
    items: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    prompt: str,
    examples: Sequence[dict[str, Any]],
    data: CanonicalData,
) -> Path | None:
    run_dir = args.output_dir / phase / f"reasoning_{reasoning_effort}"
    batches_dir = run_dir / "batches"
    manifest = build_manifest(
        phase=phase,
        reasoning_effort=reasoning_effort,
        api_provider=args.provider,
        openrouter_allow_non_zdr=args.openrouter_allow_non_zdr,
        model=args.model,
        batch_size=args.batch_size,
        max_output_tokens=args.max_output_tokens,
        prompt_path=args.prompt,
        examples_path=args.examples,
        data=data,
    )
    ensure_manifest(run_dir / "manifest.json", manifest)
    prompt_sha256 = manifest["prompt_sha256"]
    batches = list(iter_batches(items, args.batch_size))
    if args.max_batches is not None:
        batches = batches[: args.max_batches]
    client = None
    artifacts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for batch_index, batch_items in batches:
        batch_path = batches_dir / f"batch_{batch_index:05d}.json"
        artifact = None
        if batch_path.exists():
            candidate = json.loads(batch_path.read_text(encoding="utf-8"))
            try:
                validate_success_artifact(
                    candidate,
                    expected_items=batch_items,
                    api_provider=args.provider,
                    openrouter_allow_non_zdr=args.openrouter_allow_non_zdr,
                    model=args.model,
                    reasoning_effort=reasoning_effort,
                    prompt_sha256=prompt_sha256,
                )
                artifact = candidate
                print(
                    f"  {phase}/{reasoning_effort}: batch {batch_index}/{len(batches)} "
                    "loaded from saved artifact",
                    flush=True,
                )
            except BatchValidationError:
                artifact = None

        if artifact is None:
            if client is None:
                client = create_client(args.provider)
            print(
                f"  {phase}/{reasoning_effort}: calling batch "
                f"{batch_index}/{len(batches)} ({len(batch_items)} outcomes)",
                flush=True,
            )
            artifact = call_batch(
                client=client,
                api_provider=args.provider,
                openrouter_allow_non_zdr=args.openrouter_allow_non_zdr,
                model=args.model,
                reasoning_effort=reasoning_effort,
                prompt=prompt,
                prompt_sha256=prompt_sha256,
                examples=examples,
                items=batch_items,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                batch_path=batch_path,
                batch_index=batch_index,
            )

        ordered_predictions = validate_success_artifact(
            artifact,
            expected_items=batch_items,
            api_provider=args.provider,
            openrouter_allow_non_zdr=args.openrouter_allow_non_zdr,
            model=args.model,
            reasoning_effort=reasoning_effort,
            prompt_sha256=prompt_sha256,
        )
        prediction_by_id = {
            prediction["id"]: prediction["label"] for prediction in ordered_predictions
        }
        for item in batch_items:
            prediction = int(prediction_by_id[item["id"]])
            rows.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "true": int(item["label"]),
                    "pred": prediction,
                    "correct": prediction == int(item["label"]),
                    "batch_index": batch_index,
                    "response_id": artifact.get("response_id"),
                    "response_model": artifact.get("response_model"),
                    "response_provider": artifact.get("response_provider"),
                }
            )
        artifacts.append(artifact)

    completed_all_batches = len(batches) == (len(items) + args.batch_size - 1) // args.batch_size
    if not completed_all_batches:
        partial_path = run_dir / "partial_predictions.csv"
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print(f"  Partial run saved to {partial_path}")
        return None

    predictions_path = run_dir / "predictions.csv"
    predictions_frame = pd.DataFrame(rows)
    predictions_frame.to_csv(predictions_path, index=False)
    metrics = compute_metrics(items, predictions_frame["pred"].tolist())
    metrics.update(
        {
            "phase": phase,
            "reasoning_effort": reasoning_effort,
            "model": args.model,
            "completed_at_utc": utc_now(),
            "usage": sum_usage(artifacts),
            "successful_batches": len(artifacts),
            "fallback_predictions": 0,
        }
    )
    metrics_path = run_dir / "metrics.json"
    atomic_write_json(metrics_path, metrics)
    print(
        f"  {phase}/{reasoning_effort}: accuracy={metrics['accuracy']:.4f}, "
        f"macro F1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    return metrics_path


def paired_bootstrap_difference(
    true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    samples: int = 2000,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(true)
    accuracy_differences = np.empty(samples, dtype=float)
    macro_f1_differences = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.integers(0, count, size=count)
        sampled_true = true[sampled]
        first_sampled = first[sampled]
        second_sampled = second[sampled]
        accuracy_differences[index] = accuracy_score(
            sampled_true, second_sampled
        ) - accuracy_score(sampled_true, first_sampled)
        macro_f1_differences[index] = f1_score(
            sampled_true,
            second_sampled,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ) - f1_score(
            sampled_true,
            first_sampled,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "ci_2.5_percentile": float(np.percentile(values, 2.5)),
            "ci_97.5_percentile": float(np.percentile(values, 97.5)),
            "probability_difference_gt_zero": float(np.mean(values > 0)),
        }

    return {
        "accuracy_low_minus_none": summarize(accuracy_differences),
        "macro_f1_low_minus_none": summarize(macro_f1_differences),
    }


def compare_validation(args: argparse.Namespace) -> Path:
    run_frames: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for effort in ("none", "low"):
        run_dir = args.output_dir / "validation" / f"reasoning_{effort}"
        predictions_path = run_dir / "predictions.csv"
        metrics_path = run_dir / "metrics.json"
        if not predictions_path.exists() or not metrics_path.exists():
            raise ExperimentError(
                "Both validation runs must finish before comparison. Missing "
                f"artifacts for reasoning={effort}."
            )
        run_frames[effort] = pd.read_csv(predictions_path)
        metrics[effort] = json.loads(metrics_path.read_text(encoding="utf-8"))

    first = run_frames["none"]
    second = run_frames["low"]
    if first["id"].tolist() != second["id"].tolist():
        raise ExperimentError("Validation IDs differ between reasoning settings.")
    if not np.array_equal(first["true"], second["true"]):
        raise ExperimentError("Validation labels differ between reasoning settings.")

    true = first["true"].to_numpy(dtype=int)
    none_predictions = first["pred"].to_numpy(dtype=int)
    low_predictions = second["pred"].to_numpy(dtype=int)
    disagreement = none_predictions != low_predictions
    none_only_correct = int(
        np.sum((none_predictions == true) & (low_predictions != true))
    )
    low_only_correct = int(
        np.sum((none_predictions != true) & (low_predictions == true))
    )
    comparison = {
        "comparison_role": "validation-only exploratory comparison",
        "selection_rule": (
            "Choose the higher validation accuracy; if accuracies are exactly equal, "
            "choose the higher validation macro F1; if both are equal, choose none."
        ),
        "none": metrics["none"],
        "low": metrics["low"],
        "absolute_difference_low_minus_none": {
            "accuracy": float(metrics["low"]["accuracy"] - metrics["none"]["accuracy"]),
            "macro_f1": float(metrics["low"]["macro_f1"] - metrics["none"]["macro_f1"]),
        },
        "paired_disagreement": {
            "n": int(np.sum(disagreement)),
            "percent": float(100 * np.mean(disagreement)),
            "none_only_correct": none_only_correct,
            "low_only_correct": low_only_correct,
        },
        "paired_bootstrap": paired_bootstrap_difference(
            true, none_predictions, low_predictions
        ),
        "created_at_utc": utc_now(),
    }
    comparison_path = args.output_dir / "validation_comparison.json"
    atomic_write_json(comparison_path, comparison)
    summary = pd.DataFrame(
        [
            {
                "reasoning_effort": effort,
                "n": metrics[effort]["n"],
                "accuracy": metrics[effort]["accuracy"],
                "macro_f1": metrics[effort]["macro_f1"],
                "weighted_f1": metrics[effort]["weighted_f1"],
                "cohen_kappa": metrics[effort]["cohen_kappa"],
                **{
                    f"tokens_{key}": value
                    for key, value in metrics[effort].get("usage", {}).items()
                },
            }
            for effort in ("none", "low")
        ]
    )
    summary.to_csv(args.output_dir / "validation_comparison.csv", index=False)
    return comparison_path


def select_effort(comparison: dict[str, Any]) -> str:
    none_accuracy = float(comparison["none"]["accuracy"])
    low_accuracy = float(comparison["low"]["accuracy"])
    if low_accuracy > none_accuracy:
        return "low"
    if none_accuracy > low_accuracy:
        return "none"
    none_macro_f1 = float(comparison["none"]["macro_f1"])
    low_macro_f1 = float(comparison["low"]["macro_f1"])
    if low_macro_f1 > none_macro_f1:
        return "low"
    return "none"


def freeze_configuration(
    args: argparse.Namespace,
    data: CanonicalData,
    prompt_sha256: str,
    examples_sha256: str,
) -> Path:
    settings = provider_settings(args.provider)
    comparison_path = args.output_dir / "validation_comparison.json"
    if not comparison_path.exists():
        raise ExperimentError("Run the complete validation comparison before freezing.")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    effort = select_effort(comparison)
    frozen = {
        "status": "frozen before test evaluation",
        "frozen_at_utc": utc_now(),
        "model": args.model,
        "provider": settings["name"],
        "base_url": settings["base_url"],
        "routing_policy": request_routing_policy(
            args.provider, args.openrouter_allow_non_zdr
        ),
        "reasoning_effort": effort,
        "batch_size": args.batch_size,
        "max_output_tokens": args.max_output_tokens,
        "prompt_sha256": prompt_sha256,
        "few_shot_sha256": examples_sha256,
        "dataset_sha256": data.dataset_sha256,
        "split_sha256": data.split_sha256,
        "selection_rule": comparison["selection_rule"],
        "validation_accuracy": comparison[effort]["accuracy"],
        "validation_macro_f1": comparison[effort]["macro_f1"],
        "test_evaluations_allowed": 1,
    }
    frozen_path = args.output_dir / "frozen_configuration.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        if existing != frozen:
            # Ignore timestamps when deciding whether an existing freeze is identical.
            existing_without_time = dict(existing)
            frozen_without_time = dict(frozen)
            existing_without_time.pop("frozen_at_utc", None)
            frozen_without_time.pop("frozen_at_utc", None)
            if existing_without_time != frozen_without_time:
                raise ExperimentError(
                    "A different frozen configuration already exists. Use a new output "
                    "directory rather than overwriting the prespecified test plan."
                )
        return frozen_path
    atomic_write_json(frozen_path, frozen)
    return frozen_path


def validate_frozen_configuration(
    args: argparse.Namespace,
    data: CanonicalData,
    prompt_sha256: str,
    examples_sha256: str,
) -> dict[str, Any]:
    frozen_path = args.output_dir / "frozen_configuration.json"
    if not frozen_path.exists():
        raise ExperimentError("Frozen configuration is required before a test run.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected = {
        "provider": provider_settings(args.provider)["name"],
        "base_url": provider_settings(args.provider)["base_url"],
        "routing_policy": request_routing_policy(
            args.provider, args.openrouter_allow_non_zdr
        ),
        "model": args.model,
        "batch_size": args.batch_size,
        "max_output_tokens": args.max_output_tokens,
        "prompt_sha256": prompt_sha256,
        "few_shot_sha256": examples_sha256,
        "dataset_sha256": data.dataset_sha256,
        "split_sha256": data.split_sha256,
    }
    mismatches = {
        key: {"frozen": frozen.get(key), "requested": value}
        for key, value in expected.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise ExperimentError(
            f"Requested test configuration differs from frozen settings: {mismatches}"
        )
    if frozen.get("reasoning_effort") not in ("none", "low"):
        raise ExperimentError("Frozen reasoning effort is invalid.")
    return frozen


def write_preflight(
    args: argparse.Namespace,
    data: CanonicalData,
    prompt: str,
    examples_payload: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    example_audit: Sequence[dict[str, Any]],
) -> Path:
    settings = provider_settings(args.provider)
    saved_test_check = verify_saved_primary_test_split(data, args.saved_test_predictions)
    preflight = {
        "experiment_role": (
            "Exploratory prompt experiment; does not replace or modify the historical "
            "prompt, historical GPT results, or manuscript."
        ),
        "created_at_utc": utc_now(),
        "dataset": str(args.dataset),
        "dataset_sha256": data.dataset_sha256,
        "split_seed": SPLIT_SEED,
        "counts": {
            "train": len(data.train),
            "validation": len(data.validation),
            "test": len(data.test),
        },
        "class_counts": {
            split: {
                str(label): int(count)
                for label, count in getattr(data, split)["labels"]
                .value_counts()
                .sort_index()
                .items()
            }
            for split in ("train", "validation", "test")
        },
        "split_sha256": data.split_sha256,
        "saved_primary_test_split_check": saved_test_check,
        "prompt_path": str(args.prompt),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "few_shot_path": str(args.examples),
        "few_shot_sha256": sha256_file(args.examples),
        "few_shot_provenance": examples_payload.get("provenance"),
        "few_shot_count": len(examples),
        "few_shot_exact_match_audit": list(example_audit),
        "model": args.model,
        "provider": settings["name"],
        "base_url": settings["base_url"],
        "model_pin": settings["model_pin"],
        "routing_policy": request_routing_policy(
            args.provider, args.openrouter_allow_non_zdr
        ),
        "non_zdr_study_data_guardrail": (
            "Validation and test phases require "
            "--acknowledge-non-zdr-study-data when non-ZDR routing is enabled."
            if args.provider == "openrouter" and args.openrouter_allow_non_zdr
            else None
        ),
        "reasoning_efforts_for_validation": ["none", "low"],
        "batch_size": args.batch_size,
        "max_output_tokens": args.max_output_tokens,
        "test_guardrail": (
            "No test API call is allowed until validation results are compared, a "
            "configuration is frozen, and --confirm-test-pass is supplied."
        ),
    }
    path = args.output_dir / "preflight.json"
    atomic_write_json(path, preflight)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "smoke", "validation", "freeze", "test"),
        default="prepare",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "openrouter"),
        default="openai",
        help="API gateway. OpenRouter runs are stored separately by default.",
    )
    parser.add_argument(
        "--openrouter-allow-non-zdr",
        action="store_true",
        help=(
            "Permit an OpenRouter endpoint without zero-data-retention. Provider "
            "data collection remains denied."
        ),
    )
    parser.add_argument(
        "--acknowledge-non-zdr-study-data",
        action="store_true",
        help=(
            "Required in addition to --openrouter-allow-non-zdr before validation "
            "or test outcome text can be sent."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Debugging only: stop after this many batches and do not compute metrics.",
    )
    parser.add_argument(
        "--saved-test-predictions", type=Path, default=DEFAULT_SAVED_TEST
    )
    parser.add_argument(
        "--confirm-test-pass",
        action="store_true",
        help="Required for the single frozen test-set API pass.",
    )
    args = parser.parse_args()
    settings = provider_settings(args.provider)
    if args.output_dir is None:
        if args.provider == "openrouter" and args.openrouter_allow_non_zdr:
            args.output_dir = DEFAULT_OPENROUTER_NON_ZDR_OUTPUT_DIR
        else:
            args.output_dir = settings["default_output_dir"]
    if args.model is None:
        args.model = settings["default_model"]
    if args.openrouter_allow_non_zdr and args.provider != "openrouter":
        parser.error("--openrouter-allow-non-zdr requires --provider openrouter")
    if args.acknowledge_non_zdr_study_data and not args.openrouter_allow_non_zdr:
        parser.error(
            "--acknowledge-non-zdr-study-data requires --openrouter-allow-non-zdr"
        )
    if (
        args.provider == "openrouter"
        and args.openrouter_allow_non_zdr
        and args.phase in ("validation", "test")
        and not args.acknowledge_non_zdr_study_data
    ):
        parser.error(
            "Non-ZDR OpenRouter routing for validation or test data is blocked. "
            "Review the privacy implications, then explicitly add "
            "--acknowledge-non-zdr-study-data if authorized."
        )
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be at least 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    prompt = args.prompt.read_text(encoding="utf-8")
    examples_payload, examples = load_examples(args.examples)
    data = load_canonical_data(args.dataset)
    example_audit = audit_examples(examples, data)
    preflight_path = write_preflight(
        args, data, prompt, examples_payload, examples, example_audit
    )
    print(f"Preflight checks passed: {preflight_path}")
    print(
        "Canonical split: "
        f"train={len(data.train)}, validation={len(data.validation)}, test={len(data.test)}"
    )

    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
    examples_sha256 = sha256_file(args.examples)
    if args.phase == "prepare":
        return
    if args.phase == "smoke":
        for effort in ("none", "low"):
            run_items(
                phase="smoke",
                reasoning_effort=effort,
                items=SMOKE_ITEMS,
                args=args,
                prompt=prompt,
                examples=examples,
                data=data,
            )
        return
    if args.phase == "validation":
        validation_items = make_split_items(data.validation, "validation")
        for effort in ("none", "low"):
            run_items(
                phase="validation",
                reasoning_effort=effort,
                items=validation_items,
                args=args,
                prompt=prompt,
                examples=examples,
                data=data,
            )
        if args.max_batches is None:
            comparison_path = compare_validation(args)
            print(f"Validation comparison saved: {comparison_path}")
        return
    if args.phase == "freeze":
        frozen_path = freeze_configuration(
            args, data, prompt_sha256, examples_sha256
        )
        print(f"Frozen configuration saved: {frozen_path}")
        return
    if args.phase == "test":
        if not args.confirm_test_pass:
            raise ExperimentError(
                "Test phase is blocked. Review and freeze validation results, then "
                "rerun with --confirm-test-pass for the single held-out test pass."
            )
        frozen = validate_frozen_configuration(
            args, data, prompt_sha256, examples_sha256
        )
        test_items = make_split_items(data.test, "test")
        run_items(
            phase="test",
            reasoning_effort=frozen["reasoning_effort"],
            items=test_items,
            args=args,
            prompt=prompt,
            examples=examples,
            data=data,
        )
        return
    raise AssertionError(f"Unhandled phase: {args.phase}")


if __name__ == "__main__":
    try:
        main()
    except ExperimentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
