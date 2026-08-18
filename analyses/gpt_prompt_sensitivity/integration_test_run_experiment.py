#!/usr/bin/env python3
"""Local tests for the exploratory GPT-5.2 experiment runner."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_experiment as experiment


class FakeResponse:
    def __init__(self, predictions, response_number):
        self.output_text = json.dumps({"predictions": predictions})
        self.id = f"response_{response_number}"
        self.model = experiment.DEFAULT_MODEL
        self.status = "completed"
        self.usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "output_tokens_details": {"reasoning_tokens": 5},
        }

    def model_dump(self, mode="json"):
        del mode
        return {
            "id": self.id,
            "model": self.model,
            "status": self.status,
            "output_text": self.output_text,
            "usage": self.usage,
        }


class FakeResponses:
    def __init__(self, labels_by_id):
        self.labels_by_id = labels_by_id
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["text"]["format"]["schema"]
        ids = schema["properties"]["predictions"]["items"]["properties"]["id"][
            "enum"
        ]
        predictions = [
            {"id": item_id, "label": self.labels_by_id[item_id]} for item_id in ids
        ]
        return FakeResponse(predictions, len(self.calls))


class FakeClient:
    def __init__(self, labels_by_id):
        self.responses = FakeResponses(labels_by_id)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not experiment.DEFAULT_DATASET.is_file():
            raise unittest.SkipTest(
                "Restricted full dataset is not configured; set OUTCOME_DATASET_CSV."
            )
        if not experiment.DEFAULT_SAVED_TEST.is_file():
            raise unittest.SkipTest(
                "Retained primary predictions are not configured; set "
                "OUTCOME_PRIMARY_PREDICTIONS."
            )
        cls.data = experiment.load_canonical_data(experiment.DEFAULT_DATASET)
        cls.examples_payload, cls.examples = experiment.load_examples(
            experiment.DEFAULT_EXAMPLES
        )

    def test_canonical_split_matches_saved_primary_rows(self):
        self.assertEqual(len(self.data.train), 15763)
        self.assertEqual(len(self.data.validation), 2252)
        self.assertEqual(len(self.data.test), 4503)
        check = experiment.verify_saved_primary_test_split(
            self.data, experiment.DEFAULT_SAVED_TEST
        )
        self.assertTrue(check["texts_match"])
        self.assertTrue(check["labels_match"])

    def test_examples_do_not_match_validation_or_test(self):
        audit = experiment.audit_examples(self.examples, self.data)
        self.assertEqual(len(audit), len(self.examples))
        for row in audit:
            self.assertNotIn("validation", row["exact_match_splits"])
            self.assertNotIn("test", row["exact_match_splits"])

    def test_contrastive_examples_do_not_match_validation_or_test(self):
        path = experiment.SCRIPT_DIR / "few_shot_examples_contrastive18.json"
        _, examples = experiment.load_examples(path)
        self.assertEqual(len(examples), 18)
        self.assertEqual({item["label"] for item in examples}, {0, 1, 2})
        audit = experiment.audit_examples(examples, self.data)
        for row in audit:
            self.assertNotIn("validation", row["exact_match_splits"])
            self.assertNotIn("test", row["exact_match_splits"])

    def test_strict_payload_validation_reorders_by_input_id(self):
        payload = {
            "predictions": [
                {"id": "outcome_2", "label": 1},
                {"id": "outcome_1", "label": 0},
            ]
        }
        validated = experiment.validate_prediction_payload(
            payload, ["outcome_1", "outcome_2"]
        )
        self.assertEqual(
            validated,
            [
                {"id": "outcome_1", "label": 0},
                {"id": "outcome_2", "label": 1},
            ],
        )

    def test_strict_payload_validation_rejects_missing_or_invalid_values(self):
        invalid_payloads = [
            {"predictions": [{"id": "outcome_1", "label": 0}]},
            {
                "predictions": [
                    {"id": "outcome_1", "label": 0},
                    {"id": "outcome_1", "label": 1},
                ]
            },
            {
                "predictions": [
                    {"id": "outcome_1", "label": True},
                    {"id": "outcome_2", "label": 1},
                ]
            },
            {
                "predictions": [
                    {"id": "outcome_1", "label": 0, "confidence": 0.9},
                    {"id": "outcome_2", "label": 1},
                ]
            },
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(experiment.BatchValidationError):
                    experiment.validate_prediction_payload(
                        payload, ["outcome_1", "outcome_2"]
                    )

    def test_api_request_and_resume_without_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            args = argparse.Namespace(
                output_dir=output_dir,
                batch_size=2,
                max_batches=None,
                provider="openai",
                openrouter_allow_non_zdr=False,
                model=experiment.DEFAULT_MODEL,
                max_output_tokens=4000,
                max_retries=2,
                prompt=experiment.DEFAULT_PROMPT,
                examples=experiment.DEFAULT_EXAMPLES,
            )
            fake_client = FakeClient(
                {item["id"]: item["label"] for item in experiment.SMOKE_ITEMS}
            )
            prompt_text = experiment.DEFAULT_PROMPT.read_text(encoding="utf-8")

            with patch.object(experiment, "create_client", return_value=fake_client):
                metrics_path = experiment.run_items(
                    phase="mock_smoke",
                    reasoning_effort="low",
                    items=experiment.SMOKE_ITEMS,
                    args=args,
                    prompt=prompt_text,
                    examples=self.examples,
                    data=self.data,
                )
            self.assertIsNotNone(metrics_path)
            self.assertEqual(len(fake_client.responses.calls), 2)
            for call in fake_client.responses.calls:
                self.assertEqual(call["model"], experiment.DEFAULT_MODEL)
                self.assertEqual(call["reasoning"], {"effort": "low"})
                self.assertFalse(call["store"])
                self.assertEqual(call["text"]["format"]["type"], "json_schema")
                self.assertTrue(call["text"]["format"]["strict"])
                self.assertEqual(len(call["input"]), 3)
                self.assertEqual(call["input"][-1]["role"], "user")

            with patch.object(
                experiment,
                "create_client",
                side_effect=AssertionError("resume should not create an API client"),
            ):
                resumed_metrics = experiment.run_items(
                    phase="mock_smoke",
                    reasoning_effort="low",
                    items=experiment.SMOKE_ITEMS,
                    args=args,
                    prompt=prompt_text,
                    examples=self.examples,
                    data=self.data,
                )
            self.assertEqual(metrics_path, resumed_metrics)

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["macro_f1"], 1.0)
            self.assertEqual(metrics["fallback_predictions"], 0)
            self.assertEqual(metrics["usage"]["total_tokens"], 240)

    def test_test_configuration_must_be_frozen(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = argparse.Namespace(
                output_dir=Path(temporary_directory),
                provider="openai",
                openrouter_allow_non_zdr=False,
                model=experiment.DEFAULT_MODEL,
                batch_size=50,
                max_output_tokens=4000,
            )
            with self.assertRaises(experiment.ExperimentError):
                experiment.validate_frozen_configuration(
                    args,
                    self.data,
                    experiment.sha256_file(experiment.DEFAULT_PROMPT),
                    experiment.sha256_file(experiment.DEFAULT_EXAMPLES),
                )

    def test_validation_selection_rule_and_freeze(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            comparison = {
                "selection_rule": "accuracy, then macro F1, then none",
                "none": {"accuracy": 0.80, "macro_f1": 0.82},
                "low": {"accuracy": 0.80, "macro_f1": 0.84},
            }
            experiment.atomic_write_json(
                output_dir / "validation_comparison.json", comparison
            )
            args = argparse.Namespace(
                output_dir=output_dir,
                provider="openai",
                openrouter_allow_non_zdr=False,
                model=experiment.DEFAULT_MODEL,
                batch_size=50,
                max_output_tokens=4000,
            )
            prompt_hash = experiment.sha256_file(experiment.DEFAULT_PROMPT)
            example_hash = experiment.sha256_file(experiment.DEFAULT_EXAMPLES)
            frozen_path = experiment.freeze_configuration(
                args, self.data, prompt_hash, example_hash
            )
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertEqual(frozen["reasoning_effort"], "low")
            validated = experiment.validate_frozen_configuration(
                args, self.data, prompt_hash, example_hash
            )
            self.assertEqual(validated, frozen)

    def test_openrouter_request_is_explicit_and_isolated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            args = argparse.Namespace(
                output_dir=output_dir,
                batch_size=50,
                max_batches=None,
                provider="openrouter",
                openrouter_allow_non_zdr=False,
                model=experiment.DEFAULT_OPENROUTER_MODEL,
                max_output_tokens=4000,
                max_retries=2,
                prompt=experiment.DEFAULT_PROMPT,
                examples=experiment.DEFAULT_EXAMPLES,
            )
            fake_client = FakeClient(
                {item["id"]: item["label"] for item in experiment.SMOKE_ITEMS}
            )
            prompt_text = experiment.DEFAULT_PROMPT.read_text(encoding="utf-8")

            with patch.object(experiment, "create_client", return_value=fake_client):
                metrics_path = experiment.run_items(
                    phase="mock_smoke",
                    reasoning_effort="none",
                    items=experiment.SMOKE_ITEMS,
                    args=args,
                    prompt=prompt_text,
                    examples=self.examples,
                    data=self.data,
                )

            self.assertIsNotNone(metrics_path)
            self.assertEqual(len(fake_client.responses.calls), 1)
            call = fake_client.responses.calls[0]
            self.assertEqual(call["model"], experiment.DEFAULT_OPENROUTER_MODEL)
            self.assertEqual(call["reasoning"], {"effort": "none"})
            self.assertNotIn("prompt_cache_key", call)
            self.assertEqual(
                call["extra_body"],
                {
                    "provider": {
                        "require_parameters": True,
                        "data_collection": "deny",
                        "zdr": True,
                    }
                },
            )
            self.assertTrue(call["text"]["format"]["strict"])

            manifest = json.loads(
                (output_dir / "mock_smoke/reasoning_none/manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["provider"], "OpenRouter")
            self.assertEqual(manifest["base_url"], experiment.OPENROUTER_BASE_URL)
            self.assertEqual(
                manifest["openrouter_canonical_slug_at_implementation"],
                experiment.OPENROUTER_CANONICAL_SLUG,
            )

    def test_openrouter_non_zdr_policy_remains_separate(self):
        self.assertEqual(
            experiment.request_routing_policy("openrouter", False),
            {
                "require_parameters": True,
                "data_collection": "deny",
                "zdr": True,
            },
        )
        self.assertEqual(
            experiment.request_routing_policy("openrouter", True),
            {
                "require_parameters": True,
                "data_collection": "deny",
            },
        )


if __name__ == "__main__":
    unittest.main()
