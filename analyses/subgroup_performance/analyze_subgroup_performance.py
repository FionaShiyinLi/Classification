#!/usr/bin/env python3
"""Reproduce the 16-subgroup-stratified Bioformer results in Table S10.

The script reads restricted row-level inputs but writes only aggregate counts
and percentages. It does not train or evaluate a 16-way classifier.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
load_dotenv(REPO_ROOT / ".env")
DEFAULT_SUBGROUP_DATA = Path(
    os.getenv(
        "OUTCOME_BINARY_DATASET_CSV",
        REPO_ROOT / "restricted_data" / "binary.outcome.csv",
    )
)
DEFAULT_TEST_MANIFEST = Path(
    os.getenv(
        "OUTCOME_TEST_MANIFEST",
        REPO_ROOT / "restricted_data" / "leakage_similarity_test_items.csv",
    )
)
DEFAULT_PREDICTIONS = Path(
    os.getenv(
        "OUTCOME_PRIMARY_PREDICTIONS",
        REPO_ROOT
        / "results/outputs_hparam_search_base16/"
        "lr3e-05_wu0.04_uf8_rd1.0/best_model/test_predictions.csv",
    )
)
DEFAULT_MAPPING = REPO_ROOT / "config" / "subgroup_mapping.json"
DEFAULT_OUTPUT = REPO_ROOT / "private_outputs" / "subgroup_performance"

REVIEW_COL = "CDSR.id"
OUTCOME_ID_COL = "outcome.id"
TEXT_COL = "outcome"
THREE_CLASS_COL = "outcome.class"
LABEL_NAMES = {0: "Objective", 1: "Semi-objective", 2: "Subjective"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subgroup-data", type=Path, default=DEFAULT_SUBGROUP_DATA)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_mapping(path: Path) -> tuple[dict[int, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapped = {
        int(row["code"]): int(row["mapped_label_id"])
        for row in payload["subgroups"]
    }
    names = {int(row["code"]): str(row["name"]) for row in payload["subgroups"]}
    if set(mapped) != set(range(1, 17)):
        raise ValueError("Mapping must contain subgroup codes 1 through 16")
    return mapped, names


def load_predictions(path: Path, manifest: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(manifest):
        raise ValueError(
            f"Prediction count {len(frame)} does not match test count {len(manifest)}"
        )
    text_column = "text" if "text" in frame.columns else TEXT_COL
    if text_column in frame.columns:
        expected = manifest[TEXT_COL].astype(str).str.strip().tolist()
        observed = frame[text_column].astype(str).str.strip().tolist()
        if observed != expected:
            raise ValueError("Prediction text/order does not match the test manifest")
    if "true" in frame.columns and not np.array_equal(
        frame["true"].to_numpy(dtype=int),
        manifest[THREE_CLASS_COL].to_numpy(dtype=int),
    ):
        raise ValueError("Prediction labels do not match the test manifest")
    prediction_columns = [
        column
        for column in ("pred", "pred_Bioformer-8L", "prediction")
        if column in frame.columns
    ]
    if len(prediction_columns) != 1:
        raise ValueError(
            "Predictions must contain exactly one of pred, pred_Bioformer-8L, "
            "or prediction"
        )
    predictions = frame[prediction_columns[0]].to_numpy(dtype=int)
    if not set(np.unique(predictions)).issubset({0, 1, 2}):
        raise ValueError("Predictions contain labels outside 0, 1, and 2")
    return predictions


def main() -> None:
    args = parse_args()
    mapped, subgroup_names = load_mapping(args.mapping)
    manifest = pd.read_csv(
        args.test_manifest,
        usecols=[REVIEW_COL, OUTCOME_ID_COL, TEXT_COL, THREE_CLASS_COL],
    )
    if len(manifest) != 4_503:
        raise ValueError(f"Expected 4,503 test outcomes, found {len(manifest)}")

    subgroup = pd.read_csv(
        args.subgroup_data,
        usecols=[REVIEW_COL, OUTCOME_ID_COL, TEXT_COL, THREE_CLASS_COL],
    ).rename(
        columns={TEXT_COL: "subgroup_text", THREE_CLASS_COL: "subgroup_code"}
    )
    if subgroup.duplicated([REVIEW_COL, OUTCOME_ID_COL]).any():
        raise ValueError("Subgroup data contain duplicate review/outcome identifiers")
    data = manifest.merge(
        subgroup,
        on=[REVIEW_COL, OUTCOME_ID_COL],
        how="left",
        validate="one_to_one",
    )
    if data["subgroup_code"].isna().any():
        raise ValueError("Some test outcomes have no subgroup assignment")
    text_matches = (
        data[TEXT_COL].astype(str).str.strip()
        == data["subgroup_text"].astype(str).str.strip()
    )
    if not bool(text_matches.all()):
        raise ValueError("Subgroup and test-manifest outcome texts do not match")
    data["subgroup_code"] = data["subgroup_code"].astype(int)
    if not set(data["subgroup_code"]).issubset(set(range(1, 17))):
        raise ValueError("Test data contain subgroup codes outside 1 through 16")
    data["mapped_class"] = data["subgroup_code"].map(mapped)
    if not np.array_equal(
        data["mapped_class"].to_numpy(dtype=int),
        data[THREE_CLASS_COL].to_numpy(dtype=int),
    ):
        raise ValueError("Subgroup mapping does not reproduce the test reference labels")

    data["prediction"] = load_predictions(args.predictions, manifest)
    data["correct"] = data["prediction"] == data["mapped_class"]
    rows = []
    for code in range(1, 17):
        part = data[data["subgroup_code"] == code]
        n = len(part)
        row = {
            "subgroup_code": code,
            "subgroup_name": subgroup_names[code],
            "mapped_class_id": mapped[code],
            "mapped_class": LABEL_NAMES[mapped[code]],
            "test_outcomes_n": n,
            "correct_n": int(part["correct"].sum()),
            "correct_percent": 100 * float(part["correct"].mean()),
        }
        for label, name in LABEL_NAMES.items():
            count = int((part["prediction"] == label).sum())
            key = name.lower().replace("-", "_")
            row[f"predicted_{key}_n"] = count
            row[f"predicted_{key}_percent"] = 100 * count / n
        rows.append(row)

    output = pd.DataFrame(rows)
    total_correct = int(data["correct"].sum())
    prediction_counts = {
        label: int((data["prediction"] == label).sum()) for label in LABEL_NAMES
    }
    if (len(data), total_correct, prediction_counts) != (
        4_503,
        4_122,
        {0: 340, 1: 1_973, 2: 2_190},
    ):
        raise ValueError(
            "Aggregate prediction checks do not match the primary checkpoint"
        )

    metadata = {
        "analysis": "primary Bioformer predictions stratified by the original 16 subgroups",
        "test_outcomes_n": len(data),
        "correct_n": total_correct,
        "accuracy": total_correct / len(data),
        "prediction_counts": {
            LABEL_NAMES[label]: count for label, count in prediction_counts.items()
        },
        "mapping_file": args.mapping.name,
        "outputs_contain_source_text": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "subgroup_performance.csv", index=False)
    (args.output_dir / "subgroup_performance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
