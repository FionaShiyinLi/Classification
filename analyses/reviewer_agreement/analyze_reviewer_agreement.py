#!/usr/bin/env python3
"""Reproduce Table S8 and the independently double-coded-subset kappas.

Only aggregate JSON and CSV files are written. Source outcome text is read from
the restricted local inputs but is never included in an output artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
load_dotenv(REPO_ROOT / ".env")
DEFAULT_BINARY_DATA = Path(
    os.getenv(
        "OUTCOME_BINARY_DATASET_CSV",
        REPO_ROOT / "restricted_data" / "binary.outcome.csv",
    )
)
DEFAULT_THREE_CLASS_DATA = Path(
    os.getenv(
        "OUTCOME_DATASET_CSV",
        REPO_ROOT / "restricted_data" / "outcome_3cls.csv",
    )
)
DEFAULT_MAPPING = REPO_ROOT / "config" / "subgroup_mapping.json"
DEFAULT_OUTPUT = REPO_ROOT / "private_outputs" / "reviewer_agreement"

TEXT_COL = "outcome"
INITIAL_SUBGROUP_COL = "outcome.class"
SECOND_SUBGROUP_COL = "KR"
THREE_CLASS_COL = "outcome.class"
LABEL_NAMES = {0: "Objective", 1: "Semi-objective", 2: "Subjective"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-data", type=Path, default=DEFAULT_BINARY_DATA)
    parser.add_argument(
        "--three-class-data", type=Path, default=DEFAULT_THREE_CLASS_DATA
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-subset-n", type=int, default=948)
    return parser.parse_args()


def load_mapping(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {
        int(row["code"]): int(row["mapped_label_id"])
        for row in payload["subgroups"]
    }
    if set(mapping) != set(range(1, 17)):
        raise ValueError("Mapping must contain subgroup codes 1 through 16")
    return mapping


def analytic_distribution(path: Path) -> dict[int, int]:
    frame = pd.read_csv(path, usecols=[TEXT_COL, THREE_CLASS_COL])
    frame[TEXT_COL] = frame[TEXT_COL].astype("string").str.strip()
    frame[THREE_CLASS_COL] = pd.to_numeric(
        frame[THREE_CLASS_COL], errors="coerce"
    )
    frame = frame[
        frame[TEXT_COL].notna()
        & frame[TEXT_COL].ne("")
        & frame[THREE_CLASS_COL].isin([0, 1, 2])
    ].copy()
    frame[THREE_CLASS_COL] = frame[THREE_CLASS_COL].astype(int)
    frame = frame.drop_duplicates(
        subset=[TEXT_COL, THREE_CLASS_COL], keep="first"
    )
    return {
        label: int((frame[THREE_CLASS_COL] == label).sum())
        for label in LABEL_NAMES
    }


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    frame = pd.read_csv(
        args.binary_data,
        usecols=[TEXT_COL, INITIAL_SUBGROUP_COL, SECOND_SUBGROUP_COL],
    )
    frame["initial_subgroup"] = pd.to_numeric(
        frame[INITIAL_SUBGROUP_COL], errors="coerce"
    )
    frame["second_subgroup"] = pd.to_numeric(
        frame[SECOND_SUBGROUP_COL], errors="coerce"
    )
    subset = frame[
        frame["initial_subgroup"].isin(range(1, 17))
        & frame["second_subgroup"].isin(range(1, 17))
    ].copy()
    subset["initial_subgroup"] = subset["initial_subgroup"].astype(int)
    subset["second_subgroup"] = subset["second_subgroup"].astype(int)
    if len(subset) != args.expected_subset_n:
        raise ValueError(
            f"Expected {args.expected_subset_n} double-coded rows, found {len(subset)}"
        )

    subset["initial_three_class"] = subset["initial_subgroup"].map(mapping)
    subset["second_three_class"] = subset["second_subgroup"].map(mapping)
    subgroup_kappa = float(
        cohen_kappa_score(subset["initial_subgroup"], subset["second_subgroup"])
    )
    three_class_kappa = float(
        cohen_kappa_score(
            subset["initial_three_class"], subset["second_three_class"]
        )
    )

    full_counts = analytic_distribution(args.three_class_data)
    subset_counts = {
        label: int((subset["initial_three_class"] == label).sum())
        for label in LABEL_NAMES
    }
    full_n = sum(full_counts.values())
    subset_n = len(subset)
    rows = []
    for label, name in LABEL_NAMES.items():
        full_pct = 100 * full_counts[label] / full_n
        subset_pct = 100 * subset_counts[label] / subset_n
        rows.append(
            {
                "outcome_class": name,
                "full_analytic_corpus_n": full_counts[label],
                "full_analytic_corpus_percent": full_pct,
                "double_coded_subset_n": subset_counts[label],
                "double_coded_subset_percent": subset_pct,
                "percentage_point_difference": subset_pct - full_pct,
            }
        )
    rows.append(
        {
            "outcome_class": "Total",
            "full_analytic_corpus_n": full_n,
            "full_analytic_corpus_percent": 100.0,
            "double_coded_subset_n": subset_n,
            "double_coded_subset_percent": 100.0,
            "percentage_point_difference": 0.0,
        }
    )

    result = {
        "analysis": "independently double-coded subset agreement",
        "double_coded_subset_n": subset_n,
        "subgroup_kappa": subgroup_kappa,
        "mapped_three_class_kappa": three_class_kappa,
        "mapping_file": args.mapping.name,
        "outputs_contain_source_text": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "reviewer_agreement_class_distribution.csv", index=False
    )
    (args.output_dir / "reviewer_agreement.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
