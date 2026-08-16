#!/usr/bin/env python3
"""Fail if the proposed public repository contains common privacy hazards."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024
PROHIBITED_NAMES = {
    ".env",
    "aligned_test_predictions.csv",
    "leakage_similarity_test_items.csv",
    "review_level_split_assignments.csv",
    "review_level_test_predictions.csv",
}
PROHIBITED_SUFFIXES = {
    ".doc",
    ".docx",
    ".pdf",
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
}
PROHIBITED_PARTS = {"cache", "batches", "checkpoint", "private_outputs"}
SECRET_PATTERN = re.compile(rb"(?:sk-or-v1-|sk-)[A-Za-z0-9_-]{20,}")
# Split the literal so the audit script does not flag its own detector.
ABSOLUTE_USER_PATH = re.compile(rb"/" + rb"Users/" + rb"[^/\s]+/")


def candidate_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line]


def audit() -> list[str]:
    failures: list[str] = []
    files = candidate_files()
    for path in files:
        relative = path.relative_to(REPO_ROOT)
        parts = set(relative.parts)
        if path.name in PROHIBITED_NAMES:
            failures.append(f"prohibited file: {relative}")
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            failures.append(f"prohibited binary/document type: {relative}")
        if parts.intersection(PROHIBITED_PARTS) or any(
            part.startswith("checkpoint-") for part in relative.parts
        ):
            failures.append(f"prohibited generated-output path: {relative}")
        if path.is_file() and path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            failures.append(f"file exceeds 10 MiB: {relative}")
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if SECRET_PATTERN.search(content):
            failures.append(f"possible API secret: {relative}")
        if ABSOLUTE_USER_PATH.search(content):
            failures.append(f"machine-specific /Users path: {relative}")
        if path.suffix == ".json":
            try:
                json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                failures.append(f"invalid JSON: {relative}")

    sample = REPO_ROOT / "outcome_3cls.csv"
    if not sample.is_file():
        failures.append("missing public sample: outcome_3cls.csv")
    else:
        with sample.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
        if len(rows) > 500:
            failures.append(
                f"outcome_3cls.csv has {len(rows) - 1} data rows; expected a small sample"
            )
        required = {"CDSR.id", "outcome.id", "outcome", "outcome.class"}
        if not rows or not required.issubset(set(rows[0])):
            failures.append("outcome_3cls.csv does not have the documented sample schema")
    return failures


def main() -> int:
    failures = audit()
    if failures:
        print("Public-repository audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Public-repository audit passed: no prohibited files or obvious secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
