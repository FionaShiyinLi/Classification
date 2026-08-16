#!/usr/bin/env python3
"""Leakage-risk audit for the outcome-classification item-level split.

This script recreates the manuscript's fixed stratified 70/10/20 item-level
split and reports review overlap and near-duplicate text similarity between
test and training outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


TEXT_COL = "outcome"
LABEL_COL = "outcome.class"
REVIEW_COL = "CDSR.id"
OUTCOME_ID_COL = "outcome.id"
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
load_dotenv(REPO_ROOT / ".env")
DEFAULT_DATA = Path(
    os.getenv(
        "OUTCOME_DATASET_CSV",
        REPO_ROOT / "restricted_data" / "outcome_3cls.csv",
    )
)
DEFAULT_OUTPUT = REPO_ROOT / "private_outputs" / "leakage"

_DASH_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")


def normalize_text(value: str) -> str:
    text = str(value).lower().strip()
    text = _DASH_RE.sub("-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def char_ngrams(value: str, n: int = 5) -> set[str]:
    text = normalize_text(value)
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def load_and_clean(path: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(path)
    df = raw[[REVIEW_COL, OUTCOME_ID_COL, TEXT_COL, LABEL_COL]].copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df[(df[TEXT_COL].str.len() > 0) & (df[LABEL_COL].isin([0, 1, 2]))].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    pre_dedup_n = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL, LABEL_COL], keep="first").reset_index(drop=True)
    return df, pre_dedup_n


def stratified_split(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _, group in df.groupby(LABEL_COL):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.10 * n))
        n_test = n - n_train - n_val

        if n >= 1 and n_train == 0:
            n_train = 1
            n_test = n - n_train - n_val
        if n >= 10 and n_val == 0:
            n_val = 1
            n_test = n - n_train - n_val
        if n_test < 0:
            give = min(n_val, -n_test)
            n_val -= give
            n_test += give
        if n_test < 0:
            give = min(max(n_train - 1, 0), -n_test)
            n_train -= give
            n_test += give

        train_idx += idx[:n_train].tolist()
        val_idx += idx[n_train : n_train + n_val].tolist()
        test_idx += idx[n_train + n_val : n_train + n_val + n_test].tolist()

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def similarity_bin(value: float) -> str:
    if value < 0.25:
        return "<0.25"
    if value < 0.50:
        return "0.25-<0.50"
    if value < 0.75:
        return "0.50-<0.75"
    if value < 0.90:
        return "0.75-<0.90"
    return ">=0.90"


def add_nearest_train_similarity(train: pd.DataFrame, test: pd.DataFrame, ngram: int = 5) -> pd.DataFrame:
    train_sets = [char_ngrams(text, ngram) for text in train[TEXT_COL]]
    test_sets = [char_ngrams(text, ngram) for text in test[TEXT_COL]]

    inverted_index: dict[str, list[int]] = defaultdict(list)
    for i, shingles in enumerate(train_sets):
        for shingle in shingles:
            inverted_index[shingle].append(i)

    nearest_similarity: list[float] = []
    nearest_index: list[int] = []

    for shingles in test_sets:
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(inverted_index.get(shingle, ()))

        best_similarity = 0.0
        best_index = -1
        for i in candidates:
            train_shingles = train_sets[i]
            union_size = len(shingles | train_shingles)
            similarity = len(shingles & train_shingles) / union_size if union_size else 0.0
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i

        nearest_similarity.append(best_similarity)
        nearest_index.append(best_index)

    audited = test.copy()
    audited["nearest_train_jaccard_5gram"] = nearest_similarity
    audited["nearest_train_outcome"] = [
        train.loc[i, TEXT_COL] if i >= 0 else "" for i in nearest_index
    ]
    audited["nearest_train_label"] = [
        int(train.loc[i, LABEL_COL]) if i >= 0 else None for i in nearest_index
    ]
    audited["nearest_train_review"] = [
        train.loc[i, REVIEW_COL] if i >= 0 else "" for i in nearest_index
    ]
    audited["similarity_bin"] = audited["nearest_train_jaccard_5gram"].map(similarity_bin)
    return audited


def build_summary(
    source_data: Path,
    pre_dedup_n: int,
    df: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    audited_test: pd.DataFrame,
    ngram: int,
) -> dict:
    train_reviews = set(train[REVIEW_COL])
    test_reviews = set(test[REVIEW_COL])
    train_exact_text = set(train[TEXT_COL])
    train_exact_text_label = set(zip(train[TEXT_COL], train[LABEL_COL]))

    train_norm = train[TEXT_COL].map(normalize_text)
    test_norm = test[TEXT_COL].map(normalize_text)
    train_norm_text = set(train_norm)
    train_norm_text_label = set(zip(train_norm, train[LABEL_COL]))

    bins = []
    for label in ["<0.25", "0.25-<0.50", "0.50-<0.75", "0.75-<0.90", ">=0.90"]:
        subset = audited_test[audited_test["similarity_bin"] == label]
        bins.append(
            {
                "similarity_bin": label,
                "n": int(len(subset)),
                "percent": float(100 * len(subset) / len(audited_test)),
                "median_similarity": (
                    float(np.median(subset["nearest_train_jaccard_5gram"]))
                    if len(subset)
                    else None
                ),
            }
        )

    jaccard_ge_075 = int((audited_test["nearest_train_jaccard_5gram"] >= 0.75).sum())
    jaccard_ge_090 = int((audited_test["nearest_train_jaccard_5gram"] >= 0.90).sum())

    return {
        "source_data": str(source_data),
        "pre_dedup_valid_rows": int(pre_dedup_n),
        "post_dedup_rows": int(len(df)),
        "exact_text_label_duplicates_removed": int(pre_dedup_n - len(df)),
        "split_sizes": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "class_counts_after_dedup": {
            str(k): int(v) for k, v in df[LABEL_COL].value_counts().sort_index().items()
        },
        "unique_reviews_after_dedup": int(df[REVIEW_COL].nunique()),
        "train_unique_reviews": int(train[REVIEW_COL].nunique()),
        "test_unique_reviews": int(test[REVIEW_COL].nunique()),
        "test_reviews_also_in_train": int(len(test_reviews & train_reviews)),
        "test_review_overlap_percent_reviews": float(
            100 * len(test_reviews & train_reviews) / len(test_reviews)
        ),
        "test_items_from_reviews_also_in_train": int(test[REVIEW_COL].isin(train_reviews).sum()),
        "test_item_review_overlap_percent": float(
            100 * test[REVIEW_COL].isin(train_reviews).sum() / len(test)
        ),
        "test_items_with_exact_text_in_train_any_label": int(test[TEXT_COL].isin(train_exact_text).sum()),
        "test_items_with_exact_text_and_label_in_train": int(
            sum((text, label) in train_exact_text_label for text, label in zip(test[TEXT_COL], test[LABEL_COL]))
        ),
        "test_items_with_normalized_text_in_train_any_label": int(test_norm.isin(train_norm_text).sum()),
        "test_items_with_normalized_text_and_label_in_train": int(
            sum((text, label) in train_norm_text_label for text, label in zip(test_norm, test[LABEL_COL]))
        ),
        "ngram": ngram,
        "test_items_nearest_jaccard_ge_0_75": jaccard_ge_075,
        "test_item_nearest_jaccard_ge_0_75_percent": float(100 * jaccard_ge_075 / len(audited_test)),
        "test_items_nearest_jaccard_ge_0_90": jaccard_ge_090,
        "test_item_nearest_jaccard_ge_0_90_percent": float(100 * jaccard_ge_090 / len(audited_test)),
        "similarity_bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument(
        "--write-row-level",
        action="store_true",
        help=(
            "Write the restricted row-level similarity file. By default only "
            "text-free aggregate summaries are written."
        ),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df, pre_dedup_n = load_and_clean(args.data)
    train, validation, test = stratified_split(df, args.seed)
    audited_test = add_nearest_train_similarity(train, test, args.ngram)
    summary = build_summary(args.data, pre_dedup_n, df, train, validation, test, audited_test, args.ngram)

    if args.write_row_level:
        audited_test.to_csv(
            args.outdir / "leakage_similarity_test_items.csv", index=False
        )
    pd.DataFrame(summary["similarity_bins"]).to_csv(
        args.outdir / "leakage_similarity_summary.csv", index=False
    )
    with open(args.outdir / "leakage_similarity_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
