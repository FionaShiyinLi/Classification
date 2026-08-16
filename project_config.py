"""Shared reproducibility settings for the manuscript analyses.

The model revisions below identify the exact Hugging Face snapshots retained in
the final local audit environment.  Training schedules are intentionally kept
analysis-specific because the recorded historical runs used different
schedulers; changing a scheduler would require rerunning the affected results.
"""

from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")

MODEL_REVISIONS = {
    "bioformers/bioformer-8L": "e9d5990ea54f382afbb390c9bcce9d3a0c035873",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract": "d673b8835373c6fa116d6d8006b33d48734e305d",
    "dmis-lab/biobert-base-cased-v1.2": "67c9c25b46986521ca33df05d8540da1210b3256",
    "allenai/scibert_scivocab_uncased": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
    "answerdotai/ModernBERT-base": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
}

# Explicitly recorded from the retained training arguments.
OPTIMIZER_NAME = "adamw_torch_fused"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0

# Historical schedules used by the reported analyses.
PRIMARY_SCHEDULER = "linear"
COMPARATOR_SCHEDULER = "cosine"
ABLATION_SCHEDULER = "cosine"
