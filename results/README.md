# Curated paper results

This directory contains reviewed, text-free result artifacts corresponding to
the manuscript and supplement.

- Root JSON files support the core model, sweep, ablation, comparison, original
  GPT, and aligned ensemble results.
- `analyses/` contains reviewer-requested and post hoc aggregate outputs.

The following are intentionally excluded: outcome text, row-level predictions,
split assignments, nearest-training text, GPT raw requests/responses, request
logs, model checkpoints, optimizer state, and absolute machine-local paths.

The obsolete heuristic 0.70 hybrid and the older non-aligned ensemble artifact
are not part of the curated results. The current hybrid files under
`analyses/hybrid_threshold/` record validation selection of 0.55 and the single
locked test evaluation.

