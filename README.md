# Automated outcome-type classification for outcome-stratified meta-analysis

Code accompanying the manuscript on automated classification of clinical trial outcomes into `objective`, `semi-objective`, and `subjective` categories using Cochrane review data.

## Expected environment

- Python 3.9+
- PyTorch
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`
- `scipy`
- `openai` for the LLM evaluation path only

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data availability

`outcome_3cls.csv` in this repository should remain a small sample file for format inspection and pipeline smoke tests only. The full dataset used in the study cannot be shared publicly because of source-material redistribution restrictions. The full dataset is available from the corresponding author upon reasonable request.

## Included result artifacts

The repository may include a small set of paper-facing summary artifacts in `results/`:

- `results/primary_checkpoint_metrics.json`
- `results/search_results.json`
- `results/ablation_results.json`
- `results/model_comparison_results.json`
- `results/evaluation_results.json`
- `results/ensemble_results.json`

These JSON files are intended to document the final metrics reported in the manuscript. They are not substitutes for full training outputs or model checkpoints.

## Reproducing local analyses

Main evaluation:

```bash
python3 evaluate_all_methods.py
```

Hyperparameter sweep summary:

```bash
python3 evaluate_all_methods_param_search.py
```

Cross-model comparison:

```bash
python3 train_other_models.py
```

To evaluate only the fixed Bioformer baseline row without retraining comparator backbones:

```bash
OUTCOME_SKIP_COMPARATOR_TRAINING=1 python3 train_other_models.py
```

Ensemble experiment:

```bash
python3 train_ensemble.py
```

Ablation study:

```bash
python3 ablation_study.py
```

