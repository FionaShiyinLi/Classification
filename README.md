# Biomedical language models outperform prompted GPT-5.2 for automated outcome classification in systematic reviews

Code and text-free aggregate results accompanying the manuscript by Li et al.
The study classifies systematic-review outcome descriptions into the
study-specific operational `Objective`, `Semi-objective`, and `Subjective`
classes.

## What is included

- Core Bioformer training, model comparison, ablation, ensemble, and historical
  GPT evaluation scripts.
- Reviewer-requested and post hoc analyses under `analyses/`.
- Versioned prompts and demonstration examples for the GPT-5.2 sensitivities.
- Text-free aggregate result artifacts under `results/`.
- A 142-record sample CSV for schema inspection and smoke tests only.

The public sample cannot reproduce the manuscript's numerical results. The full
annotated dataset contains third-party Cochrane outcome text and is not
redistributed.

## Repository layout

```text
.
├── outcome_3cls.csv                 # small public schema sample
├── evaluate_all_methods.py          # primary + historical GPT evaluation
├── evaluate_primary_checkpoint.py   # exact primary-checkpoint evaluation
├── evaluate_all_methods_param_search.py
├── train_other_models.py
├── train_ensemble.py
├── ablation_study.py
├── project_config.py                # optimizer, scheduler, model revisions
├── config/                          # model and subgroup mappings
├── analyses/                        # reviewer/post hoc workflows
├── results/                         # curated text-free paper results
├── scripts/audit_public_repository.py
└── tests/
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the result-to-code map and
analysis order.

## Installation

The pinned requirements describe the environment used for the final post hoc
audits. The manuscript notes that every package version from the original
historical runs could not be recovered.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The primary retained checkpoint records Transformers 4.57.6. Exact Hugging Face
model snapshot revisions are recorded in `config/model_revisions.json` and used
by the training scripts.

## Restricted inputs

Keep restricted files outside Git or in the ignored `restricted_data/`
directory. Paths may be supplied through `.env` or command-line arguments.

```bash
cp .env.example .env
```

Common variables are:

```text
OUTCOME_DATASET_CSV=/absolute/private/path/outcome_3cls.csv
OUTCOME_BINARY_DATASET_CSV=/absolute/private/path/binary.outcome.csv
OUTCOME_TEST_MANIFEST=/absolute/private/path/leakage_similarity_test_items.csv
OUTCOME_REFERENCE_CHECKPOINT=/absolute/private/path/best_model
OUTCOME_PRIMARY_PREDICTIONS=/absolute/private/path/test_predictions.csv
```

API keys belong only in `.env`; `.env` is ignored and must never be committed.
Repository entry points load this file automatically. If the restricted dataset
path is not configured, manuscript workflows fail instead of silently using the
142-record public sample. New run outputs go to ignored `private_outputs/` by
default; copy an aggregate into `results/` only after checking it against the
manuscript.

## Main entry points

| Result family | Command |
|---|---|
| Original GPT-5.2 comparison | `python evaluate_all_methods.py` |
| Primary checkpoint | `python evaluate_primary_checkpoint.py` |
| Sixteen-run sweep | `python evaluate_all_methods_param_search.py` |
| PubMedBERT/BioBERT/SciBERT | `python train_other_models.py` |
| Ablations | `python ablation_study.py` |
| Five-seed ensemble | `python train_ensemble.py` |
| Leakage and 5-gram similarity | `python analyses/leakage/leakage_risk_audit.py` |
| Table 1/2 uncertainty | `python analyses/uncertainty/run_table_uncertainty_analysis.py` |
| Review-separated validation | `python analyses/review_separated/run_review_level_validation.py` |
| GPT batch-size sensitivity | `python analyses/gpt_batch_size/run_gpt_batch_size_ablation.py` |
| GPT prompt sensitivity | `python analyses/gpt_prompt_sensitivity/run_experiment.py` |
| GPT cluster comparisons | `python analyses/gpt_prompt_sensitivity/analyze_cluster_results.py` |
| Validation-selected hybrid | `python analyses/hybrid_threshold/run_threshold_experiment.py select`, then `python analyses/hybrid_threshold/run_threshold_experiment.py test` |
| ModernBERT | `python analyses/modernbert/run_modernbert_baseline.py` |
| Similarity-stratified performance | `python analyses/similarity/run_similarity_stratified_analysis.py` |
| Reviewer agreement | `python analyses/reviewer_agreement/analyze_reviewer_agreement.py` |
| Sixteen-subgroup performance | `python analyses/subgroup_performance/analyze_subgroup_performance.py` |

The historical heuristic 0.70 hybrid is not part of the core evaluator or the
curated results. The current manuscript result is produced only by the
validation-only threshold workflow, which selected 0.55 before its one locked
test evaluation. The complete validation threshold table retains 0.70 as an
inferior candidate for transparency.

## Training schedules

The code preserves the schedules recorded in the retained runs:

- Linear: primary sweep, ensemble, and review-separated validation.
- Cosine: comparator encoders, ablations, and ModernBERT.

Changing these schedules would define new experiments and would require
replacing the corresponding reported results.

## Results and privacy

`results/` contains curated aggregate artifacts only. Row-level prediction
files, split assignments, nearest-training text, raw GPT requests/responses,
model checkpoints, and preprocessing reports remain ignored in
`private_outputs/` or other private locations.

Before every commit, run:

```bash
python scripts/audit_public_repository.py
python -m unittest discover -s tests -v
```

The public audit checks the sample size, common secret patterns, machine-local
paths, large files, model binaries, and prohibited text-bearing artifact names.

## Data availability

The analyzed outcome descriptions were derived from published Cochrane Reviews
and contain third-party copyrighted material that the authors do not have
permission to redistribute as a complete text-level annotated dataset.
Researchers should independently obtain the underlying material through the
Cochrane Library and applicable data-request process, subject to its access and
reuse terms. The authors received no special access privileges.
