# Reproducibility guide

## Scope

The repository separates public code and aggregate results from restricted
outcome text. Reproducing manuscript numbers requires the full private inputs
and retained checkpoints; `outcome_3cls.csv` at repository root is only a small
format sample.

## Result-to-code map

| Manuscript result | Code | Public aggregate output |
|---|---|---|
| Primary Bioformer metrics | `evaluate_primary_checkpoint.py` | `results/primary_checkpoint_metrics.json` |
| Hyperparameter sweep/Table 3 | `evaluate_all_methods_param_search.py` | `results/search_results.json` |
| Encoder comparison/Table 2 | `train_other_models.py` | `results/model_comparison_results.json` |
| Table 1/2 CIs and paired tests | `analyses/uncertainty/run_table_uncertainty_analysis.py` | `results/analyses/uncertainty/` |
| Ablations/Table 4 | `ablation_study.py` | `results/ablation_results.json` |
| Five-seed ensemble | `train_ensemble.py` | `results/ensemble_results_aligned_primary.json` |
| Review-separated validation | `analyses/review_separated/run_review_level_validation.py` | `results/analyses/review_separated/` |
| GPT batch-size sensitivity | `analyses/gpt_batch_size/run_gpt_batch_size_ablation.py` | `results/analyses/gpt_batch_size/` |
| GPT prompt sensitivities | `analyses/gpt_prompt_sensitivity/run_experiment.py` | `results/analyses/gpt_prompt_sensitivity/` |
| GPT cluster-aware comparisons | `analyses/gpt_prompt_sensitivity/analyze_cluster_results.py` | same directory |
| Validation-selected 0.55 hybrid | `analyses/hybrid_threshold/run_threshold_experiment.py` | `results/analyses/hybrid_threshold/` |
| ModernBERT | `analyses/modernbert/run_modernbert_baseline.py` | `results/analyses/modernbert/` |
| Leakage/near-duplicate audit | `analyses/leakage/leakage_risk_audit.py` | `results/analyses/leakage/` |
| Similarity-stratified/Table S11 | `analyses/similarity/run_similarity_stratified_analysis.py` | `results/analyses/similarity/` |
| Double-coding/Table S8 | `analyses/reviewer_agreement/analyze_reviewer_agreement.py` | `results/analyses/reviewer_agreement/` |
| Sixteen subgroups/Table S10 | `analyses/subgroup_performance/analyze_subgroup_performance.py` | `results/analyses/subgroup_performance/` |

## Recommended analysis order

1. Provide the restricted full three-class and 16-subgroup datasets.
2. Recreate the fixed seed-42 split and primary checkpoint predictions.
3. Run the leakage audit with `--write-row-level` into `private_outputs/` when
   the private test manifest is needed downstream.
4. Run the Table 1/2 uncertainty analysis; its private aligned prediction file
   is used by ModernBERT and similarity analyses.
5. Run the remaining independent analyses.
6. Copy only reviewed, text-free aggregate artifacts into `results/analyses/`.
7. Run the public repository audit and unit tests before staging.

## Statistical conventions

- Accuracy and macro F1 uncertainty use percentile cluster bootstrap intervals
  with Cochrane review (`CDSR.id`) as the resampling unit where stated.
- Table 2 comparisons use paired permutation tests at the Cochrane-review level
  and Holm correction across the planned comparisons.
- GPT 10-versus-18 and GPT-versus-Bioformer differences use identical
  review-cluster bootstrap resamples.
- The hybrid threshold is selected on validation data by highest hybrid
  accuracy, then higher macro F1, lower routing rate, and lower threshold. The
  selected 0.55 threshold is then evaluated once on the test set.

## Model and optimizer provenance

Exact retained Hugging Face snapshot identifiers are in
`config/model_revisions.json`. `project_config.py` explicitly records fused
AdamW, beta values, epsilon, gradient clipping, and the analysis-specific
schedulers.

The final audit environment is pinned in `requirements.txt`. This is not a
claim that every package version from the original historical runs was
recoverable. The primary checkpoint records Transformers 4.57.6; the retained
review-separated checkpoint records 4.57.1, and the earlier GPT sensitivity
run metadata record OpenAI SDK 2.7.1.

## GPT prompt provenance

The versioned prompt and example files are under
`analyses/gpt_prompt_sensitivity/`. Their SHA-256 hashes match those reported in
the supplement. The 18-example authored set has no normalized exact match in
train, validation, or test. In the 10-example set, the two short generic phrases
“Serious adverse events” and “Hospital readmission” also occur in training;
neither occurs as an exact validation or test example. Therefore the repository
does not claim that the 10-example set has zero overlap with the entire corpus.

## Private artifacts

Never publish the following:

- the complete annotated datasets;
- row-level outcome predictions or split assignments;
- files containing test text and nearest-training text;
- GPT request batches, raw provider responses, or API logs;
- preprocessing reports containing outcome text;
- model checkpoints or optimizer state without a separate licensing and
  memorization review; or
- `.env` or any API credential.

