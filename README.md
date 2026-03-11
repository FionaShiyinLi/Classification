# Automated outcome-type classification for outcome-stratified meta-analysis: a feasibility study using Cochrane reviews

# Abstract
Outcome classification into objective, semi-objective, and subjective categories is important for outcome-stratified meta-analysis and for understanding how heterogeneity varies across studies. Previous work suggests that heterogeneity differs by outcome type, but applying this approach to large evidence bases requires classifying many thousands of outcomes. Manual classification is labor-intensive, time-consuming, and subject to disagreement between reviewers, creating a need for scalable automated methods. We assessed whether automated approaches can classify clinical trial outcomes into these three categories and compared fine-tuned biomedical language models with prompt-based large language models. We evaluated these methods on 22,518 unique clinical trial outcomes from 3,929 Cochrane meta-analyses, derived from 43,339 raw entries after cleaning and deduplication. Fine-tuned Bioformer-8L achieved 90.07% accuracy (95% CI: 89.27%–90.94%; macro F1 0.905; Cohen’s κ = 0.82), with strong performance across all three classes, whereas few-shot prompting with GPT-5.2 reached 57.9% accuracy (macro F1 0.58). The fine-tuned model was also substantially faster than the API-based model. These findings support the use of fine-tuned biomedical language models for scalable outcome-type classification in large evidence-synthesis workflows.

## Expected environment

- Python 3.9+ (3.10+ recommended)
- PyTorch
- transformers
- datasets
- scikit-learn
- pandas
- numpy
- scipy
- openai (for OpenRouter-based LLM evaluation only)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce paper artifacts


1) Hyperparameter search (paper Section 3.1.3):

```bash
python3 evaluate_all_methods_param_search.py
```

2) Main evaluation (fine-tuned model + optional LLM + hybrid):

```bash
python3 evaluate_all_methods.py
```

3) Ensemble experiment:

```bash
python3 train_ensemble.py
```

4) Cross-model comparison:

```bash
python3 train_other_models.py
```


5) Ablation:

```bash
python3 ablation_study.py
```
