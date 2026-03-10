# FINETUNE Code Package


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

Run all commands from this `finetune_code/` directory.

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
