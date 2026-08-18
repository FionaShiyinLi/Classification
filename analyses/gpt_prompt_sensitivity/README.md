# Exploratory GPT-5.2 taxonomy-aligned prompt

This experiment is intentionally separate from the historical GPT-5.2
evaluation. It does not edit `evaluate_all_methods.py`, historical result
files, the original prompt, or any manuscript.

## Prespecified workflow

1. Rebuild and verify the canonical seed-42 train/validation/test split.
2. Run the supplied three-item smoke test with reasoning `none` and `low`.
3. Compare `none` and `low` on all 2,252 validation outcomes.
4. Select the higher validation accuracy. If accuracy is exactly equal, select
   the higher validation macro F1. If both are equal, select `none`.
5. Freeze the prompt, examples, model snapshot, reasoning setting, batch size,
   dataset hash, and split hashes.
6. Make one held-out test pass with the frozen setting only.

The default provider remains direct OpenAI with the pinned snapshot
`gpt-5.2-2025-12-11`. An explicitly selected OpenRouter mode uses
`openai/gpt-5.2`; OpenRouter's model catalog identifies its canonical slug as
`openai/gpt-5.2-20251211`. OpenRouter results remain a distinct gateway
condition and are not presented as direct-OpenAI results. Every API batch uses
strict Structured Outputs with item IDs. Failed or malformed batches are never
replaced with fallback labels; they are logged and must succeed on a resumed
run.

## Commands

```bash
python3 analyses/gpt_prompt_sensitivity/run_experiment.py --phase prepare
python3 analyses/gpt_prompt_sensitivity/run_experiment.py --phase smoke
python3 analyses/gpt_prompt_sensitivity/run_experiment.py --phase validation
python3 analyses/gpt_prompt_sensitivity/run_experiment.py --phase freeze
python3 analyses/gpt_prompt_sensitivity/run_experiment.py --phase test --confirm-test-pass
```

For OpenRouter, add `--provider openrouter` to every command. Its default
artifacts are isolated under the ignored
`private_outputs/gpt_prompt_sensitivity/` directory. OpenRouter requests require
supported parameters and request no-data-collection and zero-data-retention
routing.

At present, OpenRouter may have no GPT-5.2 endpoint satisfying zero-data
retention. The independently authored smoke items can be tested with
`--openrouter-allow-non-zdr`; those artifacts remain under the same ignored
private-output root. This mode still
denies provider data collection. Validation and test outcome text remain
blocked unless `--acknowledge-non-zdr-study-data` is also supplied explicitly.

## Contrastive 18-example variant

`few_shot_examples_contrastive18.json` is a second exploratory calibration
variant. It uses the same taxonomy prompt and runner but must use a new output
directory. Several initially proposed phrases occurred verbatim in validation
or test data, so this file uses independently authored equivalents that preserve
the intended contrastive boundaries without normalized exact-text overlap with
held-out outcomes. It does not replace `few_shot_examples.json` or either prior
experiment.

The 18-example file has no normalized exact-text match in train, validation,
or test. In the 10-example file, the short generic phrases “Serious adverse
events” and “Hospital readmission” also occur in training, but neither occurs
as an exact validation or test example. This is recorded explicitly rather
than claiming zero overlap with the entire corpus.

After both test runs, reproduce the review-cluster confidence intervals and
paired comparisons with:

```bash
python3 analyses/gpt_prompt_sensitivity/analyze_cluster_results.py
```

Direct OpenAI phases require `OPENAI_API_KEY`. OpenRouter phases prefer
`OPENROUTER_API_KEY` and also accept an OpenRouter credential in
`OPENAI_API_KEY` for compatibility with an existing local `.env` file (see
`.env.example`; never commit the real key). Raw batch responses and row-level
predictions are written under `private_outputs/gpt_prompt_sensitivity/`, which
is ignored because it contains restricted outcome text. Only reviewed
aggregate metrics are copied to `results/analyses/gpt_prompt_sensitivity/`.
