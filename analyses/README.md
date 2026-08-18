# Analysis workflows

Each subdirectory contains one bounded manuscript analysis. Scripts read
restricted inputs through explicit command-line arguments or the variables in
`.env.example`. Analyses that create row-level or model artifacts write to the
ignored `private_outputs/` directory by default.

Only reviewed, text-free summaries are copied to `results/analyses/`.

| Directory | Purpose |
|---|---|
| `leakage/` | Review overlap, exact matches, and character 5-gram Jaccard audit |
| `uncertainty/` | Cluster-bootstrap CIs and paired Table 2 comparisons |
| `review_separated/` | Internal validation with non-overlapping Cochrane reviews |
| `gpt_batch_size/` | Prompt batch-size sensitivity |
| `gpt_prompt_sensitivity/` | Taxonomy-aligned 10- and 18-example GPT analyses |
| `hybrid_threshold/` | Validation-only threshold selection and locked test run |
| `modernbert/` | ModernBERT post hoc comparator |
| `similarity/` | Primary performance by nearest-training similarity |
| `reviewer_agreement/` | Double-coded-subset agreement and Table S8 |
| `subgroup_performance/` | Sixteen-subgroup prediction distribution and Table S10 |

