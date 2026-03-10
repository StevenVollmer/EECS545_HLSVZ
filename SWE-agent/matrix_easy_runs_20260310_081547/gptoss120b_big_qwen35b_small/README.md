# Matrix Batch Results

- variants: `6`
- projects: `1`
- issues: `1`

## Variant Aggregate

| Variant | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| big_coder_only | 1 | 0/1 | 0.0/10 | 0.0/5 | 4.0/5 | 1.0/5 | 0.0 | 0.0 | 0.0 | 1.0 |
| big_planner_big_coder | 1 | 0/1 | 0.0/10 | 0.0/5 | 4.0/5 | 1.0/5 | 0.0 | 0.0 | 0.0 | 1.0 |
| big_planner_big_coder_big_reviewer | 1 | 0/1 | 0.0/10 | 0.0/5 | 4.0/5 | 1.0/5 | 0.0 | 0.0 | 0.0 | 1.0 |
| big_planner_small_coder | 1 | 0/1 | 0.0/10 | 0.0/5 | 4.0/5 | 1.0/5 | 0.0 | 0.0 | 0.0 | 1.0 |
| big_planner_small_coder_big_reviewer | 1 | 0/1 | 0.0/10 | 0.0/5 | 4.0/5 | 1.0/5 | 0.0 | 0.0 | 0.0 | 1.0 |
| small_coder_only | 1 | 1/1 | 6.0/10 | 5.0/5 | 3.0/5 | 5.0/5 | 71092.0 | 1411.0 | 2586990.0 | 17.0 |

## Project Index

| Project | Issues | Variants | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pylint-dev__astroid | 1 | 6 | 1.0/10 | 0.8/5 | 3.8/5 | 11848.7 | 235.2 | 431165.0 | small_coder_only | [pylint-dev__astroid](./projects/pylint-dev__astroid/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
