# Matrix Batch Results

- variants: `6`
- projects: `1`
- issues: `1`

## Variant Aggregate

| Variant | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| big_coder_only | 1 | 1/1 | 10.0/10 | 5.0/5 | 5.0/5 | 5.0/5 | 15748.0 | 567.0 | 590870.0 | 8.0 |
| big_planner_big_coder | 1 | 1/1 | 5.0/10 | 4.0/5 | 1.0/5 | 5.0/5 | 108989.0 | 2922.0 | 4019155.0 | 29.0 |
| big_planner_big_coder_big_reviewer | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 124475.0 | 2884.0 | 4558505.0 | 34.0 |
| big_planner_small_coder | 1 | 0/1 | 1.0/10 | 0.0/5 | 3.0/5 | 4.0/5 | 32628.0 | 912.0 | 1205820.0 | 13.0 |
| big_planner_small_coder_big_reviewer | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 73467.0 | 1969.0 | 1472407.0 | 28.0 |
| small_coder_only | 1 | 1/1 | 4.0/10 | 4.0/5 | 1.0/5 | 5.0/5 | 75537.0 | 1526.0 | 707301.0 | 19.0 |

## Project Index

| Project | Issues | Variants | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pylint-dev__astroid | 1 | 6 | 5.3/10 | 3.8/5 | 2.0/5 | 71807.3 | 1796.7 | 2092343.0 | big_coder_only | [pylint-dev__astroid](./projects/pylint-dev__astroid/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
