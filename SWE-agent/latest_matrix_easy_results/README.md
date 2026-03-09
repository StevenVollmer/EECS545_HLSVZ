# Matrix Batch Results

- variants: `7`
- projects: `1`
- issues: `1`

## Variant Aggregate

| Variant | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| all_3_big | 1 | 1/1 | 6.0/10 | 5.0/5 | 0.0/5 | 5.0/5 | 292673.0 | 4786.0 | 10578575.0 | 45.0 |
| big_coder_only | 1 | 0/1 | 0.0/10 | 0.0/5 | 2.0/5 | 4.0/5 | 62808.0 | 1865.0 | 2328830.0 | 11.0 |
| big_planner_big_coder | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 98603.0 | 3177.0 | 3673495.0 | 29.0 |
| big_planner_small_coder | 1 | 0/1 | 4.0/10 | 3.0/5 | 0.0/5 | 5.0/5 | 99774.0 | 3357.0 | 1631376.0 | 26.0 |
| big_planner_small_coder_big_reviewer | 1 | 0/1 | 3.0/10 | 1.0/5 | 2.0/5 | 4.0/5 | 85796.0 | 2199.0 | 1606098.0 | 22.0 |
| big_planner_small_coder_small_reviewer | 1 | 0/1 | 5.0/10 | 3.0/5 | 2.0/5 | 5.0/5 | 45789.0 | 2082.0 | 1304249.0 | 18.0 |
| small_coder_only | 1 | 0/1 | 0.0/10 | 0.0/5 | 0.0/5 | 4.0/5 | 115210.0 | 1933.0 | 1071684.0 | 17.0 |

## Project Index

| Project | Issues | Variants | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pydicom__pydicom | 1 | 7 | 3.4/10 | 2.4/5 | 1.0/5 | 114379.0 | 2771.3 | 3170615.3 | all_3_big | [pydicom__pydicom](./projects/pydicom__pydicom/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
