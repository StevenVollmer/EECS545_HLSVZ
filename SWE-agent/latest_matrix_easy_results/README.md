# Matrix Batch Results

- presets: `0`
- variants: `7`
- configs: `7`
- projects: `1`
- issues: `1`
- observed runs: `7`

## Primary Aggregate

These metrics prioritize issue alignment, focused editing, validation after edits, and execution stability.

| Config | Instances | Submitted | Manual Submit | Validated After Edit | Aligned Edits | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| all_3_big | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 15.0/20 | 5.0/5 | 5.0/5 | 5.0/5 | 0.0/5 | 292673.0 | 4786.0 | 10578575.00 | 45.0 |
| big_coder_only | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 11.0/20 | 3.0/5 | 4.0/5 | 2.0/5 | 2.0/5 | 62808.0 | 1865.0 | 2328830.00 | 11.0 |
| big_planner_big_coder | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 17.0/20 | 5.0/5 | 5.0/5 | 5.0/5 | 2.0/5 | 98603.0 | 3177.0 | 3673495.00 | 29.0 |
| big_planner_small_coder | 1 | 0/1 | 0/1 | 1/1 | 1/1 | 15.0/20 | 5.0/5 | 5.0/5 | 4.0/5 | 1.0/5 | 99774.0 | 3357.0 | 1631376.00 | 26.0 |
| big_planner_small_coder_big_reviewer | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 11.0/20 | 3.0/5 | 4.0/5 | 2.0/5 | 2.0/5 | 85796.0 | 2199.0 | 1606098.00 | 22.0 |
| big_planner_small_coder_small_reviewer | 1 | 0/1 | 0/1 | 0/1 | 1/1 | 16.0/20 | 5.0/5 | 5.0/5 | 3.0/5 | 3.0/5 | 45789.0 | 2082.0 | 1304249.00 | 18.0 |
| small_coder_only | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 10.0/20 | 3.0/5 | 4.0/5 | 2.0/5 | 1.0/5 | 115210.0 | 1933.0 | 1071684.00 | 17.0 |

## Issue Index

| Issue | Project | Configs Run | Submitted | Validated After Edit | Aligned Edits | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Analysis | Exit Mix |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pydicom__pydicom-1458 | pydicom__pydicom | 7 | 2/7 | 3/7 | 4/7 | 13.6/20 | 4.1/5 | 4.6/5 | 3.3/5 | 1.6/5 | 114379.0 | 2771.3 | 3170615.29 | big_planner_big_coder | 17.0/20 | Uncaught ValidationError=1, exit_format=2, submitted=2, submitted (exit_format)=2 |

## Project Index

| Project | Issues | Configs | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pydicom__pydicom | 1 | 7 | 13.6/20 | 4.1/5 | 4.6/5 | 3.3/5 | 1.6/5 | 114379.0 | 2771.3 | 3170615.29 | big_planner_big_coder | [pydicom__pydicom](./projects/pydicom__pydicom/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
