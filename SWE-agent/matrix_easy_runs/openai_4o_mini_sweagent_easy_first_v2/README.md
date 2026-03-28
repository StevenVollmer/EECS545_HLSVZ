# Matrix Batch Results

- presets: `0`
- variants: `1`
- configs: `1`
- projects: `1`
- issues: `1`
- observed runs: `1`

## Primary Aggregate

These metrics prioritize issue alignment, focused editing, validation after edits, and execution stability.

| Config | Instances | Submitted | Manual Submit | Validated After Edit | Aligned Edits | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| small_coder_only | 1 | 1/1 | 1/1 | 0/1 | 0/1 | 10.0/20 | 1.0/5 | 4.0/5 | 3.0/5 | 2.0/5 | 11491.0 | 149.0 | 11789.00 | 4.0 |

## Issue Index

| Issue | Project | Configs Run | Submitted | Validated After Edit | Aligned Edits | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Analysis | Exit Mix |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pydicom__pydicom-1458 | pydicom__pydicom | 1 | 1/1 | 0/1 | 0/1 | 10.0/20 | 1.0/5 | 4.0/5 | 3.0/5 | 2.0/5 | 11491.0 | 149.0 | 11789.00 | small_coder_only | 10.0/20 | submitted=1 |

## Project Index

| Project | Issues | Configs | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pydicom__pydicom | 1 | 1 | 10.0/20 | 1.0/5 | 4.0/5 | 3.0/5 | 2.0/5 | 11491.0 | 149.0 | 11789.00 | small_coder_only | [pydicom__pydicom](./projects/pydicom__pydicom/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
