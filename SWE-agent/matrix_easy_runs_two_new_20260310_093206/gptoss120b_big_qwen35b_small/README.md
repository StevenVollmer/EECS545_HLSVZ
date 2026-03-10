# Matrix Batch Results

- presets: `0`
- variants: `6`
- configs: `6`
- projects: `2`
- issues: `2`
- observed runs: `12`

## Variant Aggregate

| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| big_coder_only | 2 | 1/2 | 4.5/10 | 3.5/5 | 2.5/5 | 4.0/5 | 474228.5 | 1503.0 | 57268140.0 | 28.0 |
| big_planner_big_coder | 2 | 0/2 | 3.0/10 | 3.0/5 | 0.0/5 | 5.0/5 | 200765.0 | 4513.5 | 25175040.0 | 27.5 |
| big_planner_big_coder_big_reviewer | 2 | 1/2 | 4.5/10 | 4.0/5 | 0.0/5 | 4.0/5 | 527890.0 | 4138.5 | 64340040.0 | 45.0 |
| big_planner_small_coder | 2 | 2/2 | 6.5/10 | 5.0/5 | 0.5/5 | 5.0/5 | 230181.5 | 3322.0 | 9532272.5 | 31.0 |
| big_planner_small_coder_big_reviewer | 2 | 1/2 | 5.5/10 | 4.5/5 | 0.5/5 | 5.0/5 | 468034.0 | 7414.0 | 32866017.5 | 57.0 |
| small_coder_only | 2 | 2/2 | 5.5/10 | 5.0/5 | 2.5/5 | 5.0/5 | 61344.5 | 1556.0 | 2255977.5 | 14.5 |

## Issue Index

| Issue | Project | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pvlib__pvlib-python-1072 | pvlib__pvlib-python | 6 | 2/6 | 4.2/10 | 4.0/5 | 0.5/5 | 459855.0 | 4482.3 | 47409307.5 | big_planner_small_coder | 6.0/10 | skipped (stuck_repetition)=2, skipped (submitted (exit_cost))=1, skipped (submitted (exit_format))=1, skipped (submitted)=2 |
| pylint-dev__astroid-1866 | pylint-dev__astroid | 6 | 5/6 | 5.7/10 | 4.3/5 | 1.5/5 | 194292.8 | 3000.0 | 16403188.3 | big_coder_only | 8.0/10 | skipped (stuck_repetition)=1, skipped (submitted)=5 |

## Project Index

| Project | Issues | Configs | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pvlib__pvlib-python | 1 | 6 | 4.2/10 | 4.0/5 | 0.5/5 | 459855.0 | 4482.3 | 47409307.5 | big_planner_small_coder | [pvlib__pvlib-python](./projects/pvlib__pvlib-python/README.md) |
| pylint-dev__astroid | 1 | 6 | 5.7/10 | 4.3/5 | 1.5/5 | 194292.8 | 3000.0 | 16403188.3 | big_coder_only | [pylint-dev__astroid](./projects/pylint-dev__astroid/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
