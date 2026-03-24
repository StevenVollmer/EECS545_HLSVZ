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
| big_coder_only | 2 | 1/2 | 2.5/10 | 2.5/5 | 3.5/5 | 3.0/5 | 24731.0 | 606.5 | 908040.0 | 8.5 |
| big_planner_big_coder | 2 | 2/2 | 6.5/10 | 5.0/5 | 2.0/5 | 5.0/5 | 88241.0 | 2188.0 | 3241595.0 | 19.5 |
| big_planner_big_coder_big_reviewer | 2 | 1/2 | 5.5/10 | 4.0/5 | 1.0/5 | 5.0/5 | 136822.5 | 3316.5 | 5020942.5 | 30.5 |
| big_planner_small_coder | 2 | 1/2 | 6.0/10 | 4.5/5 | 1.0/5 | 5.0/5 | 337971.5 | 4529.5 | 3279014.5 | 35.5 |
| big_planner_small_coder_big_reviewer | 2 | 0/2 | 4.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 456851.5 | 7978.0 | 4386918.5 | 50.5 |
| small_coder_only | 2 | 2/2 | 5.0/10 | 4.0/5 | 3.0/5 | 5.0/5 | 28320.0 | 830.0 | 269820.0 | 9.0 |

## Issue Index

| Issue | Project | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pvlib__pvlib-python-1072 | pvlib__pvlib-python | 6 | 2/6 | 4.7/10 | 3.5/5 | 1.5/5 | 207834.7 | 3306.0 | 3092766.0 | small_coder_only | 9.0/10 | exit_format=1, review_stopped=1, submitted=2, submitted (exit_cost)=1, submitted (exit_format)=1 |
| pylint-dev__astroid-1866 | pylint-dev__astroid | 6 | 5/6 | 5.2/10 | 4.5/5 | 2.0/5 | 149811.2 | 3176.8 | 2609344.2 | big_planner_small_coder | 8.0/10 | submitted=5, submitted (exit_cost)=1 |

## Project Index

| Project | Issues | Configs | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pvlib__pvlib-python | 1 | 6 | 4.7/10 | 3.5/5 | 1.5/5 | 207834.7 | 3306.0 | 3092766.0 | small_coder_only | [pvlib__pvlib-python](./projects/pvlib__pvlib-python/README.md) |
| pylint-dev__astroid | 1 | 6 | 5.2/10 | 4.5/5 | 2.0/5 | 149811.2 | 3176.8 | 2609344.2 | big_planner_small_coder | [pylint-dev__astroid](./projects/pylint-dev__astroid/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
