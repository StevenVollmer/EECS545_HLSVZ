# Matrix Batch Results

- variants: `14`
- projects: `4`
- issues: `10`

## Variant Aggregate

| Variant | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| big_coder_only | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| big_planner_big_coder | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| big_planner_big_coder_big_reviewer | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| big_planner_big_coder_small_reviewer | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| big_planner_small_coder | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| big_planner_small_coder_big_reviewer | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| big_planner_small_coder_small_reviewer | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| small_coder_only | 10 | 4/10 | 3.7/10 | 3.1/5 | 1.7/5 | 4.8/5 | 328540.1 | 2873.5 | 3008583.9 | 27.3 |
| small_planner_big_coder | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| small_planner_big_coder_big_reviewer | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| small_planner_big_coder_small_reviewer | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| small_planner_small_coder | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| small_planner_small_coder_big_reviewer | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| small_planner_small_coder_small_reviewer | 10 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Project Index

| Project | Issues | Variants | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pvlib__pvlib-python | 3 | 14 | 2.7/10 | 2.3/5 | 1.3/5 | 830583.3 | 4882.3 | 7563132.0 | small_coder_only | [pvlib__pvlib-python](./projects/pvlib__pvlib-python/README.md) |
| pylint-dev__astroid | 5 | 14 | 4.0/10 | 4.0/5 | 1.6/5 | 134271.6 | 2364.2 | 1251000.0 | small_coder_only | [pylint-dev__astroid](./projects/pylint-dev__astroid/README.md) |
| pyvista__pyvista | 1 | 14 | 5.0/10 | 2.0/5 | 2.0/5 | 102006.0 | 1191.0 | 939492.0 | small_coder_only | [pyvista__pyvista](./projects/pyvista__pyvista/README.md) |
| sqlfluff__sqlfluff | 1 | 14 | 4.0/10 | 2.0/5 | 3.0/5 | 20287.0 | 1076.0 | 201951.0 | small_coder_only | [sqlfluff__sqlfluff](./projects/sqlfluff__sqlfluff/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
