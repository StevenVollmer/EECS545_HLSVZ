# Matrix Batch Results

- presets: `2`
- variants: `6`
- configs: `12`
- projects: `2`
- issues: `2`
- observed runs: `24`

## Variant Aggregate

| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gptoss120b_big_qwen35b_small/big_coder_only | 2 | 1/2 | 4.5/10 | 3.5/5 | 2.5/5 | 4.0/5 | 474228.5 | 1503.0 | 57268140.0 | 28.0 |
| gptoss120b_big_qwen35b_small/big_planner_big_coder | 2 | 0/2 | 3.0/10 | 3.0/5 | 0.0/5 | 5.0/5 | 200765.0 | 4513.5 | 25175040.0 | 27.5 |
| gptoss120b_big_qwen35b_small/big_planner_big_coder_big_reviewer | 2 | 1/2 | 4.5/10 | 4.0/5 | 0.0/5 | 4.0/5 | 527890.0 | 4138.5 | 64340040.0 | 45.0 |
| gptoss120b_big_qwen35b_small/big_planner_small_coder | 2 | 2/2 | 6.5/10 | 5.0/5 | 0.5/5 | 5.0/5 | 230181.5 | 3322.0 | 9532272.5 | 31.0 |
| gptoss120b_big_qwen35b_small/big_planner_small_coder_big_reviewer | 2 | 1/2 | 5.5/10 | 4.5/5 | 0.5/5 | 5.0/5 | 468034.0 | 7414.0 | 32866017.5 | 57.0 |
| gptoss120b_big_qwen35b_small/small_coder_only | 2 | 2/2 | 5.5/10 | 5.0/5 | 2.5/5 | 5.0/5 | 61344.5 | 1556.0 | 2255977.5 | 14.5 |
| qwen_local_35b_9b/big_coder_only | 2 | 1/2 | 2.5/10 | 2.5/5 | 3.5/5 | 3.0/5 | 24731.0 | 606.5 | 908040.0 | 8.5 |
| qwen_local_35b_9b/big_planner_big_coder | 2 | 2/2 | 6.5/10 | 5.0/5 | 2.0/5 | 5.0/5 | 88241.0 | 2188.0 | 3241595.0 | 19.5 |
| qwen_local_35b_9b/big_planner_big_coder_big_reviewer | 2 | 1/2 | 5.5/10 | 4.0/5 | 1.0/5 | 5.0/5 | 136822.5 | 3316.5 | 5020942.5 | 30.5 |
| qwen_local_35b_9b/big_planner_small_coder | 2 | 1/2 | 6.0/10 | 4.5/5 | 1.0/5 | 5.0/5 | 337971.5 | 4529.5 | 3279014.5 | 35.5 |
| qwen_local_35b_9b/big_planner_small_coder_big_reviewer | 2 | 0/2 | 4.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 456851.5 | 7978.0 | 4386918.5 | 50.5 |
| qwen_local_35b_9b/small_coder_only | 2 | 2/2 | 5.0/10 | 4.0/5 | 3.0/5 | 5.0/5 | 28320.0 | 830.0 | 269820.0 | 9.0 |

## Issue Index

| Issue | Project | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pvlib__pvlib-python-1072 | pvlib__pvlib-python | 12 | 4/12 | 4.4/10 | 3.8/5 | 1.0/5 | 333844.8 | 3894.2 | 25251036.8 | qwen_local_35b_9b/small_coder_only | 9.0/10 | exit_format=1, review_stopped=1, skipped (stuck_repetition)=2, skipped (submitted (exit_cost))=1, skipped (submitted (exit_format))=1, skipped (submitted)=2, submitted=2, submitted (exit_cost)=1, submitted (exit_format)=1 |
| pylint-dev__astroid-1866 | pylint-dev__astroid | 12 | 10/12 | 5.4/10 | 4.4/5 | 1.8/5 | 172052.0 | 3088.4 | 9506266.2 | gptoss120b_big_qwen35b_small/big_coder_only | 8.0/10 | skipped (stuck_repetition)=1, skipped (submitted)=5, submitted=5, submitted (exit_cost)=1 |

## Project Index

| Project | Issues | Configs | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pvlib__pvlib-python | 1 | 12 | 4.4/10 | 3.8/5 | 1.0/5 | 333844.8 | 3894.2 | 25251036.8 | qwen_local_35b_9b/small_coder_only | [pvlib__pvlib-python](./projects/pvlib__pvlib-python/README.md) |
| pylint-dev__astroid | 1 | 12 | 5.4/10 | 4.4/5 | 1.8/5 | 172052.0 | 3088.4 | 9506266.2 | gptoss120b_big_qwen35b_small/big_coder_only | [pylint-dev__astroid](./projects/pylint-dev__astroid/README.md) |

## Files

- `summary.csv`: one row per `(variant, instance)` pair
- `summary.json`: JSON version of the same table
- `projects/<project>/README.md`: per-project comparisons across all variants
