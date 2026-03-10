# pvlib__pvlib-python

- issues: `1`
- presets: `0`
- variants: `6`
- configs: `6`
- observed runs: `6`

## Variant Aggregate

| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| big_coder_only | 1 | 0/1 | 0.0/10 | 0.0/5 | 4.0/5 | 1.0/5 | 4343.0 | 191.0 | 165375.0 | 4.0 |
| big_planner_big_coder | 1 | 1/1 | 6.0/10 | 5.0/5 | 0.0/5 | 5.0/5 | 126601.0 | 2927.0 | 4635925.0 | 24.0 |
| big_planner_big_coder_big_reviewer | 1 | 0/1 | 5.0/10 | 3.0/5 | 1.0/5 | 5.0/5 | 115711.0 | 3386.0 | 4286905.0 | 30.0 |
| big_planner_small_coder | 1 | 0/1 | 4.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 618296.0 | 7253.0 | 5782760.0 | 53.0 |
| big_planner_small_coder_big_reviewer | 1 | 0/1 | 4.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 370265.0 | 5482.0 | 3568757.0 | 47.0 |
| small_coder_only | 1 | 1/1 | 9.0/10 | 5.0/5 | 4.0/5 | 5.0/5 | 11792.0 | 597.0 | 116874.0 | 5.0 |

## Issue Aggregate

| Issue | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pvlib__pvlib-python-1072 | 6 | 2/6 | 4.7/10 | 3.5/5 | 1.5/5 | 207834.7 | 3306.0 | 3092766.0 | small_coder_only | 9.0/10 | exit_format=1, review_stopped=1, submitted=2, submitted (exit_cost)=1, submitted (exit_format)=1 |

## Instance Details

| Instance | Preset | Variant | Exit | Quality | Completion | Efficiency | Grounding | In Tok | Out Tok | Rel Cost | Validations | Good Edits | Failed Edits | Submitted | Steps |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| pvlib__pvlib-python-1072 |  | big_coder_only | exit_format | 0/10 | 0/5 | 4/5 | 1/5 | 4343 | 191 | 165375.0 | 0 | 0 | 0 | False | 4 |
| pvlib__pvlib-python-1072 |  | big_planner_big_coder | submitted | 6/10 | 5/5 | 0/5 | 5/5 | 126601 | 2927 | 4635925.0 | 4 | 1 | 2 | True | 24 |
| pvlib__pvlib-python-1072 |  | big_planner_big_coder_big_reviewer | review_stopped | 5/10 | 3/5 | 1/5 | 5/5 | 115711 | 3386 | 4286905.0 | 8 | 1 | 1 | False | 30 |
| pvlib__pvlib-python-1072 |  | big_planner_small_coder | submitted (exit_cost) | 4/10 | 4/5 | 0/5 | 5/5 | 618296 | 7253 | 5782760.0 | 4 | 2 | 8 | False | 53 |
| pvlib__pvlib-python-1072 |  | big_planner_small_coder_big_reviewer | submitted (exit_format) | 4/10 | 4/5 | 0/5 | 5/5 | 370265 | 5482 | 3568757.0 | 1 | 13 | 4 | False | 47 |
| pvlib__pvlib-python-1072 |  | small_coder_only | submitted | 9/10 | 5/5 | 4/5 | 5/5 | 11792 | 597 | 116874.0 | 2 | 1 | 0 | True | 5 |
