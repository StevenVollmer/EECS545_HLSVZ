# pylint-dev__astroid

- issues: `1`
- variants: `6`

## Variant Aggregate

| Variant | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| big_coder_only | 1 | 1/1 | 10.0/10 | 5.0/5 | 5.0/5 | 5.0/5 | 15748.0 | 567.0 | 590870.0 | 8.0 |
| big_planner_big_coder | 1 | 1/1 | 5.0/10 | 4.0/5 | 1.0/5 | 5.0/5 | 108989.0 | 2922.0 | 4019155.0 | 29.0 |
| big_planner_big_coder_big_reviewer | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 124475.0 | 2884.0 | 4558505.0 | 34.0 |
| big_planner_small_coder | 1 | 0/1 | 1.0/10 | 0.0/5 | 3.0/5 | 4.0/5 | 32628.0 | 912.0 | 1205820.0 | 13.0 |
| big_planner_small_coder_big_reviewer | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 73467.0 | 1969.0 | 1472407.0 | 28.0 |
| small_coder_only | 1 | 1/1 | 4.0/10 | 4.0/5 | 1.0/5 | 5.0/5 | 75537.0 | 1526.0 | 707301.0 | 19.0 |

## Instance Details

| Instance | Variant | Exit | Quality | Completion | Efficiency | Grounding | In Tok | Out Tok | Rel Cost | Validations | Good Edits | Failed Edits | Submitted | Steps |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| pylint-dev__astroid-1978 | big_coder_only | submitted | 10/10 | 5/5 | 5/5 | 5/5 | 15748 | 567 | 590870.0 | 1 | 1 | 0 | True | 8 |
| pylint-dev__astroid-1978 | big_planner_big_coder | submitted | 5/10 | 4/5 | 1/5 | 5/5 | 108989 | 2922 | 4019155.0 | 4 | 0 | 3 | True | 29 |
| pylint-dev__astroid-1978 | big_planner_big_coder_big_reviewer | submitted | 6/10 | 5/5 | 1/5 | 5/5 | 124475 | 2884 | 4558505.0 | 9 | 2 | 2 | True | 34 |
| pylint-dev__astroid-1978 | big_planner_small_coder | stuck_repetition | 1/10 | 0/5 | 3/5 | 4/5 | 32628 | 912 | 1205820.0 | 0 | 0 | 0 | False | 13 |
| pylint-dev__astroid-1978 | big_planner_small_coder_big_reviewer | submitted | 6/10 | 5/5 | 1/5 | 5/5 | 73467 | 1969 | 1472407.0 | 6 | 1 | 2 | True | 28 |
| pylint-dev__astroid-1978 | small_coder_only | submitted | 4/10 | 4/5 | 1/5 | 5/5 | 75537 | 1526 | 707301.0 | 0 | 1 | 5 | True | 19 |
