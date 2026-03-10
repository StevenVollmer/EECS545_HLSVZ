# pylint-dev__astroid

- issues: `1`
- presets: `0`
- variants: `6`
- configs: `6`
- observed runs: `6`

## Variant Aggregate

| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| big_coder_only | 1 | 1/1 | 8.0/10 | 3.0/5 | 5.0/5 | 4.0/5 | 10882.0 | 299.0 | 1377600.0 | 5.0 |
| big_planner_big_coder | 1 | 0/1 | 3.0/10 | 3.0/5 | 0.0/5 | 5.0/5 | 99407.0 | 5757.0 | 13310520.0 | 19.0 |
| big_planner_big_coder_big_reviewer | 1 | 1/1 | 5.0/10 | 5.0/5 | 0.0/5 | 5.0/5 | 509951.0 | 4347.0 | 62237400.0 | 42.0 |
| big_planner_small_coder | 1 | 1/1 | 7.0/10 | 5.0/5 | 0.0/5 | 5.0/5 | 383039.0 | 4687.0 | 14539575.0 | 43.0 |
| big_planner_small_coder_big_reviewer | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 127088.0 | 1965.0 | 5649235.0 | 28.0 |
| small_coder_only | 1 | 1/1 | 5.0/10 | 5.0/5 | 3.0/5 | 5.0/5 | 35390.0 | 945.0 | 1304800.0 | 11.0 |

## Issue Aggregate

| Issue | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pylint-dev__astroid-1866 | 6 | 5/6 | 5.7/10 | 4.3/5 | 1.5/5 | 194292.8 | 3000.0 | 16403188.3 | big_coder_only | 8.0/10 | skipped (stuck_repetition)=1, skipped (submitted)=5 |

## Instance Details

| Instance | Preset | Variant | Exit | Quality | Completion | Efficiency | Grounding | In Tok | Out Tok | Rel Cost | Validations | Good Edits | Failed Edits | Submitted | Steps |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| pylint-dev__astroid-1866 |  | big_coder_only | skipped (submitted) | 8/10 | 3/5 | 5/5 | 4/5 | 10882 | 299 | 1377600.0 | 1 | 0 | 0 | True | 5 |
| pylint-dev__astroid-1866 |  | big_planner_big_coder | skipped (stuck_repetition) | 3/10 | 3/5 | 0/5 | 5/5 | 99407 | 5757 | 13310520.0 | 2 | 1 | 4 | False | 19 |
| pylint-dev__astroid-1866 |  | big_planner_big_coder_big_reviewer | skipped (submitted) | 5/10 | 5/5 | 0/5 | 5/5 | 509951 | 4347 | 62237400.0 | 6 | 4 | 8 | True | 42 |
| pylint-dev__astroid-1866 |  | big_planner_small_coder | skipped (submitted) | 7/10 | 5/5 | 0/5 | 5/5 | 383039 | 4687 | 14539575.0 | 8 | 3 | 1 | True | 43 |
| pylint-dev__astroid-1866 |  | big_planner_small_coder_big_reviewer | skipped (submitted) | 6/10 | 5/5 | 1/5 | 5/5 | 127088 | 1965 | 5649235.0 | 9 | 2 | 3 | True | 28 |
| pylint-dev__astroid-1866 |  | small_coder_only | skipped (submitted) | 5/10 | 5/5 | 3/5 | 5/5 | 35390 | 945 | 1304800.0 | 3 | 1 | 2 | True | 11 |
