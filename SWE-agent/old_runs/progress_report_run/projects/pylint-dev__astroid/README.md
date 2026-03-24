# pylint-dev__astroid

- issues: `1`
- presets: `2`
- variants: `6`
- configs: `12`
- observed runs: `12`

## Variant Aggregate

| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gptoss120b_big_qwen35b_small/big_coder_only | 1 | 1/1 | 8.0/10 | 3.0/5 | 5.0/5 | 4.0/5 | 10882.0 | 299.0 | 1377600.0 | 5.0 |
| gptoss120b_big_qwen35b_small/big_planner_big_coder | 1 | 0/1 | 3.0/10 | 3.0/5 | 0.0/5 | 5.0/5 | 99407.0 | 5757.0 | 13310520.0 | 19.0 |
| gptoss120b_big_qwen35b_small/big_planner_big_coder_big_reviewer | 1 | 1/1 | 5.0/10 | 5.0/5 | 0.0/5 | 5.0/5 | 509951.0 | 4347.0 | 62237400.0 | 42.0 |
| gptoss120b_big_qwen35b_small/big_planner_small_coder | 1 | 1/1 | 7.0/10 | 5.0/5 | 0.0/5 | 5.0/5 | 383039.0 | 4687.0 | 14539575.0 | 43.0 |
| gptoss120b_big_qwen35b_small/big_planner_small_coder_big_reviewer | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 127088.0 | 1965.0 | 5649235.0 | 28.0 |
| gptoss120b_big_qwen35b_small/small_coder_only | 1 | 1/1 | 5.0/10 | 5.0/5 | 3.0/5 | 5.0/5 | 35390.0 | 945.0 | 1304800.0 | 11.0 |
| qwen_local_35b_9b/big_coder_only | 1 | 1/1 | 5.0/10 | 5.0/5 | 3.0/5 | 5.0/5 | 45119.0 | 1022.0 | 1650705.0 | 13.0 |
| qwen_local_35b_9b/big_planner_big_coder | 1 | 1/1 | 7.0/10 | 5.0/5 | 4.0/5 | 5.0/5 | 49881.0 | 1449.0 | 1847265.0 | 15.0 |
| qwen_local_35b_9b/big_planner_big_coder_big_reviewer | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 157934.0 | 3247.0 | 5754980.0 | 31.0 |
| qwen_local_35b_9b/big_planner_small_coder | 1 | 1/1 | 8.0/10 | 5.0/5 | 2.0/5 | 5.0/5 | 57647.0 | 1806.0 | 775269.0 | 18.0 |
| qwen_local_35b_9b/big_planner_small_coder_big_reviewer | 1 | 0/1 | 4.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 543438.0 | 10474.0 | 5205080.0 | 54.0 |
| qwen_local_35b_9b/small_coder_only | 1 | 1/1 | 1.0/10 | 3.0/5 | 2.0/5 | 5.0/5 | 44848.0 | 1063.0 | 422766.0 | 13.0 |

## Issue Aggregate

| Issue | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pylint-dev__astroid-1866 | 12 | 10/12 | 5.4/10 | 4.4/5 | 1.8/5 | 172052.0 | 3088.4 | 9506266.2 | gptoss120b_big_qwen35b_small/big_coder_only | 8.0/10 | skipped (stuck_repetition)=1, skipped (submitted)=5, submitted=5, submitted (exit_cost)=1 |

## Instance Details

| Instance | Preset | Variant | Exit | Quality | Completion | Efficiency | Grounding | In Tok | Out Tok | Rel Cost | Validations | Good Edits | Failed Edits | Submitted | Steps |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| pylint-dev__astroid-1866 | gptoss120b_big_qwen35b_small | big_coder_only | skipped (submitted) | 8/10 | 3/5 | 5/5 | 4/5 | 10882 | 299 | 1377600.0 | 1 | 0 | 0 | True | 5 |
| pylint-dev__astroid-1866 | qwen_local_35b_9b | big_coder_only | submitted | 5/10 | 5/5 | 3/5 | 5/5 | 45119 | 1022 | 1650705.0 | 5 | 1 | 2 | True | 13 |
| pylint-dev__astroid-1866 | gptoss120b_big_qwen35b_small | big_planner_big_coder | skipped (stuck_repetition) | 3/10 | 3/5 | 0/5 | 5/5 | 99407 | 5757 | 13310520.0 | 2 | 1 | 4 | False | 19 |
| pylint-dev__astroid-1866 | qwen_local_35b_9b | big_planner_big_coder | submitted | 7/10 | 5/5 | 4/5 | 5/5 | 49881 | 1449 | 1847265.0 | 6 | 1 | 0 | True | 15 |
| pylint-dev__astroid-1866 | gptoss120b_big_qwen35b_small | big_planner_big_coder_big_reviewer | skipped (submitted) | 5/10 | 5/5 | 0/5 | 5/5 | 509951 | 4347 | 62237400.0 | 6 | 4 | 8 | True | 42 |
| pylint-dev__astroid-1866 | qwen_local_35b_9b | big_planner_big_coder_big_reviewer | submitted | 6/10 | 5/5 | 1/5 | 5/5 | 157934 | 3247 | 5754980.0 | 11 | 1 | 3 | True | 31 |
| pylint-dev__astroid-1866 | gptoss120b_big_qwen35b_small | big_planner_small_coder | skipped (submitted) | 7/10 | 5/5 | 0/5 | 5/5 | 383039 | 4687 | 14539575.0 | 8 | 3 | 1 | True | 43 |
| pylint-dev__astroid-1866 | qwen_local_35b_9b | big_planner_small_coder | submitted | 8/10 | 5/5 | 2/5 | 5/5 | 57647 | 1806 | 775269.0 | 3 | 1 | 1 | True | 18 |
| pylint-dev__astroid-1866 | gptoss120b_big_qwen35b_small | big_planner_small_coder_big_reviewer | skipped (submitted) | 6/10 | 5/5 | 1/5 | 5/5 | 127088 | 1965 | 5649235.0 | 9 | 2 | 3 | True | 28 |
| pylint-dev__astroid-1866 | qwen_local_35b_9b | big_planner_small_coder_big_reviewer | submitted (exit_cost) | 4/10 | 4/5 | 0/5 | 5/5 | 543438 | 10474 | 5205080.0 | 5 | 2 | 5 | False | 54 |
| pylint-dev__astroid-1866 | gptoss120b_big_qwen35b_small | small_coder_only | skipped (submitted) | 5/10 | 5/5 | 3/5 | 5/5 | 35390 | 945 | 1304800.0 | 3 | 1 | 2 | True | 11 |
| pylint-dev__astroid-1866 | qwen_local_35b_9b | small_coder_only | submitted | 1/10 | 3/5 | 2/5 | 5/5 | 44848 | 1063 | 422766.0 | 0 | 0 | 3 | True | 13 |
