# pvlib__pvlib-python

- issues: `1`
- presets: `2`
- variants: `6`
- configs: `12`
- observed runs: `12`

## Variant Aggregate

| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gptoss120b_big_qwen35b_small/big_coder_only | 1 | 0/1 | 1.0/10 | 4.0/5 | 0.0/5 | 4.0/5 | 937575.0 | 2707.0 | 113158680.0 | 51.0 |
| gptoss120b_big_qwen35b_small/big_planner_big_coder | 1 | 0/1 | 3.0/10 | 3.0/5 | 0.0/5 | 5.0/5 | 302123.0 | 3270.0 | 37039560.0 | 36.0 |
| gptoss120b_big_qwen35b_small/big_planner_big_coder_big_reviewer | 1 | 0/1 | 4.0/10 | 3.0/5 | 0.0/5 | 3.0/5 | 545829.0 | 3930.0 | 66442680.0 | 48.0 |
| gptoss120b_big_qwen35b_small/big_planner_small_coder | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 77324.0 | 1957.0 | 4524970.0 | 19.0 |
| gptoss120b_big_qwen35b_small/big_planner_small_coder_big_reviewer | 1 | 0/1 | 5.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 808980.0 | 12863.0 | 60082800.0 | 86.0 |
| gptoss120b_big_qwen35b_small/small_coder_only | 1 | 1/1 | 6.0/10 | 5.0/5 | 2.0/5 | 5.0/5 | 87299.0 | 2167.0 | 3207155.0 | 18.0 |
| qwen_local_35b_9b/big_coder_only | 1 | 0/1 | 0.0/10 | 0.0/5 | 4.0/5 | 1.0/5 | 4343.0 | 191.0 | 165375.0 | 4.0 |
| qwen_local_35b_9b/big_planner_big_coder | 1 | 1/1 | 6.0/10 | 5.0/5 | 0.0/5 | 5.0/5 | 126601.0 | 2927.0 | 4635925.0 | 24.0 |
| qwen_local_35b_9b/big_planner_big_coder_big_reviewer | 1 | 0/1 | 5.0/10 | 3.0/5 | 1.0/5 | 5.0/5 | 115711.0 | 3386.0 | 4286905.0 | 30.0 |
| qwen_local_35b_9b/big_planner_small_coder | 1 | 0/1 | 4.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 618296.0 | 7253.0 | 5782760.0 | 53.0 |
| qwen_local_35b_9b/big_planner_small_coder_big_reviewer | 1 | 0/1 | 4.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 370265.0 | 5482.0 | 3568757.0 | 47.0 |
| qwen_local_35b_9b/small_coder_only | 1 | 1/1 | 9.0/10 | 5.0/5 | 4.0/5 | 5.0/5 | 11792.0 | 597.0 | 116874.0 | 5.0 |

## Issue Aggregate

| Issue | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pvlib__pvlib-python-1072 | 12 | 4/12 | 4.4/10 | 3.8/5 | 1.0/5 | 333844.8 | 3894.2 | 25251036.8 | qwen_local_35b_9b/small_coder_only | 9.0/10 | exit_format=1, review_stopped=1, skipped (stuck_repetition)=2, skipped (submitted (exit_cost))=1, skipped (submitted (exit_format))=1, skipped (submitted)=2, submitted=2, submitted (exit_cost)=1, submitted (exit_format)=1 |

## Instance Details

| Instance | Preset | Variant | Exit | Quality | Completion | Efficiency | Grounding | In Tok | Out Tok | Rel Cost | Validations | Good Edits | Failed Edits | Submitted | Steps |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| pvlib__pvlib-python-1072 | gptoss120b_big_qwen35b_small | big_coder_only | skipped (submitted (exit_cost)) | 1/10 | 4/5 | 0/5 | 4/5 | 937575 | 2707 | 113158680.0 | 11 | 2 | 7 | False | 51 |
| pvlib__pvlib-python-1072 | qwen_local_35b_9b | big_coder_only | exit_format | 0/10 | 0/5 | 4/5 | 1/5 | 4343 | 191 | 165375.0 | 0 | 0 | 0 | False | 4 |
| pvlib__pvlib-python-1072 | gptoss120b_big_qwen35b_small | big_planner_big_coder | skipped (stuck_repetition) | 3/10 | 3/5 | 0/5 | 5/5 | 302123 | 3270 | 37039560.0 | 2 | 5 | 14 | False | 36 |
| pvlib__pvlib-python-1072 | qwen_local_35b_9b | big_planner_big_coder | submitted | 6/10 | 5/5 | 0/5 | 5/5 | 126601 | 2927 | 4635925.0 | 4 | 1 | 2 | True | 24 |
| pvlib__pvlib-python-1072 | gptoss120b_big_qwen35b_small | big_planner_big_coder_big_reviewer | skipped (stuck_repetition) | 4/10 | 3/5 | 0/5 | 3/5 | 545829 | 3930 | 66442680.0 | 4 | 15 | 6 | False | 48 |
| pvlib__pvlib-python-1072 | qwen_local_35b_9b | big_planner_big_coder_big_reviewer | review_stopped | 5/10 | 3/5 | 1/5 | 5/5 | 115711 | 3386 | 4286905.0 | 8 | 1 | 1 | False | 30 |
| pvlib__pvlib-python-1072 | gptoss120b_big_qwen35b_small | big_planner_small_coder | skipped (submitted) | 6/10 | 5/5 | 1/5 | 5/5 | 77324 | 1957 | 4524970.0 | 4 | 1 | 1 | True | 19 |
| pvlib__pvlib-python-1072 | qwen_local_35b_9b | big_planner_small_coder | submitted (exit_cost) | 4/10 | 4/5 | 0/5 | 5/5 | 618296 | 7253 | 5782760.0 | 4 | 2 | 8 | False | 53 |
| pvlib__pvlib-python-1072 | gptoss120b_big_qwen35b_small | big_planner_small_coder_big_reviewer | skipped (submitted (exit_format)) | 5/10 | 4/5 | 0/5 | 5/5 | 808980 | 12863 | 60082800.0 | 30 | 1 | 1 | False | 86 |
| pvlib__pvlib-python-1072 | qwen_local_35b_9b | big_planner_small_coder_big_reviewer | submitted (exit_format) | 4/10 | 4/5 | 0/5 | 5/5 | 370265 | 5482 | 3568757.0 | 1 | 13 | 4 | False | 47 |
| pvlib__pvlib-python-1072 | gptoss120b_big_qwen35b_small | small_coder_only | skipped (submitted) | 6/10 | 5/5 | 2/5 | 5/5 | 87299 | 2167 | 3207155.0 | 5 | 2 | 1 | True | 18 |
| pvlib__pvlib-python-1072 | qwen_local_35b_9b | small_coder_only | submitted | 9/10 | 5/5 | 4/5 | 5/5 | 11792 | 597 | 116874.0 | 2 | 1 | 0 | True | 5 |
