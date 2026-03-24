# pvlib__pvlib-python

- issues: `1`
- presets: `0`
- variants: `6`
- configs: `6`
- observed runs: `6`

## Variant Aggregate

| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| big_coder_only | 1 | 0/1 | 1.0/10 | 4.0/5 | 0.0/5 | 4.0/5 | 937575.0 | 2707.0 | 113158680.0 | 51.0 |
| big_planner_big_coder | 1 | 0/1 | 3.0/10 | 3.0/5 | 0.0/5 | 5.0/5 | 302123.0 | 3270.0 | 37039560.0 | 36.0 |
| big_planner_big_coder_big_reviewer | 1 | 0/1 | 4.0/10 | 3.0/5 | 0.0/5 | 3.0/5 | 545829.0 | 3930.0 | 66442680.0 | 48.0 |
| big_planner_small_coder | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 77324.0 | 1957.0 | 4524970.0 | 19.0 |
| big_planner_small_coder_big_reviewer | 1 | 0/1 | 5.0/10 | 4.0/5 | 0.0/5 | 5.0/5 | 808980.0 | 12863.0 | 60082800.0 | 86.0 |
| small_coder_only | 1 | 1/1 | 6.0/10 | 5.0/5 | 2.0/5 | 5.0/5 | 87299.0 | 2167.0 | 3207155.0 | 18.0 |

## Issue Aggregate

| Issue | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pvlib__pvlib-python-1072 | 6 | 2/6 | 4.2/10 | 4.0/5 | 0.5/5 | 459855.0 | 4482.3 | 47409307.5 | big_planner_small_coder | 6.0/10 | skipped (stuck_repetition)=2, skipped (submitted (exit_cost))=1, skipped (submitted (exit_format))=1, skipped (submitted)=2 |

## Instance Details

| Instance | Preset | Variant | Exit | Quality | Completion | Efficiency | Grounding | In Tok | Out Tok | Rel Cost | Validations | Good Edits | Failed Edits | Submitted | Steps |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| pvlib__pvlib-python-1072 |  | big_coder_only | skipped (submitted (exit_cost)) | 1/10 | 4/5 | 0/5 | 4/5 | 937575 | 2707 | 113158680.0 | 11 | 2 | 7 | False | 51 |
| pvlib__pvlib-python-1072 |  | big_planner_big_coder | skipped (stuck_repetition) | 3/10 | 3/5 | 0/5 | 5/5 | 302123 | 3270 | 37039560.0 | 2 | 5 | 14 | False | 36 |
| pvlib__pvlib-python-1072 |  | big_planner_big_coder_big_reviewer | skipped (stuck_repetition) | 4/10 | 3/5 | 0/5 | 3/5 | 545829 | 3930 | 66442680.0 | 4 | 15 | 6 | False | 48 |
| pvlib__pvlib-python-1072 |  | big_planner_small_coder | skipped (submitted) | 6/10 | 5/5 | 1/5 | 5/5 | 77324 | 1957 | 4524970.0 | 4 | 1 | 1 | True | 19 |
| pvlib__pvlib-python-1072 |  | big_planner_small_coder_big_reviewer | skipped (submitted (exit_format)) | 5/10 | 4/5 | 0/5 | 5/5 | 808980 | 12863 | 60082800.0 | 30 | 1 | 1 | False | 86 |
| pvlib__pvlib-python-1072 |  | small_coder_only | skipped (submitted) | 6/10 | 5/5 | 2/5 | 5/5 | 87299 | 2167 | 3207155.0 | 5 | 2 | 1 | True | 18 |
