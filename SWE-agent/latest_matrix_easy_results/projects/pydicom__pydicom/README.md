# pydicom__pydicom

- issues: `1`
- variants: `7`

## Variant Aggregate

| Variant | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| all_3_big | 1 | 1/1 | 6.0/10 | 5.0/5 | 0.0/5 | 5.0/5 | 292673.0 | 4786.0 | 10578575.0 | 45.0 |
| big_coder_only | 1 | 0/1 | 0.0/10 | 0.0/5 | 2.0/5 | 4.0/5 | 62808.0 | 1865.0 | 2328830.0 | 11.0 |
| big_planner_big_coder | 1 | 1/1 | 6.0/10 | 5.0/5 | 1.0/5 | 5.0/5 | 98603.0 | 3177.0 | 3673495.0 | 29.0 |
| big_planner_small_coder | 1 | 0/1 | 4.0/10 | 3.0/5 | 0.0/5 | 5.0/5 | 99774.0 | 3357.0 | 1631376.0 | 26.0 |
| big_planner_small_coder_big_reviewer | 1 | 0/1 | 3.0/10 | 1.0/5 | 2.0/5 | 4.0/5 | 85796.0 | 2199.0 | 1606098.0 | 22.0 |
| big_planner_small_coder_small_reviewer | 1 | 0/1 | 5.0/10 | 3.0/5 | 2.0/5 | 5.0/5 | 45789.0 | 2082.0 | 1304249.0 | 18.0 |
| small_coder_only | 1 | 0/1 | 0.0/10 | 0.0/5 | 0.0/5 | 4.0/5 | 115210.0 | 1933.0 | 1071684.0 | 17.0 |

## Instance Details

| Instance | Variant | Exit | Quality | Completion | Efficiency | Grounding | In Tok | Out Tok | Rel Cost | Validations | Good Edits | Failed Edits | Submitted | Steps |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| pydicom__pydicom-1458 | all_3_big | submitted | 6/10 | 5/5 | 0/5 | 5/5 | 292673 | 4786 | 10578575.0 | 8 | 1 | 2 | True | 45 |
| pydicom__pydicom-1458 | big_coder_only | exit_format | 0/10 | 0/5 | 2/5 | 4/5 | 62808 | 1865 | 2328830.0 | 0 | 0 | 0 | False | 11 |
| pydicom__pydicom-1458 | big_planner_big_coder | submitted | 6/10 | 5/5 | 1/5 | 5/5 | 98603 | 3177 | 3673495.0 | 3 | 1 | 2 | True | 29 |
| pydicom__pydicom-1458 | big_planner_small_coder | Uncaught ValidationError | 4/10 | 3/5 | 0/5 | 5/5 | 99774 | 3357 | 1631376.0 | 2 | 1 | 4 | False | 26 |
| pydicom__pydicom-1458 | big_planner_small_coder_big_reviewer | submitted (exit_format) | 3/10 | 1/5 | 2/5 | 4/5 | 85796 | 2199 | 1606098.0 | 0 | 0 | 0 | False | 22 |
| pydicom__pydicom-1458 | big_planner_small_coder_small_reviewer | submitted (exit_format) | 5/10 | 3/5 | 2/5 | 5/5 | 45789 | 2082 | 1304249.0 | 0 | 1 | 0 | False | 18 |
| pydicom__pydicom-1458 | small_coder_only | exit_format | 0/10 | 0/5 | 0/5 | 4/5 | 115210 | 1933 | 1071684.0 | 0 | 0 | 0 | False | 17 |
