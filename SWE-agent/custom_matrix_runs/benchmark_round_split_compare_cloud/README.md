# Custom Matrix Report: benchmark_round_split_compare_cloud

- Matrix root: `/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud`
- Runs: `80`
- Strict resolved rate: `0.675`
- Observed resolved rate: `0.675`
- Avg total score: `83.84`
- Avg relative cost to 4o-mini: `6.855`

## By Architecture

| Architecture | Runs | Strict Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost |
| --- | --- | --- | --- | --- | --- |
| planner_coder | 30 | 0.633 | 83.13 | 6.140 | 19.39 |
| planner_coder_reviewer | 30 | 0.700 | 83.07 | 6.405 | 18.32 |
| single | 20 | 0.700 | 86.05 | 8.603 | 14.69 |

## By Config And Architecture

| Config | Architecture | Runs | Strict Resolve | Observed Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost | Avg Turns | Avg Parse Err | Avg Tool Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 10 | 0.400 | 0.400 | 66.90 | 3.148 | 21.22 | 34.1 | 6.7 | 4.1 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 10 | 0.700 | 0.700 | 89.20 | 3.042 | 29.32 | 10.6 | 1.2 | 1.2 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 10 | 0.800 | 0.800 | 93.30 | 12.229 | 7.63 | 12.1 | 2.3 | 1.9 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 10 | 0.600 | 0.600 | 77.90 | 3.391 | 22.95 | 18.7 | 4.0 | 2.5 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 10 | 0.700 | 0.700 | 78.30 | 3.179 | 24.66 | 30.9 | 8.4 | 8.4 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 10 | 0.800 | 0.800 | 93.00 | 12.644 | 7.35 | 17.5 | 3.7 | 3.1 |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 10 | 0.600 | 0.600 | 78.60 | 3.453 | 22.58 | 17.3 | 5.5 | 1.5 |
| single::openai/openai/gpt-oss-120b | single | 10 | 0.800 | 0.800 | 93.50 | 13.752 | 6.80 | 12.9 | 3.4 | 2.0 |

## Size Split Comparison

| Config | Planner Size | Coder Size | Reviewer Size | Mixed Sizes | Runs | Strict Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 0 | no | 10 | 0.400 | 66.90 | 3.148 | 21.22 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 5 | 4 | 0 | yes | 10 | 0.700 | 89.20 | 3.042 | 29.32 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 0 | no | 10 | 0.800 | 93.30 | 12.229 | 7.63 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 4 | no | 10 | 0.600 | 77.90 | 3.391 | 22.95 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 5 | 4 | 5 | yes | 10 | 0.700 | 78.30 | 3.179 | 24.66 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 5 | no | 10 | 0.800 | 93.00 | 12.644 | 7.35 |

## Hypothesis Check: PCR >= PC >= Single

| Case | Single Score | PC Score | PCR Score | Order Holds |
| --- | --- | --- | --- | --- |
| board_rollup_001 | 99 | 99 | 99 | yes |
| budget_snapshot_001 | 72 | 73 | 68 | no |
| digest_preview_001 | 98 | 94 | 98 | no |
| incident_brief_001 | 99 | 99 | 99 | yes |
| label_formatter_001 | 99 | 100 | 99 | no |
| nested_app_001 | 98 | 99 | 98 | no |
| owner_recap_001 | 99 | 99 | 99 | yes |
| shipment_preview_001 | 78 | 79 | 78 | no |
| simple_mean_bug_001 | 100 | 100 | 100 | yes |
| workspace_digest_001 | 99 | 99 | 99 | yes |

## Mixed-Size vs Big-Only

| Case | Best Mixed Config | Mixed Score | Mixed Cost | Best Big-Only Config | Big Score | Big Cost | Similar Or Better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.018 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.406 | yes |
| budget_snapshot_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 73 | 3.041 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 72 | 12.207 | yes |
| digest_preview_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 86 | 3.324 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.675 | no |
| incident_brief_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.036 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.167 | yes |
| label_formatter_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.017 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.154 | yes |
| nested_app_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.079 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.194 | yes |
| owner_recap_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.038 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.436 | yes |
| shipment_preview_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 55 | 3.046 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 79 | 12.169 | no |
| simple_mean_bug_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.073 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.179 | yes |
| workspace_digest_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.041 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.145 | yes |

## By Case

| Case | Runs | Strict Resolve | Observed Resolve | Avg Score | Best Config | Best Architecture | Best Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | 8 | 0.875 | 0.875 | 89.75 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| budget_snapshot_001 | 8 | 0.000 | 0.000 | 66.38 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 73 |
| digest_preview_001 | 8 | 0.500 | 0.500 | 80.38 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 |
| incident_brief_001 | 8 | 1.000 | 1.000 | 98.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| label_formatter_001 | 8 | 0.875 | 0.875 | 91.38 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| nested_app_001 | 8 | 0.625 | 0.625 | 77.50 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| owner_recap_001 | 8 | 0.875 | 0.875 | 87.25 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| shipment_preview_001 | 8 | 0.000 | 0.000 | 49.75 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 79 |
| simple_mean_bug_001 | 8 | 1.000 | 1.000 | 98.75 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| workspace_digest_001 | 8 | 1.000 | 1.000 | 98.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |

## Top Runs

- `simple_mean_bug_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.073`
- `label_formatter_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.154`
- `simple_mean_bug_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.179`
- `simple_mean_bug_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | cost `3.358`
- `simple_mean_bug_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | cost `3.509`
- `label_formatter_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.017`
- `board_rollup_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.018`
- `incident_brief_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.036`
- `owner_recap_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.038`
- `workspace_digest_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.041`
- `nested_app_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.079`
- `label_formatter_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `99` | strict `True` | observed `True` | cost `3.079`

## Best Failures

- `shipment_preview_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder` | score `79` | cost `12.169` | only part of the success validation passed; 11 protocol/parse errors; 3 tool errors; patch exists but does not satisfy success checks
- `shipment_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `78` | cost `12.474` | only part of the success validation passed; 12 protocol/parse errors; 10 tool errors; long run (48 turns)
- `shipment_preview_001` | `single::openai/openai/gpt-oss-120b` | `single` | score `78` | cost `13.340` | only part of the success validation passed; 16 protocol/parse errors; 7 tool errors; long run (41 turns)
- `budget_snapshot_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `73` | cost `3.041` | only part of the success validation passed; 1 tool errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder` | score `72` | cost `12.207` | only part of the success validation passed; 2 protocol/parse errors; 2 tool errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `single::openai/openai/gpt-oss-120b` | `single` | score `72` | cost `13.775` | only part of the success validation passed; 2 protocol/parse errors; 1 tool errors; patch exists but does not satisfy success checks
- `digest_preview_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `70` | cost `3.029` | only part of the success validation passed; edited code without post-edit validation; patch exists but does not satisfy success checks
- `digest_preview_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `69` | cost `3.443` | only part of the success validation passed; edited code without post-edit validation; loop control triggered; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `69` | cost `3.480` | only part of the success validation passed; edited code without post-edit validation; loop control triggered; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `68` | cost `12.620` | only part of the success validation passed; edited code without post-edit validation; 7 protocol/parse errors; 3 tool errors
- `budget_snapshot_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `67` | cost `3.341` | only part of the success validation passed; edited code without post-edit validation; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `66` | cost `3.143` | only part of the success validation passed; edited code without post-edit validation; loop control triggered; patch exists but does not satisfy success checks

## Per-Case Comparison

### board_rollup_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.018 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/board_rollup |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 28 | no | no | 3.139 | 60 | 7 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.236 | 15 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/board_rollup |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.406 | 16 | 0 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/board_rollup |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.473 | 17 | 0 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/board_rollup |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.185 | 9 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.508 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/board_rollup |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.977 | 8 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/board_rollup |

### budget_snapshot_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 73 | no | no | 3.041 | 8 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/budget_snapshot |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 66 | no | no | 3.143 | 14 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 44 | no | no | 3.139 | 60 | 31 | 28 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/budget_snapshot |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 67 | no | no | 3.341 | 9 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/budget_snapshot |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 69 | no | no | 3.480 | 15 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/budget_snapshot |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 | no | no | 12.207 | 13 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 68 | no | no | 12.620 | 25 | 7 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/budget_snapshot |
| single::openai/openai/gpt-oss-120b | single | 72 | no | no | 13.775 | 8 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/budget_snapshot |

### digest_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 70 | no | no | 3.029 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/digest_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 63 | no | no | 3.152 | 49 | 5 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 86 | yes | yes | 3.324 | 60 | 53 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/digest_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 65 | no | no | 3.368 | 19 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/digest_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 69 | no | no | 3.443 | 11 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/digest_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 94 | yes | yes | 12.207 | 8 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.675 | 10 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/digest_preview |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.456 | 12 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/digest_preview |

### incident_brief_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.036 | 11 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/incident_brief |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.167 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.130 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/incident_brief |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.412 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/incident_brief |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.480 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/incident_brief |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.216 | 15 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.629 | 13 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/incident_brief |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.791 | 11 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/incident_brief |

### label_formatter_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.017 | 7 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/label_formatter |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.154 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.079 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/label_formatter |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 38 | no | no | 3.399 | 20 | 0 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/label_formatter |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.500 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/label_formatter |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 12.074 | 7 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.517 | 8 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/label_formatter |
| single::openai/openai/gpt-oss-120b | single | 99 | yes | yes | 13.805 | 7 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/label_formatter |

### nested_app_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.079 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/nested_app |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 45 | no | no | 3.112 | 60 | 0 | 20 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 45 | no | no | 3.305 | 60 | 0 | 35 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/nested_app |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 96 | yes | yes | 3.392 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/nested_app |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 41 | no | no | 3.443 | 25 | 0 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/nested_app |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.194 | 12 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.658 | 10 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/nested_app |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.945 | 10 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/nested_app |

### owner_recap_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.038 | 12 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/owner_recap |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 12 | no | no | 3.138 | 60 | 55 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 95 | yes | yes | 3.176 | 16 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/owner_recap |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.436 | 20 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/owner_recap |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.486 | 13 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/owner_recap |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.371 | 15 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.784 | 18 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/owner_recap |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.755 | 13 | 3 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/owner_recap |

### shipment_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 55 | no | no | 3.046 | 32 | 12 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/shipment_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 57 | no | no | 3.154 | 60 | 0 | 12 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 22 | no | no | 3.131 | 60 | 0 | 9 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/shipment_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 17 | no | no | 3.386 | 60 | 40 | 9 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/shipment_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 12 | no | no | 3.193 | 60 | 55 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/shipment_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 79 | no | no | 12.169 | 26 | 11 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 78 | no | no | 12.474 | 48 | 12 | 10 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/shipment_preview |
| single::openai/openai/gpt-oss-120b | single | 78 | no | no | 13.340 | 41 | 16 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/shipment_preview |

### simple_mean_bug_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.073 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/simple_mean_bug |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.179 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 95 | yes | yes | 3.111 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/simple_mean_bug |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 3.358 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/simple_mean_bug |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.509 | 5 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/simple_mean_bug |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 12.449 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.967 | 7 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/simple_mean_bug |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 14.056 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/simple_mean_bug |

### workspace_digest_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.041 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/workspace_digest |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.145 | 11 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.164 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/workspace_digest |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.416 | 17 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/workspace_digest |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.523 | 8 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/workspace_digest |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.217 | 10 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.611 | 27 | 8 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/workspace_digest |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.623 | 12 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/workspace_digest |
