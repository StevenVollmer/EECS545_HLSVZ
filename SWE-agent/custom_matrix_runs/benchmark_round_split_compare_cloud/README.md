# Custom Matrix Report: benchmark_round_split_compare_cloud

- Matrix root: `/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud`
- Runs: `80`
- Strict resolved rate: `0.662`
- Observed resolved rate: `0.662`
- Avg total score: `82.39`
- Avg relative cost to 4o-mini: `6.658`

## By Architecture

| Architecture | Runs | Strict Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost |
| --- | --- | --- | --- | --- | --- |
| planner_coder | 30 | 0.600 | 78.60 | 6.079 | 18.10 |
| planner_coder_reviewer | 30 | 0.700 | 86.27 | 6.399 | 19.25 |
| single | 20 | 0.700 | 82.25 | 7.915 | 514.88 |

## By Config And Architecture

| Config | Architecture | Runs | Strict Resolve | Observed Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost | Avg Turns | Avg Parse Err | Avg Tool Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 10 | 0.500 | 0.500 | 71.30 | 3.142 | 22.66 | 24.5 | 5.8 | 4.2 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 10 | 0.500 | 0.500 | 72.70 | 3.022 | 24.03 | 26.5 | 14.9 | 1.1 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 10 | 0.800 | 0.800 | 91.80 | 12.075 | 7.60 | 12.4 | 2.7 | 1.8 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 10 | 0.600 | 0.600 | 82.10 | 3.391 | 24.25 | 20.6 | 6.5 | 3.7 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 10 | 0.700 | 0.700 | 83.10 | 3.188 | 26.08 | 29.5 | 4.5 | 6.9 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 10 | 0.800 | 0.800 | 93.60 | 12.619 | 7.42 | 15.0 | 3.8 | 2.5 |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 10 | 0.600 | 0.600 | 82.80 | 3.455 | 23.91 | 18.5 | 5.3 | 2.6 |
| single::openai/openai/gpt-oss-120b | single | 10 | 0.800 | 0.800 | 81.70 | 12.376 | 1005.86 | 15.7 | 6.0 | 2.1 |

## Size Split Comparison

| Config | Planner Size | Coder Size | Reviewer Size | Mixed Sizes | Runs | Strict Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 0 | no | 10 | 0.500 | 71.30 | 3.142 | 22.66 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 5 | 4 | 0 | yes | 10 | 0.500 | 72.70 | 3.022 | 24.03 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 0 | no | 10 | 0.800 | 91.80 | 12.075 | 7.60 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 4 | no | 10 | 0.600 | 82.10 | 3.391 | 24.25 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 5 | 4 | 5 | yes | 10 | 0.700 | 83.10 | 3.188 | 26.08 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 5 | no | 10 | 0.800 | 93.60 | 12.619 | 7.42 |

## Hypothesis Check: PCR >= PC >= Single

| Case | Single Score | PC Score | PCR Score | Order Holds |
| --- | --- | --- | --- | --- |
| board_rollup_001 | 99 | 99 | 99 | yes |
| budget_snapshot_001 | 99 | 73 | 72 | no |
| digest_preview_001 | 98 | 98 | 98 | yes |
| incident_brief_001 | 99 | 100 | 100 | yes |
| label_formatter_001 | 99 | 99 | 99 | yes |
| nested_app_001 | 98 | 98 | 98 | yes |
| owner_recap_001 | 98 | 98 | 98 | yes |
| shipment_preview_001 | 59 | 60 | 79 | yes |
| simple_mean_bug_001 | 100 | 100 | 100 | yes |
| workspace_digest_001 | 99 | 99 | 99 | yes |

## Mixed-Size vs Big-Only

| Case | Best Mixed Config | Mixed Score | Mixed Cost | Best Big-Only Config | Big Score | Big Cost | Similar Or Better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.008 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.112 | yes |
| budget_snapshot_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 69 | 3.017 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 73 | 12.096 | yes |
| digest_preview_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 96 | 3.237 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.027 | yes |
| incident_brief_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.018 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.183 | yes |
| label_formatter_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 97 | 3.155 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.100 | yes |
| nested_app_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 45 | 3.286 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.013 | no |
| owner_recap_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 98 | 3.026 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 98 | 3.421 | yes |
| shipment_preview_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 58 | 3.139 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 79 | 12.682 | no |
| simple_mean_bug_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.090 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.191 | yes |
| workspace_digest_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.025 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.200 | yes |

## By Case

| Case | Runs | Strict Resolve | Observed Resolve | Avg Score | Best Config | Best Architecture | Best Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | 8 | 1.000 | 1.000 | 98.75 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 |
| budget_snapshot_001 | 8 | 0.125 | 0.125 | 66.12 | single::openai/openai/gpt-oss-120b | single | 99 |
| digest_preview_001 | 8 | 0.500 | 0.500 | 82.25 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 |
| incident_brief_001 | 8 | 1.000 | 1.000 | 98.75 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| label_formatter_001 | 8 | 0.875 | 0.875 | 90.12 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 |
| nested_app_001 | 8 | 0.375 | 0.375 | 64.25 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 |
| owner_recap_001 | 8 | 0.875 | 0.875 | 86.25 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 |
| shipment_preview_001 | 8 | 0.000 | 0.000 | 50.75 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 79 |
| simple_mean_bug_001 | 8 | 1.000 | 1.000 | 99.12 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| workspace_digest_001 | 8 | 0.875 | 0.875 | 87.50 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |

## Top Runs

- `simple_mean_bug_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.090`
- `incident_brief_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.183`
- `simple_mean_bug_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.191`
- `simple_mean_bug_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | cost `3.221`
- `incident_brief_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | cost `3.307`
- `simple_mean_bug_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | cost `3.509`
- `board_rollup_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.008`
- `incident_brief_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.018`
- `workspace_digest_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.025`
- `label_formatter_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.100`
- `board_rollup_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.112`
- `workspace_digest_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `99` | strict `True` | observed `True` | cost `3.124`

## Best Failures

- `shipment_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `79` | cost `12.682` | only part of the success validation passed; 9 protocol/parse errors; 2 tool errors; run ended without submit (reviewer_rejected)
- `budget_snapshot_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder` | score `73` | cost `12.096` | only part of the success validation passed; 7 protocol/parse errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `72` | cost `12.440` | only part of the success validation passed; 3 protocol/parse errors; 3 tool errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `69` | cost `3.017` | only part of the success validation passed; edited code without post-edit validation; loop control triggered; patch exists but does not satisfy success checks
- `digest_preview_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `69` | cost `3.453` | only part of the success validation passed; edited code without post-edit validation; loop control triggered; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `69` | cost `3.478` | only part of the success validation passed; edited code without post-edit validation; loop control triggered; patch exists but does not satisfy success checks
- `digest_preview_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `68` | cost `3.011` | only part of the success validation passed; edited code without post-edit validation; 1 protocol/parse errors; loop control triggered
- `digest_preview_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `66` | cost `3.108` | only part of the success validation passed; edited code without post-edit validation; 2 tool errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `66` | cost `3.357` | only part of the success validation passed; edited code without post-edit validation; loop control triggered; patch exists but does not satisfy success checks
- `digest_preview_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `65` | cost `3.401` | only part of the success validation passed; edited code without post-edit validation; 3 tool errors; loop control triggered
- `shipment_preview_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder` | score `60` | cost `12.102` | success validation failed; 10 protocol/parse errors; 5 tool errors; patch exists but does not satisfy success checks
- `shipment_preview_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `59` | cost `3.281` | success validation failed; 53 protocol/parse errors; 15 tool errors; long run (60 turns)

## Per-Case Comparison

### board_rollup_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.008 | 8 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/board_rollup |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.112 | 9 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.145 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/board_rollup |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.400 | 12 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/board_rollup |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.476 | 12 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/board_rollup |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 12.011 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.897 | 34 | 12 | 10 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/board_rollup |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.977 | 8 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/board_rollup |

### budget_snapshot_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 69 | no | no | 3.017 | 16 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/budget_snapshot |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 40 | no | no | 3.120 | 8 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 41 | no | no | 3.141 | 21 | 2 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/budget_snapshot |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 66 | no | no | 3.357 | 12 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/budget_snapshot |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 69 | no | no | 3.478 | 13 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/budget_snapshot |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 73 | no | no | 12.096 | 21 | 7 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 72 | no | no | 12.440 | 16 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/budget_snapshot |
| single::openai/openai/gpt-oss-120b | single | 99 | yes | yes | 13.600 | 20 | 14 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/budget_snapshot |

### digest_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 68 | no | no | 3.011 | 19 | 1 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/digest_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 66 | no | no | 3.108 | 9 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 96 | yes | yes | 3.237 | 42 | 38 | 6 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/digest_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 65 | no | no | 3.401 | 12 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/digest_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 69 | no | no | 3.453 | 28 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/digest_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.027 | 10 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.314 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/digest_preview |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.390 | 11 | 3 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/digest_preview |

### incident_brief_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.018 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/incident_brief |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.183 | 11 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 3.168 | 9 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/incident_brief |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 3.307 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/incident_brief |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.480 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/incident_brief |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.024 | 10 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.612 | 13 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/incident_brief |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.859 | 13 | 4 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/incident_brief |

### label_formatter_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 38 | no | no | 2.976 | 60 | 46 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/label_formatter |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.100 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 97 | yes | yes | 3.155 | 60 | 0 | 8 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/label_formatter |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 92 | yes | yes | 3.365 | 12 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/label_formatter |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.503 | 8 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/label_formatter |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.974 | 9 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 12.442 | 8 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/label_formatter |
| single::openai/openai/gpt-oss-120b | single | 99 | yes | yes | 13.815 | 7 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/label_formatter |

### nested_app_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 44 | no | no | 3.014 | 60 | 43 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/nested_app |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 45 | no | no | 3.160 | 60 | 0 | 20 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 45 | no | no | 3.286 | 60 | 0 | 27 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/nested_app |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 45 | no | no | 3.474 | 60 | 0 | 19 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/nested_app |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 41 | no | no | 3.442 | 22 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/nested_app |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.013 | 9 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.557 | 12 | 4 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/nested_app |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.856 | 14 | 4 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/nested_app |

### owner_recap_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 98 | yes | yes | 3.026 | 15 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/owner_recap |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 8 | no | no | 3.124 | 60 | 58 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 3.260 | 20 | 1 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/owner_recap |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 98 | yes | yes | 3.421 | 11 | 5 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/owner_recap |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 94 | yes | yes | 3.443 | 18 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/owner_recap |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.039 | 12 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.722 | 15 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/owner_recap |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.822 | 17 | 6 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/owner_recap |

### shipment_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 13 | no | no | 3.031 | 60 | 58 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/shipment_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 57 | no | no | 3.121 | 60 | 0 | 13 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 58 | no | no | 3.139 | 60 | 2 | 18 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/shipment_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 59 | no | no | 3.411 | 60 | 59 | 11 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/shipment_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 59 | no | no | 3.281 | 60 | 53 | 15 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/shipment_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 60 | no | no | 12.102 | 25 | 10 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 79 | no | no | 12.682 | 23 | 9 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/shipment_preview |
| single::openai/openai/gpt-oss-120b | single | 21 | no | no | 13.393 | 60 | 25 | 11 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/shipment_preview |

### simple_mean_bug_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.090 | 5 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/simple_mean_bug |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.191 | 5 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 100 | yes | yes | 3.221 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/simple_mean_bug |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.361 | 6 | 1 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/simple_mean_bug |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.509 | 5 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/simple_mean_bug |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.371 | 8 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.877 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/simple_mean_bug |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 14.044 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/simple_mean_bug |

### workspace_digest_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.025 | 11 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/workspace_digest |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.200 | 15 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.124 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/workspace_digest |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 98 | yes | yes | 3.412 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/workspace_digest |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.487 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/workspace_digest |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.091 | 12 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.647 | 13 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/workspace_digest |
| single::openai/openai/gpt-oss-120b | single | 10 | no | no | 0.000 | 0 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/workspace_digest |
