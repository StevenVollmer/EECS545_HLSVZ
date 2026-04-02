# Custom Matrix Report: benchmark_round_split_compare

- Matrix root: `/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare`
- Runs: `69`
- Strict resolved rate: `0.667`
- Observed resolved rate: `0.667`
- Avg total score: `82.68`
- Avg relative cost to 4o-mini: `5.539`

## By Architecture

| Architecture | Runs | Strict Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost |
| --- | --- | --- | --- | --- | --- |
| planner_coder | 38 | 0.658 | 82.47 | 4.983 | 33.38 |
| planner_coder_reviewer | 31 | 0.677 | 82.94 | 6.220 | 19.65 |

## By Config And Architecture

| Config | Architecture | Runs | Strict Resolve | Observed Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost | Avg Turns | Avg Parse Err | Avg Tool Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 8 | 0.625 | 0.625 | 81.88 | 0.954 | 85.76 | 15.1 | 2.8 | 0.9 |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 10 | 0.500 | 0.500 | 71.90 | 3.112 | 23.08 | 26.2 | 12.9 | 4.1 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 10 | 0.700 | 0.700 | 82.70 | 3.012 | 27.43 | 16.9 | 1.4 | 1.8 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 10 | 0.800 | 0.800 | 93.30 | 12.047 | 7.74 | 11.7 | 1.7 | 1.6 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 1 | 0.000 | 0.000 | 45.00 | 1.006 | 44.72 | 10.0 | 0.0 | 1.0 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 10 | 0.500 | 0.500 | 66.20 | 3.380 | 19.62 | 27.2 | 19.1 | 0.7 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 10 | 0.800 | 0.800 | 93.30 | 3.167 | 29.46 | 11.9 | 1.2 | 1.1 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 10 | 0.800 | 0.800 | 93.10 | 12.636 | 7.37 | 17.8 | 4.6 | 2.1 |

## Size Split Comparison

| Config | Planner Size | Coder Size | Reviewer Size | Mixed Sizes | Runs | Strict Resolve | Avg Score | Avg Rel Cost | Avg Score/Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | 4 | 1 | 0 | yes | 8 | 0.625 | 81.88 | 0.954 | 85.76 |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 0 | no | 10 | 0.500 | 71.90 | 3.112 | 23.08 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 5 | 4 | 0 | yes | 10 | 0.700 | 82.70 | 3.012 | 27.43 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 0 | no | 10 | 0.800 | 93.30 | 12.047 | 7.74 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 1 | 4 | yes | 1 | 0.000 | 45.00 | 1.006 | 44.72 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 4 | no | 10 | 0.500 | 66.20 | 3.380 | 19.62 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 5 | 4 | 5 | yes | 10 | 0.800 | 93.30 | 3.167 | 29.46 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 5 | no | 10 | 0.800 | 93.10 | 12.636 | 7.37 |

## Hypothesis Check: PCR >= PC >= Single

| Case | Single Score | PC Score | PCR Score | Order Holds |
| --- | --- | --- | --- | --- |
| board_rollup_001 | n/a | 99 | 99 | n/a |
| budget_snapshot_001 | n/a | 72 | 70 | n/a |
| digest_preview_001 | n/a | 100 | 98 | n/a |
| incident_brief_001 | n/a | 99 | 98 | n/a |
| label_formatter_001 | n/a | 99 | 99 | n/a |
| nested_app_001 | n/a | 98 | 99 | n/a |
| owner_recap_001 | n/a | 98 | 98 | n/a |
| shipment_preview_001 | n/a | 79 | 78 | n/a |
| simple_mean_bug_001 | n/a | 100 | 100 | n/a |
| workspace_digest_001 | n/a | 99 | 100 | n/a |

## Mixed-Size vs Big-Only

| Case | Best Mixed Config | Mixed Score | Mixed Cost | Best Big-Only Config | Big Score | Big Cost | Similar Or Better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 2.991 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.090 | yes |
| budget_snapshot_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 72 | 2.989 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 72 | 11.972 | yes |
| digest_preview_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.012 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.633 | yes |
| incident_brief_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 98 | 3.014 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.070 | yes |
| label_formatter_001 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | 99 | 0.957 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.316 | yes |
| nested_app_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 99 | 3.153 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.148 | yes |
| owner_recap_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 98 | 3.211 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.059 | yes |
| shipment_preview_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 78 | 3.151 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 79 | 11.999 | yes |
| simple_mean_bug_001 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | 100 | 0.956 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.184 | yes |
| workspace_digest_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.028 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.391 | yes |

## By Case

| Case | Runs | Strict Resolve | Observed Resolve | Avg Score | Best Config | Best Architecture | Best Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | 7 | 1.000 | 1.000 | 98.00 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 |
| budget_snapshot_001 | 7 | 0.000 | 0.000 | 68.86 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 |
| digest_preview_001 | 6 | 0.667 | 0.667 | 67.67 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| incident_brief_001 | 7 | 1.000 | 1.000 | 97.86 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| label_formatter_001 | 8 | 0.750 | 0.750 | 84.88 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 |
| nested_app_001 | 7 | 0.571 | 0.571 | 69.29 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 |
| owner_recap_001 | 7 | 0.714 | 0.714 | 87.43 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 |
| shipment_preview_001 | 7 | 0.000 | 0.000 | 54.86 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 79 |
| simple_mean_bug_001 | 7 | 1.000 | 1.000 | 99.00 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| workspace_digest_001 | 6 | 1.000 | 1.000 | 98.83 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 |

## Top Runs

- `simple_mean_bug_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `0.956`
- `digest_preview_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.012`
- `simple_mean_bug_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.058`
- `simple_mean_bug_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | cost `3.184`
- `simple_mean_bug_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | cost `3.235`
- `workspace_digest_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | cost `3.391`
- `label_formatter_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `0.957`
- `board_rollup_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `2.991`
- `workspace_digest_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.028`
- `incident_brief_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.070`
- `board_rollup_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `99` | strict `True` | observed `True` | cost `3.090`
- `label_formatter_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `99` | strict `True` | observed `True` | cost `3.109`

## Best Failures

- `shipment_preview_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder` | score `79` | cost `11.999` | only part of the success validation passed; 3 protocol/parse errors; 3 tool errors; patch exists but does not satisfy success checks
- `shipment_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `78` | cost `3.151` | only part of the success validation passed; 1 protocol/parse errors; 2 tool errors; loop control triggered
- `shipment_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `78` | cost `12.535` | only part of the success validation passed; 9 protocol/parse errors; 5 tool errors; long run (40 turns)
- `owner_recap_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `76` | cost `3.399` | only part of the success validation passed; 17 protocol/parse errors; 4 tool errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `72` | cost `2.989` | only part of the success validation passed; 1 protocol/parse errors; 1 tool errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder` | score `72` | cost `11.972` | only part of the success validation passed; 3 protocol/parse errors; 2 tool errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `70` | cost `12.923` | only part of the success validation passed; 20 protocol/parse errors; 1 tool errors; long run (39 turns)
- `budget_snapshot_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b` | `planner_coder` | score `69` | cost `0.958` | only part of the success validation passed; edited code without post-edit validation; 1 tool errors; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `68` | cost `3.169` | only part of the success validation passed; edited code without post-edit validation; 1 protocol/parse errors; loop control triggered
- `budget_snapshot_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `66` | cost `3.079` | only part of the success validation passed; edited code without post-edit validation; loop control triggered; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `65` | cost `3.357` | only part of the success validation passed; edited code without post-edit validation; 1 protocol/parse errors; loop control triggered
- `shipment_preview_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b` | `planner_coder` | score `59` | cost `0.953` | success validation failed; 10 protocol/parse errors; 2 tool errors; loop control triggered

## Per-Case Comparison

### board_rollup_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 96 | yes | yes | 0.950 | 10 | 1 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder/board_rollup |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 2.991 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/board_rollup |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.090 | 11 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.160 | 12 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/board_rollup |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 96 | yes | yes | 3.418 | 12 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/board_rollup |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 12.027 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.553 | 7 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/board_rollup |

### budget_snapshot_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 69 | no | no | 0.958 | 8 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder/budget_snapshot |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 72 | no | no | 2.989 | 9 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/budget_snapshot |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 66 | no | no | 3.079 | 14 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 68 | no | no | 3.169 | 24 | 1 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/budget_snapshot |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 65 | no | no | 3.357 | 18 | 1 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/budget_snapshot |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 | no | no | 11.972 | 17 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 70 | no | no | 12.923 | 39 | 20 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/budget_snapshot |

### digest_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.012 | 9 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/digest_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 7 | no | no | 3.115 | 60 | 57 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 95 | yes | yes | 3.180 | 12 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/digest_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 12 | no | no | 3.382 | 60 | 57 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/digest_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 94 | yes | yes | 12.084 | 9 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.633 | 9 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/digest_preview |

### incident_brief_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 97 | yes | yes | 0.954 | 18 | 4 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder/incident_brief |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 98 | yes | yes | 3.014 | 9 | 3 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/incident_brief |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.070 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 3.190 | 9 | 3 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/incident_brief |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 97 | yes | yes | 3.360 | 12 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/incident_brief |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.995 | 12 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.634 | 12 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/incident_brief |

### label_formatter_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 99 | yes | yes | 0.957 | 17 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder/label_formatter |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 45 | no | no | 1.006 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder_reviewer/label_formatter |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 46 | no | no | 2.975 | 33 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/label_formatter |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 94 | yes | yes | 3.083 | 13 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.109 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/label_formatter |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.316 | 14 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/label_formatter |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 11.837 | 7 | 1 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.370 | 13 | 3 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/label_formatter |

### nested_app_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 41 | no | no | 0.951 | 19 | 4 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder/nested_app |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 97 | yes | yes | 3.041 | 16 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/nested_app |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 44 | no | no | 3.076 | 9 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.153 | 9 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/nested_app |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 8 | no | no | 3.387 | 60 | 58 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/nested_app |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.148 | 13 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.482 | 9 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/nested_app |

### owner_recap_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 94 | yes | yes | 0.957 | 18 | 3 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder/owner_recap |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 95 | yes | yes | 3.000 | 11 | 7 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/owner_recap |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 55 | no | no | 3.106 | 60 | 16 | 15 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 3.211 | 8 | 7 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/owner_recap |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 76 | no | no | 3.399 | 19 | 17 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/owner_recap |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.059 | 12 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 96 | yes | yes | 12.771 | 31 | 7 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/owner_recap |

### shipment_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 59 | no | no | 0.953 | 24 | 10 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder/shipment_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 21 | no | no | 3.011 | 60 | 2 | 10 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/shipment_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 56 | no | no | 3.108 | 60 | 56 | 18 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 78 | no | no | 3.151 | 19 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/shipment_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 13 | no | no | 3.418 | 60 | 58 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/shipment_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 79 | no | no | 11.999 | 20 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 78 | no | no | 12.535 | 40 | 9 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/shipment_preview |

### simple_mean_bug_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->ollama/qwen3.5:9b | planner_coder | 100 | yes | yes | 0.956 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen_planner_ollama_qwen_coder/planner_coder/simple_mean_bug |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.058 | 5 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/simple_mean_bug |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.184 | 5 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 100 | yes | yes | 3.235 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/simple_mean_bug |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 96 | yes | yes | 3.370 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/simple_mean_bug |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.236 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 12.769 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/simple_mean_bug |

### workspace_digest_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.028 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder/workspace_digest |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.207 | 20 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.113 | 14 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/workspace_digest |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 3.391 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_qwen/planner_coder_reviewer/workspace_digest |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.112 | 13 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.690 | 12 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare/umich_gptoss_120b/planner_coder_reviewer/workspace_digest |
