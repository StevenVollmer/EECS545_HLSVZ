# Custom Matrix Report: benchmark_round_split_compare_cloud

- Matrix root: `/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud`
- Runs: `160`
- Strict resolved rate: `0.744`
- Observed resolved rate: `0.744`
- Avg total score: `88.78`
- Avg relative compute burden to 4o-mini: `6.918`

## By Architecture

| Architecture | Runs | Strict Resolve | Avg Score | Avg Rel Compute | Avg Score/Compute |
| --- | --- | --- | --- | --- | --- |
| planner_coder | 60 | 0.750 | 87.25 | 6.157 | 20.80 |
| planner_coder_reviewer | 60 | 0.733 | 89.97 | 6.510 | 20.22 |
| single | 40 | 0.750 | 89.28 | 8.670 | 15.72 |

## By Config And Architecture

| Config | Architecture | Runs | Strict Resolve | Observed Resolve | Avg Score | Avg Rel Compute | Avg Score/Compute | Avg Turns | Avg Parse Err | Avg Tool Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 20 | 0.650 | 0.650 | 79.95 | 3.163 | 25.27 | 21.7 | 4.2 | 4.2 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 20 | 0.800 | 0.800 | 91.15 | 3.064 | 29.74 | 14.2 | 0.9 | 2.4 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 20 | 0.800 | 0.800 | 90.65 | 12.244 | 7.40 | 13.6 | 3.6 | 2.2 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 20 | 0.600 | 0.600 | 84.65 | 3.421 | 24.81 | 20.5 | 0.2 | 4.3 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 20 | 0.800 | 0.800 | 92.25 | 3.224 | 28.62 | 12.2 | 0.1 | 2.2 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 20 | 0.800 | 0.800 | 93.00 | 12.887 | 7.21 | 15.2 | 2.0 | 3.0 |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 20 | 0.700 | 0.700 | 86.05 | 3.467 | 24.78 | 20.1 | 1.6 | 4.8 |
| single::openai/openai/gpt-oss-120b | single | 20 | 0.800 | 0.800 | 92.50 | 13.873 | 6.66 | 13.3 | 3.1 | 2.4 |

## Size Split Comparison

| Config | Planner Size | Coder Size | Reviewer Size | Mixed Sizes | Runs | Strict Resolve | Avg Score | Avg Rel Compute | Avg Score/Compute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 0 | no | 20 | 0.650 | 79.95 | 3.163 | 25.27 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 5 | 4 | 0 | yes | 20 | 0.800 | 91.15 | 3.064 | 29.74 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 0 | no | 20 | 0.800 | 90.65 | 12.244 | 7.40 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 4 | no | 20 | 0.600 | 84.65 | 3.421 | 24.81 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 5 | 4 | 5 | yes | 20 | 0.800 | 92.25 | 3.224 | 28.62 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 5 | no | 20 | 0.800 | 93.00 | 12.887 | 7.21 |

## Hypothesis Check: PCR >= PC >= Single

| Case | Single Score | PC Score | PCR Score | Order Holds |
| --- | --- | --- | --- | --- |
| board_rollup_001 | 99 | 99 | 99 | yes |
| budget_snapshot_001 | 72 | 72 | 73 | yes |
| contact_card_001 | 72 | 72 | 72 | yes |
| digest_preview_001 | 98 | 99 | 96 | no |
| incident_brief_001 | 99 | 100 | 100 | yes |
| invoice_footer_001 | 98 | 98 | 99 | yes |
| label_formatter_001 | 99 | 100 | 100 | yes |
| median_window_001 | 99 | 99 | 99 | yes |
| milestone_rollup_001 | 99 | 99 | 99 | yes |
| nested_app_001 | 98 | 99 | 99 | yes |
| owner_recap_001 | 98 | 98 | 95 | no |
| owner_sort_001 | 100 | 99 | 99 | no |
| priority_snapshot_001 | 99 | 99 | 99 | yes |
| renewal_preview_001 | 79 | 94 | 79 | no |
| risk_score_001 | 100 | 99 | 99 | no |
| shipment_preview_001 | 60 | 60 | 80 | yes |
| simple_mean_bug_001 | 100 | 100 | 100 | yes |
| status_slug_001 | 100 | 100 | 100 | yes |
| team_digest_001 | 99 | 99 | 99 | yes |
| workspace_digest_001 | 99 | 99 | 99 | yes |

## Mixed-Size vs Big-Only

| Case | Best Mixed Config | Mixed Score | Mixed Cost | Best Big-Only Config | Big Score | Big Cost | Similar Or Better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.057 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.163 | yes |
| budget_snapshot_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 73 | 3.130 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 72 | 12.243 | yes |
| contact_card_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 71 | 3.030 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 72 | 12.234 | yes |
| digest_preview_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.058 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.134 | yes |
| incident_brief_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.075 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.181 | yes |
| invoice_footer_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 98 | 3.299 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 99 | 12.967 | yes |
| label_formatter_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.008 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 99 | 12.093 | yes |
| median_window_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.050 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.188 | yes |
| milestone_rollup_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.072 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.418 | yes |
| nested_app_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.055 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.206 | yes |
| owner_recap_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 95 | 3.086 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 12.396 | yes |
| owner_sort_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.084 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.147 | yes |
| priority_snapshot_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.059 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.387 | yes |
| renewal_preview_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 94 | 3.031 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 80 | 3.155 | yes |
| risk_score_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.086 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.314 | yes |
| shipment_preview_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 80 | 3.271 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 79 | 3.496 | yes |
| simple_mean_bug_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.142 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.216 | yes |
| status_slug_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.078 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 3.333 | yes |
| team_digest_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.078 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.447 | yes |
| workspace_digest_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.050 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.454 | yes |

## By Case

| Case | Runs | Strict Resolve | Observed Resolve | Avg Score | Best Config | Best Architecture | Best Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | 8 | 1.000 | 1.000 | 98.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| budget_snapshot_001 | 8 | 0.000 | 0.000 | 67.50 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 73 |
| contact_card_001 | 8 | 0.000 | 0.000 | 57.62 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 |
| digest_preview_001 | 8 | 0.875 | 0.875 | 93.00 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| incident_brief_001 | 8 | 1.000 | 1.000 | 98.88 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| invoice_footer_001 | 8 | 0.500 | 0.500 | 71.50 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 |
| label_formatter_001 | 8 | 1.000 | 1.000 | 98.75 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| median_window_001 | 8 | 1.000 | 1.000 | 98.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| milestone_rollup_001 | 8 | 1.000 | 1.000 | 98.00 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| nested_app_001 | 8 | 0.625 | 0.625 | 78.38 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| owner_recap_001 | 8 | 0.750 | 0.750 | 83.25 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 |
| owner_sort_001 | 8 | 1.000 | 1.000 | 98.75 | single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 |
| priority_snapshot_001 | 8 | 1.000 | 1.000 | 98.38 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| renewal_preview_001 | 8 | 0.125 | 0.125 | 80.50 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 94 |
| risk_score_001 | 8 | 1.000 | 1.000 | 98.62 | single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 |
| shipment_preview_001 | 8 | 0.000 | 0.000 | 61.62 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 80 |
| simple_mean_bug_001 | 8 | 1.000 | 1.000 | 99.00 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| status_slug_001 | 8 | 1.000 | 1.000 | 98.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| team_digest_001 | 8 | 1.000 | 1.000 | 98.38 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| workspace_digest_001 | 8 | 1.000 | 1.000 | 97.50 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |

## Top Runs

- `label_formatter_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | compute `3.008`
- `status_slug_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | compute `3.078`
- `label_formatter_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | compute `3.097`
- `incident_brief_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | compute `3.181`
- `simple_mean_bug_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | compute `3.216`
- `status_slug_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | compute `3.333`
- `incident_brief_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | compute `3.357`
- `simple_mean_bug_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | compute `3.377`
- `simple_mean_bug_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | compute `3.509`
- `risk_score_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | compute `3.510`
- `status_slug_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | compute `3.511`
- `owner_sort_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | compute `3.520`

## Best Failures

- `renewal_preview_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `80` | compute `3.155` | only part of the success validation passed; 1 tool errors; patch exists but does not satisfy success checks
- `shipment_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `80` | compute `3.271` | only part of the success validation passed; 5 tool errors; run ended without submit (reviewer_rejected); patch exists but does not satisfy success checks
- `renewal_preview_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `79` | compute `3.421` | only part of the success validation passed; 1 protocol/parse errors; 1 tool errors; run ended without submit (reviewer_rejected)
- `owner_recap_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `79` | compute `3.471` | only part of the success validation passed; 3 protocol/parse errors; 3 tool errors; run ended without submit (reviewer_rejected)
- `shipment_preview_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `79` | compute `3.496` | only part of the success validation passed; 1 protocol/parse errors; 6 tool errors; patch exists but does not satisfy success checks
- `renewal_preview_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder` | score `79` | compute `12.175` | only part of the success validation passed; 2 protocol/parse errors; 2 tool errors; patch exists but does not satisfy success checks
- `renewal_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `79` | compute `12.710` | only part of the success validation passed; 1 protocol/parse errors; 2 tool errors; patch exists but does not satisfy success checks
- `renewal_preview_001` | `single::openai/openai/gpt-oss-120b` | `single` | score `79` | compute `13.911` | only part of the success validation passed; 1 protocol/parse errors; 2 tool errors; patch exists but does not satisfy success checks
- `shipment_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `78` | compute `12.733` | only part of the success validation passed; 6 protocol/parse errors; 7 tool errors; long run (47 turns)
- `renewal_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `77` | compute `3.184` | only part of the success validation passed; edited code without post-edit validation; patch exists but does not satisfy success checks
- `renewal_preview_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `77` | compute `3.475` | only part of the success validation passed; edited code without post-edit validation; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `73` | compute `3.130` | only part of the success validation passed; loop control triggered; run ended without submit (reviewer_rejected); patch exists but does not satisfy success checks

## Per-Case Comparison

### board_rollup_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.057 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/board_rollup |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.163 | 9 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.224 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/board_rollup |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.425 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/board_rollup |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.468 | 33 | 0 | 14 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/board_rollup |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.246 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.844 | 7 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/board_rollup |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.957 | 8 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/board_rollup |

### budget_snapshot_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 70 | no | no | 3.049 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/budget_snapshot |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 43 | no | no | 3.169 | 9 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 73 | no | no | 3.130 | 13 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/budget_snapshot |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 69 | no | no | 3.421 | 23 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/budget_snapshot |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 69 | no | no | 3.443 | 18 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/budget_snapshot |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 | no | no | 12.243 | 14 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 72 | no | no | 12.714 | 18 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/budget_snapshot |
| single::openai/openai/gpt-oss-120b | single | 72 | no | no | 13.787 | 18 | 7 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/budget_snapshot |

### contact_card_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 71 | no | no | 3.030 | 60 | 12 | 12 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/contact_card |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 44 | no | no | 3.148 | 60 | 1 | 14 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/contact_card |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 45 | no | no | 3.267 | 60 | 0 | 12 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/contact_card |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 40 | no | no | 3.420 | 60 | 0 | 29 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/contact_card |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 45 | no | no | 3.432 | 60 | 0 | 14 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/contact_card |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 | no | no | 12.234 | 25 | 8 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/contact_card |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 72 | no | no | 12.849 | 22 | 2 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/contact_card |
| single::openai/openai/gpt-oss-120b | single | 72 | no | no | 13.823 | 19 | 5 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/contact_card |

### digest_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.058 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/digest_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 95 | yes | yes | 3.174 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 94 | yes | yes | 3.208 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/digest_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 68 | no | no | 3.466 | 18 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/digest_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 96 | yes | yes | 3.422 | 8 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/digest_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.134 | 11 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 96 | yes | yes | 13.074 | 29 | 7 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/digest_preview |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.640 | 14 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/digest_preview |

### incident_brief_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.075 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/incident_brief |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.181 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.224 | 13 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/incident_brief |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 3.357 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/incident_brief |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.480 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/incident_brief |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.295 | 10 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.868 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/incident_brief |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.841 | 16 | 2 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/incident_brief |

### invoice_footer_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 45 | no | no | 3.068 | 18 | 0 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/invoice_footer |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 45 | no | no | 3.131 | 27 | 0 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/invoice_footer |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 3.299 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/invoice_footer |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 44 | no | no | 3.499 | 60 | 0 | 10 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/invoice_footer |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 45 | no | no | 3.464 | 60 | 0 | 29 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/invoice_footer |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.223 | 12 | 2 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/invoice_footer |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 12.967 | 9 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/invoice_footer |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.749 | 9 | 1 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/invoice_footer |

### label_formatter_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.008 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/label_formatter |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 98 | yes | yes | 3.176 | 18 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 100 | yes | yes | 3.097 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/label_formatter |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 98 | yes | yes | 3.431 | 12 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/label_formatter |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.500 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/label_formatter |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 12.093 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.517 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/label_formatter |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.889 | 8 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/label_formatter |

### median_window_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.050 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/median_window |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.188 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/median_window |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.224 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/median_window |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.411 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/median_window |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.513 | 6 | 1 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/median_window |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.371 | 6 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/median_window |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.820 | 17 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/median_window |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.948 | 8 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/median_window |

### milestone_rollup_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.072 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/milestone_rollup |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 98 | yes | yes | 3.130 | 17 | 4 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/milestone_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.285 | 10 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/milestone_rollup |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.418 | 12 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/milestone_rollup |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.476 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/milestone_rollup |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.279 | 9 | 1 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/milestone_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 94 | yes | yes | 12.981 | 25 | 1 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/milestone_rollup |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 14.062 | 19 | 6 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/milestone_rollup |

### nested_app_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.055 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/nested_app |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 45 | no | no | 3.145 | 60 | 0 | 20 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.201 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/nested_app |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 45 | no | no | 3.521 | 60 | 0 | 20 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/nested_app |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 45 | no | no | 3.432 | 60 | 0 | 18 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/nested_app |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.206 | 15 | 4 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.879 | 13 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/nested_app |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.930 | 13 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/nested_app |

### owner_recap_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 95 | yes | yes | 3.086 | 12 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/owner_recap |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 13 | no | no | 3.167 | 60 | 58 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 95 | yes | yes | 3.190 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/owner_recap |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 79 | no | no | 3.471 | 19 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/owner_recap |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 94 | yes | yes | 3.443 | 18 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/owner_recap |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.396 | 13 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 94 | yes | yes | 12.922 | 11 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/owner_recap |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.862 | 11 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/owner_recap |

### owner_sort_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.084 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/owner_sort |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.147 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/owner_sort |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.237 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/owner_sort |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.342 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/owner_sort |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.520 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/owner_sort |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.368 | 8 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/owner_sort |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.993 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/owner_sort |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 14.023 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/owner_sort |

### priority_snapshot_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.059 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/priority_snapshot |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 98 | yes | yes | 3.161 | 11 | 9 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/priority_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.239 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/priority_snapshot |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.387 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/priority_snapshot |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.471 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/priority_snapshot |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.247 | 13 | 2 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/priority_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.948 | 14 | 1 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/priority_snapshot |
| single::openai/openai/gpt-oss-120b | single | 97 | yes | yes | 13.624 | 24 | 6 | 6 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/priority_snapshot |

### renewal_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 94 | yes | yes | 3.031 | 16 | 7 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/renewal_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 80 | no | no | 3.155 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/renewal_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 77 | no | no | 3.184 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/renewal_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 79 | no | no | 3.421 | 11 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/renewal_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 77 | no | no | 3.475 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/renewal_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 79 | no | no | 12.175 | 13 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/renewal_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 79 | no | no | 12.710 | 16 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/renewal_preview |
| single::openai/openai/gpt-oss-120b | single | 79 | no | no | 13.911 | 10 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/renewal_preview |

### risk_score_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.086 | 7 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/risk_score |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 98 | yes | yes | 3.130 | 21 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/risk_score |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.235 | 7 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/risk_score |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.314 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/risk_score |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.510 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/risk_score |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.277 | 6 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/risk_score |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.933 | 7 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/risk_score |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 14.059 | 8 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/risk_score |

### shipment_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 60 | no | no | 3.068 | 60 | 0 | 10 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/shipment_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 58 | no | no | 3.187 | 8 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 80 | no | no | 3.271 | 28 | 0 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/shipment_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 79 | no | no | 3.496 | 35 | 1 | 6 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/shipment_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 58 | no | no | 3.296 | 40 | 30 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/shipment_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 20 | no | no | 12.171 | 60 | 36 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 78 | no | no | 12.733 | 47 | 6 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/shipment_preview |
| single::openai/openai/gpt-oss-120b | single | 60 | no | no | 13.299 | 24 | 8 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/shipment_preview |

### simple_mean_bug_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.142 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/simple_mean_bug |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.216 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 3.266 | 7 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/simple_mean_bug |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 3.377 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/simple_mean_bug |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.509 | 5 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/simple_mean_bug |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 12.173 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.926 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/simple_mean_bug |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 14.078 | 8 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/simple_mean_bug |

### status_slug_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 3.078 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/status_slug |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 96 | yes | yes | 3.175 | 8 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/status_slug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.229 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/status_slug |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 3.333 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/status_slug |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.511 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/status_slug |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.292 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/status_slug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 13.012 | 8 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/status_slug |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.934 | 9 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/status_slug |

### team_digest_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.078 | 10 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/team_digest |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 97 | yes | yes | 3.146 | 13 | 13 | 6 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/team_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 3.255 | 9 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/team_digest |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.447 | 11 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/team_digest |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.488 | 25 | 0 | 8 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/team_digest |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.204 | 13 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/team_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 12.950 | 9 | 1 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/team_digest |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 13.960 | 21 | 7 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/team_digest |

### workspace_digest_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 3.050 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/workspace_digest |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 94 | yes | yes | 3.168 | 60 | 0 | 25 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 95 | yes | yes | 3.213 | 17 | 0 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/workspace_digest |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.454 | 19 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/workspace_digest |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.484 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/workspace_digest |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.250 | 14 | 4 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 13.091 | 19 | 4 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/workspace_digest |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 14.088 | 12 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/workspace_digest |
