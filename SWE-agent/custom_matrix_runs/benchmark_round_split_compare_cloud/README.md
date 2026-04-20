# Custom Matrix Report: benchmark_round_split_compare_cloud

- Matrix root: `/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud`
- Runs: `160`
- Strict resolved rate: `0.744`
- Observed resolved rate: `0.744`
- Avg total score: `88.88`
- Avg relative compute burden to 4o-mini: `8.357`

## By Architecture

| Architecture | Runs | Strict Resolve | Avg Score | Avg Rel Compute | Avg Score/Compute |
| --- | --- | --- | --- | --- | --- |
| planner_coder | 60 | 0.750 | 87.35 | 5.344 | 30.63 |
| planner_coder_reviewer | 60 | 0.733 | 90.10 | 7.583 | 22.71 |
| single | 40 | 0.750 | 89.35 | 14.039 | 14.03 |

## By Config And Architecture

| Config | Architecture | Runs | Strict Resolve | Observed Resolve | Avg Score | Avg Rel Compute | Avg Score/Compute | Avg Turns | Avg Parse Err | Avg Tool Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 20 | 0.650 | 0.650 | 80.10 | 2.273 | 35.22 | 21.7 | 4.2 | 4.2 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 20 | 0.800 | 0.800 | 91.25 | 1.865 | 49.03 | 14.2 | 0.9 | 2.4 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 20 | 0.800 | 0.800 | 90.70 | 11.894 | 7.63 | 13.6 | 3.6 | 2.2 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 20 | 0.600 | 0.600 | 84.90 | 3.339 | 25.80 | 20.5 | 0.2 | 4.3 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 20 | 0.800 | 0.800 | 92.40 | 2.526 | 36.82 | 12.2 | 0.1 | 2.2 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 20 | 0.800 | 0.800 | 93.00 | 16.884 | 5.52 | 15.2 | 2.0 | 3.0 |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 20 | 0.700 | 0.700 | 86.20 | 3.529 | 24.30 | 20.1 | 1.6 | 4.8 |
| single::openai/openai/gpt-oss-120b | single | 20 | 0.800 | 0.800 | 92.50 | 24.548 | 3.76 | 13.3 | 3.1 | 2.4 |

## Size Split Comparison

| Config | Planner Size | Coder Size | Reviewer Size | Mixed Sizes | Runs | Strict Resolve | Avg Score | Avg Rel Compute | Avg Score/Compute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 0 | no | 20 | 0.650 | 80.10 | 2.273 | 35.22 |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 5 | 4 | 0 | yes | 20 | 0.800 | 91.25 | 1.865 | 49.03 |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 0 | no | 20 | 0.800 | 90.70 | 11.894 | 7.63 |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 4 | 4 | 4 | no | 20 | 0.600 | 84.90 | 3.339 | 25.80 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 5 | 4 | 5 | yes | 20 | 0.800 | 92.40 | 2.526 | 36.82 |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 5 | 5 | 5 | no | 20 | 0.800 | 93.00 | 16.884 | 5.52 |

## Hypothesis Check: PCR >= PC >= Single

| Case | Single Score | PC Score | PCR Score | Order Holds |
| --- | --- | --- | --- | --- |
| board_rollup_001 | 99 | 99 | 99 | yes |
| budget_snapshot_001 | 72 | 72 | 74 | yes |
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
| renewal_preview_001 | 79 | 95 | 79 | no |
| risk_score_001 | 100 | 99 | 99 | no |
| shipment_preview_001 | 60 | 60 | 80 | yes |
| simple_mean_bug_001 | 100 | 100 | 100 | yes |
| status_slug_001 | 100 | 100 | 100 | yes |
| team_digest_001 | 99 | 99 | 99 | yes |
| workspace_digest_001 | 99 | 99 | 99 | yes |

## Mixed-Size vs Big-Only

| Case | Best Mixed Config | Mixed Score | Mixed Cost | Best Big-Only Config | Big Score | Big Cost | Similar Or Better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.834 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 2.275 | yes |
| budget_snapshot_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 74 | 2.136 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 72 | 11.890 | yes |
| contact_card_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 71 | 1.725 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 72 | 11.820 | yes |
| digest_preview_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.841 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 11.040 | yes |
| incident_brief_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.910 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 2.347 | yes |
| invoice_footer_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 99 | 2.836 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 99 | 17.508 | yes |
| label_formatter_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 1.632 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 2.328 | yes |
| median_window_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.807 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 2.377 | yes |
| milestone_rollup_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.898 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.329 | yes |
| nested_app_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.828 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 11.596 | yes |
| owner_recap_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 95 | 1.957 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | 98 | 13.073 | yes |
| owner_sort_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.948 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 2.207 | yes |
| priority_snapshot_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.843 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.201 | yes |
| renewal_preview_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 95 | 1.729 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 80 | 2.241 | yes |
| risk_score_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.957 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 2.136 | yes |
| shipment_preview_001 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | 80 | 2.719 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 79 | 3.652 | yes |
| simple_mean_bug_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 2.185 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 2.494 | yes |
| status_slug_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 1.922 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 100 | 2.975 | yes |
| team_digest_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.923 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.448 | yes |
| workspace_digest_001 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 1.807 | planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | 99 | 3.476 | yes |

## By Case

| Case | Runs | Strict Resolve | Observed Resolve | Avg Score | Best Config | Best Architecture | Best Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| board_rollup_001 | 8 | 1.000 | 1.000 | 98.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| budget_snapshot_001 | 8 | 0.000 | 0.000 | 67.88 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 74 |
| contact_card_001 | 8 | 0.000 | 0.000 | 57.75 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 |
| digest_preview_001 | 8 | 0.875 | 0.875 | 93.25 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| incident_brief_001 | 8 | 1.000 | 1.000 | 98.88 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| invoice_footer_001 | 8 | 0.500 | 0.500 | 72.00 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 |
| label_formatter_001 | 8 | 1.000 | 1.000 | 99.00 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| median_window_001 | 8 | 1.000 | 1.000 | 98.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| milestone_rollup_001 | 8 | 1.000 | 1.000 | 98.00 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| nested_app_001 | 8 | 0.625 | 0.625 | 78.38 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| owner_recap_001 | 8 | 0.750 | 0.750 | 83.38 | planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 |
| owner_sort_001 | 8 | 1.000 | 1.000 | 98.75 | single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 |
| priority_snapshot_001 | 8 | 1.000 | 1.000 | 98.38 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| renewal_preview_001 | 8 | 0.125 | 0.125 | 80.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 95 |
| risk_score_001 | 8 | 1.000 | 1.000 | 98.75 | single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 |
| shipment_preview_001 | 8 | 0.000 | 0.000 | 61.88 | planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 80 |
| simple_mean_bug_001 | 8 | 1.000 | 1.000 | 99.00 | planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| status_slug_001 | 8 | 1.000 | 1.000 | 98.62 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 |
| team_digest_001 | 8 | 1.000 | 1.000 | 98.38 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |
| workspace_digest_001 | 8 | 1.000 | 1.000 | 97.50 | planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 |

## Top Runs

- `label_formatter_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | compute `1.632`
- `status_slug_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | compute `1.922`
- `label_formatter_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | compute `2.003`
- `incident_brief_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | compute `2.347`
- `simple_mean_bug_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `100` | strict `True` | observed `True` | compute `2.494`
- `status_slug_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | compute `2.975`
- `incident_brief_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | compute `3.076`
- `simple_mean_bug_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `100` | strict `True` | observed `True` | compute `3.160`
- `simple_mean_bug_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | compute `3.702`
- `risk_score_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | compute `3.706`
- `status_slug_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | compute `3.712`
- `owner_sort_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `100` | strict `True` | observed `True` | compute `3.749`

## Best Failures

- `renewal_preview_001` | `planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder` | score `80` | compute `2.241` | only part of the success validation passed; 1 tool errors; patch exists but does not satisfy success checks
- `shipment_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `80` | compute `2.719` | only part of the success validation passed; 5 tool errors; run ended without submit (reviewer_rejected); patch exists but does not satisfy success checks
- `renewal_preview_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `79` | compute `3.341` | only part of the success validation passed; 1 protocol/parse errors; 1 tool errors; run ended without submit (reviewer_rejected)
- `owner_recap_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `79` | compute `3.548` | only part of the success validation passed; 3 protocol/parse errors; 3 tool errors; run ended without submit (reviewer_rejected)
- `shipment_preview_001` | `planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `planner_coder_reviewer` | score `79` | compute `3.652` | only part of the success validation passed; 1 protocol/parse errors; 6 tool errors; patch exists but does not satisfy success checks
- `renewal_preview_001` | `planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder` | score `79` | compute `11.355` | only part of the success validation passed; 2 protocol/parse errors; 2 tool errors; patch exists but does not satisfy success checks
- `renewal_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `79` | compute `15.513` | only part of the success validation passed; 1 protocol/parse errors; 2 tool errors; patch exists but does not satisfy success checks
- `renewal_preview_001` | `single::openai/openai/gpt-oss-120b` | `single` | score `79` | compute `24.839` | only part of the success validation passed; 1 protocol/parse errors; 2 tool errors; patch exists but does not satisfy success checks
- `shipment_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `78` | compute `15.689` | only part of the success validation passed; 6 protocol/parse errors; 7 tool errors; long run (47 turns)
- `renewal_preview_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `77` | compute `2.359` | only part of the success validation passed; edited code without post-edit validation; patch exists but does not satisfy success checks
- `renewal_preview_001` | `single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct` | `single` | score `77` | compute `3.562` | only part of the success validation passed; edited code without post-edit validation; patch exists but does not satisfy success checks
- `budget_snapshot_001` | `planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b` | `planner_coder_reviewer` | score `74` | compute `2.136` | only part of the success validation passed; run ended without submit (reviewer_rejected); patch exists but does not satisfy success checks

## Per-Case Comparison

### board_rollup_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.834 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/board_rollup |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 2.275 | 9 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.525 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/board_rollup |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.357 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/board_rollup |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.535 | 33 | 0 | 14 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/board_rollup |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.914 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/board_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 16.554 | 7 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/board_rollup |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 25.196 | 8 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/board_rollup |

### budget_snapshot_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 70 | no | no | 1.802 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/budget_snapshot |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 43 | no | no | 2.297 | 9 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 74 | no | no | 2.136 | 13 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/budget_snapshot |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 70 | no | no | 3.339 | 23 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/budget_snapshot |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 70 | no | no | 3.433 | 18 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/budget_snapshot |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 | no | no | 11.890 | 14 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/budget_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 72 | no | no | 15.549 | 18 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/budget_snapshot |
| single::openai/openai/gpt-oss-120b | single | 72 | no | no | 23.876 | 18 | 7 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/budget_snapshot |

### contact_card_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 71 | no | no | 1.725 | 60 | 12 | 12 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/contact_card |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 44 | no | no | 2.213 | 60 | 1 | 14 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/contact_card |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 45 | no | no | 2.703 | 60 | 0 | 12 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/contact_card |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 41 | no | no | 3.335 | 60 | 0 | 29 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/contact_card |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 45 | no | no | 3.385 | 60 | 0 | 14 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/contact_card |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 72 | no | no | 11.820 | 25 | 8 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/contact_card |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 72 | no | no | 16.591 | 22 | 2 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/contact_card |
| single::openai/openai/gpt-oss-120b | single | 72 | no | no | 24.155 | 19 | 5 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/contact_card |

### digest_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.841 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/digest_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 95 | yes | yes | 2.320 | 10 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 95 | yes | yes | 2.459 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/digest_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 69 | no | no | 3.526 | 18 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/digest_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 96 | yes | yes | 3.344 | 8 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/digest_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.040 | 11 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/digest_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 96 | yes | yes | 18.341 | 29 | 7 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/digest_preview |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 22.739 | 14 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/digest_preview |

### incident_brief_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.910 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/incident_brief |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 2.347 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.527 | 13 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/incident_brief |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 3.076 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/incident_brief |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.583 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/incident_brief |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.292 | 10 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/incident_brief |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 16.740 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/incident_brief |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 24.297 | 16 | 2 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/incident_brief |

### invoice_footer_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 46 | no | no | 1.880 | 18 | 0 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/invoice_footer |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 46 | no | no | 2.140 | 27 | 0 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/invoice_footer |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.836 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/invoice_footer |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 45 | no | no | 3.663 | 60 | 0 | 10 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/invoice_footer |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 45 | no | no | 3.517 | 60 | 0 | 29 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/invoice_footer |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.729 | 12 | 2 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/invoice_footer |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 17.508 | 9 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/invoice_footer |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 23.585 | 9 | 1 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/invoice_footer |

### label_formatter_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 1.632 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/label_formatter |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 2.328 | 18 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 100 | yes | yes | 2.003 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/label_formatter |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.383 | 12 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/label_formatter |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.667 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/label_formatter |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 10.725 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/label_formatter |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 14.011 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/label_formatter |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 24.672 | 8 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/label_formatter |

### median_window_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.807 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/median_window |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 2.377 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/median_window |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.524 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/median_window |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.299 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/median_window |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.720 | 6 | 1 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/median_window |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.879 | 6 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/median_window |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 16.369 | 17 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/median_window |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 25.126 | 8 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/median_window |

### milestone_rollup_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.898 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/milestone_rollup |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 98 | yes | yes | 2.138 | 17 | 4 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/milestone_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.780 | 10 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/milestone_rollup |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.329 | 12 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/milestone_rollup |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.566 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/milestone_rollup |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.166 | 9 | 1 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/milestone_rollup |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 94 | yes | yes | 17.616 | 25 | 1 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/milestone_rollup |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 26.012 | 19 | 6 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/milestone_rollup |

### nested_app_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.828 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/nested_app |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 45 | no | no | 2.201 | 60 | 0 | 20 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.430 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/nested_app |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 45 | no | no | 3.752 | 60 | 0 | 20 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/nested_app |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 45 | no | no | 3.384 | 60 | 0 | 18 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/nested_app |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.596 | 15 | 4 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/nested_app |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 16.823 | 13 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/nested_app |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 24.990 | 13 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/nested_app |

### owner_recap_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 95 | yes | yes | 1.957 | 12 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/owner_recap |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 13 | no | no | 2.291 | 60 | 58 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 95 | yes | yes | 2.387 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/owner_recap |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 79 | no | no | 3.548 | 19 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/owner_recap |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 95 | yes | yes | 3.433 | 18 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/owner_recap |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 13.073 | 13 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/owner_recap |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 94 | yes | yes | 17.159 | 11 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/owner_recap |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 24.459 | 11 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/owner_recap |

### owner_sort_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.948 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/owner_sort |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 2.207 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/owner_sort |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.578 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/owner_sort |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.014 | 9 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/owner_sort |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.749 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/owner_sort |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.860 | 8 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/owner_sort |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 17.711 | 9 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/owner_sort |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 25.710 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/owner_sort |

### priority_snapshot_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.843 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/priority_snapshot |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 98 | yes | yes | 2.264 | 11 | 9 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/priority_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.589 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/priority_snapshot |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.201 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/priority_snapshot |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.548 | 6 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/priority_snapshot |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.920 | 13 | 2 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/priority_snapshot |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 17.366 | 14 | 1 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/priority_snapshot |
| single::openai/openai/gpt-oss-120b | single | 97 | yes | yes | 22.616 | 24 | 6 | 6 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/priority_snapshot |

### renewal_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 95 | yes | yes | 1.729 | 16 | 7 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/renewal_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 80 | no | no | 2.241 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/renewal_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 77 | no | no | 2.359 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/renewal_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 79 | no | no | 3.341 | 11 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/renewal_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 77 | no | no | 3.562 | 10 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/renewal_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 79 | no | no | 11.355 | 13 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/renewal_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 79 | no | no | 15.513 | 16 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/renewal_preview |
| single::openai/openai/gpt-oss-120b | single | 79 | no | no | 24.839 | 10 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/renewal_preview |

### risk_score_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.957 | 7 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/risk_score |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 2.136 | 21 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/risk_score |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.572 | 7 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/risk_score |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 2.898 | 11 | 0 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/risk_score |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.706 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/risk_score |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.148 | 6 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/risk_score |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 17.246 | 7 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/risk_score |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 25.995 | 8 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/risk_score |

### shipment_preview_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 60 | no | no | 1.879 | 60 | 0 | 10 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/shipment_preview |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 58 | no | no | 2.372 | 8 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 80 | no | no | 2.719 | 28 | 0 | 5 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/shipment_preview |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 79 | no | no | 3.652 | 35 | 1 | 6 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/shipment_preview |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 59 | no | no | 2.823 | 40 | 30 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/shipment_preview |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 21 | no | no | 11.330 | 60 | 36 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/shipment_preview |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 78 | no | no | 15.689 | 47 | 6 | 7 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/shipment_preview |
| single::openai/openai/gpt-oss-120b | single | 60 | no | no | 20.089 | 24 | 8 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/shipment_preview |

### simple_mean_bug_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 2.185 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/simple_mean_bug |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 2.494 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 2.698 | 7 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/simple_mean_bug |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 3.160 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/simple_mean_bug |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.702 | 5 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/simple_mean_bug |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 99 | yes | yes | 11.343 | 7 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/simple_mean_bug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 17.188 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/simple_mean_bug |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 26.140 | 8 | 1 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/simple_mean_bug |

### status_slug_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 100 | yes | yes | 1.922 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/status_slug |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 96 | yes | yes | 2.321 | 8 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/status_slug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.548 | 8 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/status_slug |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 100 | yes | yes | 2.975 | 7 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/status_slug |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 100 | yes | yes | 3.712 | 6 | 0 | 0 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/status_slug |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 12.270 | 7 | 1 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/status_slug |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 17.863 | 8 | 2 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/status_slug |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 25.021 | 9 | 2 | 1 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/status_slug |

### team_digest_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.923 | 10 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/team_digest |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 97 | yes | yes | 2.203 | 13 | 13 | 6 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/team_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 99 | yes | yes | 2.656 | 9 | 0 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/team_digest |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.448 | 11 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/team_digest |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.616 | 25 | 0 | 8 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/team_digest |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.583 | 13 | 3 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/team_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 17.377 | 9 | 1 | 3 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/team_digest |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 25.224 | 21 | 7 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/team_digest |

### workspace_digest_001
| Config | Architecture | Score | Strict Success | Observed Success | Rel Cost | Turns | Parse Err | Tool Err | Run Dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_coder::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 99 | yes | yes | 1.807 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder/workspace_digest |
| planner_coder::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder | 94 | yes | yes | 2.294 | 60 | 0 | 25 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/openai/gpt-oss-120b | planner_coder_reviewer | 95 | yes | yes | 2.481 | 17 | 0 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_planner_umich_qwen_coder/planner_coder_reviewer/workspace_digest |
| planner_coder_reviewer::openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct->openai/Qwen/Qwen3-VL-30B-A3B-Instruct | planner_coder_reviewer | 99 | yes | yes | 3.476 | 19 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/planner_coder_reviewer/workspace_digest |
| single::openai/Qwen/Qwen3-VL-30B-A3B-Instruct | single | 99 | yes | yes | 3.602 | 10 | 0 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_qwen/single/workspace_digest |
| planner_coder::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder | 98 | yes | yes | 11.939 | 14 | 4 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder/workspace_digest |
| planner_coder_reviewer::openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b->openai/openai/gpt-oss-120b | planner_coder_reviewer | 98 | yes | yes | 18.474 | 19 | 4 | 4 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/planner_coder_reviewer/workspace_digest |
| single::openai/openai/gpt-oss-120b | single | 98 | yes | yes | 26.216 | 12 | 3 | 2 | SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/umich_gptoss_120b/single/workspace_digest |
