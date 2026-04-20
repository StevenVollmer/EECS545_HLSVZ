# Raw Results Data

All experimental results in one place for the meeting. Three data sources.

---

## Source 1: Linear Agent (7 harder cases, April 19)

Cases: numeric_drift_sum, pagination_drift, path_normalizer_cache, retry_cap, search_hit_localize, stable_ranking, weighted_median

Models: qwen = Qwen3-VL-30B, gpt = gpt-oss-120b

| Case | qwen | qw→qw | qw→qw+C | gpt→qw | gpt→qw+R | gpt→qw+C | gpt→qw+CR | gpt→gpt | gpt→gpt+C | gpt |
|------|------|--------|----------|--------|----------|----------|-----------|---------|-----------|-----|
| numeric_drift_sum | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| pagination_drift | FAIL | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | PASS | PASS |
| path_normalizer_cache | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| retry_cap | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| search_hit_localize | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| stable_ranking | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| weighted_median | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

### Summary

| Config | Solved | Rate | Compute (relative) | Description |
|--------|--------|------|-------------------|-------------|
| qwen | 4/7 | 57% | 3.17 | Small coder solo |
| qw→qw | 4/7 | 57% | 2.52 | Weak planner + weak coder |
| qw→qw+C | 4/7 | 57% | 2.85 | Weak planner + critic + weak coder |
| gpt→qw | 4/7 | 57% | 2.03 | Strong planner + weak coder |
| gpt→qw+R | 5/7 | 71% | 2.71 | Strong planner + weak coder + reviewer |
| **gpt→qw+C** | **6/7** | **86%** | **2.16** | **Strong planner + critic + weak coder** |
| gpt→qw+CR | 6/7 | 86% | 2.47 | Strong planner + critic + weak coder + reviewer |
| gpt→gpt | 6/7 | 86% | 13.37 | Strong planner + strong coder |
| gpt→gpt+C | 6/7 | 86% | 14.10 | Strong planner + critic + strong coder |
| gpt | 6/7 | 86% | 24.42 | Strong coder solo |

### Key observations
- Critic lifts gpt→qw from 57% to 86% (+29 pp) at +6% compute
- Critic matches gpt solo ceiling at 11× less compute
- Reviewer adds +14 pp but costs +33% more compute than critic
- Critic on weak planner = no effect (garbage in, garbage out)
- Critic on strong coder = no effect (already at ceiling)

---

## Source 2: MCTS Critic Ablation (c1, 20 cases, April 20)

Cases: board_rollup, budget_snapshot, contact_card, digest_preview, incident_brief, invoice_footer, label_formatter, median_window, milestone_rollup, nested_app, owner_recap, owner_sort, priority_snapshot, renewal_preview, risk_score, shipment_preview, simple_mean_bug, status_slug, team_digest, workspace_digest

Models: coder = Qwen3-VL-30B, planner/reviewer/critic = gpt-oss-120b

| Case | C_baseline | C_plan_critic | C_critic_gate |
|------|-----------|---------------|---------------|
| board_rollup | PASS | PASS | PASS |
| budget_snapshot | FAIL | FAIL | FAIL |
| contact_card | FAIL | FAIL | FAIL |
| digest_preview | FAIL (error) | PASS | FAIL |
| incident_brief | FAIL (error) | PASS | PASS |
| invoice_footer | FAIL (error) | FAIL | FAIL |
| label_formatter | PASS | PASS | PASS |
| median_window | PASS | PASS | PASS |
| milestone_rollup | PASS | PASS | PASS |
| nested_app | PASS | FAIL | FAIL |
| owner_recap | PASS | FAIL | PASS |
| owner_sort | PASS | PASS | PASS |
| priority_snapshot | PASS | PASS | PASS |
| renewal_preview | FAIL | FAIL | FAIL |
| risk_score | PASS | PASS | PASS |
| shipment_preview | FAIL | FAIL | FAIL |
| simple_mean_bug | PASS | PASS | PASS |
| status_slug | PASS | PASS | PASS |
| team_digest | PASS | PASS | PASS |
| workspace_digest | PASS | PASS | FAIL |

### Summary

| Variant | Solved | Rate | Description |
|---------|--------|------|-------------|
| C_baseline | 13/20 | 65% | MCTS + auto-accept |
| C_plan_critic | 13/20 | 65% | MCTS + critic warnings (no revision) |
| C_critic_gate | 12/20 | 60% | MCTS + critic submission gate (realistic) |

Note: C_baseline had 3 docker errors (digest_preview, incident_brief, invoice_footer). True baseline likely 14-16/20 without infra issues. C_both failed due to docker exhaustion (discard).

### Key observations
- Plan critic as warnings = neutral (does not hurt MCTS, unlike revision which dropped to 50%)
- Critic gate = realistic deployment number (-5% from auto-accept)
- Run-to-run variance is high: same config produced 65-85% across different runs

---

## Source 3: Steven's Combined Results (c1+c2+c3, from combined_results/)

From `combined_results/final_results.csv`. Cases: custom_cases (c1), custom_cases_2 (c2), custom_cases_3 (c3), 20 each.

| Variant | Description | c1 | c2 | c3 | Cost (USD) |
|---------|-------------|----|----|----|----|
| A | 9b MCTS | 75% | 60% | 70% | $0.0034 |
| B | Linear (120b plan + 30b code) | 80% | 60% | 80% | $0.0020 |
| C | Mixed MCTS (120b plan + 30b code) | 85% | 70% | 75% | $0.0040 |
| D | C + LLM value function | 70% | 70% | — | $0.0040 |
| E | C + value fn + hindsight | 85% | 60% | — | $0.0034 |
| F | 9b code + 120b plan, no search | 60% | 70% | — | $0.0041 |
| G | 9b MCTS + hindsight feedback | 75% | 80% | — | $0.0030 |
| H | 9b MCTS + self-eval value fn | 70% | 70% | — | $0.0039 |
| I | 9b full swe-search replica | 75% | 70% | — | $0.0038 |
| J | 30b flat + full swe-search | 75% | 60% | — | $0.0042 |
| K | Bare UCB1 (minimal swe-search) | 55% | 55% | — | $0.0027 |

Pareto-optimal (c1+c2 avg): **B** (70%, $0.0019) and **G** (77.5%, $0.0030)

### Key observations
- Variant B (our linear agent) is most cost-efficient
- Variant C (mixed MCTS) gets highest c1 accuracy (85%)
- MCTS helps small models (A=75% vs K=55% on same 9b model)
- Hindsight feedback helps 9b models (G=77.5% avg vs A=67.5% avg)
- LLM value function alone doesn't help (D ≤ C)
- All variants use auto-accept (inflated vs real deployment)

---

## Unified View: Best Configs Across All Sources

| Config | Source | Solve Rate | Cost/Compute | Realistic? | Novel Contribution |
|--------|--------|-----------|--------------|------------|-------------------|
| B (linear) | Steven | 80% c1 | $0.0020 | yes | Planner effect |
| **B + critic** | Rafe | **86% (7-case)** | **~$0.0022** | **yes** | **Plan auditing** |
| C (mixed MCTS) | Steven | 85% c1 | $0.0040 | no (auto-accept) | MCTS + mixed models |
| C + critic gate | Rafe | 60% c1 | ~$0.0042 | yes | Honest MCTS numbers |
| G (9b hindsight) | Steven | 77.5% avg | $0.0030 | partial | Hindsight feedback |
| gpt solo | Rafe | 86% (7-case) | 24.42 relative | yes | Ceiling |

---

## What To Present Tomorrow

### Figure 1: Pareto frontier (cost vs solve rate)
Points: B, B+critic, C, C+gate, G, A, K
x-axis: cost (USD or relative compute)
y-axis: solve rate (%)
Highlight: B+critic dominates — near-C accuracy at B's cost

### Figure 2: Audit mechanism comparison (linear agent)
Bar chart: gpt→qw, gpt→qw+R, gpt→qw+C, gpt→qw+CR
Shows: critic > reviewer, stacking doesn't help

### Figure 3: Auto-accept impact (MCTS)
Bar chart: C_baseline (auto-accept), C_critic_gate (realistic)
Shows: 5-20% inflation from auto-accept

### Table 1: Full ablation
All variants from Steven + our critic additions

### Table 2: Per-case heatmap
Cases × configs, colored by pass/fail
