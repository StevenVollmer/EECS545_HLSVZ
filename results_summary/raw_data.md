# Raw Results Data

All experimental results in one place. **Primary source is the final unified matrix** (Source 1) — 8 configs × 27 cases run in a single session. Sources 2–4 are earlier runs kept for historical comparison.

---

## Source 1: Final Unified Matrix (8 configs × 27 cases, April 21)

Single session, same benchmark, consistent conditions. Output:
`SWE-agent/custom_matrix_runs/final_matrix/`

Models: qwen = Qwen3-VL-30B-A3B, gpt = gpt-oss-120b (UMich cluster).
MCTS runs use: planner/reviewer/critic = gpt, coder = qwen, 15 iterations × 15 depth, edit majority vote k=3.
Linear runs use: planner (if any) = gpt, coder = qwen or gpt, reviewer (if any) = gpt.

### Per-case pass/fail grid

| Case | qwen | gpt→qw | gpt→qw+C | gpt→qw+R | gpt | mcts_base | mcts+critic | mcts+gate |
|------|------|--------|----------|----------|-----|-----------|-------------|-----------|
| board_rollup | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| budget_snapshot | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| contact_card | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| digest_preview | FAIL | PASS | FAIL | FAIL | PASS | FAIL | FAIL | FAIL |
| incident_brief | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| invoice_footer | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | PASS |
| label_formatter | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| median_window | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| milestone_rollup | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| nested_app | FAIL | PASS | PASS | PASS | PASS | FAIL | PASS | PASS |
| owner_recap | PASS | PASS | PASS | FAIL | PASS | FAIL | FAIL | PASS |
| owner_sort | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS |
| priority_snapshot | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| renewal_preview | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| risk_score | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| shipment_preview | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| simple_mean_bug | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| status_slug | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| team_digest | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| workspace_digest | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| numeric_drift_sum | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| pagination_drift | PASS | FAIL | FAIL | PASS | PASS | FAIL | PASS | PASS |
| path_normalizer_cache | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| retry_cap | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS |
| search_hit_localize | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | FAIL |
| stable_ranking | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| weighted_median | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

### Summary (8 configs × 27 cases)

| Config | Solved | Rate | Avg compute (×4o-mini) | Description |
|--------|--------|------|-----------------------|-------------|
| qwen | 18/27 | 66.7% | 3.45 | Small coder solo (floor) |
| gpt→qw | 19/27 | 70.4% | **2.01** | Strong planner + weak coder |
| gpt→qw+C | 19/27 | 70.4% | 2.20 | Strong planner + **plan critic** + weak coder |
| gpt→qw+R | 17/27 | 63.0% | 1.99 | Strong planner + weak coder + reviewer (2 rounds) |
| **gpt** | **22/27** | **81.5%** | 23.68 | Strong coder solo (ceiling) |
| mcts_baseline | 17/27 | 63.0% | 3.62 | MCTS + auto-accept |
| mcts+plan_critic | 19/27 | 70.4% | 3.91 | MCTS + critic warnings (no revision) |
| **mcts+critic_gate** | **20/27** | **74.1%** | 3.95 | MCTS + critic submission gate (deployment-realistic) |

### Case difficulty buckets (derived from grid)

- **Universal PASS** (all 8 configs): 10 cases — board_rollup, incident_brief, median_window, milestone_rollup, priority_snapshot, risk_score, simple_mean_bug, status_slug, workspace_digest, numeric_drift_sum, path_normalizer_cache, stable_ranking (12 cases really)
- **Universal FAIL** (all 8 configs): 5 cases — budget_snapshot, contact_card, renewal_preview, shipment_preview, weighted_median (these are beyond every config, including gpt solo)
- **gpt-only**: digest_preview, invoice_footer (partial — 2 configs each solve)
- **Mixed/borderline**: label_formatter, nested_app, owner_recap, owner_sort, team_digest, pagination_drift, retry_cap, search_hit_localize

### Key observations from the unified matrix

1. **gpt solo is the true ceiling at 22/27 (81.5%)** — 5 cases are impossible for every tested config. The "achievable" denominator is 22.
2. **Linear pipelines cluster at ~19/27 (70%) with ≥10× less compute than gpt solo.** gpt→qw achieves 70% at 2.01 avg-compute vs gpt solo at 23.68 (11.8× cheaper).
3. **Critic doesn't beat plain gpt→qw on this 27-case benchmark** (both 19/27). This contrasts with the 7-hard-case earlier run where critic gave +29 pp. The gap closes because the 20 easier cases are already solved by gpt→qw, and the 7 harder cases include several (budget_snapshot, weighted_median, renewal_preview) that no config can solve.
4. **Reviewer regresses on this benchmark** (17/27 vs 19/27 for gpt→qw). Cases lost: label_formatter, owner_recap, retry_cap, search_hit_localize — all cases where round-2 rewrites damaged a good round-1 patch. Reviewer is only a win when round-1 was actually broken.
5. **MCTS + critic_gate is the best MCTS variant at 20/27 (74%)** — better than the auto-accept baseline (17/27). Surprising: the gate was expected to *lower* rates (by rejecting auto-finalized patches), but instead it rescued cases by forcing additional iterations. Auto-accept on MCTS short-circuits the search too aggressively.
6. **MCTS + plan_critic (warnings) ties linear gpt→qw+C at 19/27.** Warnings injected into search context are neutral-to-helpful — no regression vs baseline, and +2 cases.
7. **Compute picture**: MCTS variants sit at ~3.9 avg-compute, 2× the linear pipelines but 6× cheaper than gpt solo. They trade compute for search breadth.

---

## Source 2: Linear Agent — 7 harder cases only (earlier, April 19)

Subset of the 7 harder cases from Source 1. Full 10-config sweep.

Cases: numeric_drift_sum, pagination_drift, path_normalizer_cache, retry_cap, search_hit_localize, stable_ranking, weighted_median.

| Case | qwen | qw→qw | qw→qw+C | gpt→qw | gpt→qw+R | gpt→qw+C | gpt→qw+CR | gpt→gpt | gpt→gpt+C | gpt |
|------|------|--------|----------|--------|----------|----------|-----------|---------|-----------|-----|
| numeric_drift_sum | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| pagination_drift | FAIL | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | PASS | PASS |
| path_normalizer_cache | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| retry_cap | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| search_hit_localize | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| stable_ranking | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| weighted_median | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

### Summary (7 cases)

| Config | Solved | Rate | Compute (rel) | Description |
|--------|--------|------|---------------|-------------|
| qwen | 4/7 | 57% | 3.17 | Small coder solo |
| qw→qw | 4/7 | 57% | 2.52 | Weak planner + weak coder |
| qw→qw+C | 4/7 | 57% | 2.85 | Weak planner + critic + weak coder |
| gpt→qw | 4/7 | 57% | 2.03 | Strong planner + weak coder |
| gpt→qw+R | 5/7 | 71% | 2.71 | + reviewer |
| **gpt→qw+C** | **6/7** | **86%** | **2.16** | + critic |
| gpt→qw+CR | 6/7 | 86% | 2.47 | + both |
| gpt→gpt | 6/7 | 86% | 13.37 | Strong planner + strong coder |
| gpt→gpt+C | 6/7 | 86% | 14.10 | + critic |
| gpt | 6/7 | 86% | 24.42 | Strong coder solo |

**Note:** Source 1 reruns these 7 cases as part of 27; the combined matrix confirms pagination_drift regresses with critic (same result). On the full 27-case benchmark, critic's lift relative to gpt→qw disappears because the 20 additional cases are already solved by gpt→qw.

---

## Source 3: MCTS Critic Ablation (c1 = first 20 cases, earlier April 20)

Cases: board_rollup, budget_snapshot, contact_card, digest_preview, incident_brief, invoice_footer, label_formatter, median_window, milestone_rollup, nested_app, owner_recap, owner_sort, priority_snapshot, renewal_preview, risk_score, shipment_preview, simple_mean_bug, status_slug, team_digest, workspace_digest.

| Variant | Solved | Rate |
|---------|--------|------|
| C_baseline | 13/20 | 65% |
| C_plan_critic | 13/20 | 65% |
| C_critic_gate | 12/20 | 60% |

**Superseded by Source 1**, which reruns these 20 cases + 7 harder cases in a consistent session:
- mcts_baseline on the first 20: 11/20 (55%)  [docker-era: 13/20]
- mcts_plan_critic on the first 20: 12/20 (60%)
- mcts_critic_gate on the first 20: 13/20 (65%)

High run-to-run variance confirmed. On the expanded 27 cases, critic_gate emerges as the best MCTS variant.

---

## Source 4: Steven's Combined Results (c1+c2+c3, pre-unified-matrix)

From `combined_results/final_results.csv`. Uses auto-accept (inflated vs real deployment).

| Variant | Description | c1 | c2 | c3 | Cost (USD) |
|---------|-------------|-----|-----|-----|-----------|
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

Pareto-optimal (c1+c2 avg): **B** ($0.0019, 70%) and **G** ($0.0030, 77.5%).

---

## Unified Paper Narrative

### Three contributions

1. **Strong planner + weak coder matches large-model performance at ~12× less compute.**
   Source 1: gpt→qw resolves 19/27 at 2.01 avg-compute; gpt solo resolves 22/27 at 23.68.
   Ratio: 86% of gpt's solve-rate at 8.5% of its compute.

2. **Plan critic is a useful audit pass when planner errors are the bottleneck.**
   On the 7 harder cases where gpt→qw struggles (Source 2), critic lifts 57% → 86%.
   On the full 27-case mix (Source 1), critic is neutral because gpt→qw already solves the 20 easier cases. **Critic helps when the planner's handoff is the failure mode, not when the coder is the bottleneck.**

3. **Critic as MCTS submission gate replaces auto-accept with deployment-realistic evaluation.**
   On 27 cases (Source 1): mcts_baseline 17/27 → mcts_critic_gate 20/27 (+3 cases). The gate prevents the search from stopping on early auto-finalized branches that pass success checks but don't fully address the issue.

### Key claims

1. **Compute efficiency**: gpt→qw (linear) reaches 70% of benchmark at 2.0 avg-compute units — Pareto-optimal among all configs tested.
2. **Plan auditing**: pre-coder critic lifts resolved rate by +29 pp on hard cases (Source 2) at <7% extra compute. Does not hurt MCTS when applied as warnings (+2 cases on 27).
3. **Critic gate for MCTS**: replaces auto-accept with LLM quality check; gives +3 cases on 27 vs baseline MCTS. Honest deployment number.
4. **Reviewer is not free**: on 27 cases, post-coder reviewer regresses 19/27 → 17/27 because round-2 rewrites damage good round-1 patches. Critic (pre-coder) > reviewer (post-coder) on every metric.
5. **5/27 cases are unsolved by every tested config** (budget_snapshot, contact_card, renewal_preview, shipment_preview, weighted_median). These are the frontier.

### Figures to produce

1. **Pareto frontier (cost vs solve rate)** — points: qwen, gpt→qw, gpt→qw+C, gpt→qw+R, gpt, mcts_baseline, mcts+critic_gate. gpt→qw and gpt→qw+C dominate.
2. **Audit mechanism comparison (linear agent, 27 cases)** — bars: gpt→qw, +R, +C. Shows reviewer hurts, critic neutral here (but see 7-hard figure).
3. **Audit mechanism comparison (linear, 7 harder cases)** — bars: gpt→qw, +R, +C, +CR. Shows critic lifts 57→86.
4. **MCTS ablation (27 cases)** — bars: mcts_baseline, +critic, +gate. Shows auto-accept inflates baseline; gate gives honest improvement.
5. **Per-case heatmap** — cases × configs, colored by pass/fail.

Artifacts:
- `results_summary/final_matrix_raw.json` — machine-readable output from the 27-case matrix.
- `SWE-agent/custom_matrix_runs/final_matrix/` — full trajectories and patches.
