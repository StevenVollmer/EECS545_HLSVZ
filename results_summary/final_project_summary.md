# EECS545 Project Summary: Multi-Role Agent Architectures, Performance, and Next Analyses

## 1) Architectures attempted in this project

### A. Custom linear/role-loop runner (`SWE-agent/scripts/custom/`)

- **Core runner**: `run_custom_swebench.py` (standalone, bypasses core SWE-agent class stack).
- **Architectures supported**:
  - `single` (coder only),
  - `planner_coder`,
  - `planner_coder_reviewer`.
- **Design choice**: planner/reviewer are one-shot JSON handoff stages around an iterative coder loop (low orchestration overhead, clean ablations).
- **Strength**: controlled comparison of role value and model assignment with low infrastructure complexity.

### B. Tree-search agent development (`SWE-agent/scripts/tree_search_custom/`)

- **Core idea**: MCTS-like search over tool-action trajectories (`run_tree_search.py`, launched via `run_mcts.py` presets).
- **Search mechanics**: UCB1 selection, branch expansion with stochastic candidate actions, loop-state/value signals, git-state backtracking per branch.
- **Development path** (ablation scripts):
  - Phase-2 and Phase-3 focused on **thinking mode**, **failure surfacing**, and robustness on hard subsets.
  - Separate scripts for run comparison, regression checks, and branch visualization.
- **Role integration**: tree search is combined with planner/coder/reviewer role assignment in several presets.

### C. Combined (SWE-search-inspired) agent (`SWE-agent/scripts/combined/`)

- **Core runner**: `run_combined.py` (extends tree-search runner).
- **SWE-search-inspired additions**:
  1. LLM node value function,
  2. hindsight feedback from failed branches,
  3. mean trajectory reward for final branch selection.
- **Mixed-size role assignment**:
  - planner/reviewer on larger reasoning model(s),
  - coder/value function on smaller/cheaper model(s).
- **Ablation orchestration**: `run_combined_ablation.py` (variants C–K on c1/c2; c3 held out for final checks).

### D. Combined variant families (from ablations)

- **Baselines**: A (9b MCTS), B (linear split model), K (minimal swe-search).
- **Our mixed-role search variants**: C, D, E.
- **9b technique variants**: F, G, H, I.
- **30b swe-search variant**: J.

## 2) Custom cases summary and why they exist

The project intentionally uses `custom_cases`, `custom_cases_2`, and `custom_cases_3` because direct SWE-bench work was too difficult and not very illuminating for iterative small-model agent development.

- **Set sizes**: 20 cases each in c1/c2/c3 (60 total).
- **Difficulty mix**:
  - c1: tiered easy/medium/hard set with explicit showcase tags (planner-favored, reviewer-favored, coder/basic).
  - c2: 7 easy / 7 medium / 6 hard.
  - c3: 7 easy / 7 medium / 6 hard, used as held-out final set.
- **Why this helps**:
  - deterministic, fast fixtures,
  - controlled architecture ablations,
  - stronger signal for planner/reviewer contributions,
  - easier regression and failure-mode analysis than opaque SWE-bench failures.

## 3) Performance summary (ablative + final)

Primary sources used:
- `combined_results/results.csv`
- `combined_results/tables.tex`
- `combined_results/figures/fig1..fig4`

### Headline solve-rate results (current combined evaluator outputs)

- **A (9b MCTS)**: c1 75%, c2 60%, c3 70%.
- **B (linear split)**: c1 90%, c2 60%, c3 80%.
- **C (mixed MCTS)**: c1 85%, c2 70%, c3 75%.

### Important caveat on B(c1)

- Current `eval_combined.py` now falls back to trajectory `submitted` status for B(c1) because strict patch replay against evolving fixture repos is unreliable.
- This can inflate B(c1) relative to strict success-check evaluation.
- Paper text should explicitly mark B(c1) as **proxy-evaluated unless strict rejudge is rerun from frozen fixtures/commits**.

### Ablation trends (c1+c2 average solve-rate)

- Top group: **C and G** at 77.5%.
- Strong middle: B (75%), E and I (72.5%), D/H (70%).
- Lower: A/J (67.5%), F (65%), K (55%).

### Efficiency vs accuracy (current estimated USD metric)

- New Pareto frontier export (`combined_results/pareto_frontier.csv`) identifies **B** and **G** as non-dominated under current metrics.
- **G (9b hindsight)** is the strongest low-cost/high-accuracy frontier point among search-heavy variants.

## 4) Analysis-tool improvements completed in this pass

### Updated evaluator (`SWE-agent/scripts/combined/eval_combined.py`)

Added exports to support paper analysis:
- `--instance-csv` → per-instance metrics (`combined_results/instance_metrics.csv`)
- `--efficiency-csv` → run-level efficiency/accuracy table (`combined_results/efficiency_accuracy.csv`)
- `--frontier-csv` → c1+c2 Pareto table (`combined_results/pareto_frontier.csv`)

Also updated reporting/tables to use **estimated USD cost** consistently.

### Regenerated artifacts

- `combined_results/results.csv`
- `combined_results/tables.tex`
- `combined_results/instance_metrics.csv`
- `combined_results/efficiency_accuracy.csv`
- `combined_results/pareto_frontier.csv`
- `combined_results/figures/fig1_main_results.png`
- `combined_results/figures/fig2_ablation.png`
- `combined_results/figures/fig3_efficiency.png`
- `combined_results/figures/fig4_token_breakdown.png`

## 5) Key findings so far

1. **Role decomposition is useful**, but not all additions help equally on every set.
2. **Search + hindsight can be a strong efficiency play** (G on frontier).
3. **Value-function additions are mixed**: can help in some settings, but not uniformly better than simpler variants.
4. **Evaluation protocol quality is itself a major risk**: strict-vs-proxy scoring can shift conclusions (notably B(c1)).
5. **Custom cases successfully expose planner/reviewer effects** in ways SWE-bench did not for early small-model iteration.

## 6) Recommended final tests/analyses before paper lock

1. **Freeze benchmark fixtures by commit/hash** for c1/c2/c3 and rerun strict patch judging for all headline variants.
2. **Repeat-run variance**: run top variants (at least A/B/C/G/E) with 3 seeds per set; report confidence intervals.
3. **Reviewer value audit**: compare accepted vs revised branches and false-accept/false-reject rates (build on `reviewer_audits`).
4. **Case-stratified performance**: report easy/medium/hard and planner-favored vs reviewer-favored splits.
5. **Cost sensitivity analysis**: re-rank methods under alternate pricing assumptions (local cluster vs API-equivalent).
6. **Generalization check**: keep c3 untouched until final model/parameter selection is fixed; then run exactly once for final claim.

## 7) High-value findings these tests could reveal

- Whether C’s gains are robust or mainly set-specific.
- Whether G remains Pareto-optimal under strict/frozen judging.
- How much of the apparent linear baseline strength is true correctness vs submission proxy effects.
- Whether planner gains and reviewer gains are additive or redundant by case type.
- Whether value-function overhead is justified after variance is accounted for.
