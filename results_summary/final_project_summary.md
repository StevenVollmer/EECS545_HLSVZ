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

### D. Combined variant glossary (A through K, explicit)

- **A — 9b MCTS baseline (ours):** 9b planner/coder/reviewer with MCTS and our custom controls; no hindsight/value-fn extras.
- **B — Rafe linear split baseline:** 120b planner/reviewer + 30b coder, effectively linearized search behavior.
- **C — Mixed-size MCTS (ours):** 120b planner/reviewer + 30b coder with our MCTS configuration.
- **D — C + 30b value function:** mixed-size MCTS plus LLM value scoring.
- **E — D + hindsight feedback:** mixed-size MCTS + value scoring + dead-branch hindsight hints.
- **F — 9b coder + 120b planner/reviewer:** isolates planner/reviewer strength with a smaller coder.
- **G — 9b + hindsight:** all-9b run with cross-branch hindsight feedback.
- **H — 9b + self-eval value fn:** all-9b run with 9b value function (no hindsight).
- **I — 9b full swe-search style:** 9b value function + hindsight (swe-search-style package on 9b).
- **J — 30b full swe-search style:** flat 30b role assignment + value fn + hindsight.
- **K — minimal swe-search baseline:** stripped-down single-agent/UCB1-style baseline with swe-search-style signals but without our custom orchestration improvements.

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

Primary sources used (latest run set):
- `combined_results/final_results.csv`
- `combined_results/final_tables.tex`
- `combined_results/final_efficiency_accuracy.csv`
- `combined_results/final_pareto_frontier.csv`
- `combined_results/figures/`

### Headline solve-rate results (final-labeled outputs)

- **A (9b MCTS)**: c1 75%, c2 60%, c3 70%.
- **B (linear split)**: c1 80%, c2 60%, c3 80%.
- **C (mixed MCTS)**: c1 85%, c2 70%, c3 75%.

### Final ablation trends (c1+c2 average solve-rate)

- Top: **C and G** at 77.5%.
- Next: **E and I** at 72.5%.
- Middle: **B, D, H** at 70%.
- Lower: **A and J** at 67.5%, **F** at 65%, **K** at 55%.

### Efficiency vs accuracy (final estimated USD metric)

- `combined_results/final_pareto_frontier.csv` identifies **B** and **G** as non-dominated under current metrics.
- **G (9b hindsight)** is the strongest low-cost/high-accuracy frontier point among search-heavy variants.

## 4) Analysis-tool improvements completed in this pass

### Updated evaluator (`SWE-agent/scripts/combined/eval_combined.py`)

Added exports to support paper analysis:
- `--instance-csv` → per-instance metrics (`combined_results/final_instance_metrics.csv`)
- `--efficiency-csv` → run-level efficiency/accuracy table (`combined_results/final_efficiency_accuracy.csv`)
- `--frontier-csv` → c1+c2 Pareto table (`combined_results/final_pareto_frontier.csv`)

Also updated reporting/tables to use **estimated USD cost** consistently.

### Regenerated / analyzed artifacts (final set)

- `combined_results/final_results.csv`
- `combined_results/final_tables.tex`
- `combined_results/final_instance_metrics.csv`
- `combined_results/final_efficiency_accuracy.csv`
- `combined_results/final_pareto_frontier.csv`
- `combined_results/figures/fig1_main_results.png`
- `combined_results/figures/fig2_ablation.png`
- `combined_results/figures/fig3_efficiency.png`
- `combined_results/figures/fig4_token_breakdown.png`
- `combined_results/figures/fig5_set_heatmap.png`

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

## 8) CLI runbook (c1/c2, c3, and analysis)

### Prereqs

- Run from repo root.
- Keep **exactly one local-GPU terminal** for local 9b runs.
- Remote-only runs can execute in parallel terminals.

### A) c1/c2 runs (recommended minimum: A/B/C/E/G)

#### Seed 1

```bash
# Terminal L (local only)
python SWE-agent/scripts/combined/run_combined_ablation.py --execute --resume --group local --only A_c1,A_c2,G_c1,G_c2 --output-root SWE-agent/tree_search_runs/combined_seed1
```

```bash
# Terminal R1 (remote)
python SWE-agent/scripts/combined/run_combined_ablation.py --execute --resume --group remote --only B_c1,B_c2 --output-root SWE-agent/tree_search_runs/combined_seed1
```

```bash
# Terminal R2 (remote)
python SWE-agent/scripts/combined/run_combined_ablation.py --execute --resume --group remote --only C_c1,C_c2 --output-root SWE-agent/tree_search_runs/combined_seed1
```

```bash
# Terminal R3 (remote)
python SWE-agent/scripts/combined/run_combined_ablation.py --execute --resume --group remote --only E_c1,E_c2 --output-root SWE-agent/tree_search_runs/combined_seed1
```

Repeat the same four commands for seeds 2 and 3 by replacing:

- `combined_seed1` → `combined_seed2`
- `combined_seed1` → `combined_seed3`

### B) c3 final evaluation (run after architecture/model freeze)

#### Seed 1

```bash
# Terminal L (local only)
python SWE-agent/scripts/combined/run_combined_ablation.py --execute --resume --group local --only A_c3 --output-root SWE-agent/tree_search_runs/combined_seed1_c3
```

```bash
# Terminal R1 (remote)
python SWE-agent/scripts/combined/run_combined_ablation.py --execute --resume --group remote --only B_c3 --output-root SWE-agent/tree_search_runs/combined_seed1_c3
```

```bash
# Terminal R2 (remote)
python SWE-agent/scripts/combined/run_combined_ablation.py --execute --resume --group remote --only C_c3 --output-root SWE-agent/tree_search_runs/combined_seed1_c3
```

Repeat for additional seeds with `combined_seed2_c3`, `combined_seed3_c3` if needed.

### C) Full analysis scripts (in place) and commands

Scripts:

- `SWE-agent/scripts/combined/eval_combined.py`
- `SWE-agent/scripts/combined/plot_results.py`
- `SWE-agent/scripts/combined/audit_reviewer.py`
- `SWE-agent/scripts/combined/eval_patches.py`

Per-seed evaluation + artifact generation:

```bash
python SWE-agent/scripts/combined/eval_combined.py \
  --combined-root SWE-agent/tree_search_runs/combined_seed1 \
  --csv combined_results/seed1_results.csv \
  --instance-csv combined_results/seed1_instance_metrics.csv \
  --efficiency-csv combined_results/seed1_efficiency_accuracy.csv \
  --frontier-csv combined_results/seed1_pareto_frontier.csv \
  --latex combined_results/seed1_tables.tex
```

```bash
python SWE-agent/scripts/combined/plot_results.py \
  --combined-root SWE-agent/tree_search_runs/combined_seed1 \
  --output-dir combined_results/figures_seed1 \
  --format png
```

To process all seeds quickly:

```bash
for s in 1 2 3; do
  python SWE-agent/scripts/combined/eval_combined.py \
    --combined-root SWE-agent/tree_search_runs/combined_seed${s} \
    --csv combined_results/seed${s}_results.csv \
    --instance-csv combined_results/seed${s}_instance_metrics.csv \
    --efficiency-csv combined_results/seed${s}_efficiency_accuracy.csv \
    --frontier-csv combined_results/seed${s}_pareto_frontier.csv \
    --latex combined_results/seed${s}_tables.tex
done
```

Generate publication figures directly from finalized CSV outputs:

```bash
python SWE-agent/scripts/combined/plot_results.py \
  --results-csv combined_results/final_results.csv \
  --output-dir combined_results/figures \
  --format png
```
