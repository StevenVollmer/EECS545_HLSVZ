# Analysis Scripts Overhaul

## Efficiency Metric
**Parameter-normalized tokens (BPT)** replaces USD cost.
`cost_bpt = Σ_role (tokens_in + tokens_out) × params_B`
where 9b→9, 30b→30, 120b→120. Proportional to FLOPs; standard in scaling literature.
Paper axis: `"Estimated FLOPs (B-param·tokens)"`.

## Files Changed / Created

| File | Change |
|------|--------|
| `SWE-agent/scripts/combined/eval_combined.py` | Add BPT metric, scan combined_results/, new export functions |
| `SWE-agent/scripts/combined/audit_reviewer.py` | Also scan combined_results/, add reviewer_performance CSV |
| `SWE-agent/scripts/combined/extract_mcts_trees.py` | **New** — extract MCTS tree stats + example trees from traj files |

## New CSV Outputs (in combined_results/)

| File | Content |
|------|---------|
| `efficiency_frontier_bpt.csv` | Per-instance: tokens by role+model, cost_bpt, solve, mcts_iterations, duration |
| `efficiency_frontier_bpt_runs.csv` | Per-run: solve_rate, avg_cost_bpt, bpt_per_solve, Pareto (c2+c3 only) |
| `reviewer_performance.csv` | Confusion matrix per (reviewer_size × variant × case_set) |
| `mcts_branch_stats.csv` | Per-instance tree stats for MCTS runs |
| `mcts_example_trees.json` | 2–3 selected tree structures for paper figure |
| `steps_to_solve.csv` | Per-instance: steps_used (raw), steps_allowed, solve_fraction, cost_bpt |
| `instance_overlap.csv` | Wide-format solved booleans per variant; pairwise Jaccard overlap |
| `resource_waste.csv` | Per-instance: solved, cost_bpt, steps, stopped_reason |
| `ablation_features.csv` | Per binary feature: mean solve rate / BPT with/without feature |

## Key Implementation Notes
- `stats["turns"]` in traj gives actual coder iterations (steps_used)
- `mcts_tree.nodes` in traj is already serialized — no log parsing needed
- `_variant_letter()` and `_case_set()` updated to handle `A_strict_c1_...` format
- Pareto frontier uses c2+c3 only (MCTS was tuned on c1)
- Ablation covers: mcts, planner, reviewer, strict_gate, hindsight, value_fn, coder_size
