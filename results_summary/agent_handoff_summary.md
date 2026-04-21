# Agent Handoff: Figure Editing Summary

**Date:** April 21, 2026  
**Status:** All data collection complete. Figure generation scripts exist and run cleanly. `fig_a` and `fig_d` need the edits described below before they are paper-ready.

---

## Project Overview

EECS545 ML course project evaluating multi-role agent structures (planner → coder → reviewer) vs. single-role baselines on a custom 60-instance SWE-bench-style benchmark. The primary metric is solve rate vs. BPT (B-param·tokens — a model-size-normalized compute unit). All 27 experimental runs are complete across three case sets (c1/c2/c3, 20 instances each).

---

## What Was Being Worked On Before a Few Hours Ago

From recent git history:

- **Linear baseline re-runs** (`commit 1dd5190: Mid-re-run of linear baseline agents`): The L (Qwen 9b), M (Qwen 30b), and N (GPT-OSS 120b) linear agent variants were being re-run across all 3 case sets. This filled gaps and ensured apples-to-apples comparison against the multi-role and MCTS variants.
- **Final analysis plan** (`commit 1cfac78`): Written to `results_summary/` — defined the intended comparison structure (linear efficiency frontier, role ablations, MCTS critic gate).
- **All runs confirmed complete** (`commit b08b735`): All 27 variant × case-set directories organized into `combined_results/`.
- **Summary data regenerated + plotting script rewritten** (`commit 33800cc: All collection runs complete. Mid-development of plotting scripts`): `eval_combined.py` regenerated all 8 CSVs in `Summary_Data/` (old versions archived to `Summary_Data/Archive/`). `plot_results.py` was heavily revised (969 lines changed) — all 7 figure functions exist and generate output, but visual polish, filtering, and labeling are **incomplete** for fig_a and fig_d specifically.

---

## Data Collection Status

**All 27 runs complete.** Directories: `combined_results/{VARIANT}_{case_set}_{description}/`

Aggregate CSVs that feed the figures (all in `Summary_Data/`):

| File | Rows | Used by |
|------|------|---------|
| `efficiency_frontier_bpt_runs.csv` | 53 | fig_a |
| `steps_to_solve.csv` | 1,041 | fig_d |
| `ablation_features.csv` | 7 | fig_c |
| `mcts_branch_stats.csv` | 388 | fig_g |
| `resource_waste.csv` | 1,041 | fig_f |
| `instance_overlap_wide.csv` | 61 | fig_e |
| `instance_overlap_wide_pairwise.csv` | 631 | fig_e fallback |

Key columns in `efficiency_frontier_bpt_runs.csv`: `run_id, variant, case_set, solve_rate, avg_cost_bpt, bpt_per_solve, ...`  
Key columns in `steps_to_solve.csv`: `instance_id, run_id, variant, case_set, solved, steps_used, cost_bpt, ...`

---

## Figure Generation Status

**Script:** `SWE-agent/scripts/combined/plot_results.py`  
**Output dir:** `combined_results/figures/`  
**Run command:**
```bash
python SWE-agent/scripts/combined/plot_results.py \
    --data-dir Summary_Data \
    --output-dir combined_results/figures \
    --format png
```

| Figure | Function | Status | Action needed |
|--------|----------|--------|---------------|
| `fig_a_efficiency_frontier` | `fig_efficiency_frontier()` | **Needs edits** | Drop soft variants, drop swe-search, add descriptive labels |
| `fig_b_reviewer_audit` | `fig_reviewer_audit()` | Done | None |
| `fig_c_ablation_features` | `fig_ablation_features()` | Done | None |
| `fig_d_steps_to_solve` | `fig_steps_to_solve()` | **Needs edits + new variant** | Drop swe-search; add BPT-axis version |
| `fig_e_instance_overlap` | `fig_instance_overlap()` | Done | None |
| `fig_f_resource_waste` | `fig_resource_waste()` | Done | None |
| `fig_g_mcts_branching` | `fig_mcts_branching()` | Done | None |

---

## Editing Instructions — fig_a_efficiency_frontier

**File:** `SWE-agent/scripts/combined/plot_results.py`  
**Function:** `fig_efficiency_frontier()` (~lines 116–180)

### Change 1: Drop soft (non-strict) reviewer variants

The plot currently shows both `A` and `A_strict`, `B` and `B_strict`, etc. The soft variants (no `_strict` suffix) differ only in whether the reviewer gate is enforced — keeping both clutters the plot. **Keep only `_strict` variants** for multi-role/MCTS configs, plus the linear baselines (L, M — which have no strict/soft distinction) and P.

Update `SKIP_VARIANTS` inside `fig_efficiency_frontier()` (currently just `{"N"}`) to:

```python
SKIP_VARIANTS = {"N", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}
# Keeps: A_strict, B_strict, C_strict, F_strict, G_strict, L, M, P
```

### Change 2: Drop swe-search variants

H, I, J, K are swe-search replications included in the same set of SKIP_VARIANTS above. No separate change needed beyond the combined set above.

### Change 3: Replace single-letter annotations with full descriptive labels

Currently each scatter point is labeled with just the variant key (e.g., `"A_strict"`). This is unreadable to anyone without the variant table memorized.

Add a `VARIANT_DISPLAY_LABELS` dict near the top of the file (next to the existing `VARIANT_GROUPS` dict):

```python
VARIANT_DISPLAY_LABELS: dict[str, str] = {
    "L":        "L: 9b linear (baseline)",
    "M":        "M: 30b linear (baseline)",
    "B_strict": "B: 120b plan + 30b code, strict reviewer",
    "A_strict": "A: 9b MCTS, strict reviewer",
    "G_strict": "G: 9b MCTS + hindsight, strict reviewer",
    "F_strict": "F: 9b coder + 120b planner (no search), strict reviewer",
    "C_strict": "C: Mixed MCTS (120b plan + 9b code), strict reviewer",
    "P":        "P: Best combined (mixed MCTS + hindsight)",
}
```

In the annotation loop inside `fig_efficiency_frontier()`, change:

```python
# BEFORE:
ax.annotate(v, (bpt, rate), textcoords="offset points", xytext=(5, 3),
            fontsize=7.5, fontweight="bold", color=_variant_color(v))

# AFTER:
label = VARIANT_DISPLAY_LABELS.get(v, v)
ax.annotate(label, (bpt, rate), textcoords="offset points", xytext=(5, 3),
            fontsize=7.5, fontweight="bold", color=_variant_color(v))
```

Also increase figure width to accommodate longer labels:

```python
# BEFORE:
fig, ax = plt.subplots(figsize=(9, 5.5))

# AFTER:
fig, ax = plt.subplots(figsize=(12, 6))
```

---

## Editing Instructions — fig_d_steps_to_solve

**File:** `SWE-agent/scripts/combined/plot_results.py`  
**Function:** `fig_steps_to_solve()` (~lines 340–424)

### Change 1: Drop swe-search group

`STEP_GROUPS` (lines 74–83) maps H, I, J, K → `"swe-search"`. Remove this group from both the normalization denominator and the output bars.

In `fig_steps_to_solve()`, update the `rows_c23` filter and add a `group_avgs` filter:

```python
# BEFORE:
rows_c23 = [r for r in rows if r["case_set"] in ("c2", "c3")]

# AFTER:
rows_c23 = [r for r in rows if r["case_set"] in ("c2", "c3")
            and STEP_GROUPS.get(r["variant"], "swe-search") != "swe-search"]
```

And after the existing `group_avgs = {g: v for g, v in group_avgs.items() if any(v)}` line, add:

```python
group_avgs = {g: v for g, v in group_avgs.items() if g != "swe-search"}
```

### Change 2: Add a new BPT-axis version (fig_d_bpt_to_solve)

Create a new function `fig_steps_to_solve_bpt()` immediately after `fig_steps_to_solve()`. This is structurally identical but bins by compute cost (BPT) instead of step count. The purpose is to compare both plots and pick whichever tells the cleaner story for the paper: steps shows *agent behavior* (do some agents solve faster?); BPT shows *compute cost directly*.

**Binning logic:**
- Source column: `float(r["cost_bpt"]) / 1e6` (M B-param·tokens)
- Bin edges: `[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]`
- Bin labels: `["<0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5", "0.5–0.75", "0.75–1.0", "1.0–1.5", ">1.5"]`
- Last bin is open-ended (`>= 1.5`)
- Skip instances with `cost_bpt == 0` (missing data)

Sample binning loop (replace the steps-based loop in fig_steps_to_solve):

```python
BIN_EDGES  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
bin_labels = ["<0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5",
              "0.5–0.75", "0.75–1.0", "1.0–1.5", ">1.5"]
n_bins     = len(bin_labels)

for r in rows_f:
    bpt_val = float(r.get("cost_bpt", 0))
    if bpt_val <= 0:
        continue
    bpt_m = bpt_val / 1e6
    v = r["variant"]
    g = STEP_GROUPS.get(v, "swe-search")
    if g == "swe-search":
        continue
    if g not in group_counts:
        group_counts[g] = [0] * n_bins
    # find bin index
    idx = n_bins - 1  # default to last bin (open-ended)
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= bpt_m < BIN_EDGES[i + 1]:
            idx = i
            break
    group_counts[g][idx] += 1
```

**Axis/title/filename differences from fig_d:**
```python
ax.set_xlabel("Compute per solved instance (M B-param·tokens)", fontsize=11)
ax.set_ylabel("Avg instances solved per agent", fontsize=11)
ax.set_title("Compute-to-Solve Distribution by Agent Configuration (c2+c3, avg per agent)",
             fontsize=12, fontweight="bold", pad=10)
out = out_dir / f"fig_d_bpt_to_solve.{fmt}"
```

**Wire into main():** Call `fig_steps_to_solve_bpt(data_dir, out_dir, fmt)` immediately after the existing `fig_steps_to_solve(...)` call in `main()`.

---

## Variant Reference

| Variant | Description | Models | Role structure |
|---------|-------------|--------|----------------|
| L | 9b linear baseline | Qwen 9b | Coder only |
| M | 30b linear baseline | Qwen 30b | Coder only |
| N | 120b linear baseline *(excluded from all plots)* | GPT-OSS 120b | Coder only |
| B / B_strict | Multi-role linear | 120b planner + 30b coder + 120b reviewer | Planner → Coder → Reviewer |
| A / A_strict | 9b MCTS | Qwen 9b (all roles) | MCTS coder + reviewer gate |
| G / G_strict | 9b MCTS + hindsight | Qwen 9b | MCTS + hindsight context from prior failures |
| F / F_strict | 9b coder + 120b planner, no search | 120b planner + 9b coder + 120b reviewer | Linear pipeline, mixed model sizes |
| C / C_strict | Mixed-size MCTS | 120b planner + 9b coder + 120b reviewer | MCTS with large planner |
| H, I, J, K | swe-search variants *(excluded from all plots)* | Various | swe-search replication |
| P | Best combined | 120b planner + 9b coder + 120b reviewer | Mixed MCTS + hindsight |

**`_strict` suffix** = strict reviewer gate: reviewer must explicitly accept the patch before it is finalized. Without it (soft), the agent auto-accepts after a fixed number of rounds.

**For fig_a:** plot only `A_strict`, `B_strict`, `C_strict`, `F_strict`, `G_strict`, `L`, `M`, `P`.  
**For fig_d:** plot all groups except `"swe-search"`.
