#!/usr/bin/env python3
"""Generate paper figures from combined-agent ablation results.

Figures produced:
  fig_a_efficiency_frontier  — Scatter: solve rate vs BPT (c2+c3 held-out avg)
  fig_b_reviewer_audit       — Stacked bar: TP/FP/FN/TN % by reviewer model size
  fig_c_ablation_features    — Dual panel: accuracy lift + % efficiency change per feature
  fig_d_steps_to_solve       — Histogram: avg solved per agent by iteration bin, by group
  fig_e_instance_overlap     — Venn diagram: solved/failed instance overlap for A/B/C
  fig_f_resource_waste       — Dual histogram: compute distribution for solved vs failed
  fig_g_mcts_branching       — MCTS search depth and branching analysis

Usage (from project root):
  python SWE-agent/scripts/combined/plot_results.py \\
      --data-dir Summary_Data \\
      --audit-csv combined_results/reviewer_audits/audit_results.csv \\
      --output-dir combined_results/figures \\
      --format png
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Variant / group metadata
# ---------------------------------------------------------------------------

VARIANT_GROUPS: dict[str, str] = {
    "L": "linear",       "M": "linear",       "N": "linear",
    "B": "baseline",     "B_strict": "baseline",
    "H": "swesearch",    "I": "swesearch",     "J": "swesearch",  "K": "swesearch",
    "A": "ours_9b",      "A_strict": "ours_9b",
    "G": "ours_9b",      "G_strict": "ours_9b",
    "F": "ours_mixed",   "F_strict": "ours_mixed",
    "C": "ours_mixed",   "C_strict": "ours_mixed",
    "D": "ours_mixed",   "E": "ours_mixed",
    "P": "ours_best",
}

VARIANT_DISPLAY_LABELS: dict[str, str] = {
    "L":        "L: 9b C",
    "M":        "M: 30b C",
    "B_strict": "B: 120b P + 30b C + R",
    "A_strict": "A: 9b P+C+R (MCTS)",
    "G_strict": "G: 9b P+C+R (MCTS + hindsight)",
    "F_strict": "F: 120b P + 9b C + R",
    "C_strict": "C: 120b P + 9b C + R (MCTS)",
    "P":        "P: 120b P + 9b C + R (MCTS + hindsight)",
}

GROUP_COLORS: dict[str, str] = {
    "linear":     "#c9c9c9",
    "baseline":   "#9b9b9b",
    "swesearch":  "#e8a838",
    "ours_9b":    "#5b9bd5",
    "ours_mixed": "#70ad47",
    "ours_best":  "#c00000",
}

GROUP_LABELS: dict[str, str] = {
    "linear":     "Linear baselines (L/M/N)",
    "baseline":   "Multi-role linear (B)",
    "swesearch":  "swe-search variants",
    "ours_9b":    "9b MCTS (ours)",
    "ours_mixed": "Mixed-size (ours)",
    "ours_best":  "Best combined (P)",
}

FEATURE_NAMES: dict[str, str] = {
    "mcts":        "MCTS search",
    "planner":     "Planner role",
    "reviewer":    "Reviewer role",
    "strict_gate": "Strict reviewer gate",
    "hindsight":   "Hindsight feedback",
    "value_fn":    "Value function",
}

# Groups used for steps-to-solve aggregation (fig_d).
# 5 architectural categories. Auto-accept runs (rafe_mcts_baseline,
# rafe_mcts_plan_critic) are intentionally omitted — they default to
# "swe-search" and are filtered out in fig_steps_to_solve.
STEP_GROUPS: dict[str, str] = {
    # 1. Linear single-coder (no planning, no search)
    "L":             "Linear single-coder",
    "M":             "Linear single-coder",
    "rafe_qwen":     "Linear single-coder",
    "rafe_gpt120b":  "Linear single-coder",
    # 2. Planner + coder (no quality gate after coding)
    "rafe_gpt_plan_qwen_code": "Planner + coder",
    # 3. Planner + coder + reviewer (post-code reviewer gate)
    # B_strict and F_strict excluded: combined_results counts only coder-role turns,
    # not total pipeline turns — incomparable with final_matrix step counts.
    "rafe_gpt_plan_qwen_code_rev": "Planner + coder + reviewer",
    # 4. Planner + critic + coder (plan-level critic before coding)
    "rafe_gpt_plan_critic_qwen":   "Planner + critic + coder",
    # 5. MCTS (no auto-accept; all variants use a reviewer/critic gate)
    "A_strict":              "MCTS",
    "G_strict":              "MCTS",
    "C_strict":              "MCTS",
    "rafe_mcts_critic_gate": "MCTS",
}

STEP_GROUP_COLORS: dict[str, str] = {
    "Linear single-coder":         "#c9c9c9",
    "Planner + coder":             "#70ad47",
    "Planner + coder + reviewer":  "#2e75b6",
    "Planner + critic + coder":    "#ed7d31",
    "MCTS":                        "#c00000",
}

# Variants tracked in mcts_branch_stats (MCTS variants only)
MCTS_VARIANTS = {"A", "A_strict", "C", "C_strict", "E", "G", "G_strict", "P"}

# Variants included in resource-waste figures: strict-gate reviewer OR no-reviewer linear baseline.
# Excludes auto-accept MCTS (rafe_mcts_baseline, rafe_mcts_plan_critic) and soft-gate variants
# (A, B, C, D, E, F, G, H, I, J, K, P).
_STRICT_GATE_VARIANTS: frozenset[str] = frozenset([
    # Strict-gate MCTS
    "A_strict", "C_strict", "G_strict", "rafe_mcts_critic_gate",
    # Strict-gate linear (reviewer/critic gate)
    "B_strict", "F_strict", "rafe_gpt_plan_qwen_code_rev", "rafe_gpt_plan_critic_qwen",
    # No-reviewer linear baselines (single-pass, not soft-gate)
    "L", "M", "N", "rafe_qwen", "rafe_gpt120b", "rafe_gpt_plan_qwen_code",
])

# Within _STRICT_GATE_VARIANTS, which use MCTS search
_WASTE_MCTS_VARIANTS: frozenset[str] = frozenset([
    "A_strict", "C_strict", "G_strict", "rafe_mcts_critic_gate",
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _variant_color(variant: str) -> str:
    group = VARIANT_GROUPS.get(variant, "swesearch")
    return GROUP_COLORS[group]


# ---------------------------------------------------------------------------
# Fig a — Efficiency frontier: solve rate vs BPT (no Pareto overlay)
# ---------------------------------------------------------------------------

def fig_efficiency_frontier(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    rows = _load_csv(data_dir / "efficiency_frontier_bpt_runs.csv")

    # Average c2+c3 per variant (held-out sets); skip N (120b linear) — noted in paper
    SKIP_VARIANTS = {"N", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}
    # Keeps: A_strict, B_strict, C_strict, F_strict, G_strict, L, M, P
    by_variant: dict[str, dict[str, list]] = {}
    for r in rows:
        if r["case_set"] not in ("c2", "c3"):
            continue
        v = r["variant"]
        if v in SKIP_VARIANTS:
            continue
        d = by_variant.setdefault(v, {"rate": [], "bpt": []})
        d["rate"].append(float(r["solve_rate"]) * 100)
        d["bpt"].append(float(r["avg_cost_bpt"]) / 1e6)

    if not by_variant:
        print("fig_a: no c2+c3 data found")
        return

    points = []
    for v, d in by_variant.items():
        rate = sum(d["rate"]) / len(d["rate"])
        bpt  = sum(d["bpt"])  / len(d["bpt"])
        points.append((v, rate, bpt))

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    xs = [bpt  for _, _, bpt  in points]
    ys = [rate for _, rate, _ in points]
    cs = [_variant_color(v) for v, _, _ in points]
    ax.scatter(xs, ys, c=cs, s=90, zorder=4, edgecolors="white", linewidths=0.6)

    texts = []
    for v, rate, bpt in points:
        label = VARIANT_DISPLAY_LABELS.get(v, v)
        texts.append(ax.text(bpt, rate, label,
                             fontsize=7.5, fontweight="bold",
                             color=_variant_color(v)))

    # --- 10-case audit-study reference points (Part 2 of results_summary/README) ---
    # Rescaled into BPT units using the gpt→qwen ≈ F anchor (both are 120b plan + 9b
    # code, no search): gpt→qwen at 2.04 rel-compute ≈ F at 0.60 M BPT → factor 0.294.
    # gpt-solo (24.62 rel) falls off-scale and is omitted.
    AUDIT_SCALE = 0.60 / 2.04
    AUDIT_COLOR = "#8a4fbd"
    audit_points = [
        (70.0, 2.04, "S: 120b P + 30b C"),
        (80.0, 2.15, "T: 120b P + 120b Cr + 30b C"),
        (80.0, 2.60, "U: 120b P + 30b C + 120b R"),
        (50.0, 3.37, "Q: 30b C"),
    ]
    audit_xs = [comp * AUDIT_SCALE for _, comp, _ in audit_points]
    audit_ys = [rate                for rate, _, _ in audit_points]
    ax.scatter(audit_xs, audit_ys, marker="^", s=70, c=AUDIT_COLOR,
               zorder=4, edgecolors="white", linewidths=0.6)
    for (rate, comp, label), bpt_eq in zip(audit_points, audit_xs):
        texts.append(ax.text(bpt_eq, rate, label,
                             fontsize=7.0, fontweight="bold", color=AUDIT_COLOR))

    ax.set_xlabel("Avg compute per instance (M B-param·tokens)", fontsize=11)
    ax.set_ylabel("Avg solve rate (%)", fontsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    try:
        from adjustText import adjust_text
        # Combine matrix + audit points so adjustText knows about all scatter markers
        all_x = list(xs) + list(audit_xs)
        all_y = list(ys) + list(audit_ys)
        adjust_text(texts, x=all_x, y=all_y, ax=ax,
                    expand=(1.6, 1.9),
                    force_text=(0.8, 1.1),
                    force_static=(0.8, 1.0),
                    force_pull=(0.02, 0.02),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
                    only_move={"points": "xy", "texts": "xy", "static": "xy"},
                    max_move=60,
                    iter_lim=500)
    except ImportError:
        pass

    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS["ours_9b"],    label="40-case matrix (held-out)"),
        mpatches.Patch(color=AUDIT_COLOR,                label="10-case audit study (rescaled)"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9, framealpha=0.9, frameon=False)

    ax.set_title("Efficiency Frontier: Solve Rate vs. Compute",
                 fontsize=12, fontweight="bold", pad=10)
    fig.text(0.5, 0.02,
             "Role abbreviations: P = planner, Cr = critic, C = coder, R = reviewer",
             ha="center", fontsize=8.5, style="italic", color="#444")

    fig.tight_layout()
    out = out_dir / f"fig_a_efficiency_frontier.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig b — Reviewer outcomes: TP / FP / FN / TN as % of evaluations
# ---------------------------------------------------------------------------

def fig_reviewer_audit(audit_csv: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load_csv(audit_csv)
    sizes = ["9b", "30b", "120b"]

    agg: dict[str, dict[str, int]] = {s: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for s in sizes}
    for r in rows:
        sz = r.get("reviewer_size", "")
        if sz not in agg:
            continue
        for k in ("tp", "fp", "fn", "tn"):
            agg[sz][k] += int(r.get(k, 0))

    # Compute % of total evaluations for each outcome
    pcts: dict[str, dict[str, float]] = {}
    for sz in sizes:
        total = sum(agg[sz].values())
        pcts[sz] = {k: agg[sz][k] / total * 100 for k in ("tp", "fp", "fn", "tn")}

    # Order: Good accepts (TP), Good rejects (TN), False accepts (FP), False rejects (FN)
    outcomes     = ["tp",             "tn",             "fp",              "fn"]
    labels_out   = ["Good accepts",   "Good rejects",   "False accepts",   "False rejects"]
    colors_out   = ["#70ad47",        "#a9d18e",        "#e8a838",         "#c00000"]

    x = np.arange(len(sizes))
    width = 0.55
    bottoms = np.zeros(len(sizes))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_by_outcome: dict[str, Any] = {}
    for outcome, label, color in zip(outcomes, labels_out, colors_out):
        vals = np.array([pcts[s][outcome] for s in sizes])
        bars_by_outcome[outcome] = ax.bar(
            x, vals, width, bottom=bottoms,
            color=color, label=label, zorder=3, edgecolor="white", linewidth=0.5
        )
        # Annotate segments that are large enough to label
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 4:
                ax.text(x[i], b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        bottoms = bottoms + vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} reviewer" for s in sizes], fontsize=10)
    ax.set_ylabel("% of evaluations", fontsize=11)
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("Reviewer Accuracy by Model Size\n(blind evaluation — no oracle test access)",
                 fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    out = out_dir / f"fig_b_reviewer_audit.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig c — Ablation feature impact (dual panel, efficiency normalized to %)
# ---------------------------------------------------------------------------

def fig_ablation_features(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load_csv(data_dir / "ablation_features.csv")

    # Sort by feature_effect_solve descending
    rows_sorted = sorted(rows, key=lambda r: float(r["feature_effect_solve"]), reverse=True)

    labels      = [FEATURE_NAMES.get(r["feature"], r["feature"]) for r in rows_sorted]
    effect_acc  = [float(r["feature_effect_solve"]) * 100 for r in rows_sorted]
    low_power   = [bool(r.get("low_power_warning", "")) for r in rows_sorted]

    # Normalize efficiency: % reduction in BPT/solve vs. baseline (feature off)
    # Positive = feature is more efficient (cheaper per solve) vs. without it
    # Negative = feature adds compute burden per solve
    effect_eff_pct = []
    for r in rows_sorted:
        on_val  = float(r["mean_bpt_per_solve_on"])
        off_val = float(r["mean_bpt_per_solve_off"])
        pct = (off_val - on_val) / off_val * 100 if off_val > 0 else 0.0
        effect_eff_pct.append(pct)

    y = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.6 * len(labels) + 2.5), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    def _bar_colors(vals: list[float]) -> list[str]:
        return ["#70ad47" if v >= 0 else "#c00000" for v in vals]

    # Left: accuracy lift
    bars1 = ax1.barh(y, effect_acc, color=_bar_colors(effect_acc),
                     edgecolor="white", linewidth=0.4, zorder=3)
    for bar, val, lp in zip(bars1, effect_acc, low_power):
        sign   = "+" if val >= 0 else ""
        suffix = "*" if lp else ""
        ax1.text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
                 f"{sign}{val:.1f}pp{suffix}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=8.5, fontweight="bold")
    ax1.axvline(0, color="black", linewidth=0.8, zorder=2)
    ax1.set_xlabel("Accuracy lift (percentage points)", fontsize=10)
    ax1.set_title("Solve Rate Impact", fontsize=11, fontweight="bold")
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: efficiency impact (% change in BPT/solve vs baseline)
    bars2 = ax2.barh(y, effect_eff_pct, color=_bar_colors(effect_eff_pct),
                     edgecolor="white", linewidth=0.4, zorder=3)
    for bar, val, lp in zip(bars2, effect_eff_pct, low_power):
        sign   = "+" if val >= 0 else ""
        suffix = "*" if lp else ""
        ax2.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height() / 2,
                 f"{sign}{val:.0f}%{suffix}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=8.5, fontweight="bold")
    ax2.axvline(0, color="black", linewidth=0.8, zorder=2)
    ax2.set_xlabel("% reduction in BPT/solve vs. baseline\n(positive = more efficient per solved instance)", fontsize=10)
    ax2.set_title("Efficiency Impact", fontsize=11, fontweight="bold")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)

    if any(low_power):
        fig.text(0.5, 0.01, "* low statistical power (n < 3 runs)",
                 ha="center", fontsize=8, style="italic", color="gray")

    fig.suptitle("Feature Ablation: Impact on Solve Rate and Compute Efficiency",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / f"fig_c_ablation_features.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig d — Steps-to-solve: avg per agent by bin (bin=2, max=18)
# ---------------------------------------------------------------------------

def fig_steps_to_solve(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load_csv(data_dir / "steps_to_solve.csv")

    # Filter: solved=1, c1 only (includes both combined_results c1 and final_matrix c1)
    # Exclude swe-search and soft (non-strict) variants via STEP_GROUPS membership.
    _EXCLUDE_GROUPS = {"swe-search"}

    rows_f = [r for r in rows
              if r["solved"] in ("1", "True") and r["case_set"] == "c1"
              and STEP_GROUPS.get(r["variant"], "swe-search") not in _EXCLUDE_GROUPS]

    if not rows_f:
        print("fig_d: no solved instances in c1")
        return

    # Bins of width 2, steps 1–18
    BIN_W    = 2
    MAX_STEP = 18
    bin_starts  = list(range(1, MAX_STEP + 1, BIN_W))   # [1,3,5,...,17]
    bin_labels  = [f"{b}–{b+BIN_W-1}" for b in bin_starts]
    n_bins      = len(bin_starts)

    # Denominator: unique run_ids per group across all c1 rows (not just solved)
    group_counts:  dict[str, list[int]] = {}
    group_run_ids: dict[str, set[str]]  = defaultdict(set)

    for r in rows:
        if r["case_set"] != "c1":
            continue
        g = STEP_GROUPS.get(r["variant"], "swe-search")
        if g in _EXCLUDE_GROUPS:
            continue
        group_run_ids[g].add(r["run_id"])

    for r in rows_f:
        v     = r["variant"]
        g     = STEP_GROUPS.get(v, "swe-search")
        steps = int(r.get("effective_steps") or r["steps_used"])
        if steps > MAX_STEP or steps < 1:
            continue
        if g not in group_counts:
            group_counts[g] = [0] * n_bins
        for i, b in enumerate(bin_starts):
            if b <= steps <= b + BIN_W - 1:
                group_counts[g][i] += 1
                break

    # Normalize by number of run_ids in each group
    group_avgs: dict[str, list[float]] = {}
    for g, counts in group_counts.items():
        n = len(group_run_ids[g]) or 1
        group_avgs[g] = [c / n for c in counts]

    # Drop groups with no solves
    group_avgs = {g: v for g, v in group_avgs.items() if any(v)}
    group_avgs = {g: v for g, v in group_avgs.items() if g != "swe-search"}

    if not group_avgs:
        print("fig_d: no data after grouping")
        return

    n_groups = len(group_avgs)
    fig, ax = plt.subplots(figsize=(max(12, n_groups * 1.2), 5.5))
    x        = np.arange(n_bins)
    bar_width = 0.8 / n_groups

    for i, (g, avgs) in enumerate(group_avgs.items()):
        offset = (i - n_groups / 2 + 0.5) * bar_width
        color  = STEP_GROUP_COLORS.get(g, "#aaaaaa")
        ax.bar(x + offset, avgs, bar_width * 0.9, color=color,
               label=g, zorder=3, edgecolor="white", linewidth=0.4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("Steps used to solve (iteration bin)", fontsize=11)
    ax.set_ylabel("Avg instances solved per agent", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right",
              ncol=2 if n_groups > 6 else 1)
    ax.set_title(
        "Steps-to-Solve Distribution by Agent Configuration\n"
        "(c1, 20 instances, avg solved per agent; strict variants only)",
        fontsize=12, fontweight="bold", pad=10,
    )

    fig.tight_layout()
    out = out_dir / f"fig_d_steps_to_solve.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig d (bpt) — Compute-to-solve: avg per agent by BPT bin
# ---------------------------------------------------------------------------

def fig_steps_to_solve_bpt(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load_csv(data_dir / "steps_to_solve.csv")

    # Filter: solved=1, held-out c2+c3
    rows_f = [r for r in rows
              if r["solved"] in ("1", "True") and r["case_set"] in ("c2", "c3")]

    if not rows_f:
        print("fig_d_bpt: no solved instances in c2+c3")
        return

    BIN_EDGES  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
    bin_labels = ["<0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5",
                  "0.5–0.75", "0.75–1.0", "1.0–1.5", ">1.5"]
    n_bins     = len(bin_labels)

    group_counts:  dict[str, list[int]] = {}
    group_run_ids: dict[str, set[str]]  = defaultdict(set)

    rows_c23 = [r for r in rows if r["case_set"] in ("c2", "c3")
                and STEP_GROUPS.get(r["variant"], "swe-search") != "swe-search"]
    for r in rows_c23:
        g = STEP_GROUPS.get(r["variant"], "swe-search")
        group_run_ids[g].add(r["run_id"])

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
        idx = n_bins - 1  # default to last bin (open-ended)
        for i in range(len(BIN_EDGES) - 1):
            if BIN_EDGES[i] <= bpt_m < BIN_EDGES[i + 1]:
                idx = i
                break
        group_counts[g][idx] += 1

    # Normalize by number of run_ids in each group
    group_avgs: dict[str, list[float]] = {}
    for g, counts in group_counts.items():
        n = len(group_run_ids[g]) or 1
        group_avgs[g] = [c / n for c in counts]

    group_avgs = {g: v for g, v in group_avgs.items() if any(v)}
    group_avgs = {g: v for g, v in group_avgs.items() if g != "swe-search"}

    if not group_avgs:
        print("fig_d_bpt: no data after grouping")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x        = np.arange(n_bins)
    n_groups = len(group_avgs)
    bar_width = 0.8 / n_groups

    for i, (g, avgs) in enumerate(group_avgs.items()):
        offset = (i - n_groups / 2 + 0.5) * bar_width
        color  = STEP_GROUP_COLORS.get(g, "#aaaaaa")
        ax.bar(x + offset, avgs, bar_width * 0.9, color=color,
               label=g, zorder=3, edgecolor="white", linewidth=0.4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("Compute per solved instance (M B-param·tokens)", fontsize=11)
    ax.set_ylabel("Avg instances solved per agent", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.set_title("Compute-to-Solve Distribution by Agent Configuration (c2+c3, avg per agent)",
                 fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    out = out_dir / f"fig_d_bpt_to_solve.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig e — Instance overlap (Venn diagram) — fixed region ordering
# ---------------------------------------------------------------------------

def fig_instance_overlap(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    rows     = _load_csv(data_dir / "instance_overlap_wide.csv")
    rows_f   = [r for r in rows if r["case_set"] in ("c2", "c3")]

    def _venn_subsets(col_a: str, col_b: str, col_c: str,
                      pos: str = "1") -> tuple[int, ...]:
        """Return venn3 7-tuple (Abc,aBc,ABc,abC,AbC,aBC,ABC) for solved (pos='1') or failed."""
        neg = "0" if pos == "1" else "1"
        a = {r["instance_id"] for r in rows_f if r.get(col_a, neg) == pos}
        b = {r["instance_id"] for r in rows_f if r.get(col_b, neg) == pos}
        c = {r["instance_id"] for r in rows_f if r.get(col_c, neg) == pos}
        abc     = len(a & b & c)
        ab_only = len(a & b) - abc
        ac_only = len(a & c) - abc
        bc_only = len(b & c) - abc
        only_a  = len(a - b - c)
        only_b  = len(b - a - c)
        only_c  = len(c - a - b)
        # venn3 order: Abc, aBc, ABc, abC, AbC, aBC, ABC
        return (only_a, only_b, ab_only, only_c, ac_only, bc_only, abc)

    try:
        from matplotlib_venn import venn3
        from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
        has_venn = True
    except ImportError:
        has_venn = False

    if not has_venn:
        _fig_overlap_fallback(data_dir, out_dir, fmt)
        return

    solved_sub = _venn_subsets("A", "B", "C", pos="1")
    failed_sub = _venn_subsets("A", "B", "C", pos="0")
    n_total    = len(rows_f)

    # Equal-circle layout (replaces deprecated venn3_unweighted)
    _equal_layout = DefaultLayoutAlgorithm(fixed_subset_sizes=(1, 1, 1, 1, 1, 1, 1))

    fig, (ax_s, ax_f) = plt.subplots(1, 2, figsize=(12, 5.5))

    venn3(subsets=solved_sub,
          set_labels=("A\n9b MCTS", "B\nMulti-role\nlinear", "C\nMixed MCTS"),
          set_colors=("#5b9bd5", "#9b9b9b", "#c00000"), alpha=0.55, ax=ax_s,
          layout_algorithm=_equal_layout)
    ax_s.set_title(f"Instances Solved (c2+c3, n={n_total})",
                   fontsize=10, fontweight="bold")

    venn3(subsets=failed_sub,
          set_labels=("A\n9b MCTS", "B\nMulti-role\nlinear", "C\nMixed MCTS"),
          set_colors=("#5b9bd5", "#9b9b9b", "#c00000"), alpha=0.55, ax=ax_f,
          layout_algorithm=_equal_layout)
    ax_f.set_title(f"Instances Failed (c2+c3, n={n_total})",
                   fontsize=10, fontweight="bold")

    fig.suptitle("Instance Overlap: Which Problems Each Agent Solved vs. Failed",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = out_dir / f"fig_e_instance_overlap.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _fig_overlap_fallback(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows   = _load_csv(data_dir / "instance_overlap_pairwise.csv")
    rows_f = [r for r in rows if r["case_set"] in ("c2", "c3")]

    pair_jac: dict[tuple[str, str], list[float]] = {}
    for r in rows_f:
        key = (r["variant_a"], r["variant_b"])
        pair_jac.setdefault(key, []).append(float(r["jaccard"]))
    pair_avg = {k: sum(v) / len(v) for k, v in pair_jac.items()}

    variants = sorted({v for a, b in pair_avg for v in (a, b)})
    mat = np.zeros((len(variants), len(variants)))
    for i, a in enumerate(variants):
        for j, b in enumerate(variants):
            mat[i, j] = pair_avg.get((a, b), pair_avg.get((b, a), 0.0 if i != j else 1.0))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(variants)))
    ax.set_yticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(variants, fontsize=8)
    for i in range(len(variants)):
        for j in range(len(variants)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Jaccard similarity (solved)")
    ax.set_title("Pairwise Instance Overlap (c2+c3)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = out_dir / f"fig_e_instance_overlap.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved (fallback heatmap): {out}")


# ---------------------------------------------------------------------------
# Fig f — Resource waste: solved vs failed compute distributions
# ---------------------------------------------------------------------------

def _eff_steps(r: dict) -> int:
    eff = r.get("effective_steps", "")
    return int(eff) if eff else int(r["steps_used"])


def _plot_steps_hist(
    ax: "Any",
    solved_steps: list[int],
    failed_steps: list[int],
    *,
    bins: "Any",
    title: str,
    color_s: str = "#4472c4",
    color_f: str = "#c00000",
    alpha: float = 0.65,
) -> None:
    import numpy as np
    ax.hist(solved_steps, bins=bins, color=color_s, alpha=alpha,
            label=f"Solved (n={len(solved_steps)})", zorder=3)
    ax.hist(failed_steps, bins=bins, color=color_f, alpha=alpha,
            label=f"Failed (n={len(failed_steps)})", zorder=3)
    if solved_steps:
        med_s = float(np.median(solved_steps))
        ax.axvline(med_s, color=color_s, linestyle="--", linewidth=1.5,
                   label=f"Solved median={med_s:.0f}")
    if failed_steps:
        med_f = float(np.median(failed_steps))
        ax.axvline(med_f, color=color_f, linestyle="--", linewidth=1.5,
                   label=f"Failed median={med_f:.0f}")
    ax.set_xlabel("Total model calls (coder)", fontsize=11)
    ax.set_ylabel("Instance count", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(__import__("matplotlib.ticker", fromlist=["MaxNLocator"]).MaxNLocator(integer=True))


def fig_resource_waste(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows   = _load_csv(data_dir / "resource_waste.csv")
    rows_f = [r for r in rows
              if r["case_set"] in ("c1", "c2", "c3")
              and r["variant"] in _STRICT_GATE_VARIANTS]

    solved_steps = [_eff_steps(r) for r in rows_f if r["solved"] == "1" and _eff_steps(r) > 0]
    failed_steps = [_eff_steps(r) for r in rows_f if r["solved"] == "0" and _eff_steps(r) > 0]
    solved_bpt   = [float(r["cost_bpt"]) / 1e6 for r in rows_f
                    if r["solved"] == "1" and float(r["cost_bpt"]) > 0]
    failed_bpt   = [float(r["cost_bpt"]) / 1e6 for r in rows_f
                    if r["solved"] == "0" and float(r["cost_bpt"]) > 0]

    if not solved_steps or not failed_steps:
        print("fig_f: insufficient data")
        return

    color_s = "#4472c4"
    color_f = "#c00000"

    # Fig f1: steps (all strict-gate variants combined) — bins cover full observed range
    all_steps_max = max(max(solved_steps), max(failed_steps))
    step_bins = range(1, all_steps_max + 2)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    _plot_steps_hist(ax1, solved_steps, failed_steps,
                     bins=step_bins,
                     title="Resource Consumption: Model Calls, Solved vs Failed\n(all variants, aggregated)")
    fig1.tight_layout()
    out1 = out_dir / f"fig_f1_resource_consumption_steps.{fmt}"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {out1}")

    # Fig f2: BPT (log scale)
    if solved_bpt and failed_bpt:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        log_min = math.floor(math.log10(min(min(solved_bpt), min(failed_bpt))))
        log_max = math.ceil( math.log10(max(max(solved_bpt), max(failed_bpt))))
        bpt_bins = np.logspace(log_min, log_max, 20)
        ax2.hist(solved_bpt, bins=bpt_bins, color=color_s, alpha=0.65, label="Solved", zorder=3)
        ax2.hist(failed_bpt, bins=bpt_bins, color=color_f, alpha=0.65, label="Failed", zorder=3)
        med_s_bpt = float(np.median(solved_bpt))
        med_f_bpt = float(np.median(failed_bpt))
        ax2.axvline(med_s_bpt, color=color_s, linestyle="--", linewidth=1.5,
                    label=f"Solved median={med_s_bpt:.2f}M")
        ax2.axvline(med_f_bpt, color=color_f, linestyle="--", linewidth=1.5,
                    label=f"Failed median={med_f_bpt:.2f}M")
        ratio = med_f_bpt / med_s_bpt if med_s_bpt > 0 else float("nan")
        ax2.set_xscale("log")
        ax2.set_xlabel("Compute per instance (M B-param·tokens, log scale)", fontsize=11)
        ax2.set_ylabel("Instance count", fontsize=11)
        ax2.set_title(f"Resource Consumption: Compute, Solved vs Failed  (failed uses {ratio:.1f}× median)",
                      fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8.5, framealpha=0.9)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax2.set_axisbelow(True)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        out2 = out_dir / f"fig_f2_resource_waste_compute.{fmt}"
        fig2.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out2}")


def fig_resource_waste_by_arch(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    """Separate resource-waste step histograms for MCTS and linear strict-gate variants."""
    import matplotlib.pyplot as plt

    rows   = _load_csv(data_dir / "resource_waste.csv")
    rows_f = [r for r in rows
              if r["case_set"] in ("c1", "c2", "c3")
              and r["variant"] in _STRICT_GATE_VARIANTS]

    mcts_solved = [_eff_steps(r) for r in rows_f
                   if r["variant"] in _WASTE_MCTS_VARIANTS and r["solved"] == "1" and _eff_steps(r) > 0]
    mcts_failed = [_eff_steps(r) for r in rows_f
                   if r["variant"] in _WASTE_MCTS_VARIANTS and r["solved"] == "0" and _eff_steps(r) > 0]
    lin_solved  = [_eff_steps(r) for r in rows_f
                   if r["variant"] not in _WASTE_MCTS_VARIANTS and r["solved"] == "1" and _eff_steps(r) > 0]
    lin_failed  = [_eff_steps(r) for r in rows_f
                   if r["variant"] not in _WASTE_MCTS_VARIANTS and r["solved"] == "0" and _eff_steps(r) > 0]

    # Fig f1linear: linear strict-gate variants
    if lin_solved or lin_failed:
        fig, ax = plt.subplots(figsize=(6, 5))
        lin_max = max(max(lin_solved or [0]), max(lin_failed or [0]))
        lin_bins = range(1, lin_max + 2)
        _plot_steps_hist(ax, lin_solved, lin_failed,
                         bins=lin_bins,
                         title="Resource Consumption: Linear Agents\n(strict-gate variants, all case sets)")
        fig.tight_layout()
        out = out_dir / f"fig_f1linear_resource_consumption_steps.{fmt}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    # Fig f1mcts: MCTS strict-gate variants
    if mcts_solved or mcts_failed:
        fig, ax = plt.subplots(figsize=(6, 5))
        mcts_max = max(max(mcts_solved or [0]), max(mcts_failed or [0]))
        mcts_bins = range(1, mcts_max + 2)
        _plot_steps_hist(ax, mcts_solved, mcts_failed,
                         bins=mcts_bins,
                         title="Resource Consumption: MCTS Agents\n(strict-gate variants, all case sets)")
        fig.tight_layout()
        out = out_dir / f"fig_f1mcts_resource_consumption_steps.{fmt}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig g — MCTS branching: search depth and solve timing
# ---------------------------------------------------------------------------

def fig_mcts_branching(data_dir: pathlib.Path, out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load_csv(data_dir / "mcts_branch_stats.csv")
    if not rows:
        print("fig_g: mcts_branch_stats.csv empty")
        return

    # -- Panel 1: histogram of tree_nodes, solved vs failed ----------------
    solved_nodes = [int(r["tree_nodes"]) for r in rows if r["solved"] == "True"]
    failed_nodes = [int(r["tree_nodes"]) for r in rows if r["solved"] == "False"]

    # -- Panel 2: per-variant avg tree_nodes, solved vs failed -------------
    var_data: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        v  = r["run_id"].split("_")[0]
        # Aggregate A/A_strict → A, C/C_strict → C etc.
        v_key = v.split("_")[0]  # strip _strict suffix for display grouping
        # Keep full variant name for separation
        key = r["run_id"].split("_c")[0]   # e.g. A_strict, C, G
        d   = var_data.setdefault(key, {"solved": [], "failed": []})
        nodes = int(r["tree_nodes"])
        if r["solved"] == "True":
            d["solved"].append(nodes)
        else:
            d["failed"].append(nodes)

    # Sort variants by avg solved nodes
    var_keys = sorted(var_data.keys(),
                      key=lambda k: (sum(var_data[k]["solved"]) / max(len(var_data[k]["solved"]), 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    color_s = "#4472c4"
    color_f = "#c00000"

    # Left: overlapping histograms
    node_bins = range(1, 25, 1)
    ax1.hist(solved_nodes, bins=node_bins, color=color_s, alpha=0.65,
             label=f"Solved (n={len(solved_nodes)})", zorder=3)
    ax1.hist(failed_nodes, bins=node_bins, color=color_f, alpha=0.65,
             label=f"Failed (n={len(failed_nodes)})", zorder=3)
    med_s = float(np.median(solved_nodes)) if solved_nodes else 0
    med_f = float(np.median(failed_nodes)) if failed_nodes else 0
    ax1.axvline(med_s, color=color_s, linestyle="--", linewidth=1.8,
                label=f"Solved median={med_s:.0f}")
    ax1.axvline(med_f, color=color_f, linestyle="--", linewidth=1.8,
                label=f"Failed median={med_f:.0f}")
    ax1.set_xlabel("MCTS tree nodes explored", fontsize=11)
    ax1.set_ylabel("Instance count", fontsize=11)
    ax1.set_title("Search Effort: Solved vs Failed\n(nodes = total steps explored in tree)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: grouped bar per variant — avg nodes for solved vs failed
    x = np.arange(len(var_keys))
    width = 0.38
    avg_s = [sum(var_data[k]["solved"]) / max(len(var_data[k]["solved"]), 1)
             for k in var_keys]
    avg_f = [sum(var_data[k]["failed"]) / max(len(var_data[k]["failed"]), 1)
             for k in var_keys]
    n_s   = [len(var_data[k]["solved"]) for k in var_keys]
    n_f   = [len(var_data[k]["failed"]) for k in var_keys]

    bars_s = ax2.bar(x - width / 2, avg_s, width * 0.92, color=color_s, alpha=0.8,
                     label="Solved", zorder=3, edgecolor="white")
    bars_f = ax2.bar(x + width / 2, avg_f, width * 0.92, color=color_f, alpha=0.8,
                     label="Failed", zorder=3, edgecolor="white")

    for bar, val, n in zip(bars_s, avg_s, n_s):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                 f"{val:.1f}\n(n={n})", ha="center", va="bottom", fontsize=7.5, color=color_s)
    for bar, val, n in zip(bars_f, avg_f, n_f):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                 f"{val:.1f}\n(n={n})", ha="center", va="bottom", fontsize=7.5, color=color_f)

    ax2.set_xticks(x)
    ax2.set_xticklabels(var_keys, fontsize=9, rotation=20, ha="right")
    ax2.set_ylabel("Avg tree nodes explored", fontsize=11)
    ax2.set_title("Avg Search Effort by Variant\n(solved find answer early; failed exhaust budget)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("MCTS Search Behavior: Successful vs. Failed Instances",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = out_dir / f"fig_g_mcts_branching.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",   type=pathlib.Path, default=pathlib.Path("Summary_Data"),
                        help="Directory containing Summary_Data CSVs")
    parser.add_argument("--audit-csv",  type=pathlib.Path,
                        default=pathlib.Path("combined_results/reviewer_audits/audit_results.csv"))
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("combined_results/figures"))
    parser.add_argument("--format",     choices=["pdf", "png", "svg"], default="png")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not found — pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    print("Generating figures…")
    fig_efficiency_frontier(args.data_dir, args.output_dir, args.format)
    fig_reviewer_audit(args.audit_csv,     args.output_dir, args.format)
    fig_ablation_features(args.data_dir,   args.output_dir, args.format)
    fig_steps_to_solve(args.data_dir,      args.output_dir, args.format)
    fig_steps_to_solve_bpt(args.data_dir,  args.output_dir, args.format)
    fig_instance_overlap(args.data_dir,    args.output_dir, args.format)
    fig_resource_waste(args.data_dir,      args.output_dir, args.format)
    fig_resource_waste_by_arch(args.data_dir, args.output_dir, args.format)
    fig_mcts_branching(args.data_dir,      args.output_dir, args.format)
    print(f"\nAll figures written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
