#!/usr/bin/env python3
"""Generate paper figures from combined-agent ablation results.

Figures produced:
  fig1_main_results.pdf    — Grouped bar: A/B/C solve rate on c1, c2, c3
                             (the paper's headline comparison)
  fig2_ablation.pdf        — Horizontal bar: all variants ranked by avg solve rate,
                             color-coded by what each adds over the baseline
  fig3_efficiency.pdf      — Scatter: solve rate vs. estimated USD cost,
                             per variant (avg over available test sets)
  fig4_token_breakdown.pdf — Stacked bar: token share by role for key variants

Usage:
  python plot_results.py [--output-dir figures/] [--format pdf|png]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

# ---------------------------------------------------------------------------
# Pull eval data from eval_combined
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from eval_combined import load_all_runs, _aggregate, COMBINED_RUNS, RunSummary  # noqa: E402

# ---------------------------------------------------------------------------
# Variant metadata: display label, color group, description
# ---------------------------------------------------------------------------

VARIANT_META = {
    # id prefix         label                      group         description
    "K_c1": ("K  bare swe-search",            "baseline",   "Bare UCB1+value+hindsight, no our techniques"),
    "K_c2": ("K  bare swe-search",            "baseline",   ""),
    "A_c3": ("A  9b MCTS (ours)",             "ours_9b",    "Our techniques, 9b only"),
    "G_c1": ("G  9b + hindsight",             "ours_9b",    "9b + cross-branch feedback"),
    "G_c2": ("G  9b + hindsight",             "ours_9b",    ""),
    "H_c1": ("H  9b + value fn",              "ours_9b",    "9b self-eval value function"),
    "H_c2": ("H  9b + value fn",              "ours_9b",    ""),
    "I_c1": ("I  9b + full swe-search",       "swesearch",  "9b + value + hindsight (swe-search replication)"),
    "I_c2": ("I  9b + full swe-search",       "swesearch",  ""),
    "F_c1": ("F  9b coder + 120b planner",    "ours_mixed", "Mixed sizes, 9b coder, no value fn"),
    "F_c2": ("F  9b coder + 120b planner",    "ours_mixed", ""),
    "J_c1": ("J  30b flat + swe-search",      "swesearch",  "30b all roles + value + hindsight"),
    "J_c2": ("J  30b flat + swe-search",      "swesearch",  ""),
    "B_c1": ("B  Rafe linear (120b→30b)",     "baseline",   "Rafe's best: 120b plan + 30b coder, no search"),
    "B_c2": ("B  Rafe linear (120b→30b)",     "baseline",   ""),
    "B_c3": ("B  Rafe linear (120b→30b)",     "baseline",   ""),
    "D_c1": ("D  mixed + value fn",           "ours_mixed", "C + 30b value function"),
    "D_c2": ("D  mixed + value fn",           "ours_mixed", ""),
    "E_c1": ("E  mixed + value + hindsight",  "ours_mixed", "C + value + hindsight"),
    "E_c2": ("E  mixed + value + hindsight",  "ours_mixed", ""),
    "C_c1": ("C  mixed MCTS (ours best)",     "ours_best",  "Our best: 120b plan/review + 30b coder MCTS"),
    "C_c2": ("C  mixed MCTS (ours best)",     "ours_best",  ""),
    "C_c3": ("C  mixed MCTS (ours best)",     "ours_best",  ""),
}

GROUP_COLORS = {
    "baseline":   "#9b9b9b",   # grey
    "swesearch":  "#e8a838",   # amber
    "ours_9b":    "#5b9bd5",   # blue
    "ours_mixed": "#70ad47",   # green
    "ours_best":  "#c00000",   # red — our champion
}

GROUP_LABELS = {
    "baseline":   "Baselines",
    "swesearch":  "swe-search variants",
    "ours_9b":    "Our techniques (9b)",
    "ours_mixed": "Our techniques (mixed)",
    "ours_best":  "Our best (C)",
}

# Canonical ordering for ablation (worst → best)
ABLATION_ORDER = ["K","I","J","H","G","F","B","A","D","E","C"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prefix(run_id: str) -> str:
    """Return first token (e.g. 'C_c1_mixed_mcts' → 'C_c1')."""
    parts = run_id.split("_")
    # handle 'B_c1_rafe_linear' → 'B_c1'
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _set_label(run_id: str) -> str:
    p = _prefix(run_id)
    if "c1" in p: return "c1"
    if "c2" in p: return "c2"
    if "c3" in p: return "c3"
    return "?"


def _variant_letter(run_id: str) -> str:
    return run_id.split("_")[0]


def _build_lookup(summaries: list[RunSummary]) -> dict[str, RunSummary]:
    return {_prefix(s.run_id): s for s in summaries}


def _load_summaries_from_csv(path: pathlib.Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            summaries.append(
                RunSummary(
                    run_id=row["run_id"],
                    solved=int(row["solved"]),
                    total=int(row["total"]),
                    solve_rate=float(row["solve_rate"]),
                    avg_compute=float(row["avg_compute"]),
                    compute_per_solve=float(row["compute_per_solve"]) if row["compute_per_solve"] != "inf" else float("inf"),
                    avg_tokens_in=float(row["avg_tokens_in"]),
                    avg_tokens_out=float(row["avg_tokens_out"]),
                    avg_planner_tokens=float(row["avg_planner_tokens"]),
                    avg_coder_tokens=float(row["avg_coder_tokens"]),
                    avg_reviewer_tokens=float(row["avg_reviewer_tokens"]),
                    avg_value_tokens=float(row["avg_value_tokens"]),
                    avg_tree_nodes=float(row["avg_tree_nodes"]),
                    avg_best_depth=float(row["avg_best_depth"]),
                    avg_duration_s=float(row["avg_duration_s"]),
                    coder_model=row.get("coder_model", ""),
                    planner_model=row.get("planner_model", ""),
                )
            )
    return summaries


# ---------------------------------------------------------------------------
# Figure 1: Main results (A / B / C across c1, c2, c3)
# ---------------------------------------------------------------------------

def fig_main_results(summaries: list[RunSummary], out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    lkp = _build_lookup(summaries)
    variants = ["A", "B", "C"]
    sets     = ["c1", "c2", "c3"]
    set_labels = {"c1": "Custom Set 1\n(development)", "c2": "Custom Set 2\n(held-out dev)",
                  "c3": "Custom Set 3\n(held-out final)"}
    colors = {
        "A": "#5b9bd5",
        "B": "#9b9b9b",
        "C": "#c00000",
    }
    variant_labels = {
        "A": "A  9b MCTS (ours)",
        "B": "B  Rafe linear",
        "C": "C  Mixed MCTS (ours)",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sets))
    width = 0.22
    offsets = [-width, 0, width]

    for i, v in enumerate(variants):
        rates = []
        for s in sets:
            key = f"{v}_{s}"
            if key in lkp and lkp[key].total > 0:
                rates.append(lkp[key].solve_rate * 100)
            else:
                rates.append(None)

        xs, ys = zip(*[(x[j] + offsets[i], r) for j, r in enumerate(rates) if r is not None])
        bars = ax.bar(xs, ys, width=width * 0.92, color=colors[v], label=variant_labels[v],
                      zorder=3, edgecolor="white", linewidth=0.5)
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y + 1.2, f"{y:.0f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=colors[v])

    ax.set_xticks(x)
    ax.set_xticklabels([set_labels[s] for s in sets], fontsize=10)
    ax.set_ylabel("Solve rate (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shade c3 to indicate held-out
    ax.axvspan(1.5, 2.5, alpha=0.06, color="gold", zorder=0)
    ax.text(2, 102, "held-out", ha="center", fontsize=8, color="#aa8800", style="italic")

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("Solve Rate: Baseline (A, B) vs. Our Best (C)", fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    out = out_dir / f"fig1_main_results.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Full ablation bar chart
# ---------------------------------------------------------------------------

def fig_ablation(summaries: list[RunSummary], out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Average c1+c2 for each variant letter
    by_letter: dict[str, list[float]] = {}
    for s in summaries:
        letter = _variant_letter(s.run_id)
        test_set = _set_label(s.run_id)
        if test_set in ("c1", "c2") and s.total > 0:
            by_letter.setdefault(letter, []).append(s.solve_rate * 100)

    # Build ordered list (best solve rate first)
    rows = []
    for letter in sorted(by_letter.keys()):
        if letter not in by_letter:
            continue
        rates = by_letter[letter]
        avg = sum(rates) / len(rates)
        # Pair with average cost for tradeoff annotation
        costs = [
            s.avg_compute
            for s in summaries
            if _variant_letter(s.run_id) == letter and _set_label(s.run_id) in ("c1", "c2")
        ]
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        # Look up group for color
        sample_key = next((k for k in VARIANT_META if k.startswith(letter + "_c1")), None)
        group = VARIANT_META[sample_key][1] if sample_key else "baseline"
        label = VARIANT_META[sample_key][0] if sample_key else letter
        rows.append((label, avg, avg_cost, GROUP_COLORS[group], group))

    rows.sort(key=lambda r: r[1], reverse=True)

    if not rows:
        print("No ablation data available yet.")
        return

    labels, values, costs, colors, groups = zip(*rows)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labels) + 2))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    for bar, val, cost in zip(bars, values, costs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%  (${cost:.4f})", va="center", ha="left", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Avg solve rate, c1+c2 (%)", fontsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    seen = {}
    for g in groups:
        if g not in seen:
            seen[g] = mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
    ax.legend(handles=list(seen.values()), loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_title("Ablation: Avg Solve Rate (c1+c2), annotated with Avg Cost", fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    out = out_dir / f"fig2_ablation.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Efficiency scatter (solve rate vs. estimated USD cost)
# ---------------------------------------------------------------------------

def fig_efficiency(summaries: list[RunSummary], out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np

    # Average c1+c2 per variant
    by_letter: dict[str, dict] = {}
    for s in summaries:
        letter = _variant_letter(s.run_id)
        test_set = _set_label(s.run_id)
        if test_set in ("c1", "c2") and s.total > 0 and s.avg_compute > 0:
            d = by_letter.setdefault(letter, {"rates": [], "costs": []})
            d["rates"].append(s.solve_rate * 100)
            d["costs"].append(s.avg_compute)

    points = []
    for letter, d in by_letter.items():
        rate = sum(d["rates"]) / len(d["rates"])
        cost = sum(d["costs"]) / len(d["costs"])
        sample_key = next((k for k in VARIANT_META if k.startswith(letter + "_c1")), None)
        group = VARIANT_META[sample_key][1] if sample_key else "baseline"
        short_label = letter
        points.append((cost, rate, GROUP_COLORS[group], short_label, group))

    if not points:
        print("No efficiency data available.")
        return

    sorted_points = sorted(points, key=lambda x: x[0])
    costs, rates, colors, labels, groups = zip(*sorted_points)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(costs, rates, c=colors, s=130, zorder=4, edgecolors="white", linewidths=0.8)

    for x, y, c, lbl in zip(costs, rates, colors, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=9, fontweight="bold", color=c)

    # Efficiency frontier (Pareto-optimal points in cost-min, rate-max space)
    pts_sorted = sorted(zip(costs, rates, labels), key=lambda p: p[0])
    frontier_x, frontier_y = [pts_sorted[0][0]], [pts_sorted[0][1]]
    frontier_labels = [pts_sorted[0][2]]
    for cx, cy, lbl in pts_sorted[1:]:
        if cy >= frontier_y[-1]:
            frontier_x.append(cx)
            frontier_y.append(cy)
            frontier_labels.append(lbl)
    if len(frontier_x) > 1:
        ax.plot(frontier_x, frontier_y, "k--", alpha=0.25, linewidth=1.2, zorder=2, label="Pareto frontier")
        ax.scatter(frontier_x, frontier_y, marker="*", s=180, c="gold", edgecolors="black", linewidths=0.6, zorder=5)

    ax.set_xlabel("Avg estimated cost per instance (USD)", fontsize=11)
    ax.set_ylabel("Avg solve rate, c1+c2 (%)", fontsize=11)
    ax.set_ylim(40, 95)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    import matplotlib.patches as mpatches
    seen = {}
    for g in groups:
        if g not in seen:
            seen[g] = mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
    handles = list(seen.values())
    if len(frontier_x) > 1:
        handles.append(mlines.Line2D([], [], color="k", linestyle="--", alpha=0.4, label="Pareto frontier"))
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9)
    ax.set_title("Efficiency Frontier: Solve Rate vs. Cost (c1+c2 avg)", fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    out = out_dir / f"fig3_efficiency.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Token breakdown by role (key variants only)
# ---------------------------------------------------------------------------

def fig_token_breakdown(summaries: list[RunSummary], out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    KEY_VARIANTS = ["A", "B", "C", "F", "G", "K"]
    lkp = _build_lookup(summaries)

    rows = []
    for v in KEY_VARIANTS:
        # prefer c1, fallback c2
        s = lkp.get(f"{v}_c1") or lkp.get(f"{v}_c2")
        if s is None or s.total == 0:
            continue
        total_tok = s.avg_planner_tokens + s.avg_coder_tokens + s.avg_reviewer_tokens + s.avg_value_tokens
        if total_tok == 0:
            continue
        sample_key = next((k for k in VARIANT_META if k.startswith(v + "_c1")), None)
        label = VARIANT_META[sample_key][0] if sample_key else v
        rows.append((label, s.avg_planner_tokens, s.avg_coder_tokens,
                     s.avg_reviewer_tokens, s.avg_value_tokens))

    if not rows:
        print("No token breakdown data available.")
        return

    labels, plan_t, code_t, rev_t, val_t = zip(*rows) if rows else ([], [], [], [], [])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 0.65 * len(labels) + 2))
    totals = [p + c + r + v for p, c, r, v in zip(plan_t, code_t, rev_t, val_t)]

    b1 = ax.barh(y, plan_t, label="Planner",      color="#4472c4", edgecolor="white", linewidth=0.3)
    b2 = ax.barh(y, code_t, left=plan_t,           label="Coder",   color="#ed7d31", edgecolor="white", linewidth=0.3)
    left2 = [p + c for p, c in zip(plan_t, code_t)]
    b3 = ax.barh(y, rev_t, left=left2,             label="Reviewer",color="#a9d18e", edgecolor="white", linewidth=0.3)
    left3 = [l + r for l, r in zip(left2, rev_t)]
    b4 = ax.barh(y, val_t, left=left3,             label="Value fn",color="#ffc000", edgecolor="white", linewidth=0.3)

    for i, tot in enumerate(totals):
        ax.text(tot + tot * 0.01, y[i], f"{tot/1000:.0f}k", va="center", ha="left", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Avg tokens per instance", fontsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title("Token Usage by Role (avg per instance, c1)", fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    out = out_dir / f"fig4_token_breakdown.{fmt}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: set-by-set variant heatmap
# ---------------------------------------------------------------------------

def fig_set_heatmap(summaries: list[RunSummary], out_dir: pathlib.Path, fmt: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    variants = sorted({_variant_letter(s.run_id) for s in summaries if _set_label(s.run_id) in ("c1", "c2", "c3")})
    sets = ["c1", "c2", "c3"]
    matrix = []
    for v in variants:
        row = []
        for st in sets:
            s = next((x for x in summaries if _variant_letter(x.run_id) == v and _set_label(x.run_id) == st), None)
            row.append((s.solve_rate * 100.0) if s else np.nan)
        matrix.append(row)

    if not matrix:
        print("No set heatmap data available.")
        return

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(variants) + 2))
    im = ax.imshow(arr, cmap="YlGn", aspect="auto", vmin=45, vmax=90)
    ax.set_xticks(np.arange(len(sets)))
    ax.set_xticklabels(sets)
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xlabel("Case set")
    ax.set_ylabel("Variant")
    ax.set_title("Per-set solve rates by variant (%)", fontsize=12, fontweight="bold", pad=10)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Solve rate (%)")

    fig.tight_layout()
    out = out_dir / f"fig5_set_heatmap.{fmt}"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("figures"))
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument("--combined-root", type=pathlib.Path, default=COMBINED_RUNS)
    parser.add_argument("--results-csv", type=pathlib.Path, default=None,
                        help="Use an existing results.csv/final_results.csv instead of loading runs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.results_csv:
        print(f"Loading summaries from CSV: {args.results_csv}", file=sys.stderr)
        summaries = _load_summaries_from_csv(args.results_csv)
    else:
        print("Loading runs…", file=sys.stderr)
        all_runs = load_all_runs(args.combined_root)
        summaries = [_aggregate(instances) for instances in all_runs.values()]

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not found — install it with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    print("Generating figures…")
    fig_main_results(summaries,    args.output_dir, args.format)
    fig_ablation(summaries,        args.output_dir, args.format)
    fig_efficiency(summaries,      args.output_dir, args.format)
    fig_token_breakdown(summaries, args.output_dir, args.format)
    fig_set_heatmap(summaries,     args.output_dir, args.format)
    print(f"\nAll figures written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
