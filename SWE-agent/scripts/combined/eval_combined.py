#!/usr/bin/env python3
"""Unified evaluation script for combined-agent ablation runs.

Reads .traj files from tree_search_runs/combined/ and computes per-run:
  - Solve rate
  - Token usage by role (planner, coder, reviewer, value fn)
  - Real USD cost (Together AI API rates for the specific models used)
  - MCTS tree stats: nodes, depth, avg branching factor, max branches
  - Reviewer model used

Real pricing (Together AI, April 2026):
  gpt-oss-120b          $0.15/M input   $0.60/M output
  Qwen3-VL-30B-A3B      $0.08/M input   $0.28/M output
  Qwen3.5 9B            $0.10/M input   $0.15/M output

Usage:
  python eval_combined.py                          # all runs
  python eval_combined.py --csv results.csv
  python eval_combined.py --run C_c1
  python eval_combined.py --latex tables.tex
  python eval_combined.py --json results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
COMBINED_RUNS   = ROOT / "SWE-agent/tree_search_runs/combined"
RAFE_BENCH      = ROOT / "SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud"
RAFE_BEST       = RAFE_BENCH / "umich_gptoss_planner_umich_qwen_coder/planner_coder"
CASES_C1        = ROOT / "SWE-agent/custom_cases"
CASES_C2        = ROOT / "SWE-agent/custom_cases_2"
CASES_C3        = ROOT / "SWE-agent/custom_cases_3"
# Legacy MCTS baselines (A_c1, A_c2) from original tree_search runner
LEGACY_A_C1     = ROOT / "SWE-agent/tree_search_runs/all_custom_run_v10"
LEGACY_A_C2     = ROOT / "SWE-agent/tree_search_runs/custom_cases_2_baseline_9b"

# ---------------------------------------------------------------------------
# Cost model — real USD pricing (Together AI, April 2026)
# Source: https://www.together.ai/pricing
# ---------------------------------------------------------------------------
# Each entry: (substring to match in model name, $/M input, $/M output)
_PRICE_TABLE: list[tuple[str, float, float]] = [
    ("gpt-oss-120b",    0.15, 0.60),   # Together AI exact listing
    ("Qwen3-VL-30B",    0.08, 0.28),   # Qwen3 30B A3B MoE (OpenRouter/direct)
    ("qwen3.5:9b",      0.10, 0.15),   # Together AI Qwen3.5 9B equivalent
    ("qwen3.5-9b",      0.10, 0.15),
    ("qwen3",           0.10, 0.15),   # fallback for any small Qwen3 variant
    # Legacy OpenAI models (not used in our runs but kept for completeness)
    ("gpt-4o-mini",     0.15, 0.60),
    ("gpt-4o",          2.50, 10.00),
]
# Fallback: size-based approximation when no exact match found
# Uses Llama-3.1 pricing tier from Together AI as reference
_FALLBACK_PRICE_BY_SIZE: list[tuple[float, float, float]] = [
    # (max_size_b, $/M input, $/M output)
    (10.0,  0.10, 0.15),   # ≤10b
    (40.0,  0.10, 0.30),   # ≤40b
    (80.0,  0.88, 0.88),   # ≤80b  (Llama 3.3 70B tier)
    (999.0, 0.90, 0.90),   # >80b  (Llama 3.1 405B tier)
]


def _model_size_b(model_name: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    return float(m.group(1)) if m else None


def _compute_cost_usd(tokens_in: int, tokens_out: int, model_name: str) -> float:
    """Return USD cost for the given token counts and model.

    Prices are per-million tokens from Together AI (April 2026).
    Returns dollars (e.g. 0.003 = $0.003 = 0.3 cents).
    """
    for substr, price_in, price_out in _PRICE_TABLE:
        if substr in model_name:
            return (tokens_in * price_in + tokens_out * price_out) / 1_000_000

    size_b = _model_size_b(model_name) or 0.0
    for max_b, price_in, price_out in _FALLBACK_PRICE_BY_SIZE:
        if size_b <= max_b:
            return (tokens_in * price_in + tokens_out * price_out) / 1_000_000

    return (tokens_in * 0.90 + tokens_out * 0.90) / 1_000_000


# ---------------------------------------------------------------------------
# Per-instance result
# ---------------------------------------------------------------------------

@dataclass
class InstanceResult:
    instance_id: str
    run_id: str
    solved: bool
    stopped_reason: str = ""
    duration_s: float = 0.0
    # tokens
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    planner_tokens_in: int = 0
    planner_tokens_out: int = 0
    coder_tokens_in: int = 0
    coder_tokens_out: int = 0
    reviewer_tokens_in: int = 0
    reviewer_tokens_out: int = 0
    value_tokens_in: int = 0
    value_tokens_out: int = 0
    # models
    coder_model: str = ""
    planner_model: str = ""
    reviewer_model: str = ""
    # MCTS tree stats
    tree_nodes: int = 0
    best_node_depth: int = 0
    best_node_value: float = 0.0
    mcts_iterations: int = 0
    avg_branching_factor: float = 0.0   # avg children per non-leaf node
    max_branches: int = 0               # max children at any single node
    # cost (real USD at Together AI April 2026 rates)
    cost_usd: float = 0.0


def _role_tokens(stats: dict, in_key: str = "tokens_in", out_key: str = "tokens_out") -> tuple[int, int]:
    tin  = stats.get(in_key,  0) or stats.get("input_tokens",  0)
    tout = stats.get(out_key, 0) or stats.get("output_tokens", 0)
    return int(tin), int(tout)


def _parse_traj(traj_path: pathlib.Path, run_id: str) -> InstanceResult:
    d = json.loads(traj_path.read_text())
    instance_id = d.get("instance_id", traj_path.stem)
    solved       = bool(d.get("submitted") or d.get("info", {}).get("submitted"))
    stopped      = d.get("stopped_reason", "")
    duration     = float(d.get("duration_seconds", 0.0))

    stats = d.get("stats", {})
    total_in  = int(stats.get("input_tokens",  0))
    total_out = int(stats.get("output_tokens", 0))

    rms = d.get("role_model_stats", {})
    p_stats  = rms.get("planner",  {})
    c_stats  = rms.get("coder",    {})
    r_stats  = rms.get("reviewer", {})

    p_in,  p_out  = _role_tokens(p_stats)
    c_in,  c_out  = _role_tokens(c_stats, "input_tokens", "output_tokens")
    rv_in, rv_out = _role_tokens(r_stats)
    v_in  = int(stats.get("value_fn_tokens_in",  c_stats.get("value_fn_tokens_in",  0)))
    v_out = int(stats.get("value_fn_tokens_out", c_stats.get("value_fn_tokens_out", 0)))

    coder_model    = c_stats.get("model", "")
    planner_model  = p_stats.get("model", "")
    reviewer_model = r_stats.get("model", "")

    tree_nodes   = int(stats.get("tree_nodes_created", c_stats.get("tree_nodes_created", 0)))
    best_depth   = int(stats.get("best_node_depth",    c_stats.get("best_node_depth",    0)))
    best_value   = float(stats.get("best_node_value",  c_stats.get("best_node_value",    0.0)))
    iterations   = int(stats.get("iterations",         c_stats.get("iterations",         0)))

    # Branch statistics from MCTS tree
    avg_branching = 0.0
    max_branches  = 0
    mcts_tree = d.get("mcts_tree", {})
    tree_node_list = mcts_tree.get("nodes", [])
    if tree_node_list:
        from collections import Counter
        parent_counts = Counter(
            n["parent_id"] for n in tree_node_list if n.get("parent_id") is not None
        )
        if parent_counts:
            avg_branching = sum(parent_counts.values()) / len(parent_counts)
            max_branches  = max(parent_counts.values())

    # Real USD cost at Together AI April 2026 rates
    cost_usd = 0.0
    for model, tin, tout in [
        (coder_model,    c_in,  c_out),
        (planner_model,  p_in,  p_out),
        (reviewer_model, rv_in, rv_out),
    ]:
        if model and (tin or tout):
            cost_usd += _compute_cost_usd(tin, tout, model)
    if v_in or v_out:
        cost_usd += _compute_cost_usd(v_in, v_out, coder_model or "qwen3.5:9b")

    return InstanceResult(
        instance_id=instance_id, run_id=run_id,
        solved=solved, stopped_reason=stopped, duration_s=duration,
        total_tokens_in=total_in, total_tokens_out=total_out,
        planner_tokens_in=p_in,  planner_tokens_out=p_out,
        coder_tokens_in=c_in,    coder_tokens_out=c_out,
        reviewer_tokens_in=rv_in, reviewer_tokens_out=rv_out,
        value_tokens_in=v_in,    value_tokens_out=v_out,
        coder_model=coder_model, planner_model=planner_model, reviewer_model=reviewer_model,
        tree_nodes=tree_nodes, best_node_depth=best_depth,
        best_node_value=best_value, mcts_iterations=iterations,
        avg_branching_factor=avg_branching, max_branches=max_branches,
        cost_usd=cost_usd,
    )


# ---------------------------------------------------------------------------
# Rafe B_c1 evaluation (patch eval against case.json success_checks)
# ---------------------------------------------------------------------------

def _apply_and_check(patch: str, success_checks: list, repo_path: pathlib.Path) -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_repo = pathlib.Path(tmpdir) / "repo"
        shutil.copytree(repo_path, tmp_repo)
        subprocess.run(["git", "init"], cwd=tmp_repo, capture_output=True)
        patch_file = pathlib.Path(tmpdir) / "changes.patch"
        patch_file.write_text(patch + "\n")
        r = subprocess.run(
            ["git", "apply", "--recount", "--ignore-whitespace", str(patch_file)],
            capture_output=True, text=True, cwd=tmp_repo,
        )
        if r.returncode != 0:
            return False
        for check in success_checks:
            r = subprocess.run(check["command"], shell=True, capture_output=True, text=True, cwd=tmp_repo)
            out = r.stdout + r.stderr
            if r.returncode != check.get("expect_exit_code", 0): return False
            if any(s not in out for s in check.get("stdout_contains",     [])): return False
            if any(s in out     for s in check.get("stdout_not_contains", [])): return False
        return True


def _eval_rafe_b_c1() -> list[InstanceResult]:
    if not RAFE_BEST.exists():
        return []
    results = []
    # Cost model for Rafe's run: 120b planner/reviewer + 30b coder
    planner_model  = "openai/openai/gpt-oss-120b"
    coder_model    = "openai/Qwen/Qwen3-VL-30B-A3B-Instruct"
    reviewer_model = planner_model

    for inst_dir in sorted(RAFE_BEST.iterdir()):
        if not inst_dir.is_dir():
            continue
        case_name = inst_dir.name
        pred_files = list(inst_dir.rglob("*.pred"))
        traj_files = list(inst_dir.rglob("*.traj"))
        if not pred_files:
            continue
        pred  = json.loads(pred_files[0].read_text())
        patch = pred.get("model_patch", "").strip()

        case_json = CASES_C1 / case_name / "case.json"
        solved = False
        if patch and case_json.exists():
            case_data = json.loads(case_json.read_text())[0]
            repo_path = CASES_C1 / case_name / case_data["repo_path"]
            solved = _apply_and_check(patch, case_data["evaluation"]["success_checks"], repo_path)

        # Token stats from traj if available
        p_in = c_in = rv_in = p_out = c_out = rv_out = 0
        duration = 0.0
        if traj_files:
            td = json.loads(traj_files[0].read_text())
            rms = td.get("role_model_stats", {})
            p_in,  p_out  = _role_tokens(rms.get("planner",  {}))
            c_in,  c_out  = _role_tokens(rms.get("coder",    {}), "input_tokens", "output_tokens")
            rv_in, rv_out = _role_tokens(rms.get("reviewer", {}))
            duration      = float(td.get("duration_seconds", 0.0))

        cost = (_compute_cost(p_in,  p_out,  planner_model) +
                _compute_cost(c_in,  c_out,  coder_model)   +
                _compute_cost(rv_in, rv_out, reviewer_model))

        results.append(InstanceResult(
            instance_id=f"{case_name}_001", run_id="B_c1_rafe_linear",
            solved=solved, duration_s=duration,
            planner_tokens_in=p_in,   planner_tokens_out=p_out,
            coder_tokens_in=c_in,     coder_tokens_out=c_out,
            reviewer_tokens_in=rv_in, reviewer_tokens_out=rv_out,
            total_tokens_in=p_in+c_in+rv_in, total_tokens_out=p_out+c_out+rv_out,
            coder_model=coder_model, planner_model=planner_model, reviewer_model=reviewer_model,
            relative_compute=cost,
        ))
    return results


# ---------------------------------------------------------------------------
# Legacy A baseline (9b MCTS, original tree_search runner)
# ---------------------------------------------------------------------------

def _eval_legacy_mcts(run_dir: pathlib.Path, run_id: str) -> list[InstanceResult]:
    """Load traj files from the original tree_search runner for A_c1/A_c2."""
    if not run_dir.exists():
        return []
    results = []
    for case_dir in sorted(run_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        trajs = list(case_dir.glob("*.traj"))
        if not trajs:
            continue
        results.append(_parse_traj(trajs[0], run_id))
    return results


# ---------------------------------------------------------------------------
# Aggregate per run
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    run_id: str
    solved: int
    total: int
    solve_rate: float
    avg_compute: float
    compute_per_solve: float
    avg_tokens_in: float
    avg_tokens_out: float
    avg_planner_tokens: float
    avg_coder_tokens: float
    avg_reviewer_tokens: float
    avg_value_tokens: float
    avg_tree_nodes: float
    avg_best_depth: float
    avg_duration_s: float
    coder_model: str
    planner_model: str


def _aggregate(instances: list[InstanceResult]) -> RunSummary:
    n = len(instances)
    if n == 0:
        return RunSummary(run_id="", solved=0, total=0, solve_rate=0, avg_compute=0,
                          compute_per_solve=0, avg_tokens_in=0, avg_tokens_out=0,
                          avg_planner_tokens=0, avg_coder_tokens=0, avg_reviewer_tokens=0,
                          avg_value_tokens=0, avg_tree_nodes=0, avg_best_depth=0,
                          avg_duration_s=0, coder_model="", planner_model="")
    solved = sum(1 for i in instances if i.solved)
    solve_rate = solved / n
    avg_compute = sum(i.relative_compute for i in instances) / n
    compute_per_solve = avg_compute / solve_rate if solve_rate > 0 else float("inf")

    def avg(fn): return sum(fn(i) for i in instances) / n

    models = [i for i in instances if i.coder_model]
    return RunSummary(
        run_id=instances[0].run_id,
        solved=solved, total=n, solve_rate=solve_rate,
        avg_compute=avg_compute, compute_per_solve=compute_per_solve,
        avg_tokens_in=avg(lambda i: i.total_tokens_in),
        avg_tokens_out=avg(lambda i: i.total_tokens_out),
        avg_planner_tokens=avg(lambda i: i.planner_tokens_in + i.planner_tokens_out),
        avg_coder_tokens=avg(lambda i: i.coder_tokens_in + i.coder_tokens_out),
        avg_reviewer_tokens=avg(lambda i: i.reviewer_tokens_in + i.reviewer_tokens_out),
        avg_value_tokens=avg(lambda i: i.value_tokens_in + i.value_tokens_out),
        avg_tree_nodes=avg(lambda i: i.tree_nodes),
        avg_best_depth=avg(lambda i: i.best_node_depth),
        avg_duration_s=avg(lambda i: i.duration_s),
        coder_model=models[0].coder_model if models else "",
        planner_model=models[0].planner_model if models else "",
    )


# ---------------------------------------------------------------------------
# Load all runs
# ---------------------------------------------------------------------------

def load_all_runs(combined_root: pathlib.Path, run_filter: str | None = None
                  ) -> dict[str, list[InstanceResult]]:
    runs: dict[str, list[InstanceResult]] = {}

    def _include(key: str) -> bool:
        return run_filter is None or key.upper().startswith(run_filter.upper())

    # A_c1 / A_c2 from legacy tree_search runner
    if _include("A_c1"):
        a1 = _eval_legacy_mcts(LEGACY_A_C1, "A_c1_9b_mcts")
        if a1:
            runs["A_c1_9b_mcts"] = a1
    if _include("A_c2"):
        a2 = _eval_legacy_mcts(LEGACY_A_C2, "A_c2_9b_mcts")
        if a2:
            runs["A_c2_9b_mcts"] = a2

    # B_c1 from Rafe's pred files
    if _include("B_c1"):
        b1 = _eval_rafe_b_c1()
        if b1:
            runs["B_c1_rafe_linear"] = b1

    # Combined traj runs (covers A_c3, B_c2, B_c3, C through K)
    if combined_root.exists():
        for run_dir in sorted(combined_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_filter and not run_dir.name.upper().startswith(run_filter.upper()):
                continue
            trajs = list(run_dir.rglob("*.traj"))
            if not trajs:
                continue
            instances = [_parse_traj(t, run_dir.name) for t in sorted(trajs)]
            runs[run_dir.name] = instances

    return runs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(summaries: list[RunSummary]) -> None:
    hdr = f"{'Run':<40}  {'Solved':>6}  {'Total':>5}  {'Rate':>6}  {'AvgCost':>9}  {'Cost/Solve':>10}  {'AvgTok':>7}  {'Nodes':>5}  {'Depth':>5}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for s in summaries:
        rate  = f"{s.solve_rate*100:.0f}%"
        cost  = f"{s.avg_compute:.0f}"
        cps   = f"{s.compute_per_solve:.0f}" if s.compute_per_solve < 1e9 else "—"
        tok   = f"{(s.avg_tokens_in+s.avg_tokens_out)/1000:.1f}k"
        nodes = f"{s.avg_tree_nodes:.1f}"
        depth = f"{s.avg_best_depth:.1f}"
        print(f"{s.run_id:<40}  {s.solved:>6}/{s.total:<4}  {rate:>6}  {cost:>9}  {cps:>10}  {tok:>7}  {nodes:>5}  {depth:>5}")
    print()


def export_csv(summaries: list[RunSummary], path: pathlib.Path) -> None:
    fields = list(asdict(summaries[0]).keys()) if summaries else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow(asdict(s))
    print(f"CSV written: {path}")


def export_json(all_runs: dict[str, list[InstanceResult]], path: pathlib.Path) -> None:
    data = {run_id: [asdict(i) for i in instances] for run_id, instances in all_runs.items()}
    path.write_text(json.dumps(data, indent=2))
    print(f"JSON written: {path}")


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

# Human-readable metadata for each variant letter
_VARIANT_INFO: dict[str, tuple[str, str]] = {
    # letter: (short description, role group)
    "A": ("9b MCTS (ours)",                 "Ours (9b)"),
    "B": ("Rafe linear (120b plan + 30b code)", "Baseline"),
    "C": ("Mixed MCTS — 120b plan/review + 30b code", "Ours (best)"),
    "D": ("C + 30b LLM value function",     "Ours (mixed)"),
    "E": ("C + value fn + hindsight",        "Ours (mixed)"),
    "F": ("9b code + 120b plan, no search",  "Ours (9b)"),
    "G": ("9b MCTS + hindsight feedback",    "Ours (9b)"),
    "H": ("9b MCTS + self-eval value fn",    "Ours (9b)"),
    "I": ("9b full swe-search replica",      "swe-search"),
    "J": ("30b flat + full swe-search",      "swe-search"),
    "K": ("Bare UCB1 (minimal swe-search)",  "swe-search"),
}

# Ablation ordering (best → worst for main table)
_ABL_ORDER = ["C", "E", "D", "B", "A", "F", "G", "H", "J", "I", "K"]


def _lkp_run(summaries: list[RunSummary]) -> dict[str, RunSummary]:
    """Index summaries by their two-part prefix (e.g. 'C_c1')."""
    out: dict[str, RunSummary] = {}
    for s in summaries:
        parts = s.run_id.split("_")
        key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
        out[key] = s
    return out


def _rate_cell(s: RunSummary | None) -> str:
    if s is None or s.total == 0:
        return "---"
    return f"{s.solve_rate * 100:.0f}\\%"


def _cost_cell(s: RunSummary | None) -> str:
    if s is None or s.avg_compute == 0:
        return "---"
    return f"{s.avg_compute / 1000:.0f}k"


def export_latex(summaries: list[RunSummary], path: pathlib.Path) -> None:
    lkp = _lkp_run(summaries)

    lines: list[str] = []
    lines.append("% Auto-generated by eval_combined.py — do not edit by hand")
    lines.append("% Requires \\usepackage{booktabs} in preamble")
    lines.append("")

    # ── Table 1: Main results (A / B / C across c1 / c2 / c3) ──────────────
    lines.append("% ============================================================")
    lines.append("% TABLE 1: Main results — baseline vs. our best across test sets")
    lines.append("% Usage: \\input{<this file>}  inside a table environment")
    lines.append("% ============================================================")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Solve rates on three independent test sets.")
    lines.append("    \\textbf{c3} is fully held out.")
    lines.append("    Cost is model-size-weighted token count ($\\times 10^3$).}")
    lines.append("  \\label{tab:main_results}")
    lines.append("  \\begin{tabular}{llrrrrr}")
    lines.append("    \\toprule")
    lines.append("    Sys & Description & c1 & c2 & c3 & Avg cost (k) & \\# nodes \\\\")
    lines.append("    \\midrule")

    for letter in ["A", "B", "C"]:
        desc, _ = _VARIANT_INFO.get(letter, ("", ""))
        cells = []
        costs = []
        nodes_vals = []
        for cs in ["c1", "c2", "c3"]:
            s = lkp.get(f"{letter}_{cs}")
            cells.append(_rate_cell(s))
            if s and s.avg_compute > 0:
                costs.append(s.avg_compute)
            if s and s.avg_tree_nodes > 0:
                nodes_vals.append(s.avg_tree_nodes)
        avg_cost = f"{sum(costs)/len(costs)/1000:.0f}k" if costs else "---"
        avg_nodes = f"{sum(nodes_vals)/len(nodes_vals):.1f}" if nodes_vals else "---"
        row = f"    {letter} & {desc} & {' & '.join(cells)} & {avg_cost} & {avg_nodes} \\\\"
        lines.append(row)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # ── Table 2: Full ablation (all variants, avg c1+c2) ────────────────────
    lines.append("% ============================================================")
    lines.append("% TABLE 2: Ablation — all variants, averaged over c1 + c2")
    lines.append("% ============================================================")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Ablation results averaged over test sets c1 and c2.")
    lines.append("    Solve rate is the fraction of 20 instances solved.")
    lines.append("    Cost is model-size-weighted token count ($\\times 10^3$).")
    lines.append("    Nodes and depth are MCTS tree statistics (--- for linear runs).}")
    lines.append("  \\label{tab:ablation}")
    lines.append("  \\begin{tabular}{llllrrrrr}")
    lines.append("    \\toprule")
    lines.append("    Var & Description & Group & c1 & c2 & Avg\\,\\% & Avg\\,cost & Nodes & Depth \\\\")
    lines.append("    \\midrule")

    prev_group = None
    for letter in _ABL_ORDER:
        s_c1 = lkp.get(f"{letter}_c1")
        s_c2 = lkp.get(f"{letter}_c2")
        if s_c1 is None and s_c2 is None:
            continue

        desc, group = _VARIANT_INFO.get(letter, ("", ""))
        if group != prev_group and prev_group is not None:
            lines.append("    \\midrule")
        prev_group = group

        r1 = _rate_cell(s_c1)
        r2 = _rate_cell(s_c2)

        rates = [s.solve_rate * 100 for s in [s_c1, s_c2] if s and s.total > 0]
        avg_rate = f"{sum(rates)/len(rates):.0f}\\%" if rates else "---"

        costs = [s.avg_compute for s in [s_c1, s_c2] if s and s.avg_compute > 0]
        avg_cost = f"{sum(costs)/len(costs)/1000:.0f}k" if costs else "---"

        nodes = [s.avg_tree_nodes for s in [s_c1, s_c2] if s and s.avg_tree_nodes > 0]
        avg_nodes = f"{sum(nodes)/len(nodes):.1f}" if nodes else "---"

        depths = [s.avg_best_depth for s in [s_c1, s_c2] if s and s.avg_best_depth > 0]
        avg_depth = f"{sum(depths)/len(depths):.1f}" if depths else "---"

        lines.append(f"    {letter} & {desc} & {group} & {r1} & {r2} & {avg_rate} & {avg_cost} & {avg_nodes} & {avg_depth} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"LaTeX tables written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combined-root", type=pathlib.Path, default=COMBINED_RUNS)
    parser.add_argument("--run", default=None, help="Filter runs by prefix (e.g. C_c1)")
    parser.add_argument("--csv",   type=pathlib.Path, default=None)
    parser.add_argument("--json",  type=pathlib.Path, default=None)
    parser.add_argument("--latex", type=pathlib.Path, default=None,
                        help="Write two booktabs LaTeX tables (main results + ablation)")
    args = parser.parse_args()

    print("Loading runs…", file=sys.stderr)
    all_runs = load_all_runs(args.combined_root, args.run)

    summaries = [_aggregate(instances) for instances in all_runs.values()]
    summaries.sort(key=lambda s: s.run_id)

    print_report(summaries)

    if args.csv:
        export_csv(summaries, args.csv)
    if args.json:
        export_json(all_runs, args.json)
    if args.latex:
        export_latex(summaries, args.latex)


if __name__ == "__main__":
    main()
