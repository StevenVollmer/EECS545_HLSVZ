#!/usr/bin/env python3
"""Unified evaluation script for combined-agent ablation runs.

Reads .traj files from tree_search_runs/combined/ and combined_results/ and computes:
  - Solve rate
  - Token usage by role (planner, coder, reviewer, value fn)
  - Real USD cost (Together AI API rates for the specific models used)
  - BPT cost: parameter-normalized tokens (B-param·tokens, proportional to FLOPs)
  - MCTS tree stats: nodes, depth, avg branching factor, max branches
  - Reviewer model used

Real pricing (Together AI, April 2026):
  gpt-oss-120b          $0.15/M input   $0.60/M output
  Qwen3-VL-30B-A3B      $0.08/M input   $0.28/M output
  Qwen3.5 9B            $0.10/M input   $0.15/M output

BPT metric: cost_bpt = Σ_role (tokens_in + tokens_out) × params_B
  where params_B: 9b→9, 30b→30, 120b→120
  Proportional to FLOPs (≈ FLOPs/2). Paper axis: "Estimated FLOPs (B-param·tokens)"

Usage:
  python eval_combined.py                          # all runs, terminal report
  python eval_combined.py --csv results.csv
  python eval_combined.py --run C_c1
  python eval_combined.py --latex tables.tex
  python eval_combined.py --json results.json
  python eval_combined.py --instance-csv final_instance_metrics.csv
  python eval_combined.py --efficiency-csv final_efficiency_accuracy.csv
  python eval_combined.py --frontier-csv final_pareto_frontier.csv
  python eval_combined.py --efficiency-frontier-csv efficiency_frontier_bpt.csv
  python eval_combined.py --steps-csv steps_to_solve.csv
  python eval_combined.py --overlap-csv instance_overlap.csv
  python eval_combined.py --waste-csv resource_waste.csv
  python eval_combined.py --ablation-csv ablation_features.csv
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
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
COMBINED_RUNS    = ROOT / "SWE-agent/tree_search_runs/combined"
COMBINED_RESULTS = ROOT / "combined_results"
RAFE_BENCH       = ROOT / "SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud"
RAFE_BEST        = RAFE_BENCH / "umich_gptoss_planner_umich_qwen_coder/planner_coder"
CASES_C1         = ROOT / "SWE-agent/custom_cases"
CASES_C2         = ROOT / "SWE-agent/custom_cases_2"
CASES_C3         = ROOT / "SWE-agent/custom_cases_3"
LEGACY_A_C1      = ROOT / "SWE-agent/tree_search_runs/all_custom_run_v10"
FINAL_MATRIX     = ROOT / "SWE-agent/custom_matrix_runs/final_matrix"

# First 20 alphabetical instances in final_matrix = original c1 set.
# The remaining 7 (shipment_preview … workspace_digest) are new cases not yet run elsewhere.
_FINAL_MATRIX_C1_CASES: frozenset[str] = frozenset([
    "board_rollup", "budget_snapshot", "contact_card", "digest_preview",
    "incident_brief", "invoice_footer", "label_formatter", "median_window",
    "milestone_rollup", "nested_app", "numeric_drift_sum", "owner_recap",
    "owner_sort", "pagination_drift", "path_normalizer_cache", "priority_snapshot",
    "renewal_preview", "retry_cap", "risk_score", "search_hit_localize",
])
LEGACY_A_C2      = ROOT / "SWE-agent/tree_search_runs/custom_cases_2_baseline_9b"

# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------

_PRICE_TABLE: list[tuple[str, float, float]] = [
    ("gpt-oss-120b",    0.15, 0.60),
    ("Qwen3-VL-30B",    0.08, 0.28),
    ("qwen3.5:9b",      0.10, 0.15),
    ("qwen3.5-9b",      0.10, 0.15),
    ("qwen3",           0.10, 0.15),
    ("gpt-4o-mini",     0.15, 0.60),
    ("gpt-4o",          2.50, 10.00),
]
_FALLBACK_PRICE_BY_SIZE: list[tuple[float, float, float]] = [
    (10.0,  0.10, 0.15),
    (40.0,  0.10, 0.30),
    (80.0,  0.88, 0.88),
    (999.0, 0.90, 0.90),
]


def _model_size_b(model_name: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    return float(m.group(1)) if m else None


def _compute_cost_usd(tokens_in: int, tokens_out: int, model_name: str) -> float:
    for substr, price_in, price_out in _PRICE_TABLE:
        if substr in model_name:
            return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
    size_b = _model_size_b(model_name) or 0.0
    for max_b, price_in, price_out in _FALLBACK_PRICE_BY_SIZE:
        if size_b <= max_b:
            return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
    return (tokens_in * 0.90 + tokens_out * 0.90) / 1_000_000


def _compute_cost_bpt(
    planner_in: int, planner_out: int, planner_model: str,
    coder_in: int, coder_out: int, coder_model: str,
    reviewer_in: int, reviewer_out: int, reviewer_model: str,
    value_in: int, value_out: int,
) -> float:
    """Parameter-normalized token cost (B-param·tokens ≈ FLOPs/2)."""
    def _p(model: str) -> float:
        return _model_size_b(model) or 9.0
    total = (planner_in  + planner_out)  * _p(planner_model)
    total += (coder_in    + coder_out)    * _p(coder_model)
    total += (reviewer_in + reviewer_out) * _p(reviewer_model)
    total += (value_in    + value_out)    * _p(coder_model)   # value fn uses coder model
    return total


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
    steps_used: int = 0           # actual turns/iterations executed (from stats["turns"])
    avg_branching_factor: float = 0.0
    max_branches: int = 0
    # cost
    cost_usd: float = 0.0
    cost_bpt: float = 0.0         # B-param·tokens (proportional to FLOPs)


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
    steps_used = int(stats.get("turns", 0))

    rms = d.get("role_model_stats", {})
    p_stats  = rms.get("planner",  {})
    c_stats  = rms.get("coder",    {})
    r_stats  = rms.get("reviewer", {})
    pc_stats = rms.get("plan_critic", {})  # Rafe's plan-critic role

    p_in,  p_out  = _role_tokens(p_stats)
    c_in,  c_out  = _role_tokens(c_stats, "input_tokens", "output_tokens")
    rv_in, rv_out = _role_tokens(r_stats)
    # plan_critic uses same model as planner (120b); fold into reviewer slot for BPT
    pc_in, pc_out = _role_tokens(pc_stats)
    rv_in  += pc_in
    rv_out += pc_out
    v_in  = int(stats.get("value_fn_tokens_in",  c_stats.get("value_fn_tokens_in",  0)))
    v_out = int(stats.get("value_fn_tokens_out", c_stats.get("value_fn_tokens_out", 0)))

    coder_model    = c_stats.get("model", "")
    planner_model  = p_stats.get("model", "")
    reviewer_model = r_stats.get("model", "") or pc_stats.get("model", "")

    tree_nodes   = int(stats.get("tree_nodes_created", c_stats.get("tree_nodes_created", 0)))
    best_depth   = int(stats.get("best_node_depth",    c_stats.get("best_node_depth",    0)))
    best_value   = float(stats.get("best_node_value",  c_stats.get("best_node_value",    0.0)))
    iterations   = int(stats.get("iterations",         c_stats.get("iterations",         0)))

    # Multi-reviewer-round correction: top-level stats only record the final round.
    # Sum tree_nodes, turns, and total tokens across all rounds when coder_rounds is present.
    round_stats = [r.get("stats", {}) for r in d.get("coder_rounds", [])
                   if isinstance(r.get("stats"), dict) and r.get("stats")]
    if len(round_stats) > 1:
        steps_used = sum(rs.get("turns",              0) or 0 for rs in round_stats)
        total_in   = sum(rs.get("input_tokens",       0) or 0 for rs in round_stats)
        total_out  = sum(rs.get("output_tokens",      0) or 0 for rs in round_stats)
        tree_nodes = sum(rs.get("tree_nodes_created", 0) or 0 for rs in round_stats)

    avg_branching = 0.0
    max_branches  = 0
    mcts_tree = d.get("mcts_tree", {})
    tree_node_list = mcts_tree.get("nodes", [])
    if tree_node_list:
        parent_counts = Counter(
            n["parent_id"] for n in tree_node_list if n.get("parent_id") is not None
        )
        if parent_counts:
            avg_branching = sum(parent_counts.values()) / len(parent_counts)
            max_branches  = max(parent_counts.values())

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

    cost_bpt = _compute_cost_bpt(
        p_in, p_out, planner_model or "qwen3.5:9b",
        c_in, c_out, coder_model   or "qwen3.5:9b",
        rv_in, rv_out, reviewer_model or "qwen3.5:9b",
        v_in, v_out,
    )

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
        steps_used=steps_used,
        avg_branching_factor=avg_branching, max_branches=max_branches,
        cost_usd=cost_usd, cost_bpt=cost_bpt,
    )


# ---------------------------------------------------------------------------
# Rafe B_c1 evaluation (patch eval against case.json success_checks)
# ---------------------------------------------------------------------------

def _apply_and_check(patch: str, case_data: dict[str, Any], repo_path: pathlib.Path) -> bool:
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
        for cmd in case_data.get("install_commands", []) or []:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=tmp_repo)
            if r.returncode != 0:
                return False
        for cmd in case_data.get("setup_commands", []) or []:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=tmp_repo)
            if r.returncode != 0:
                return False
        for check in case_data["evaluation"]["success_checks"]:
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

        p_in = c_in = rv_in = p_out = c_out = rv_out = 0
        duration = 0.0
        solved = False
        steps_used = 0
        if traj_files:
            td = json.loads(traj_files[0].read_text())
            rms = td.get("role_model_stats", {})
            p_in,  p_out  = _role_tokens(rms.get("planner",  {}))
            c_in,  c_out  = _role_tokens(rms.get("coder",    {}), "input_tokens", "output_tokens")
            rv_in, rv_out = _role_tokens(rms.get("reviewer", {}))
            duration      = float(td.get("duration_seconds", 0.0))
            solved = bool(td.get("submitted") or td.get("info", {}).get("submitted"))
            steps_used = int(td.get("stats", {}).get("turns", 0))

        cost_usd = (_compute_cost_usd(p_in,  p_out,  planner_model) +
                    _compute_cost_usd(c_in,  c_out,  coder_model)   +
                    _compute_cost_usd(rv_in, rv_out, reviewer_model))
        cost_bpt = _compute_cost_bpt(
            p_in, p_out, planner_model,
            c_in, c_out, coder_model,
            rv_in, rv_out, reviewer_model,
            0, 0,
        )

        results.append(InstanceResult(
            instance_id=f"{case_name}_001", run_id="B_c1_rafe_linear",
            solved=solved, duration_s=duration, steps_used=steps_used,
            planner_tokens_in=p_in,   planner_tokens_out=p_out,
            coder_tokens_in=c_in,     coder_tokens_out=c_out,
            reviewer_tokens_in=rv_in, reviewer_tokens_out=rv_out,
            total_tokens_in=p_in+c_in+rv_in, total_tokens_out=p_out+c_out+rv_out,
            coder_model=coder_model, planner_model=planner_model, reviewer_model=reviewer_model,
            cost_usd=cost_usd, cost_bpt=cost_bpt,
        ))
    return results


# ---------------------------------------------------------------------------
# Final matrix (Rafe's new agents, c1 first-20 only)
# ---------------------------------------------------------------------------

def _eval_final_matrix_variant(
    variant_name: str,
    run_id: str,
    inner_subdir: str | None = None,
) -> list[InstanceResult]:
    """Parse c1 instances (first 20 alphabetically) from final_matrix for one variant.

    MCTS variants are flat: {variant}/{instance_id}/{instance_id}.traj
    Rafe linear variants are nested: {variant}/{inner_subdir}/{case}/{instance_id}/{instance_id}.traj
    """
    base = FINAL_MATRIX / variant_name
    data_dir = (base / inner_subdir) if inner_subdir else base
    if not data_dir.exists():
        return []
    results = []
    for traj_path in sorted(data_dir.rglob("*.traj")):
        # Derive case name: strip numeric suffix (e.g. "board_rollup_001" → "board_rollup")
        stem = traj_path.stem
        suffix = stem.rsplit("_", 1)[-1]
        case_name = stem.rsplit("_", 1)[0] if suffix.isdigit() else stem
        if case_name not in _FINAL_MATRIX_C1_CASES:
            continue
        results.append(_parse_traj(traj_path, run_id))
    return results


# Variant configs: (dir_name_in_final_matrix, inner_subdir_or_None, run_id)
_FINAL_MATRIX_CONFIGS: list[tuple[str, str | None, str]] = [
    ("mcts_baseline",                            None,                    "rafe_mcts_baseline_c1"),
    ("mcts_critic_gate",                         None,                    "rafe_mcts_critic_gate_c1"),
    ("mcts_plan_critic",                         None,                    "rafe_mcts_plan_critic_c1"),
    ("umich_qwen",                               "single",                "rafe_qwen_c1"),
    ("umich_gptoss_120b",                        "single",                "rafe_gpt120b_c1"),
    ("umich_gptoss_planner_umich_qwen_coder",    "planner_coder",         "rafe_gpt_plan_qwen_code_c1"),
    ("umich_gptoss_planner_umich_qwen_coder_reviewer", "planner_coder_reviewer", "rafe_gpt_plan_qwen_code_rev_c1"),
    ("umich_gptoss_planner_critic_qwen_coder",   "planner_coder",         "rafe_gpt_plan_critic_qwen_c1"),
]


# ---------------------------------------------------------------------------
# Legacy A baseline (9b MCTS, original tree_search runner)
# ---------------------------------------------------------------------------

def _eval_legacy_mcts(run_dir: pathlib.Path, run_id: str) -> list[InstanceResult]:
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
    avg_compute: float       # avg USD per instance
    compute_per_solve: float
    avg_cost_bpt: float      # avg B-param·tokens per instance
    bpt_per_solve: float
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
        return RunSummary(run_id="", solved=0, total=0, solve_rate=0,
                          avg_compute=0, compute_per_solve=0,
                          avg_cost_bpt=0, bpt_per_solve=0,
                          avg_tokens_in=0, avg_tokens_out=0,
                          avg_planner_tokens=0, avg_coder_tokens=0,
                          avg_reviewer_tokens=0, avg_value_tokens=0,
                          avg_tree_nodes=0, avg_best_depth=0,
                          avg_duration_s=0, coder_model="", planner_model="")
    solved = sum(1 for i in instances if i.solved)
    solve_rate = solved / n
    avg_compute = sum(i.cost_usd for i in instances) / n
    compute_per_solve = avg_compute / solve_rate if solve_rate > 0 else float("inf")
    avg_bpt = sum(i.cost_bpt for i in instances) / n
    bpt_per_solve = avg_bpt / solve_rate if solve_rate > 0 else float("inf")

    def avg(fn): return sum(fn(i) for i in instances) / n

    models = [i for i in instances if i.coder_model]
    return RunSummary(
        run_id=instances[0].run_id,
        solved=solved, total=n, solve_rate=solve_rate,
        avg_compute=avg_compute, compute_per_solve=compute_per_solve,
        avg_cost_bpt=avg_bpt, bpt_per_solve=bpt_per_solve,
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

    if _include("A_c1"):
        a1 = _eval_legacy_mcts(LEGACY_A_C1, "A_c1_9b_mcts")
        if a1:
            runs["A_c1_9b_mcts"] = a1
    if _include("A_c2"):
        a2 = _eval_legacy_mcts(LEGACY_A_C2, "A_c2_9b_mcts")
        if a2:
            runs["A_c2_9b_mcts"] = a2

    if _include("B_c1"):
        b1 = _eval_rafe_b_c1()
        if b1:
            runs["B_c1_rafe_linear"] = b1

    # Final matrix: Rafe's new agents, c1 first-20 only
    if FINAL_MATRIX.exists():
        for variant_name, inner_subdir, run_id in _FINAL_MATRIX_CONFIGS:
            if _include(run_id):
                instances = _eval_final_matrix_variant(variant_name, run_id, inner_subdir)
                if instances:
                    runs[run_id] = instances

    # Scan both the old combined runs dir and the new combined_results dir
    for root in [combined_root, COMBINED_RESULTS]:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            # Skip non-run directories (figures, reviewer_audits, etc.)
            if run_dir.name in {"figures", "preliminary_results", "reviewer_audits"}:
                continue
            if run_filter and not run_dir.name.upper().startswith(run_filter.upper()):
                continue
            trajs = list(run_dir.rglob("*.traj"))
            if not trajs:
                continue
            if run_dir.name not in runs:  # don't double-load
                instances = [_parse_traj(t, run_dir.name) for t in sorted(trajs)]
                runs[run_dir.name] = instances

    return runs


# ---------------------------------------------------------------------------
# Helpers: variant/case_set extraction
# ---------------------------------------------------------------------------

def _variant_letter(run_id: str) -> str:
    """Extract variant identifier (e.g. 'A', 'A_strict', 'B_strict', 'L')."""
    parts = run_id.split("_")
    for i, p in enumerate(parts):
        if p in {"c1", "c2", "c3"}:
            return "_".join(parts[:i]) if i > 0 else run_id
    return parts[0]


def _case_set(run_id: str) -> str:
    """Extract case set token (c1/c2/c3) from run_id regardless of position."""
    for p in run_id.split("_"):
        if p in {"c1", "c2", "c3"}:
            return p
    return "unknown"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(summaries: list[RunSummary]) -> None:
    hdr = f"{'Run':<44}  {'Solved':>6}  {'Rate':>6}  {'BPT':>10}  {'BPT/Solve':>10}  {'AvgTok':>7}  {'Nodes':>5}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for s in summaries:
        rate  = f"{s.solve_rate*100:.0f}%"
        bpt   = f"{s.avg_cost_bpt/1e6:.2f}M"
        bps   = f"{s.bpt_per_solve/1e6:.2f}M" if s.bpt_per_solve < 1e15 else "—"
        tok   = f"{(s.avg_tokens_in+s.avg_tokens_out)/1000:.1f}k"
        nodes = f"{s.avg_tree_nodes:.1f}"
        print(f"{s.run_id:<44}  {s.solved:>3}/{s.total:<3}  {rate:>6}  {bpt:>10}  {bps:>10}  {tok:>7}  {nodes:>5}")
    print()


# ---------------------------------------------------------------------------
# Export: main run summary CSV
# ---------------------------------------------------------------------------

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
# Export: per-instance metrics
# ---------------------------------------------------------------------------

def export_instance_csv(all_runs: dict[str, list[InstanceResult]], path: pathlib.Path) -> None:
    rows: list[dict[str, Any]] = []
    for run_id, instances in sorted(all_runs.items()):
        for inst in instances:
            row = asdict(inst)
            row["variant"] = _variant_letter(run_id)
            row["case_set"] = _case_set(run_id)
            row["solve_int"] = 1 if inst.solved else 0
            row["total_tokens"] = inst.total_tokens_in + inst.total_tokens_out
            rows.append(row)
    if not rows:
        path.write_text("")
        print(f"Instance CSV written (empty): {path}")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Instance CSV written: {path}")


# ---------------------------------------------------------------------------
# Export: efficiency accuracy (legacy format, kept for compatibility)
# ---------------------------------------------------------------------------

def export_efficiency_csv(summaries: list[RunSummary], path: pathlib.Path) -> None:
    rows: list[dict[str, Any]] = []
    for s in sorted(summaries, key=lambda x: x.run_id):
        total_tokens = s.avg_tokens_in + s.avg_tokens_out
        efficiency_usd = ((s.solve_rate * 100.0) / s.avg_compute) if s.avg_compute > 0 else 0.0
        efficiency_bpt = ((s.solve_rate * 100.0) / s.avg_cost_bpt) if s.avg_cost_bpt > 0 else 0.0
        rows.append({
            "run_id": s.run_id,
            "variant": _variant_letter(s.run_id),
            "case_set": _case_set(s.run_id),
            "solved": s.solved,
            "total": s.total,
            "solve_rate": s.solve_rate,
            "avg_compute_usd": s.avg_compute,
            "compute_per_solve_usd": s.compute_per_solve,
            "avg_cost_bpt": s.avg_cost_bpt,
            "bpt_per_solve": s.bpt_per_solve,
            "avg_total_tokens": total_tokens,
            "avg_duration_s": s.avg_duration_s,
            "avg_tree_nodes": s.avg_tree_nodes,
            "avg_best_depth": s.avg_best_depth,
            "efficiency_score_usd": efficiency_usd,
            "efficiency_score_bpt": efficiency_bpt,
        })
    if not rows:
        path.write_text("")
        print(f"Efficiency CSV written (empty): {path}")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Efficiency CSV written: {path}")


# ---------------------------------------------------------------------------
# Export: Pareto frontier (legacy USD-based, kept for compatibility)
# ---------------------------------------------------------------------------

def export_frontier_csv(summaries: list[RunSummary], path: pathlib.Path) -> None:
    by_variant: dict[str, dict[str, Any]] = {}
    for s in summaries:
        case_set = _case_set(s.run_id)
        if case_set not in {"c1", "c2"}:
            continue
        variant = _variant_letter(s.run_id)
        d = by_variant.setdefault(variant, {"rates": [], "compute": [], "compute_per_solve": [], "runs": []})
        d["rates"].append(s.solve_rate * 100.0)
        d["compute"].append(s.avg_compute)
        if s.compute_per_solve < 1e9:
            d["compute_per_solve"].append(s.compute_per_solve)
        d["runs"].append(s.run_id)

    rows: list[dict[str, Any]] = []
    for variant, d in sorted(by_variant.items()):
        if not d["rates"] or not d["compute"]:
            continue
        rows.append({
            "variant": variant,
            "avg_solve_rate_c1_c2": sum(d["rates"]) / len(d["rates"]),
            "avg_compute_c1_c2": sum(d["compute"]) / len(d["compute"]),
            "avg_compute_per_solve_c1_c2": (
                sum(d["compute_per_solve"]) / len(d["compute_per_solve"])
                if d["compute_per_solve"] else float("inf")
            ),
            "n_sets": len(d["rates"]),
            "source_runs": ",".join(d["runs"]),
            "pareto_optimal": False,
        })

    _mark_pareto(rows, "avg_solve_rate_c1_c2", "avg_compute_c1_c2")

    if not rows:
        path.write_text("")
        print(f"Frontier CSV written (empty): {path}")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Frontier CSV written: {path}")


def _mark_pareto(rows: list[dict], rate_key: str, cost_key: str) -> None:
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if (other[rate_key] >= row[rate_key] and
                    other[cost_key] <= row[cost_key] and
                    (other[rate_key] > row[rate_key] or other[cost_key] < row[cost_key])):
                dominated = True
                break
        row["pareto_optimal"] = not dominated


# ---------------------------------------------------------------------------
# Export: BPT efficiency frontier (new, primary)
# Pareto uses c2+c3 only — MCTS was iteratively tuned on c1.
# ---------------------------------------------------------------------------

def export_efficiency_frontier_csv(
    all_runs: dict[str, list[InstanceResult]],
    instance_path: pathlib.Path,
    runs_path: pathlib.Path,
) -> None:
    """Per-instance and per-run BPT frontier CSVs."""
    inst_rows: list[dict[str, Any]] = []
    for run_id, instances in sorted(all_runs.items()):
        coder_model    = next((i.coder_model    for i in instances if i.coder_model),    "")
        planner_model  = next((i.planner_model  for i in instances if i.planner_model),  "")
        reviewer_model = next((i.reviewer_model for i in instances if i.reviewer_model), "")
        for inst in instances:
            planner_bpt  = (inst.planner_tokens_in  + inst.planner_tokens_out)  * (_model_size_b(inst.planner_model  or planner_model  or "qwen3.5:9b") or 9)
            coder_bpt    = (inst.coder_tokens_in    + inst.coder_tokens_out)    * (_model_size_b(inst.coder_model    or coder_model    or "qwen3.5:9b") or 9)
            reviewer_bpt = (inst.reviewer_tokens_in + inst.reviewer_tokens_out) * (_model_size_b(inst.reviewer_model or reviewer_model or "qwen3.5:9b") or 9)
            value_bpt    = (inst.value_tokens_in    + inst.value_tokens_out)    * (_model_size_b(inst.coder_model    or coder_model    or "qwen3.5:9b") or 9)
            inst_rows.append({
                "run_id":              run_id,
                "variant":             _variant_letter(run_id),
                "case_set":            _case_set(run_id),
                "instance_id":         inst.instance_id,
                "solved":              int(inst.solved),
                "planner_tokens_in":   inst.planner_tokens_in,
                "planner_tokens_out":  inst.planner_tokens_out,
                "planner_model":       inst.planner_model or planner_model,
                "planner_params_b":    _model_size_b(inst.planner_model or planner_model or "") or 9,
                "planner_bpt":         planner_bpt,
                "coder_tokens_in":     inst.coder_tokens_in,
                "coder_tokens_out":    inst.coder_tokens_out,
                "coder_model":         inst.coder_model or coder_model,
                "coder_params_b":      _model_size_b(inst.coder_model or coder_model or "") or 9,
                "coder_bpt":           coder_bpt,
                "reviewer_tokens_in":  inst.reviewer_tokens_in,
                "reviewer_tokens_out": inst.reviewer_tokens_out,
                "reviewer_model":      inst.reviewer_model or reviewer_model,
                "reviewer_params_b":   _model_size_b(inst.reviewer_model or reviewer_model or "") or 9,
                "reviewer_bpt":        reviewer_bpt,
                "value_tokens_in":     inst.value_tokens_in,
                "value_tokens_out":    inst.value_tokens_out,
                "value_bpt":           value_bpt,
                "total_tokens":        inst.total_tokens_in + inst.total_tokens_out,
                "cost_bpt":            inst.cost_bpt,
                "cost_usd":            inst.cost_usd,
                "mcts_iterations":     inst.mcts_iterations,
                "steps_used":          inst.steps_used,
                "duration_s":          inst.duration_s,
            })

    _write_csv(inst_rows, instance_path)
    print(f"Efficiency frontier instance CSV written: {instance_path}")

    # Run-level aggregation (Pareto on c2+c3 only)
    by_variant_c23: dict[str, dict] = {}
    run_rows: list[dict[str, Any]] = []
    summaries_all = [(run_id, _aggregate(instances)) for run_id, instances in sorted(all_runs.items())]

    for run_id, s in summaries_all:
        cs = _case_set(run_id)
        v  = _variant_letter(run_id)
        if cs in {"c2", "c3"}:
            d = by_variant_c23.setdefault(v, {"rates": [], "bpt": [], "runs": []})
            d["rates"].append(s.solve_rate * 100.0)
            d["bpt"].append(s.avg_cost_bpt)
            d["runs"].append(run_id)

        total_tokens = s.avg_tokens_in + s.avg_tokens_out
        inst_list = all_runs[run_id]
        avg_planner_bpt = sum(
            (i.planner_tokens_in + i.planner_tokens_out) * (_model_size_b(i.planner_model or "") or 9)
            for i in inst_list
        ) / max(len(inst_list), 1)
        avg_coder_bpt = sum(
            (i.coder_tokens_in + i.coder_tokens_out) * (_model_size_b(i.coder_model or "") or 9)
            for i in inst_list
        ) / max(len(inst_list), 1)
        avg_reviewer_bpt = sum(
            (i.reviewer_tokens_in + i.reviewer_tokens_out) * (_model_size_b(i.reviewer_model or "") or 9)
            for i in inst_list
        ) / max(len(inst_list), 1)
        avg_value_bpt = sum(
            (i.value_tokens_in + i.value_tokens_out) * (_model_size_b(i.coder_model or "") or 9)
            for i in inst_list
        ) / max(len(inst_list), 1)

        run_rows.append({
            "run_id":            run_id,
            "variant":           v,
            "case_set":          cs,
            "solve_rate":        s.solve_rate,
            "avg_cost_bpt":      s.avg_cost_bpt,
            "bpt_per_solve":     s.bpt_per_solve,
            "avg_planner_bpt":   avg_planner_bpt,
            "avg_coder_bpt":     avg_coder_bpt,
            "avg_reviewer_bpt":  avg_reviewer_bpt,
            "avg_value_bpt":     avg_value_bpt,
            "avg_total_tokens":  total_tokens,
            "avg_cost_usd":      s.avg_compute,
            "pareto_optimal":    False,
        })

    # Mark Pareto based on c2+c3 average
    pareto_rows: list[dict[str, Any]] = []
    for v, d in sorted(by_variant_c23.items()):
        pareto_rows.append({
            "variant": v,
            "avg_solve_rate_c2_c3": sum(d["rates"]) / len(d["rates"]),
            "avg_cost_bpt_c2_c3":   sum(d["bpt"])   / len(d["bpt"]),
            "source_runs":          ",".join(d["runs"]),
        })
    _mark_pareto(pareto_rows, "avg_solve_rate_c2_c3", "avg_cost_bpt_c2_c3")

    pareto_map = {r["variant"]: r.get("pareto_optimal", False) for r in pareto_rows}
    for row in run_rows:
        row["pareto_optimal"] = pareto_map.get(row["variant"], False)

    _write_csv(run_rows, runs_path)
    print(f"Efficiency frontier run CSV written: {runs_path}")


# ---------------------------------------------------------------------------
# Export: steps-to-solve
# ---------------------------------------------------------------------------

def export_steps_to_solve_csv(
    all_runs: dict[str, list[InstanceResult]],
    path: pathlib.Path,
) -> None:
    """Per-instance steps used (raw) and BPT — no binning."""
    rows: list[dict[str, Any]] = []
    for run_id, instances in sorted(all_runs.items()):
        for inst in instances:
            # steps_allowed: for MCTS runs mcts_iterations is the max configured,
            # for linear runs it is 0 (single pass) — use tree_nodes as fallback proxy
            steps_allowed = inst.mcts_iterations if inst.mcts_iterations > 0 else inst.tree_nodes or inst.steps_used
            solve_fraction = (inst.steps_used / steps_allowed) if steps_allowed > 0 else None
            # effective_steps: tree_nodes for MCTS (counts all branches, not just winning
            # path depth), steps_used for linear agents where the two are equivalent
            effective_steps = inst.tree_nodes if inst.tree_nodes > 0 else inst.steps_used
            rows.append({
                "run_id":           run_id,
                "variant":          _variant_letter(run_id),
                "case_set":         _case_set(run_id),
                "instance_id":      inst.instance_id,
                "solved":           int(inst.solved),
                "steps_used":       inst.steps_used,
                "effective_steps":  effective_steps,
                "tree_nodes":       inst.tree_nodes,
                "steps_allowed":    steps_allowed,
                "solve_fraction":   f"{solve_fraction:.4f}" if solve_fraction is not None else "",
                "total_tokens":     inst.total_tokens_in + inst.total_tokens_out,
                "cost_bpt":         inst.cost_bpt,
                "stopped_reason":   inst.stopped_reason,
            })
    _write_csv(rows, path)
    print(f"Steps-to-solve CSV written: {path}")


# ---------------------------------------------------------------------------
# Export: instance overlap (for Venn / UpSet plot)
# ---------------------------------------------------------------------------

def export_instance_overlap_csv(
    all_runs: dict[str, list[InstanceResult]],
    path: pathlib.Path,
) -> None:
    """Wide-format solved booleans + pairwise Jaccard overlap."""
    # Collect solved sets per (variant, case_set)
    solved_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    all_instances: dict[tuple[str, str], set[str]] = defaultdict(set)

    for run_id, instances in all_runs.items():
        v  = _variant_letter(run_id)
        cs = _case_set(run_id)
        for inst in instances:
            all_instances[(cs, inst.instance_id)].add(v)
            if inst.solved:
                solved_sets[(v, cs)].add(inst.instance_id)

    variants = sorted({v for v, _ in solved_sets.keys()})
    case_sets = sorted({cs for _, cs in solved_sets.keys()})

    # Table 1: wide format
    wide_rows: list[dict[str, Any]] = []
    all_inst_keys = sorted(all_instances.keys())  # (case_set, instance_id)
    for cs, iid in all_inst_keys:
        row: dict[str, Any] = {"instance_id": iid, "case_set": cs}
        for v in variants:
            row[v] = int(iid in solved_sets.get((v, cs), set()))
        wide_rows.append(row)

    wide_path = path.with_name(path.stem + "_wide.csv")
    _write_csv(wide_rows, wide_path)
    print(f"Instance overlap wide CSV written: {wide_path}")

    # Table 2: pairwise overlap
    pair_rows: list[dict[str, Any]] = []
    for cs in case_sets:
        all_inst_cs = {iid for (c, iid) in all_instances.keys() if c == cs}
        for i, va in enumerate(variants):
            for vb in variants[i:]:
                sa = solved_sets.get((va, cs), set())
                sb = solved_sets.get((vb, cs), set())
                n_both  = len(sa & sb)
                n_only_a = len(sa - sb)
                n_only_b = len(sb - sa)
                n_neither = len(all_inst_cs - sa - sb)
                union = len(sa | sb)
                jaccard = n_both / union if union > 0 else 0.0
                fa = all_inst_cs - sa
                fb = all_inst_cs - sb
                n_both_failed = len(fa & fb)
                pair_rows.append({
                    "variant_a":    va,
                    "variant_b":    vb,
                    "case_set":     cs,
                    "n_both_solved": n_both,
                    "n_only_a":     n_only_a,
                    "n_only_b":     n_only_b,
                    "n_neither":    n_neither,
                    "jaccard":      f"{jaccard:.4f}",
                    "n_both_failed": n_both_failed,
                })
    pair_path = path.with_name(path.stem + "_pairwise.csv")
    _write_csv(pair_rows, pair_path)
    print(f"Instance overlap pairwise CSV written: {pair_path}")


# ---------------------------------------------------------------------------
# Export: resource waste / failure analysis
# ---------------------------------------------------------------------------

def export_resource_waste_csv(
    all_runs: dict[str, list[InstanceResult]],
    path: pathlib.Path,
) -> None:
    """Per-instance resource usage split by solved/failed.

    Suggested plot: overlapping histograms (solved=blue, failed=red) with
    X = cost_bpt per instance (log scale), Y = count, plus median lines.
    Alternative: CDF curves — cleaner for skewed distributions.
    """
    rows: list[dict[str, Any]] = []
    for run_id, instances in sorted(all_runs.items()):
        steps_allowed_default = next(
            (i.mcts_iterations for i in instances if i.mcts_iterations > 0), 0
        )
        for inst in instances:
            steps_allowed = inst.mcts_iterations if inst.mcts_iterations > 0 else steps_allowed_default
            frac = (inst.steps_used / steps_allowed) if steps_allowed > 0 else None
            effective_steps = inst.tree_nodes if inst.tree_nodes > 0 else inst.steps_used
            rows.append({
                "run_id":             run_id,
                "variant":            _variant_letter(run_id),
                "case_set":           _case_set(run_id),
                "instance_id":        inst.instance_id,
                "solved":             int(inst.solved),
                "steps_used":         inst.steps_used,
                "tree_nodes":         inst.tree_nodes,
                "effective_steps":    effective_steps,
                "steps_allowed":      steps_allowed,
                "fraction_steps_used": f"{frac:.4f}" if frac is not None else "",
                "total_tokens":       inst.total_tokens_in + inst.total_tokens_out,
                "cost_bpt":           inst.cost_bpt,
                "cost_usd":           inst.cost_usd,
                "duration_s":         inst.duration_s,
                "stopped_reason":     inst.stopped_reason,
            })
    _write_csv(rows, path)
    print(f"Resource waste CSV written: {path}")


# ---------------------------------------------------------------------------
# Export: ablation feature impact
# ---------------------------------------------------------------------------

# Feature coding for all known variants.
# Binary features: mcts, planner, reviewer, strict_gate, hindsight, value_fn
# Numeric: coder_size (B params)
_VARIANT_FEATURES: dict[str, dict[str, Any]] = {
    "L":        {"mcts": 0, "planner": 0, "reviewer": 0, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 9},
    "M":        {"mcts": 0, "planner": 0, "reviewer": 0, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 30},
    "N":        {"mcts": 0, "planner": 0, "reviewer": 0, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 120},
    "A":        {"mcts": 1, "planner": 0, "reviewer": 0, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 9},
    "B":        {"mcts": 0, "planner": 1, "reviewer": 1, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 30},
    "C":        {"mcts": 1, "planner": 1, "reviewer": 1, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 30},
    "D":        {"mcts": 1, "planner": 1, "reviewer": 1, "strict_gate": 0, "hindsight": 0, "value_fn": 1, "coder_size": 30},
    "E":        {"mcts": 1, "planner": 1, "reviewer": 1, "strict_gate": 0, "hindsight": 1, "value_fn": 1, "coder_size": 30},
    "F":        {"mcts": 0, "planner": 1, "reviewer": 0, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 9},
    "G":        {"mcts": 1, "planner": 0, "reviewer": 0, "strict_gate": 0, "hindsight": 1, "value_fn": 0, "coder_size": 9},
    "H":        {"mcts": 1, "planner": 0, "reviewer": 0, "strict_gate": 0, "hindsight": 0, "value_fn": 1, "coder_size": 9},
    "I":        {"mcts": 1, "planner": 0, "reviewer": 0, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 9},
    "J":        {"mcts": 1, "planner": 0, "reviewer": 0, "strict_gate": 0, "hindsight": 0, "value_fn": 0, "coder_size": 30},
    "A_strict": {"mcts": 1, "planner": 0, "reviewer": 1, "strict_gate": 1, "hindsight": 0, "value_fn": 0, "coder_size": 9},
    "B_strict": {"mcts": 0, "planner": 1, "reviewer": 1, "strict_gate": 1, "hindsight": 0, "value_fn": 0, "coder_size": 30},
    "C_strict": {"mcts": 1, "planner": 1, "reviewer": 1, "strict_gate": 1, "hindsight": 0, "value_fn": 0, "coder_size": 30},
    "F_strict": {"mcts": 0, "planner": 1, "reviewer": 1, "strict_gate": 1, "hindsight": 0, "value_fn": 0, "coder_size": 9},
    "G_strict": {"mcts": 1, "planner": 0, "reviewer": 1, "strict_gate": 1, "hindsight": 1, "value_fn": 0, "coder_size": 9},
    "P":        {"mcts": 1, "planner": 1, "reviewer": 1, "strict_gate": 0, "hindsight": 1, "value_fn": 0, "coder_size": 30},
}

_BINARY_FEATURES = ["mcts", "planner", "reviewer", "strict_gate", "hindsight", "value_fn"]


def export_ablation_features_csv(
    all_runs: dict[str, list[InstanceResult]],
    path: pathlib.Path,
) -> None:
    """Feature-level ablation: mean solve rate and efficiency with/without each binary feature.

    Aggregates across all case sets. Low statistical power when n_runs < 3 — noted in output.
    """
    # Collect per-run solve rates and BPT
    run_stats: list[dict[str, Any]] = []
    for run_id, instances in all_runs.items():
        v = _variant_letter(run_id)
        if v not in _VARIANT_FEATURES:
            continue
        n = len(instances)
        if n == 0:
            continue
        solved = sum(1 for i in instances if i.solved)
        solve_rate = solved / n
        avg_bpt = sum(i.cost_bpt for i in instances) / n
        bpt_per_solve = avg_bpt / solve_rate if solve_rate > 0 else float("inf")
        run_stats.append({
            "variant": v,
            "solve_rate": solve_rate,
            "avg_bpt": avg_bpt,
            "bpt_per_solve": bpt_per_solve,
            "features": _VARIANT_FEATURES[v],
        })

    rows: list[dict[str, Any]] = []
    for feat in _BINARY_FEATURES:
        on  = [r for r in run_stats if r["features"].get(feat) == 1]
        off = [r for r in run_stats if r["features"].get(feat) == 0]
        if not on or not off:
            continue

        def _mean(lst, key):
            vals = [r[key] for r in lst if r[key] < 1e14]
            return sum(vals) / len(vals) if vals else 0.0

        mean_solve_on  = _mean(on,  "solve_rate")
        mean_solve_off = _mean(off, "solve_rate")
        mean_bpt_on    = _mean(on,  "avg_bpt")
        mean_bpt_off   = _mean(off, "avg_bpt")
        mean_bps_on    = _mean(on,  "bpt_per_solve")
        mean_bps_off   = _mean(off, "bpt_per_solve")

        rows.append({
            "feature":                   feat,
            "n_runs_on":                 len(on),
            "n_runs_off":                len(off),
            "mean_solve_on":             f"{mean_solve_on:.4f}",
            "mean_solve_off":            f"{mean_solve_off:.4f}",
            "feature_effect_solve":      f"{mean_solve_on - mean_solve_off:.4f}",
            "mean_bpt_on":               f"{mean_bpt_on:.0f}",
            "mean_bpt_off":              f"{mean_bpt_off:.0f}",
            "feature_effect_bpt":        f"{mean_bpt_on - mean_bpt_off:.0f}",
            "mean_bpt_per_solve_on":     f"{mean_bps_on:.0f}",
            "mean_bpt_per_solve_off":    f"{mean_bps_off:.0f}",
            "feature_effect_efficiency": f"{mean_bps_on - mean_bps_off:.0f}",
            "low_power_warning":         "n<3" if min(len(on), len(off)) < 3 else "",
            "case_sets_included":        "c1+c2+c3",
        })

    _write_csv(rows, path)
    print(f"Ablation features CSV written: {path}")


# ---------------------------------------------------------------------------
# Shared CSV writer
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

_VARIANT_INFO: dict[str, tuple[str, str]] = {
    "A": ("9b MCTS (ours)",                       "Ours (9b)"),
    "B": ("Rafe linear (120b plan + 30b code)",    "Baseline"),
    "C": ("Mixed MCTS — 120b plan/review + 30b",   "Ours (best)"),
    "D": ("C + 30b LLM value function",            "Ours (mixed)"),
    "E": ("C + value fn + hindsight",              "Ours (mixed)"),
    "F": ("9b code + 120b plan, no search",        "Ours (9b)"),
    "G": ("9b MCTS + hindsight feedback",          "Ours (9b)"),
    "H": ("9b MCTS + self-eval value fn",          "Ours (9b)"),
    "I": ("9b full swe-search replica",            "swe-search"),
    "J": ("30b flat + full swe-search",            "swe-search"),
    "K": ("Bare UCB1 (minimal swe-search)",        "swe-search"),
    "L": ("9b linear (single agent)",              "Linear baseline"),
    "M": ("30b linear (single agent)",             "Linear baseline"),
    "N": ("120b linear (single agent)",            "Linear baseline"),
    "A_strict": ("9b MCTS, strict reviewer",       "Strict (9b)"),
    "B_strict": ("Rafe linear, strict reviewer",   "Strict (baseline)"),
    "C_strict": ("Mixed MCTS, strict reviewer",    "Strict (best)"),
    "F_strict": ("9b code + 120b plan, strict",    "Strict (9b)"),
    "G_strict": ("9b MCTS + hindsight, strict",    "Strict (9b)"),
    "P":        ("Best combined (adaptive, hint)", "Ours (best)"),
}

_ABL_ORDER = ["C", "E", "D", "B", "A", "F", "G", "H", "J", "I", "K",
              "C_strict", "B_strict", "A_strict", "F_strict", "G_strict",
              "L", "M", "N", "P"]


def _lkp_run(summaries: list[RunSummary]) -> dict[str, RunSummary]:
    out: dict[str, RunSummary] = {}
    for s in summaries:
        v  = _variant_letter(s.run_id)
        cs = _case_set(s.run_id)
        out[f"{v}_{cs}"] = s
    return out


def _rate_cell(s: RunSummary | None) -> str:
    if s is None or s.total == 0:
        return "---"
    return f"{s.solve_rate * 100:.0f}\\%"


def _bpt_cell(s: RunSummary | None) -> str:
    if s is None or s.avg_cost_bpt == 0:
        return "---"
    return f"{s.avg_cost_bpt/1e6:.1f}M"


def export_latex(summaries: list[RunSummary], path: pathlib.Path) -> None:
    lkp = _lkp_run(summaries)
    lines: list[str] = []
    lines.append("% Auto-generated by eval_combined.py — do not edit by hand")
    lines.append("% Requires \\usepackage{booktabs} in preamble")
    lines.append("")

    lines.append("% TABLE 1: Main results")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Solve rates on three independent test sets.")
    lines.append("    \\textbf{c3} is fully held out.")
    lines.append("    Cost is B-param$\\cdot$tokens (BPT, proportional to FLOPs).}")
    lines.append("  \\label{tab:main_results}")
    lines.append("  \\begin{tabular}{llrrrrr}")
    lines.append("    \\toprule")
    lines.append("    Sys & Description & c1 & c2 & c3 & Avg BPT & \\# nodes \\\\")
    lines.append("    \\midrule")

    for letter in ["A", "B", "C"]:
        desc, _ = _VARIANT_INFO.get(letter, ("", ""))
        cells, bpts, nodes_vals = [], [], []
        for cs in ["c1", "c2", "c3"]:
            s = lkp.get(f"{letter}_{cs}")
            cells.append(_rate_cell(s))
            if s and s.avg_cost_bpt > 0:
                bpts.append(s.avg_cost_bpt)
            if s and s.avg_tree_nodes > 0:
                nodes_vals.append(s.avg_tree_nodes)
        avg_bpt   = f"{sum(bpts)/len(bpts)/1e6:.1f}M" if bpts else "---"
        avg_nodes = f"{sum(nodes_vals)/len(nodes_vals):.1f}" if nodes_vals else "---"
        lines.append(f"    {letter} & {desc} & {' & '.join(cells)} & {avg_bpt} & {avg_nodes} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("% TABLE 2: Full ablation")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Ablation results. BPT = B-param$\\cdot$tokens (Estimated FLOPs).}")
    lines.append("  \\label{tab:ablation}")
    lines.append("  \\begin{tabular}{llllrrrr}")
    lines.append("    \\toprule")
    lines.append("    Var & Description & Group & c1 & c2 & Avg\\,\\% & Avg\\,BPT & Depth \\\\")
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
        r1, r2 = _rate_cell(s_c1), _rate_cell(s_c2)
        rates  = [s.solve_rate * 100 for s in [s_c1, s_c2] if s and s.total > 0]
        avg_r  = f"{sum(rates)/len(rates):.0f}\\%" if rates else "---"
        bpts   = [s.avg_cost_bpt for s in [s_c1, s_c2] if s and s.avg_cost_bpt > 0]
        avg_b  = f"{sum(bpts)/len(bpts)/1e6:.1f}M" if bpts else "---"
        depths = [s.avg_best_depth for s in [s_c1, s_c2] if s and s.avg_best_depth > 0]
        avg_d  = f"{sum(depths)/len(depths):.1f}" if depths else "---"
        lines.append(f"    {letter} & {desc} & {group} & {r1} & {r2} & {avg_r} & {avg_b} & {avg_d} \\\\")

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
    parser.add_argument("--run",           default=None, help="Filter runs by prefix (e.g. C_c1)")
    parser.add_argument("--csv",           type=pathlib.Path, default=None,
                        help="Write run-level summary CSV")
    parser.add_argument("--json",          type=pathlib.Path, default=None)
    parser.add_argument("--instance-csv",  type=pathlib.Path, default=None,
                        help="Write per-instance metrics for all runs")
    parser.add_argument("--efficiency-csv", type=pathlib.Path, default=None,
                        help="Write run-level efficiency vs accuracy metrics (USD + BPT)")
    parser.add_argument("--frontier-csv", type=pathlib.Path, default=None,
                        help="Write c1+c2 Pareto frontier table by variant (USD, legacy)")
    parser.add_argument("--efficiency-frontier-csv", type=pathlib.Path, default=None,
                        help="Write BPT efficiency frontier — pass instance output path; "
                             "_runs.csv is written alongside automatically")
    parser.add_argument("--steps-csv",    type=pathlib.Path, default=None,
                        help="Write per-instance steps-to-solve with BPT")
    parser.add_argument("--overlap-csv",  type=pathlib.Path, default=None,
                        help="Write instance overlap CSVs (_wide.csv and _pairwise.csv)")
    parser.add_argument("--waste-csv",    type=pathlib.Path, default=None,
                        help="Write per-instance resource waste / failure analysis CSV")
    parser.add_argument("--ablation-csv", type=pathlib.Path, default=None,
                        help="Write ablation feature impact CSV")
    parser.add_argument("--latex",        type=pathlib.Path, default=None,
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
    if args.instance_csv:
        export_instance_csv(all_runs, args.instance_csv)
    if args.efficiency_csv:
        export_efficiency_csv(summaries, args.efficiency_csv)
    if args.frontier_csv:
        export_frontier_csv(summaries, args.frontier_csv)
    if args.efficiency_frontier_csv:
        runs_path = args.efficiency_frontier_csv.with_name(
            args.efficiency_frontier_csv.stem.replace("_instances", "") + "_runs.csv"
        )
        export_efficiency_frontier_csv(all_runs, args.efficiency_frontier_csv, runs_path)
    if args.steps_csv:
        export_steps_to_solve_csv(all_runs, args.steps_csv)
    if args.overlap_csv:
        export_instance_overlap_csv(all_runs, args.overlap_csv)
    if args.waste_csv:
        export_resource_waste_csv(all_runs, args.waste_csv)
    if args.ablation_csv:
        export_ablation_features_csv(all_runs, args.ablation_csv)
    if args.latex:
        export_latex(summaries, args.latex)


if __name__ == "__main__":
    main()
