#!/usr/bin/env python3
"""Extract MCTS tree statistics and select interesting examples from traj files.

Reads mcts_tree.nodes (already serialized in every .traj) from MCTS runs
(variants containing 'mcts' or using >1 iteration) and produces:

  mcts_branch_stats.csv     — per-instance tree metrics for all MCTS runs
  mcts_example_trees.json   — full tree structure for 2-3 selected examples

Example selection:
  1. "Wide search, never found it" — high node count, branching>1, solved=False
  2. "Efficient find"              — solved=True, best_node_depth≤3, few iterations
  3. "Hindsight rescue"            — G_strict variant, solved=True, prior dead branches

Usage:
  python extract_mcts_trees.py
  python extract_mcts_trees.py --out-dir /path/to/combined_results
  python extract_mcts_trees.py --run A_strict_c1  # single run
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any

ROOT             = pathlib.Path(__file__).resolve().parents[3]
COMBINED_RUNS    = ROOT / "SWE-agent/tree_search_runs/combined"
COMBINED_RESULTS = ROOT / "combined_results"
DEFAULT_OUT      = COMBINED_RESULTS


def _is_mcts_run(run_id: str) -> bool:
    low = run_id.lower()
    return "mcts" in low or "hindsight" in low or "combined" in low or "best" in low


@dataclass
class TreeStats:
    run_id: str
    instance_id: str
    solved: bool
    tree_nodes: int
    max_depth_reached: int
    best_node_depth: int
    best_node_value: float
    n_branch_points: int     # nodes where at least one sibling exists (parent has >1 child)
    n_dead_branches: int     # branches that terminated without submission
    max_branches_at_node: int
    avg_branching_factor: float
    total_iterations: int    # configured max (from stats.iterations)
    steps_used: int          # actual turns executed (stats.turns)
    n_hindsight_feedback: int # dead_branch_feedback_count from mcts_meta


def _analyze_tree(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not nodes:
        return dict(n_branch_points=0, n_dead_branches=0,
                    max_branches_at_node=0, avg_branching_factor=0.0,
                    max_depth_reached=0)

    parent_counts = Counter(n["parent_id"] for n in nodes if n.get("parent_id") is not None)
    branching_nodes = {pid: cnt for pid, cnt in parent_counts.items() if cnt > 1}
    n_branch_points = len(branching_nodes)
    max_branches = max(parent_counts.values()) if parent_counts else 0
    avg_branching = (sum(parent_counts.values()) / len(parent_counts)
                     if parent_counts else 0.0)

    result_node_ids: set[str] = {n["id"] for n in nodes if n.get("on_result_path")}

    # Dead branches: nodes that terminated (stopped_reason set) and are NOT on result path
    dead = sum(
        1 for n in nodes
        if n.get("stopped_reason") and n["id"] not in result_node_ids
        and n["id"] != "0"  # root
    )

    max_depth = max((n.get("depth", 0) for n in nodes), default=0)

    return dict(
        n_branch_points=n_branch_points,
        n_dead_branches=dead,
        max_branches_at_node=max_branches,
        avg_branching_factor=avg_branching,
        max_depth_reached=max_depth,
    )


def _parse_traj_for_tree(traj_path: pathlib.Path, run_id: str) -> TreeStats | None:
    try:
        d = json.loads(traj_path.read_text())
    except Exception:
        return None

    mcts_tree = d.get("mcts_tree", {})
    nodes = mcts_tree.get("nodes", [])
    if not nodes:
        return None

    stats     = d.get("stats", {})
    instance_id = d.get("instance_id", traj_path.stem)
    solved    = bool(d.get("submitted") or d.get("info", {}).get("submitted"))

    tree_info = _analyze_tree(nodes)
    best_node_depth = int(stats.get("best_node_depth", 0))
    best_node_value = float(stats.get("best_node_value", 0.0))

    # Hindsight feedback count from mcts_meta
    mcts_meta = d.get("mcts_meta", {})
    n_hindsight = int(mcts_meta.get("dead_branch_feedback_count", 0))

    return TreeStats(
        run_id=run_id,
        instance_id=instance_id,
        solved=solved,
        tree_nodes=len(nodes),
        max_depth_reached=tree_info["max_depth_reached"],
        best_node_depth=best_node_depth,
        best_node_value=best_node_value,
        n_branch_points=tree_info["n_branch_points"],
        n_dead_branches=tree_info["n_dead_branches"],
        max_branches_at_node=tree_info["max_branches_at_node"],
        avg_branching_factor=tree_info["avg_branching_factor"],
        total_iterations=int(stats.get("iterations", 0)),
        steps_used=int(stats.get("turns", 0)),
        n_hindsight_feedback=n_hindsight,
    )


def _load_full_tree(traj_path: pathlib.Path, run_id: str, reason: str) -> dict[str, Any] | None:
    try:
        d = json.loads(traj_path.read_text())
    except Exception:
        return None
    mcts_tree = d.get("mcts_tree", {})
    nodes = mcts_tree.get("nodes", [])
    if not nodes:
        return None
    return {
        "run_id":      run_id,
        "instance_id": d.get("instance_id", traj_path.stem),
        "solved":      bool(d.get("submitted") or d.get("info", {}).get("submitted")),
        "reason":      reason,
        "result_node_id": mcts_tree.get("result_node_id"),
        "nodes": nodes,
    }


def _iter_mcts_trajs(
    combined_root: pathlib.Path,
    run_filter: str | None,
) -> list[tuple[str, pathlib.Path]]:
    pairs: list[tuple[str, pathlib.Path]] = []
    seen: set[str] = set()

    for root in [combined_root, COMBINED_RESULTS]:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_dir.name in {"figures", "preliminary_results", "reviewer_audits"}:
                continue
            if not _is_mcts_run(run_dir.name):
                continue
            if run_filter and not run_dir.name.upper().startswith(run_filter.upper()):
                continue
            if run_dir.name in seen:
                continue
            seen.add(run_dir.name)
            for traj in sorted(run_dir.rglob("*.traj")):
                pairs.append((run_dir.name, traj))

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combined-root", type=pathlib.Path, default=COMBINED_RUNS)
    parser.add_argument("--out-dir",       type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--run",           default=None, help="Filter by run prefix")
    parser.add_argument("--stats-csv",     type=pathlib.Path, default=None,
                        help="Override output path for mcts_branch_stats.csv")
    parser.add_argument("--examples-json", type=pathlib.Path, default=None,
                        help="Override output path for mcts_example_trees.json")
    args = parser.parse_args()

    stats_path    = args.stats_csv    or args.out_dir / "mcts_branch_stats.csv"
    examples_path = args.examples_json or args.out_dir / "mcts_example_trees.json"

    print("Scanning MCTS traj files…", file=sys.stderr)
    pairs = _iter_mcts_trajs(args.combined_root, args.run)
    print(f"  Found {len(pairs)} traj files in MCTS runs", file=sys.stderr)

    all_stats: list[TreeStats] = []
    all_traj_paths: dict[tuple[str, str], pathlib.Path] = {}  # (run_id, instance_id) → path

    for run_id, traj_path in pairs:
        ts = _parse_traj_for_tree(traj_path, run_id)
        if ts is not None:
            all_stats.append(ts)
            all_traj_paths[(run_id, ts.instance_id)] = traj_path

    print(f"  Parsed {len(all_stats)} instances with tree data", file=sys.stderr)

    # ── Write branch stats CSV ────────────────────────────────────────────────
    if all_stats:
        fields = list(asdict(all_stats[0]).keys())
        with open(stats_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for ts in all_stats:
                w.writerow(asdict(ts))
        print(f"Branch stats CSV written: {stats_path}")
    else:
        print("No MCTS tree data found.", file=sys.stderr)
        return

    # ── Select interesting examples ───────────────────────────────────────────
    examples: list[dict[str, Any]] = []

    # 1. Wide search, never found it: failed, many nodes, has branching
    candidates_wide = sorted(
        [ts for ts in all_stats if not ts.solved and ts.max_branches_at_node > 1],
        key=lambda t: -(t.tree_nodes),
    )
    if candidates_wide:
        ts = candidates_wide[0]
        traj_p = all_traj_paths.get((ts.run_id, ts.instance_id))
        if traj_p:
            ex = _load_full_tree(traj_p, ts.run_id, "wide_search_failed")
            if ex:
                examples.append(ex)
                print(f"  Example 1 (wide/failed): {ts.run_id}/{ts.instance_id} "
                      f"nodes={ts.tree_nodes} branches={ts.max_branches_at_node}", file=sys.stderr)

    # 2. Efficient find: solved early (best_node_depth ≤ 4, few steps)
    candidates_eff = sorted(
        [ts for ts in all_stats if ts.solved and ts.best_node_depth <= 4],
        key=lambda t: t.steps_used,
    )
    if candidates_eff:
        ts = candidates_eff[0]
        traj_p = all_traj_paths.get((ts.run_id, ts.instance_id))
        if traj_p:
            ex = _load_full_tree(traj_p, ts.run_id, "efficient_find")
            if ex:
                examples.append(ex)
                print(f"  Example 2 (efficient): {ts.run_id}/{ts.instance_id} "
                      f"depth={ts.best_node_depth} steps={ts.steps_used}", file=sys.stderr)

    # 3. Hindsight rescue: G_strict variant, solved, with dead branches + hindsight feedback
    candidates_hint = sorted(
        [ts for ts in all_stats
         if ts.solved and ts.n_dead_branches > 0 and "hindsight" in ts.run_id.lower()],
        key=lambda t: -t.n_hindsight_feedback,
    )
    if candidates_hint:
        ts = candidates_hint[0]
        traj_p = all_traj_paths.get((ts.run_id, ts.instance_id))
        if traj_p:
            ex = _load_full_tree(traj_p, ts.run_id, "hindsight_rescue")
            if ex:
                examples.append(ex)
                print(f"  Example 3 (hindsight): {ts.run_id}/{ts.instance_id} "
                      f"dead_branches={ts.n_dead_branches} "
                      f"hindsight_feedback={ts.n_hindsight_feedback}", file=sys.stderr)

    examples_path.write_text(json.dumps(examples, indent=2))
    print(f"Example trees JSON written: {examples_path} ({len(examples)} examples)")


if __name__ == "__main__":
    main()
