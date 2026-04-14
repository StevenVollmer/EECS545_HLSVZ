#!/usr/bin/env python3
"""Visualize the MCTS search tree stored in a run_tree_search.py trajectory file.

Usage:
  python visualize_tree.py <path/to/instance.traj>
  python visualize_tree.py <path/to/instance.traj> --format ascii
  python visualize_tree.py <path/to/dir/>           # finds the first .traj inside

The tree shows every node the MCTS explored, not just the winning path.  Nodes
on the winning path are highlighted in green (★).  Branch points (where majority-
vote edit candidates diverged) are marked with ⎇.  Submitted nodes are marked ✓.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_traj(path: Path) -> dict[str, Any]:
    if path.is_dir():
        matches = sorted(path.rglob("*.traj"))
        if not matches:
            print(f"No .traj files found under {path}", file=sys.stderr)
            sys.exit(1)
        path = matches[0]
    return json.loads(path.read_text())


def _build_node_index(nodes: list[dict]) -> dict[str, dict]:
    return {n["id"]: n for n in nodes}


def _children_of(node_id: str, nodes: list[dict]) -> list[dict]:
    return [n for n in nodes if n["parent_id"] == node_id]


# ---------------------------------------------------------------------------
# Rich renderer
# ---------------------------------------------------------------------------

def _node_label_rich(node: dict, markup: bool = True) -> str:
    """Build a Rich markup string for one node."""
    action = node.get("action_name") or "root"
    depth = node["depth"]
    val = node.get("mean_value", 0.0)
    visits = node.get("visits", 0)
    checks = node.get("success_checks", 0)
    submitted = node.get("submitted", False)
    on_path = node.get("on_result_path", False)
    is_branch = node.get("is_branch_point", False)
    vote_counts = node.get("vote_counts") or {}
    stopped = node.get("stopped_reason", "")
    edit_made = node.get("edit_made", False)

    icons = ""
    if on_path:
        icons += "★ "
    if is_branch:
        icons += "⎇ "
    if submitted:
        icons += "✓ "

    vote_str = ""
    if vote_counts and is_branch:
        winner_votes = max(vote_counts.values()) if vote_counts else 0
        total = sum(vote_counts.values())
        vote_str = f" [{winner_votes}/{total}✗]"

    check_str = f" checks={checks}" if checks else ""
    edit_str = " +edit" if edit_made else ""
    stop_str = f" [{stopped}]" if stopped else ""

    label = f"{icons}[{action}] d={depth} val={val:.1f} v={visits}{edit_str}{check_str}{vote_str}{stop_str}"

    if not markup:
        return label

    if on_path and submitted:
        return f"[bold green]{label}[/bold green]"
    if on_path:
        return f"[green]{label}[/green]"
    if submitted:
        return f"[yellow]{label}[/yellow]"
    if stopped and stopped not in ("submitted", ""):
        return f"[dim]{label}[/dim]"
    return label


def _render_rich(traj: dict[str, Any], show_values: bool) -> None:
    try:
        from rich.tree import Tree
        from rich.console import Console
    except ImportError:
        print("rich not available — falling back to ASCII renderer", file=sys.stderr)
        _render_ascii(traj, show_values)
        return

    mcts_tree = traj.get("mcts_tree")
    if not mcts_tree:
        print("No mcts_tree in traj — was this run with an older version?", file=sys.stderr)
        sys.exit(1)

    nodes = mcts_tree["nodes"]
    result_id = mcts_tree.get("result_node_id", "0")

    # Find root(s)
    roots = [n for n in nodes if n["parent_id"] is None]
    if not roots:
        print("No root node found in mcts_tree", file=sys.stderr)
        sys.exit(1)

    console = Console()

    def build_rich_tree(node: dict, rich_parent) -> None:
        label = _node_label_rich(node, markup=True)
        branch = rich_parent.add(label)
        for child in _children_of(node["id"], nodes):
            build_rich_tree(child, branch)

    for root_node in roots:
        label = _node_label_rich(root_node, markup=True)
        rt = Tree(label)
        for child in _children_of(root_node["id"], nodes):
            build_rich_tree(child, rt)
        console.print(rt)

    # Summary
    total_nodes = len(nodes)
    branch_nodes = sum(1 for n in nodes if n.get("is_branch_point"))
    submitted_nodes = sum(1 for n in nodes if n.get("submitted"))
    on_path = sum(1 for n in nodes if n.get("on_result_path"))
    console.print(
        f"\n[bold]Tree summary:[/bold] {total_nodes} nodes  "
        f"{branch_nodes} branch points  "
        f"{submitted_nodes} submitted  "
        f"result path depth={on_path - 1}"
    )


# ---------------------------------------------------------------------------
# ASCII fallback renderer
# ---------------------------------------------------------------------------

def _render_ascii(traj: dict[str, Any], show_values: bool) -> None:
    mcts_tree = traj.get("mcts_tree")
    if not mcts_tree:
        print("No mcts_tree in traj — was this run with an older version?")
        sys.exit(1)

    nodes = mcts_tree["nodes"]
    roots = [n for n in nodes if n["parent_id"] is None]

    def print_node(node: dict, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        label = _node_label_rich(node, markup=False)
        print(prefix + connector + label)
        children = _children_of(node["id"], nodes)
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            print_node(child, child_prefix, i == len(children) - 1)

    for root_node in roots:
        print(_node_label_rich(root_node, markup=False))
        children = _children_of(root_node["id"], nodes)
        for i, child in enumerate(children):
            print_node(child, "", i == len(children) - 1)

    total_nodes = len(nodes)
    branch_nodes = sum(1 for n in nodes if n.get("is_branch_point"))
    on_path = sum(1 for n in nodes if n.get("on_result_path"))
    print(
        f"\nTree summary: {total_nodes} nodes  "
        f"{branch_nodes} branch points  "
        f"result path depth={on_path - 1}"
    )


# ---------------------------------------------------------------------------
# Vote summary
# ---------------------------------------------------------------------------

def _print_vote_summary(traj: dict[str, Any]) -> None:
    vote_summary = traj.get("vote_summary", [])
    if not vote_summary:
        return
    print("\nMajority vote summary (edit turns on result path):")
    for entry in vote_summary:
        depth = entry.get("depth", "?")
        action = entry.get("action", "?")
        winner = entry.get("winner_votes", 0)
        total = entry.get("total_samples", 0)
        unique = entry.get("unique_candidates", 0)
        print(f"  depth={depth} {action}: winner={winner}/{total} samples, {unique} unique candidates")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traj", type=Path, help="Path to .traj file or directory containing one")
    parser.add_argument("--format", choices=["rich", "ascii"], default="rich",
                        help="Rendering format (default: rich if available, else ascii)")
    parser.add_argument("--show-values", action="store_true",
                        help="Reserved for future verbose node display")
    args = parser.parse_args()

    traj = _load_traj(args.traj)

    print(f"Instance:  {traj.get('instance_id', '?')}")
    print(f"Model:     {traj.get('mcts_meta', {}).get('model', '?')}")
    print(f"Submitted: {traj.get('submitted', False)}  "
          f"stopped={traj.get('stopped_reason', '?')}  "
          f"checks={traj.get('loop_state', {}).get('satisfied_success_checks', [])}")
    print()

    if args.format == "rich":
        _render_rich(traj, args.show_values)
    else:
        _render_ascii(traj, args.show_values)

    _print_vote_summary(traj)


if __name__ == "__main__":
    main()
