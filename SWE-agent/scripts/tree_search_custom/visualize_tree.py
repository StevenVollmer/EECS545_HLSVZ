#!/usr/bin/env python3
"""Visualize the MCTS search tree stored in a run_tree_search.py trajectory file.

Usage:
  python visualize_tree.py <path/to/instance.traj>
  python visualize_tree.py <path/to/instance.traj> --format ascii
  python visualize_tree.py <path/to/dir/>           # finds the first .traj inside
  python visualize_tree.py <path/to/instance.traj> --output tree.svg   # SVG image
  python visualize_tree.py <path/to/instance.traj> --output tree.png   # PNG image

The tree shows every node the MCTS explored, not just the winning path.  Nodes
on the winning path are highlighted in green (★).  Branch points (where majority-
vote edit candidates diverged) are marked with ⎇.  Submitted nodes are marked ✓.

Image output uses the system graphviz `dot` binary when available (produces SVG,
PNG, PDF, etc.).  If `dot` is absent, a matplotlib/networkx fallback is used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _children_of(node_id: str, nodes: list[dict]) -> list[dict]:
    return [n for n in nodes if n["parent_id"] == node_id]


# ---------------------------------------------------------------------------
# Shared label / color helpers (used by both CLI and image renderers)
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


def _dot_label(node: dict) -> str:
    """Multi-line label for a Graphviz DOT node (\\n-separated)."""
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

    icons = []
    if on_path:
        icons.append("*")
    if is_branch:
        icons.append("branch")
    if submitted:
        icons.append("SUBMIT")
    icon_str = " ".join(icons)

    lines = [f"[{action}]", f"d={depth}  val={val:.1f}  v={visits}"]
    if icon_str:
        lines.append(icon_str)
    if edit_made:
        lines.append("+edit")
    if checks:
        lines.append(f"checks={checks}")
    if vote_counts and is_branch:
        winner_votes = max(vote_counts.values())
        total = sum(vote_counts.values())
        lines.append(f"votes={winner_votes}/{total}")
    if stopped and stopped not in ("submitted", ""):
        lines.append(f"[{stopped}]")

    # Escape double-quotes for DOT
    return "\\n".join(line.replace('"', "'") for line in lines)


def _dot_fillcolor(node: dict) -> str:
    on_path = node.get("on_result_path", False)
    submitted = node.get("submitted", False)
    stopped = node.get("stopped_reason", "")

    if on_path and submitted:
        return "#2ecc71"   # bright green
    if on_path:
        return "#a8d5b5"   # light green
    if submitted:
        return "#f1c40f"   # yellow
    if stopped and stopped not in ("submitted", ""):
        return "#bdc3c7"   # light grey
    return "#ecf0f1"       # near-white


# ---------------------------------------------------------------------------
# Rich renderer (CLI)
# ---------------------------------------------------------------------------

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
# ASCII fallback renderer (CLI)
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
# Image export — DOT generation
# ---------------------------------------------------------------------------

def _build_dot(traj: dict[str, Any]) -> str:
    """Generate a Graphviz DOT source string from the mcts_tree in a traj."""
    mcts_tree = traj.get("mcts_tree")
    if not mcts_tree:
        raise ValueError("No mcts_tree in traj")

    nodes = mcts_tree["nodes"]
    instance_id = traj.get("instance_id", "?")
    model = traj.get("mcts_meta", {}).get("model", "?")

    lines = [
        "digraph mcts_tree {",
        f'  label="{instance_id}  model={model}";',
        "  labelloc=t;",
        "  rankdir=TB;",
        '  node [shape=box fontname="Courier" fontsize=9];',
        '  edge [arrowsize=0.7];',
    ]

    for n in nodes:
        nid = "n" + n["id"].replace(".", "_")
        label = _dot_label(n)
        color = _dot_fillcolor(n)
        lines.append(f'  {nid} [label="{label}" style=filled fillcolor="{color}"];')

    for n in nodes:
        if n["parent_id"] is not None:
            parent = "n" + n["parent_id"].replace(".", "_")
            child = "n" + n["id"].replace(".", "_")
            lines.append(f"  {parent} -> {child};")

    lines.append("}")
    return "\n".join(lines)


def _render_image_dot(dot_src: str, output_path: Path) -> None:
    """Render DOT source to an image file using the system `dot` binary."""
    fmt = output_path.suffix.lstrip(".").lower()
    if not fmt:
        fmt = "svg"
    result = subprocess.run(
        ["dot", f"-T{fmt}", "-o", str(output_path)],
        input=dot_src.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dot failed:\n{result.stderr.decode()}")
    print(f"Tree image written to: {output_path}")


def _render_image_matplotlib(traj: dict[str, Any], output_path: Path) -> None:
    """Matplotlib/networkx fallback when `dot` is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        print(f"Cannot render image: neither 'dot' nor matplotlib/networkx available ({e})",
              file=sys.stderr)
        return

    mcts_tree = traj.get("mcts_tree")
    if not mcts_tree:
        print("No mcts_tree in traj", file=sys.stderr)
        return

    nodes = mcts_tree["nodes"]
    G = nx.DiGraph()
    node_labels: dict[str, str] = {}
    node_colors: list[str] = []

    for n in nodes:
        nid = n["id"]
        G.add_node(nid)
        action = n.get("action_name") or "root"
        node_labels[nid] = f"[{action}]\nd={n['depth']} v={n.get('visits',0)}"

    for n in nodes:
        if n["parent_id"] is not None:
            G.add_edge(n["parent_id"], n["id"])

    # Build color list in node order
    node_order = list(G.nodes())
    node_map = {n["id"]: n for n in nodes}
    for nid in node_order:
        node_colors.append(_dot_fillcolor(node_map[nid]))

    # Layout: try graphviz hierarchy, fall back to spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(max(8, len(nodes) * 1.2), max(6, len(nodes) * 0.8)))
    nx.draw(G, pos=pos, labels=node_labels, node_color=node_colors,
            node_size=2000, font_size=7, arrows=True, ax=ax,
            edge_color="#888888", width=1.2)
    instance_id = traj.get("instance_id", "?")
    ax.set_title(f"MCTS tree: {instance_id}", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Tree image written to: {output_path}")


def render_image(traj: dict[str, Any], output_path: Path) -> None:
    """Dispatch to DOT or matplotlib renderer based on availability."""
    if shutil.which("dot"):
        dot_src = _build_dot(traj)
        _render_image_dot(dot_src, output_path)
    else:
        print("'dot' binary not found — using matplotlib fallback", file=sys.stderr)
        _render_image_matplotlib(traj, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traj", type=Path, help="Path to .traj file or directory containing one")
    parser.add_argument("--format", choices=["rich", "ascii"], default="rich",
                        help="CLI rendering format (default: rich if available, else ascii)")
    parser.add_argument("--show-values", action="store_true",
                        help="Reserved for future verbose node display")
    parser.add_argument(
        "--output", type=Path, metavar="PATH", nargs="?", const=Path("AUTO"),
        help=(
            "Write a tree image to this file. Format inferred from extension "
            "(.svg, .png, .pdf). Omit a path to write <traj_stem>.svg next to the traj."
        ),
    )
    args = parser.parse_args()

    # Resolve the traj path (needed for default output name)
    traj_path = args.traj
    if traj_path.is_dir():
        matches = sorted(traj_path.rglob("*.traj"))
        if not matches:
            print(f"No .traj files found under {traj_path}", file=sys.stderr)
            sys.exit(1)
        traj_path = matches[0]

    traj = json.loads(traj_path.read_text())

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

    # Image export
    if args.output is not None:
        out_path = (
            traj_path.with_suffix(".svg")
            if args.output == Path("AUTO")
            else args.output
        )
        render_image(traj, out_path)


if __name__ == "__main__":
    main()
