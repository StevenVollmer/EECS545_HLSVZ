#!/usr/bin/env python3
"""Aggregate analysis for a run_tree_search.py batch output directory.

Produces a per-instance table and aggregate statistics to surface failure
patterns, token costs, MCTS efficiency, and loop behaviour.

Usage:
  # Summarise a single batch run
  python analyze_batch.py SWE-agent/tree_search_runs/all_custom_run_v5

  # Sort by token cost (most expensive first)
  python analyze_batch.py run_dir --sort tokens

  # Show only failed instances
  python analyze_batch.py run_dir --failures-only

  # Write summary to a file
  python analyze_batch.py run_dir --output summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _is_infra_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        needle in lowered
        for needle in (
            "ports are not available",
            "container process terminated",
            "docker: error response from daemon",
            "failed to create endpoint",
            "cannot connect to the docker daemon",
            "address already in use",
            "network is unreachable",
        )
    )


def _role_tokens(role_stats: dict[str, Any]) -> int:
    return int(role_stats.get("tokens_in", role_stats.get("input_tokens", 0)) or 0)


# ---------------------------------------------------------------------------
# Traj loading
# ---------------------------------------------------------------------------

def _find_trajs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found = sorted(root.rglob("*.traj"))
    if not found:
        print(f"No .traj files found under {root}", file=sys.stderr)
        sys.exit(1)
    return found


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Per-traj metrics extraction
# ---------------------------------------------------------------------------

def _extract(traj: dict[str, Any], path: Path) -> dict[str, Any]:
    """Pull every metric we care about into a flat dict."""
    stats = traj.get("stats") or {}
    meta  = traj.get("mcts_meta") or {}

    # Basic outcome
    instance_id   = traj.get("instance_id", path.stem)
    submitted     = bool(traj.get("submitted", traj.get("info", {}).get("submitted", False)))
    stopped       = traj.get("stopped_reason", traj.get("info", {}).get("stopped_reason", "?"))
    duration      = float(traj.get("duration_seconds", 0))
    error         = traj.get("error", "")

    # Token counts
    tok_in  = int(stats.get("input_tokens", 0))
    tok_out = int(stats.get("output_tokens", 0))
    turns   = int(stats.get("turns", 0))

    # Per-role token breakdown
    rms = traj.get("role_model_stats") or {}
    planner_in = _role_tokens(rms.get("planner") or {})
    coder_in = _role_tokens(rms.get("coder") or {})
    reviewer_in = _role_tokens(rms.get("reviewer") or {})

    # MCTS tree stats
    tree_nodes   = len((traj.get("mcts_tree") or {}).get("nodes", []))
    branch_ratio = round(tree_nodes / max(turns, 1), 2)  # >1 means branching happened

    # Vote efficiency: how many samples were actually used vs theoretical max
    vote_summary = traj.get("vote_summary") or []
    vote_total_used = sum(v.get("total_samples", 0) for v in vote_summary)
    vote_edits      = len(vote_summary)
    vote_max = int(meta.get("edit_vote_samples", 0) or 0)
    vote_savings = max(0, vote_max * vote_edits - vote_total_used) if vote_max > 0 else 0
    error_kind = str(traj.get("error_kind", "")).strip().lower()
    if error and not error_kind:
        error_kind = "infra" if _is_infra_error_text(error) else "agent"

    # Success checks
    checks_passed = list(traj.get("loop_state", {}).get("satisfied_success_checks", []))
    if not checks_passed:
        # Fall back to the winning path's last node
        nodes = (traj.get("mcts_tree") or {}).get("nodes", [])
        for n in reversed(nodes):
            if n.get("on_result_path"):
                checks_passed = list(n.get("loop_state", {}).get("satisfied_success_checks", []))
                break

    # Loop warnings: count turns where a "STOP:" message was injected
    loop_count = 0
    parse_errors = 0
    for turn in (traj.get("turns") or []):
        if turn.get("parse_error"):
            parse_errors += 1
        # Loop messages are injected as user messages appended to the turn
        for msg in turn.get("messages", []):
            if msg.get("role") == "user" and str(msg.get("content", "")).startswith("STOP:"):
                loop_count += 1

    # Patch stats
    patch = traj.get("patch", "")
    patch_lines = sum(1 for ln in patch.splitlines() if ln.startswith(("+", "-"))
                      and not ln.startswith(("+++", "---"))) if patch else 0

    # Multi-round reviewer info
    coder_rounds = traj.get("coder_rounds", [])
    n_rounds = len(coder_rounds) if coder_rounds else (1 if turns > 0 else 0)
    def _round_decision(n: int) -> str:
        if n < len(coder_rounds):
            rv = coder_rounds[n].get("review_feedback") or {}
            return str(rv.get("decision", "")).lower() or "?"
        return ""
    reviewer_r1 = _round_decision(0)
    reviewer_r2 = _round_decision(1)

    return {
        "instance_id":   instance_id,
        "submitted":     submitted,
        "stopped":       stopped,
        "duration":      duration,
        "turns":         turns,
        "tok_in":        tok_in,
        "tok_out":       tok_out,
        "planner_in":    planner_in,
        "coder_in":      coder_in,
        "reviewer_in":   reviewer_in,
        "tree_nodes":    tree_nodes,
        "branch_ratio":  branch_ratio,
        "vote_edits":    vote_edits,
        "vote_samples":  vote_total_used,
        "vote_savings":  vote_savings,
        "checks_passed": len(checks_passed),
        "loop_count":    loop_count,
        "parse_errors":  parse_errors,
        "patch_lines":   patch_lines,
        "error":         error,
        "error_kind":    error_kind,
        "coder_rounds":  n_rounds,
        "reviewer_r1":   reviewer_r1,
        "reviewer_r2":   reviewer_r2,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_W = 120

def _hr(c: str = "─") -> str:
    return c * _W


def _status(m: dict) -> str:
    if m["error"]:
        return "INFRA_ERR" if m.get("error_kind") == "infra" else "ERROR"
    if m["submitted"]:
        return "SUBMIT"
    return m["stopped"][:12] if m["stopped"] else "?"


def _tok(n: int) -> str:
    return f"{n:,}"


def _fmt_table(metrics: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []

    # Header
    lines.append(_hr("═"))
    lines.append(
        f"  {'Instance':<28}  {'Status':<14}  {'Turns':>5}  {'Time':>7}  "
        f"{'Tok-In':>9}  {'Loops':>5}  {'Votes':>7}  {'Patch':>6}  {'Checks':>6}"
    )
    lines.append(_hr("─"))

    for m in metrics:
        status = _status(m)
        flag = "✓" if m["submitted"] else "✗"
        vote_str = f"{m['vote_samples']}/{m['vote_edits']}e" if m["vote_edits"] else "  —  "
        lines.append(
            f"  {flag} {m['instance_id']:<26}  {status:<14}  {m['turns']:>5}  "
            f"{m['duration']:>6.1f}s  {_tok(m['tok_in']):>9}  {m['loop_count']:>5}  "
            f"{vote_str:>7}  {m['patch_lines']:>6}  {m['checks_passed']:>6}"
        )

    lines.append(_hr("═"))
    return lines


def _fmt_aggregate(metrics: list[dict], run_dir: Path) -> list[str]:
    lines: list[str] = []
    n = len(metrics)
    if n == 0:
        return lines

    n_submit  = sum(1 for m in metrics if m["submitted"])
    n_error   = sum(1 for m in metrics if m["error"])
    n_infra   = sum(1 for m in metrics if m.get("error_kind") == "infra")
    total_tok = sum(m["tok_in"] + m["tok_out"] for m in metrics)
    avg_turns = sum(m["turns"] for m in metrics) / n
    avg_time  = sum(m["duration"] for m in metrics) / n
    avg_tok   = sum(m["tok_in"] for m in metrics) / n

    lines.append(f"  Run dir : {run_dir}")
    lines.append(
        f"  Cases   : {n}   submitted={n_submit} ({100*n_submit/n:.0f}%)   "
        f"errors={n_error} (infra={n_infra}, agent={max(0, n_error - n_infra)})"
    )
    lines.append(f"  Avg     : {avg_turns:.1f} turns   {avg_time:.0f}s   {avg_tok:,.0f} tokens-in")
    lines.append(f"  Total tokens (in+out): {total_tok:,}")

    # Stopped-reason breakdown
    from collections import Counter
    reasons = Counter(_status(m) for m in metrics if not m["submitted"] and not m["error"])
    if reasons:
        lines.append(f"  Failures: " + "  ".join(f"{r}={c}" for r, c in reasons.most_common()))

    # Vote efficiency summary
    vote_metrics = [m for m in metrics if m["vote_edits"] > 0]
    if vote_metrics:
        total_used = sum(m["vote_samples"] for m in vote_metrics)
        total_edits = sum(m["vote_edits"] for m in vote_metrics)
        total_saved = sum(m["vote_savings"] for m in vote_metrics)
        avg_per_edit = total_used / total_edits if total_edits else 0
        lines.append(f"  Votes   : {total_edits} edit decisions, avg {avg_per_edit:.1f} samples/edit "
                     f"(early-exit saved {total_saved} model calls)")

    # Loop stats
    total_loops = sum(m["loop_count"] for m in metrics)
    loopy = [m["instance_id"] for m in metrics if m["loop_count"] > 0]
    if total_loops > 0:
        lines.append(f"  Loops   : {total_loops} warnings across {len(loopy)} instance(s): {', '.join(loopy)}")

    # Parse error stats
    total_parse = sum(m["parse_errors"] for m in metrics)
    if total_parse > 0:
        lines.append(f"  Parse errors: {total_parse} total")

    return lines


def _fmt_role_breakdown(metrics: list[dict]) -> list[str]:
    """Show per-role token breakdown for submitted vs failed instances."""
    lines: list[str] = []
    submitted = [m for m in metrics if m["submitted"]]
    failed    = [m for m in metrics if not m["submitted"] and not m["error"]]

    def _avg_role(group: list[dict], key: str) -> str:
        if not group:
            return "  —  "
        return f"{sum(m[key] for m in group)/len(group):,.0f}"

    if submitted or failed:
        lines.append(_hr("─"))
        lines.append(f"  {'':20}  {'Planner-in':>12}  {'Coder-in':>12}  {'Reviewer-in':>12}")
        if submitted:
            lines.append(f"  {'Submitted avg':20}  {_avg_role(submitted,'planner_in'):>12}  "
                         f"{_avg_role(submitted,'coder_in'):>12}  {_avg_role(submitted,'reviewer_in'):>12}")
        if failed:
            lines.append(f"  {'Failed avg':20}  {_avg_role(failed,'planner_in'):>12}  "
                         f"{_avg_role(failed,'coder_in'):>12}  {_avg_role(failed,'reviewer_in'):>12}")

    return lines


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "instance_id", "status", "submitted", "stopped_reason", "turns", "time_s",
    "tok_in", "tok_out", "loops", "vote_edits", "vote_samples", "patch_lines",
    "checks_satisfied", "coder_rounds", "reviewer_decision_r1", "reviewer_decision_r2",
    "planner_tok_in", "coder_tok_in", "reviewer_tok_in",
]


def _write_csv(metrics: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for m in metrics:
            writer.writerow({
                "instance_id":         m["instance_id"],
                "status":              _status(m),
                "submitted":           m["submitted"],
                "stopped_reason":      m["stopped"],
                "turns":               m["turns"],
                "time_s":              round(m["duration"], 1),
                "tok_in":              m["tok_in"],
                "tok_out":             m["tok_out"],
                "loops":               m["loop_count"],
                "vote_edits":          m["vote_edits"],
                "vote_samples":        m["vote_samples"],
                "patch_lines":         m["patch_lines"],
                "checks_satisfied":    m["checks_passed"],
                "coder_rounds":        m["coder_rounds"],
                "reviewer_decision_r1": m["reviewer_r1"],
                "reviewer_decision_r2": m["reviewer_r2"],
                "planner_tok_in":      m["planner_in"],
                "coder_tok_in":        m["coder_in"],
                "reviewer_tok_in":     m["reviewer_in"],
            })
    print(f"CSV written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path,
                        help="Batch output directory (contains per-instance subdirs)")
    parser.add_argument("--sort", default="instance",
                        choices=["instance", "status", "turns", "time", "tokens", "loops"],
                        help="Column to sort by (default: instance)")
    parser.add_argument("--failures-only", action="store_true",
                        help="Show only non-submitted instances")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write output to file instead of stdout")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write per-instance CSV to this path (for Excel)")
    parser.add_argument("--emit-failures", action="store_true",
                        help="Print a --filter regex of non-submitted instance IDs and exit")
    args = parser.parse_args()

    trajs = _find_trajs(args.run_dir)
    all_metrics = [_extract(_load(p), p) for p in trajs]

    if args.emit_failures:
        failed = [m["instance_id"] for m in all_metrics if not m["submitted"]]
        if not failed:
            print("(no failures found)")
        else:
            print("(" + "|".join(failed) + ")")
        sys.exit(0)

    # Sort
    sort_key = {
        "instance": lambda m: m["instance_id"],
        "status":   lambda m: (_status(m), m["instance_id"]),
        "turns":    lambda m: -m["turns"],
        "time":     lambda m: -m["duration"],
        "tokens":   lambda m: -m["tok_in"],
        "loops":    lambda m: -m["loop_count"],
    }[args.sort]
    all_metrics.sort(key=sort_key)

    if args.failures_only:
        display = [m for m in all_metrics if not m["submitted"]]
    else:
        display = all_metrics

    out: list[str] = []
    out.extend(_fmt_table(display))
    out.append("")
    out.extend(_fmt_aggregate(all_metrics, args.run_dir))
    out.extend(_fmt_role_breakdown(all_metrics))
    out.append(_hr("─"))

    output = "\n".join(out)
    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)

    if args.csv:
        _write_csv(all_metrics, args.csv)


if __name__ == "__main__":
    main()
