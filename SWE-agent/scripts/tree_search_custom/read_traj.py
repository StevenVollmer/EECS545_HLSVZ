#!/usr/bin/env python3
"""Human-readable formatter for run_tree_search.py trajectory files.

Prints the winning path of the MCTS run as a legible conversation so you can
follow the agent's reasoning turn-by-turn and spot semantic or anecdotal trends
(e.g. model confusion, unnecessary steps, vote disagreements).

Usage:
  python read_traj.py <path/to/instance.traj>
  python read_traj.py <path/to/dir/>            # first .traj in the directory
  python read_traj.py <path/to/dir/> --all      # one file per .traj found
  python read_traj.py run.traj --no-tool-output # hide raw tool outputs (headers only)
  python read_traj.py run.traj --output out.txt # write to a file instead of stdout
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

# Maximum characters shown for tool output before truncating
TOOL_OUTPUT_MAX = 800
# Maximum characters shown for assistant text
ASSISTANT_TEXT_MAX = 600
# Width for wrapped text
WRAP_WIDTH = 100


# ---------------------------------------------------------------------------
# Traj loading
# ---------------------------------------------------------------------------

def _find_trajs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    found = sorted(path.rglob("*.traj"))
    if not found:
        print(f"No .traj files found under {path}", file=sys.stderr)
        sys.exit(1)
    return found


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "─", width: int = WRAP_WIDTH) -> str:
    return char * width


def _wrap(text: str, indent: str = "  ", max_chars: int | None = None) -> str:
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n  … [{len(text) - max_chars} chars truncated]"
    lines = text.splitlines()
    out = []
    for line in lines:
        if len(line) > WRAP_WIDTH - len(indent):
            out.extend(textwrap.wrap(line, width=WRAP_WIDTH - len(indent),
                                     initial_indent=indent, subsequent_indent=indent))
        else:
            out.append(indent + line)
    return "\n".join(out)


def _tool_args_summary(name: str, arguments: dict[str, Any]) -> str:
    """Compact one-line summary of tool arguments."""
    if name in ("bash", "run"):
        cmd = str(arguments.get("cmd") or arguments.get("command") or "")
        return f"$ {cmd[:200]}"
    if name == "view":
        p = arguments.get("path", "")
        start = arguments.get("start_line", "")
        end = arguments.get("end_line", "")
        loc = f":{start}-{end}" if start else ""
        return f"{p}{loc}"
    if name in ("str_replace", "replace"):
        p = arguments.get("path") or arguments.get("file_path", "")
        old = str(arguments.get("old_str") or arguments.get("old_string", ""))[:80]
        new = str(arguments.get("new_str") or arguments.get("new_string", ""))[:80]
        return f"{p}\n    old: {old!r}\n    new: {new!r}"
    if name == "insert":
        p = arguments.get("path", "")
        line = arguments.get("insert_line", "")
        new = str(arguments.get("new_str", ""))[:80]
        return f"{p} line={line}\n    insert: {new!r}"
    if name == "submit":
        return "(submit patch)"
    if name == "handoff":
        return str(arguments.get("summary") or arguments.get("message") or "")[:200]
    # Generic fallback
    parts = []
    for k, v in arguments.items():
        parts.append(f"{k}={str(v)[:60]!r}")
    return ", ".join(parts[:4])


def _emit_turns(
    turns: list[dict[str, Any]],
    vote_by_depth: dict[int, dict],
    show_tool_output: bool,
    emit: Any,
) -> None:
    """Render a list of coder turns into emit() lines (shared by single/multi-round paths)."""
    for turn in turns:
        t = turn.get("turn", "?")
        tool_calls = turn.get("tool_calls", [])
        tool_results = turn.get("tool_results", [])
        assistant_text = (turn.get("assistant_text") or "").strip()
        parse_error = turn.get("parse_error")

        if tool_calls:
            call = tool_calls[0]
            name = call.get("name", "?")
            args = call.get("arguments") or {}
            args_summary = _tool_args_summary(name, args)

            vote_str = ""
            vote_entry = vote_by_depth.get(t)
            if vote_entry:
                w = vote_entry.get("winner_votes", 0)
                tot = vote_entry.get("total_samples", 0)
                u = vote_entry.get("unique_candidates", 1)
                vote_str = f"  ★ majority-vote {w}/{tot} samples"
                if u > 1:
                    vote_str += f", {u} unique candidates"

            is_edit = name in ("str_replace", "insert", "replace")
            edit_marker = " [EDIT]" if is_edit else ""
            emit(f"  │  Turn {t:>2}  {name}{edit_marker}{vote_str}")
            for ln in args_summary.splitlines():
                emit(f"  │          {ln}")
        elif assistant_text:
            emit(f"  │  Turn {t:>2}  (no tool call)")

        if assistant_text:
            emit(f"  │          [reasoning] {assistant_text[:ASSISTANT_TEXT_MAX]}")
        if parse_error:
            emit(f"  │          ⚠ parse error: {parse_error}")
        if tool_results and show_tool_output:
            res = tool_results[0]
            output = str(res.get("output", ""))
            is_error = res.get("is_error", False)
            err_tag = " [ERROR]" if is_error else ""
            emit(f"  │          ── output{err_tag} ──")
            emit(_wrap(output, "  │          ", max_chars=TOOL_OUTPUT_MAX))

        emit("  │")


def _format_traj(traj: dict[str, Any], show_tool_output: bool) -> str:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    # ── Header ──────────────────────────────────────────────────────────────
    instance_id = traj.get("instance_id", "?")
    arch = traj.get("architecture", traj.get("agent_architecture", "?"))
    submitted = traj.get("submitted", False)
    stopped = traj.get("stopped_reason", "?")
    checks = traj.get("loop_state", {}).get("satisfied_success_checks", [])
    duration = traj.get("duration_seconds", 0)
    meta = traj.get("mcts_meta", {})
    model = meta.get("model", "?")
    iterations = meta.get("iterations_used", meta.get("max_iterations", "?"))
    tree_nodes = len(traj.get("mcts_tree", {}).get("nodes", []))

    emit(_hr("═"))
    emit(f"  INSTANCE  : {instance_id}")
    emit(f"  MODEL     : {model}    architecture={arch}")
    emit(f"  RESULT    : submitted={submitted}  stopped={stopped}  checks={checks}")
    emit(f"  TIMING    : {duration:.1f}s    iterations={iterations}    tree_nodes={tree_nodes}")
    emit(_hr("═"))
    emit()

    # ── Planner output ───────────────────────────────────────────────────────
    handoff = traj.get("planner_handoff")
    if handoff:
        emit("  ┌─ PLANNER ─────────────────────────────────────────────────────────────")
        if isinstance(handoff, dict):
            for k, v in handoff.items():
                if v and k not in ("error",):
                    label = k.replace("_", " ").title()
                    val = v if isinstance(v, str) else json.dumps(v, indent=2)
                    emit(f"  │  {label}: {val}")
        else:
            emit(_wrap(str(handoff), "  │  "))
        emit("  └───────────────────────────────────────────────────────────────────────")
        emit()

    # ── Coder rounds ─────────────────────────────────────────────────────────
    coder_rounds = traj.get("coder_rounds", [])
    if coder_rounds:
        # Multi-round trajectory: show each round with its reviewer verdict
        for rnd in coder_rounds:
            rnd_num = rnd.get("round", "?")
            rnd_turns = rnd.get("turns", [])
            rnd_vote_summary = rnd.get("vote_summary", [])
            rnd_review = rnd.get("review_feedback") or {}
            rnd_decision = str(rnd_review.get("decision", "")).lower()
            decision_marker = "✓ accept" if rnd_decision == "accept" else ("✗ revise" if rnd_decision == "revise" else "?")
            emit(f"  ┌─ CODER ROUND {rnd_num} — {len(rnd_turns)} turn(s)  reviewer→{decision_marker} ───────────────────")
            emit("  │")
            vote_by_depth: dict[int, dict] = {vs.get("depth", -1): vs for vs in rnd_vote_summary}
            _emit_turns(rnd_turns, vote_by_depth, show_tool_output, emit)
            emit("  └───────────────────────────────────────────────────────────────────────")
            # Inline reviewer verdict for this round
            if rnd_review and any(rnd_review.values()):
                summary = rnd_review.get("summary") or rnd_review.get("primary_reason", "")
                required = rnd_review.get("required_changes", [])
                emit(f"  │reviewer│ {decision_marker.upper()}  {summary[:80]}")
                for req in required[:2]:
                    emit(f"  │        │   • {req[:90]}")
            emit()
    else:
        # Single-round (legacy) trajectory
        turns = traj.get("turns", [])
        if not turns:
            emit("  (no turns recorded)")
        else:
            emit(f"  ┌─ CODER — {len(turns)} turn(s) on result path ──────────────────────────────────")
            emit("  │")
            vote_by_depth = {vs.get("depth", -1): vs for vs in traj.get("vote_summary", [])}
            _emit_turns(turns, vote_by_depth, show_tool_output, emit)
            emit("  └───────────────────────────────────────────────────────────────────────")
            emit()

    # ── Reviewer feedback (final, for single-round trajs) ────────────────────
    review = traj.get("review_feedback")
    if not coder_rounds and review and isinstance(review, dict) and any(review.values()):
        emit("  ┌─ REVIEWER ────────────────────────────────────────────────────────────")
        for k, v in review.items():
            if v:
                label = k.replace("_", " ").title()
                val = v if isinstance(v, str) else json.dumps(v)
                emit(f"  │  {label}: {val}")
        emit("  └───────────────────────────────────────────────────────────────────────")
        emit()

    # ── Final patch ──────────────────────────────────────────────────────────
    patch = traj.get("patch", "")
    if patch and patch.strip():
        emit("  ┌─ PATCH ───────────────────────────────────────────────────────────────")
        for ln in patch.strip().splitlines():
            emit(f"  │  {ln}")
        emit("  └───────────────────────────────────────────────────────────────────────")
        emit()

    # ── Stats summary ────────────────────────────────────────────────────────
    stats = traj.get("stats", {})
    rms = traj.get("role_model_stats", {})
    emit(_hr("─"))
    emit(f"  tokens in={stats.get('input_tokens',0):,}  out={stats.get('output_tokens',0):,}"
         f"    turns={stats.get('turns',0)}  tool_calls={stats.get('tool_calls',0)}")
    for role, rstat in rms.items():
        m = rstat.get("model", "?")
        ti = rstat.get("tokens_in", rstat.get("input_tokens", 0))
        to = rstat.get("tokens_out", rstat.get("output_tokens", 0))
        emit(f"  [{role}] model={m}  tokens_in={ti}  tokens_out={to}")
    emit(_hr("─"))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("traj", type=Path,
                        help="Path to .traj file, or a directory to search")
    parser.add_argument("--all", action="store_true",
                        help="Process all .traj files found under the given directory")
    parser.add_argument("--no-tool-output", action="store_true",
                        help="Hide tool output (show headers only — much shorter)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write output to this file instead of stdout")
    args = parser.parse_args()

    trajs = _find_trajs(args.traj) if args.all or args.traj.is_dir() else [args.traj]

    out_lines: list[str] = []
    for traj_path in trajs:
        traj = _load(traj_path)
        text = _format_traj(traj, show_tool_output=not args.no_tool_output)
        out_lines.append(text)
        if len(trajs) > 1:
            out_lines.append("")  # blank separator between instances

    output = "\n".join(out_lines)

    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
