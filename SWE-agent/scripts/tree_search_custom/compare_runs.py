#!/usr/bin/env python3
"""Compare two (or more) batch run directories side by side.

Shows per-instance deltas in outcome, turns, and token cost so you can track
whether a change improved or regressed performance.

Usage:
  python compare_runs.py run_v3 run_v4
  python compare_runs.py run_v3 run_v4 run_v5          # three-way comparison
  python compare_runs.py run_v3 run_v4 --output diff.txt
  python compare_runs.py run_v3 run_v4 --changed-only  # hide unchanged rows
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loading helpers (shared with analyze_batch.py)
# ---------------------------------------------------------------------------

def _find_traj(instance_dir: Path) -> Path | None:
    for p in instance_dir.rglob("*.traj"):
        return p
    return None


def _load_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Return {instance_id: traj_dict} for all instances in run_dir."""
    results: dict[str, dict[str, Any]] = {}
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        p = _find_traj(sub)
        if p is None:
            continue
        traj = json.loads(p.read_text())
        iid = traj.get("instance_id", sub.name)
        results[iid] = traj
    return results


def _summary(traj: dict[str, Any]) -> dict[str, Any]:
    stats = traj.get("stats") or {}
    submitted = bool(traj.get("submitted", traj.get("info", {}).get("submitted", False)))
    stopped = traj.get("stopped_reason", traj.get("info", {}).get("stopped_reason", "?"))
    if traj.get("error"):
        stopped = "error"
    return {
        "submitted": submitted,
        "stopped":   stopped if not submitted else "submitted",
        "turns":     int(stats.get("turns", 0)),
        "tok_in":    int(stats.get("input_tokens", 0)),
        "duration":  float(traj.get("duration_seconds", 0)),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_W = 130

def _hr(c: str = "─") -> str:
    return c * _W


def _delta(a: int | float, b: int | float, lower_is_better: bool = True) -> str:
    d = b - a
    if d == 0:
        return "  ±0 "
    sign = "+" if d > 0 else ""
    tag = ("▼" if lower_is_better else "▲") if d < 0 else ("▲" if lower_is_better else "▼")
    if isinstance(d, float):
        return f"{tag}{sign}{d:.1f}"
    if abs(d) >= 1000:
        return f"{tag}{sign}{d/1000:.1f}K"
    return f"{tag}{sign}{d}"


def _outcome_change(old_s: str, new_s: str) -> str:
    """Arrow summary of outcome change."""
    if old_s == new_s:
        return "  ="
    if new_s == "submitted":
        return " ✓↑"   # newly solved
    if old_s == "submitted":
        return " ✗↓"   # regression
    return "  ~"        # different failure mode


def _tok_short(n: int) -> str:
    if n == 0:
        return "    —"
    if n >= 1_000:
        return f"{n/1000:5.1f}K"
    return f"{n:5d}"


def _fmt_run_name(p: Path) -> str:
    return p.name[:18]


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def _build_table(
    run_dirs: list[Path],
    runs: list[dict[str, dict[str, Any]]],
    changed_only: bool,
) -> list[str]:
    # Collect all instance ids across all runs
    all_ids = sorted({iid for r in runs for iid in r})

    lines: list[str] = []

    # Column headers
    run_labels = [_fmt_run_name(d) for d in run_dirs]
    header_instance = f"  {'Instance':<28}"
    header_runs = "".join(f"  {lbl:<20}" for lbl in run_labels)
    header_delta = "".join(f"  {'Δturns':>7}  {'Δtok-in':>8}" for _ in run_labels[1:])
    lines.append(_hr("═"))
    lines.append(header_instance + header_runs + ("  Deltas (vs run 1)" if len(runs) > 1 else ""))
    lines.append(_hr("─"))

    n_improved = n_regressed = n_unchanged = 0

    for iid in all_ids:
        summaries = [_summary(r[iid]) if iid in r else None for r in runs]

        # Skip unchanged rows if requested
        if changed_only and all(s is not None for s in summaries):
            outcomes = [s["stopped"] for s in summaries if s]
            if len(set(outcomes)) == 1:
                n_unchanged += 1
                continue

        # Instance name
        row = f"  {iid:<28}"

        # Per-run status
        for s in summaries:
            if s is None:
                row += f"  {'(missing)':<20}"
            else:
                flag = "✓" if s["submitted"] else "✗"
                cell = f"{flag} {s['stopped'][:10]}  t={s['turns']:>3}  {_tok_short(s['tok_in'])}"
                row += f"  {cell:<20}"

        # Delta columns (vs run[0])
        base = summaries[0]
        for s in summaries[1:]:
            if base is None or s is None:
                row += "  " + " " * 17
                continue
            chg = _outcome_change(base["stopped"], s["stopped"])
            dt = _delta(base["turns"], s["turns"])
            dtok = _delta(base["tok_in"], s["tok_in"])
            row += f"  {chg}{dt:>7}  {dtok:>8}"

        lines.append(row)

        # Track improvement stats
        if len(summaries) >= 2 and summaries[0] and summaries[-1]:
            chg = _outcome_change(summaries[0]["stopped"], summaries[-1]["stopped"])
            if chg.strip() == "✓↑":
                n_improved += 1
            elif chg.strip() == "✗↓":
                n_regressed += 1
            else:
                n_unchanged += 1

    lines.append(_hr("═"))

    # Summary row
    if len(runs) >= 2:
        lines.append(f"  improved={n_improved}  regressed={n_regressed}  unchanged={n_unchanged}")
        # Per-run submitted counts
        submit_counts = [sum(1 for iid in r if _summary(r[iid])["submitted"]) for r in runs]
        counts_str = "  →  ".join(
            f"{_fmt_run_name(d)}: {c}/{len(r)} submitted ({100*c/max(len(r),1):.0f}%)"
            for d, r, c in zip(run_dirs, runs, submit_counts)
        )
        lines.append(f"  {counts_str}")

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", type=Path, nargs="+",
                        help="Two or more batch run directories to compare")
    parser.add_argument("--changed-only", action="store_true",
                        help="Hide rows where outcome is identical across all runs")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write output to file instead of stdout")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write per-instance comparison CSV to this path (for Excel)")
    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        parser.error("Provide at least 2 run directories to compare")

    runs = [_load_run(d) for d in args.run_dirs]
    lines = _build_table(args.run_dirs, runs, args.changed_only)

    output = "\n".join(lines)
    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)

    if args.csv:
        _write_csv(args.run_dirs, runs, args.csv)


def _write_csv(run_dirs: list[Path], runs: list[dict[str, dict[str, Any]]], path: Path) -> None:
    all_ids = sorted({iid for r in runs for iid in r})
    run_labels = [d.name for d in run_dirs]

    columns = ["instance_id", "run", "status", "submitted", "turns", "tok_in",
               "delta_turns", "delta_tok_in", "outcome_change"]

    base_run = runs[0]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for iid in all_ids:
            base = _summary(base_run[iid]) if iid in base_run else None
            for run_dir, run in zip(run_dirs, runs):
                s = _summary(run[iid]) if iid in run else None
                if s is None:
                    continue
                if base is not None and s is not base:
                    d_turns = s["turns"] - base["turns"]
                    d_tok = s["tok_in"] - base["tok_in"]
                    chg = _outcome_change(base["stopped"], s["stopped"]).strip()
                    outcome = "improved" if chg == "✓↑" else ("regressed" if chg == "✗↓" else "unchanged")
                else:
                    d_turns = 0
                    d_tok = 0
                    outcome = "baseline"
                writer.writerow({
                    "instance_id":   iid,
                    "run":           run_dir.name,
                    "status":        s["stopped"],
                    "submitted":     s["submitted"],
                    "turns":         s["turns"],
                    "tok_in":        s["tok_in"],
                    "delta_turns":   d_turns,
                    "delta_tok_in":  d_tok,
                    "outcome_change": outcome,
                })
    print(f"CSV written to {path}")


if __name__ == "__main__":
    main()
