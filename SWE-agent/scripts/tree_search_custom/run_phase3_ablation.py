#!/usr/bin/env python3
"""Phase-3 ablation: isolate thinking mode and failure surfacing contributions.

Design
------
Two independent variables, tested in a 2×2 grid on the hard cases from custom_cases_2,
plus full-set runs on both case libraries for regression/improvement tracking.

Variables:
  thinking_mode    — Qwen3.5 reasoning tokens (reasoning_effort=low, max_tokens=768)
  failure_surfacing — inject exact AssertionError / demo-script diffs into round-2 context

Variants (run in order: cheapest → most expensive):
  1. no_think_no_surface   — old behavior, baseline for this ablation
  2. surface_only          — failure surfacing ON, thinking OFF  (expected: +1–2 on c2 hard cases)
  3. think_only            — thinking ON, surfacing OFF          (expected: +1–2 via better reasoning)
  4. think_plus_surface    — both ON                            (expected: synergy or neutral)

Test sets:
  A. custom_cases_2 hard cases (6 confirmed failures, targets wrong-impl failures)
  B. custom_cases original hard cases (4 failures, breadth check / regression guard)
  C. custom_cases_2 full 20 (variants 1+2 only — overall score comparison vs c2_v1 run)

Run order: A1 → B1 → A2 → B2 → C1 → C2 → A3 → A4

Usage
-----
  # Dry-run (print commands only):
  python SWE-agent/scripts/tree_search_custom/run_phase3_ablation.py

  # Execute all runs:
  python SWE-agent/scripts/tree_search_custom/run_phase3_ablation.py --execute

  # Execute and resume any incomplete runs:
  python SWE-agent/scripts/tree_search_custom/run_phase3_ablation.py --execute --resume

  # Run only a specific subset of variants (comma-separated):
  python SWE-agent/scripts/tree_search_custom/run_phase3_ablation.py --execute --only A1,B1,A2,B2

  # Summarize results from a completed ablation:
  python SWE-agent/scripts/tree_search_custom/run_phase3_ablation.py --summarize
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hard-case filters (confirmed multi-run failures)
# ---------------------------------------------------------------------------

C2_HARD_FILTER = (
    "(assignee_board_001|campaign_header_001|dispatch_preview_001"
    "|headline_api_001|profile_preview_001|timeline_caption_001)"
)

C1_HARD_FILTER = (
    "(budget_snapshot_001|contact_card_001|digest_preview_001|shipment_preview_001)"
)

# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------

def _cmd(
    run_id: str,
    preset: str,
    instances_path: str,
    filter_str: str | None,
    output_root: Path,
    extra_flags: list[str],
    resume: bool,
) -> list[str]:
    cmd = [
        "python3",
        "SWE-agent/scripts/tree_search_custom/run_mcts.py",
        "--preset", preset,
        "--instances-type", "file",
        "--instances-path", instances_path,
        "--output-dir", str(output_root / run_id),
        "--run-name", run_id,
        *extra_flags,
    ]
    if filter_str:
        cmd += ["--filter", filter_str]
    if resume:
        cmd += ["--resume"]
    return cmd


def _count_trajs(run_dir: Path) -> int:
    """Count .traj files in a run output directory."""
    return len(list(run_dir.rglob("*.traj"))) if run_dir.exists() else 0


def _expected_cases(filter_str: str | None, instances_path: str) -> int:
    """Count how many instances a run would process."""
    import re
    path = Path(instances_path)
    if not path.exists():
        return 0
    instance_ids: list[str] = []
    for d in path.iterdir():
        case_file = d / "case.json"
        if not d.is_dir() or not case_file.exists():
            continue
        try:
            raw = json.loads(case_file.read_text())
            entries = raw if isinstance(raw, list) else [raw]
            for entry in entries:
                iid = entry.get("instance_id", "")
                if iid:
                    instance_ids.append(iid)
        except Exception:
            instance_ids.append(d.name)  # fallback to dir name
    if filter_str is None:
        return len(instance_ids)
    pattern = re.compile(filter_str)
    return sum(1 for iid in instance_ids if pattern.search(iid))


def build_runs(output_root: Path, resume: bool) -> list[tuple[str, list[str], str | None, str]]:
    """Return (run_id, command, filter_str, instances_path) tuples in execution order."""
    runs: list[tuple[str, list[str], str | None, str]] = []

    def add(run_id: str, preset: str, path: str, filt: str | None, flags: list[str]) -> None:
        runs.append((run_id, _cmd(run_id, preset, path, filt, output_root, flags, resume), filt, path))

    c2 = "SWE-agent/custom_cases_2"
    c1 = "SWE-agent/custom_cases"

    # --- Group A: c2 hard cases, all four variants ---
    add("A1_no_think_no_surface", "standard", c2, C2_HARD_FILTER,
        ["--no-thinking-mode", "--no-failure-surfacing"])

    add("B1_c1_no_think_no_surface", "standard", c1, C1_HARD_FILTER,
        ["--no-thinking-mode", "--no-failure-surfacing"])

    add("A2_surface_only", "standard", c2, C2_HARD_FILTER,
        ["--no-thinking-mode", "--failure-surfacing"])

    add("B2_c1_surface_only", "standard", c1, C1_HARD_FILTER,
        ["--no-thinking-mode", "--failure-surfacing"])

    # --- Group C: full 20-case runs for overall score comparison ---
    add("C1_c2_standard_new", "standard", c2, None,
        ["--no-thinking-mode", "--failure-surfacing"])

    add("C2_c1_standard_new", "standard", c1, None,
        ["--no-thinking-mode", "--failure-surfacing"])

    # --- Group D: UMich cluster models on c2 hard cases ---
    # NOTE: thinking_mode has no effect on openai/* models — reasoning_effort is
    # only wired for ollama/* models. These runs test whether larger models benefit
    # from failure surfacing, and give a raw capability baseline vs 9b+thinking.
    add("D1_30b_no_surface", "umich_30b", c2, C2_HARD_FILTER,
        ["--no-failure-surfacing"])

    add("D2_30b_surface", "umich_30b", c2, C2_HARD_FILTER,
        ["--failure-surfacing"])

    add("D3_120b_no_surface", "umich_120b", c2, C2_HARD_FILTER,
        ["--no-failure-surfacing"])

    add("D4_120b_surface", "umich_120b", c2, C2_HARD_FILTER,
        ["--failure-surfacing"])

    # --- Group A3/A4: Thinking mode variants (most expensive, run last) ---
    add("A3_think_only", "standard_think", c2, C2_HARD_FILTER,
        ["--thinking-mode", "--no-failure-surfacing"])

    add("A4_think_plus_surface", "standard_think", c2, C2_HARD_FILTER,
        ["--thinking-mode", "--failure-surfacing"])

    return runs


# ---------------------------------------------------------------------------
# Summarize helper
# ---------------------------------------------------------------------------

def _summarize(output_root: Path) -> None:
    runs = sorted(output_root.iterdir()) if output_root.exists() else []
    if not runs:
        print(f"No runs found under {output_root}")
        return

    print(f"\n{'Run':<35}  {'Pass':>4}  {'Total':>5}  {'%':>5}  Notes")
    print("-" * 70)

    baselines: dict[str, int] = {}

    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        trajs = list(run_dir.rglob("*.traj"))
        passed = 0
        total = len(trajs)
        for t in trajs:
            try:
                d = json.loads(t.read_text())
                if d.get("submitted") or d.get("info", {}).get("submitted"):
                    passed += 1
            except Exception:
                pass
        pct = f"{100 * passed / total:.0f}%" if total else "—"
        note = ""
        run_id = run_dir.name
        # Compare against matching A1/B1 baseline
        if run_id.startswith("A"):
            base_key = "A1"
        elif run_id.startswith("B"):
            base_key = "B1"
        else:
            base_key = None
        if base_key:
            baselines[run_id[:2]] = passed if run_id.endswith("1_no_think_no_surface") or run_id == f"{base_key}_no_think_no_surface" else baselines.get(run_id[:2], 0)
        if base_key and base_key in baselines and not run_id.endswith("no_think_no_surface"):
            delta = passed - baselines.get(base_key, 0)
            note = f"Δ{delta:+d} vs {base_key} baseline"
        print(f"{run_id:<35}  {passed:>4}  {total:>5}  {pct:>5}  {note}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("SWE-agent/tree_search_runs/phase3_ablation"),
        help="Root folder for ablation run outputs",
    )
    parser.add_argument("--execute", action="store_true",
                        help="Run commands instead of printing only")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip instances that already have a .traj file (default: on)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated run IDs to execute (e.g. A1,B1,A2)")
    parser.add_argument("--group", choices=["local", "remote"], default=None,
                        help="local = 9b Ollama runs (A+B+C groups); remote = UMich cluster runs (D group)")
    parser.add_argument("--summarize", action="store_true",
                        help="Print result summary from completed runs and exit")
    args = parser.parse_args()

    if args.summarize:
        _summarize(args.output_root)
        return

    only_ids: set[str] | None = None
    if args.only:
        only_ids = {s.strip() for s in args.only.split(",")}
    elif args.group == "local":
        only_ids = {"A1", "A2", "A3", "A4", "B1", "B2", "C1", "C2"}
    elif args.group == "remote":
        only_ids = {"D1", "D2", "D3", "D4"}

    runs = build_runs(args.output_root, args.resume)

    print(f"Phase-3 ablation — output root: {args.output_root}")
    print(f"{'Run ID':<35}  {'Status':<22}  Command")
    print("-" * 110)
    for run_id, cmd, filt, instances_path in runs:
        if only_ids and not any(run_id.startswith(oid) for oid in only_ids):
            print(f"{run_id:<35}  {'[group skipped]':<22}")
            continue

        run_dir = args.output_root / run_id
        done = _count_trajs(run_dir)
        expected = _expected_cases(filt, instances_path)
        status = f"{done}/{expected} done"

        if args.execute and done >= expected > 0:
            print(f"{run_id:<35}  {status + ' — skipping':<22}")
            continue

        printable = " ".join(cmd)
        print(f"{run_id:<35}  {status:<22}  {printable}")
        if args.execute:
            print(f"\n{'='*60}\nStarting: {run_id}  ({done}/{expected} already done)\n{'='*60}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\n[WARNING] {run_id} exited with code {result.returncode} — continuing")

    if args.execute:
        print("\nAll runs complete.")
        _summarize(args.output_root)


if __name__ == "__main__":
    main()
