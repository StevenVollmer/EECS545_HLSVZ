#!/usr/bin/env python3
"""Run matrix sweeps with freeze detection and automatic resume."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from matrix_easy_common import default_results_root, repo_root, sweep_names


def _build_sweep_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "run_matrix_sweep.py"),
        "--sweep",
        args.sweep,
        "--results-root",
        str(args.results_root.resolve()),
    ]
    if args.variants:
        cmd.extend(["--variants", *args.variants])
    if args.instance_slice is not None:
        cmd.extend(["--instance-slice", args.instance_slice])
    if args.max_presets is not None:
        cmd.extend(["--max-presets", str(args.max_presets)])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.stop_on_error:
        cmd.append("--stop-on-error")
    return cmd


def _latest_activity_ts(results_root: Path) -> float | None:
    if not results_root.exists():
        return None

    latest: float | None = None
    patterns = [
        "*.traj",
        "*.debug.log",
        "*.info.log",
        "*.trace.log",
        "run_batch.log",
        "run_batch_exit_statuses.yaml",
        "preds.json",
        "tmppreds.json",
    ]
    for pattern in patterns:
        for path in results_root.rglob(pattern):
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
            if latest is None or mtime > latest:
                latest = mtime
    return latest


def _terminate_process_group(proc: subprocess.Popen[bytes], grace_seconds: int) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a matrix sweep with freeze detection and auto-resume.")
    parser.add_argument(
        "--sweep",
        default="default",
        help="Sweep name from config/custom_configs/matrix_easy/model_presets.yaml.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root(),
        help="Root directory where generated configs and run outputs are written.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional subset of variants to run for every preset.",
    )
    parser.add_argument(
        "--instance-slice",
        default=None,
        help="Override the instance slice in every generated config, e.g. ':1' or '5:6'.",
    )
    parser.add_argument(
        "--max-presets",
        type=int,
        default=None,
        help="Optional cap on how many presets from the sweep to run, in listed order.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--freeze-seconds",
        type=int,
        default=1800,
        help="Restart the sweep if no results-root file activity occurs for this many seconds.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="How often to check for activity while the sweep is running.",
    )
    parser.add_argument(
        "--grace-seconds",
        type=int,
        default=30,
        help="How long to wait after SIGTERM before SIGKILL when restarting a frozen run.",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=20,
        help="Maximum number of automatic restarts before giving up.",
    )
    parser.add_argument(
        "--no-restart-on-nonzero",
        action="store_true",
        help="Do not restart automatically if the child sweep exits non-zero.",
    )
    parser.add_argument("--list-sweeps", action="store_true")
    args = parser.parse_args()

    if args.list_sweeps:
        for sweep in sweep_names():
            print(sweep)
        return 0
    if args.sweep not in sweep_names():
        raise SystemExit(f"Unknown sweep '{args.sweep}'. Valid sweeps: {', '.join(sweep_names())}")
    if args.freeze_seconds < 1 or args.poll_seconds < 1 or args.grace_seconds < 1:
        raise SystemExit("freeze/poll/grace seconds must all be at least 1")
    if args.max_restarts < 0:
        raise SystemExit("--max-restarts must be non-negative")

    cmd = _build_sweep_cmd(args)
    restart_on_nonzero = not args.no_restart_on_nonzero
    restart_count = 0

    while True:
        print(f"[matrix_watchdog] starting ({restart_count=})")
        print(" ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=repo_root(), start_new_session=True)
        start_time = time.time()

        while True:
            try:
                return_code = proc.wait(timeout=args.poll_seconds)
            except subprocess.TimeoutExpired:
                return_code = None

            if return_code is not None:
                if return_code == 0:
                    print("[matrix_watchdog] child exited cleanly")
                    return 0
                print(f"[matrix_watchdog] child exited non-zero: {return_code}")
                if not restart_on_nonzero or restart_count >= args.max_restarts:
                    return return_code
                restart_count += 1
                break

            latest_activity = _latest_activity_ts(args.results_root)
            reference_time = max(start_time, latest_activity or 0.0)
            idle_seconds = time.time() - reference_time
            if idle_seconds > args.freeze_seconds:
                print(
                    "[matrix_watchdog] no activity detected for "
                    f"{int(idle_seconds)}s under {args.results_root}. Restarting child."
                )
                _terminate_process_group(proc, args.grace_seconds)
                if restart_count >= args.max_restarts:
                    print("[matrix_watchdog] max restarts reached")
                    return 124
                restart_count += 1
                break


if __name__ == "__main__":
    raise SystemExit(main())
