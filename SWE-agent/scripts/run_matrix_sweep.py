#!/usr/bin/env python3
"""Run multiple matrix_easy model presets back to back."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from matrix_easy_common import default_results_root, preset_names, repo_root, sweep_names, resolve_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one named sweep of matrix model presets.")
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--list-sweeps", action="store_true")
    args = parser.parse_args()

    if args.list_sweeps:
        for sweep in sweep_names():
            print(sweep)
        return 0

    if args.sweep not in sweep_names():
        raise SystemExit(f"Unknown sweep '{args.sweep}'. Valid sweeps: {', '.join(sweep_names())}")

    exit_code = 0
    for preset in resolve_sweep(args.sweep):
        cmd = [
            sys.executable,
            str(repo_root() / "scripts" / "run_matrix_easy.py"),
            "--preset",
            preset,
            "--run-label",
            preset,
            "--results-root",
            str(args.results_root.resolve()),
        ]
        if args.variants:
            cmd.extend(["--variants", *args.variants])
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"[matrix_sweep] preset={preset}")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=repo_root())
        if completed.returncode != 0:
            exit_code = completed.returncode
            if args.stop_on_error:
                break

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
