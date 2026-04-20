#!/usr/bin/env python3
"""Run multiple matrix_easy model presets back to back."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from matrix_easy_common import default_python_bin, default_results_root, preset_names, repo_root, sweep_names, resolve_sweep


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
        "--num-workers",
        type=int,
        default=None,
        help="Set run-batch num_workers for each preset run.",
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
    parser.add_argument("--list-sweeps", action="store_true")
    args = parser.parse_args()

    if args.list_sweeps:
        for sweep in sweep_names():
            print(sweep)
        return 0

    if args.sweep not in sweep_names():
        raise SystemExit(f"Unknown sweep '{args.sweep}'. Valid sweeps: {', '.join(sweep_names())}")

    exit_code = 0
    presets = resolve_sweep(args.sweep)
    if args.max_presets is not None:
        if args.max_presets < 1:
            raise SystemExit("--max-presets must be at least 1")
        presets = presets[: args.max_presets]

    for preset in presets:
        cmd = [
            str(default_python_bin()),
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
        if args.instance_slice is not None:
            cmd.extend(["--instance-slice", args.instance_slice])
        if args.num_workers is not None:
            cmd.extend(["--num-workers", str(args.num_workers)])
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
