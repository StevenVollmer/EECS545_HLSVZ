#!/usr/bin/env python3
"""Run the full local matrix_easy comparison set."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from matrix_easy_common import VARIANTS, config_path, default_sweagent_bin, repo_root


def run_variant(sweagent_bin: Path, variant: str, dry_run: bool) -> int:
    cfg = config_path(variant)
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config for variant '{variant}': {cfg}")

    cmd = [str(sweagent_bin), "run-batch", "--config", str(cfg)]
    env = os.environ.copy()
    env["PATH"] = f"{sweagent_bin.parent}:{env.get('PATH', '')}"

    print(f"[matrix_easy] {variant}")
    print(" ".join(cmd))
    if dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=repo_root(), env=env)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the configured matrix_easy variants.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=VARIANTS,
        help="Subset of variants to run. Defaults to the full matrix.",
    )
    parser.add_argument(
        "--sweagent-bin",
        type=Path,
        default=default_sweagent_bin(),
        help="Path to the sweagent binary.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first non-zero run-batch exit code.",
    )
    args = parser.parse_args()

    exit_code = 0
    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant '{variant}'. Valid variants: {', '.join(VARIANTS)}")
        rc = run_variant(args.sweagent_bin, variant, args.dry_run)
        if rc != 0:
            exit_code = rc
            if args.stop_on_error:
                break

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
