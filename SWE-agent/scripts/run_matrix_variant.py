#!/usr/bin/env python3
"""Print or run the exact command for one matrix_easy variant."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from matrix_easy_common import VARIANTS, config_path, default_sweagent_bin, repo_root


def build_command(sweagent_bin: Path, variant: str) -> list[str]:
    cfg = config_path(variant)
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config for variant '{variant}': {cfg}")
    return [str(sweagent_bin), "run-batch", "--config", str(cfg)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Print or run a single matrix_easy command.")
    parser.add_argument("variant", nargs="?", help="Variant name. Use --list to see valid names.")
    parser.add_argument("--list", action="store_true", help="List valid variants and exit.")
    parser.add_argument(
        "--sweagent-bin",
        type=Path,
        default=default_sweagent_bin(),
        help="Path to the sweagent binary.",
    )
    parser.add_argument("--run", action="store_true", help="Execute the command instead of only printing it.")
    args = parser.parse_args()

    if args.list:
        for variant in VARIANTS:
            print(variant)
        return 0

    if not args.variant:
        raise SystemExit("Provide a variant name or use --list.")
    if args.variant not in VARIANTS:
        raise SystemExit(f"Unknown variant '{args.variant}'. Valid variants: {', '.join(VARIANTS)}")

    cmd = build_command(args.sweagent_bin, args.variant)
    print(" ".join(shlex.quote(part) for part in cmd))

    if not args.run:
        return 0

    env = os.environ.copy()
    env["PATH"] = f"{args.sweagent_bin.parent}:{env.get('PATH', '')}"
    completed = subprocess.run(cmd, cwd=repo_root(), env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
