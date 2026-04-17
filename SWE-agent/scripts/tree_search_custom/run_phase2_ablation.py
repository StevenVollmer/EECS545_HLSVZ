#!/usr/bin/env python3
"""Build or execute Phase-2 ablation runs for tree_search_custom.

By default, prints commands only. Use --execute to run them.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _cmd(run_label: str, extra: list[str], base_output_root: Path) -> list[str]:
    return [
        "python3",
        "SWE-agent/scripts/tree_search_custom/run_mcts.py",
        "--preset",
        "hard_case_phase2",
        "--instances-type",
        "file",
        "--instances-path",
        "SWE-agent/custom_cases",
        "--filter",
        "(budget_snapshot_001|contact_card_001|digest_preview_001|nested_app_001|owner_recap_001|renewal_preview_001|shipment_preview_001)",
        "--output-dir",
        str(base_output_root / run_label),
        "--run-name",
        run_label,
        *extra,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("SWE-agent/tree_search_runs/phase2_ablation"),
        help="Root folder for ablation run outputs",
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing only")
    parser.add_argument("--resume", action="store_true", help="Add --resume to each run command")
    args = parser.parse_args()

    variants: list[tuple[str, list[str]]] = [
        ("ablation_baseline_strict", ["--reviewer-gate-mode", "strict", "--no-adaptive-branching"]),
        ("ablation_soft_gate_only", ["--reviewer-gate-mode", "soft", "--no-adaptive-branching"]),
        ("ablation_soft_gate_plus_adaptive", ["--reviewer-gate-mode", "soft", "--adaptive-branching"]),
        ("ablation_soft_gate_plus_adaptive_highvote", ["--reviewer-gate-mode", "soft", "--adaptive-branching", "--edit-vote-samples", "7"]),
    ]

    if args.resume:
        variants = [(name, flags + ["--resume"]) for name, flags in variants]

    commands = [_cmd(name, flags, args.output_root) for name, flags in variants]

    for cmd in commands:
        printable = " ".join(cmd)
        print(printable)
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

