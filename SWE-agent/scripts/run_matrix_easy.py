#!/usr/bin/env python3
"""Run the full matrix_easy comparison set for one chosen model profile."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from matrix_easy_common import (
    ALL_VARIANTS,
    DEFAULT_VARIANTS,
    build_variant_config,
    default_results_root,
    default_sweagent_bin,
    preset_names,
    repo_root,
    write_yaml,
)


def add_slot_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-role-model", help=f"Override the {prefix} slot role model name.")
    parser.add_argument(f"--{prefix}-client-model", help=f"Override the {prefix} slot client model name.")
    parser.add_argument(f"--{prefix}-api-base", help=f"Override the {prefix} slot api_base.")
    parser.add_argument(f"--{prefix}-api-key", help=f"Override the {prefix} slot api_key.")
    parser.add_argument(
        f"--{prefix}-max-input-tokens",
        type=int,
        help=f"Override the {prefix} slot max_input_tokens.",
    )


def slot_overrides_from_args(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    overrides: dict[str, dict[str, object]] = {}
    for slot in ("small", "big"):
        slot_override = {
            "role_model_name": getattr(args, f"{slot}_role_model"),
            "client_model_name": getattr(args, f"{slot}_client_model"),
            "api_base": getattr(args, f"{slot}_api_base"),
            "api_key": getattr(args, f"{slot}_api_key"),
            "max_input_tokens": getattr(args, f"{slot}_max_input_tokens"),
        }
        slot_override = {key: value for key, value in slot_override.items() if value is not None}
        if slot_override:
            overrides[slot] = slot_override
    return overrides


def generated_config_path(results_root: Path, run_label: str, variant: str) -> Path:
    return results_root / run_label / "_generated_configs" / f"{variant}.yaml"


def run_variant(
    sweagent_bin: Path,
    results_root: Path,
    run_label: str,
    variant: str,
    preset: str,
    slot_overrides: dict[str, dict[str, object]],
    instance_slice: str | None,
    num_workers: int | None,
    dry_run: bool,
) -> int:
    generated_config = generated_config_path(results_root, run_label, variant)
    config = build_variant_config(
        variant,
        preset,
        results_root / run_label,
        slot_overrides,
        instance_slice=instance_slice,
        num_workers=num_workers,
    )
    write_yaml(generated_config, config)

    cmd = [str(sweagent_bin), "run-batch", "--config", str(generated_config)]
    env = os.environ.copy()
    env["PATH"] = f"{sweagent_bin.parent}:{env.get('PATH', '')}"

    print(f"[matrix_easy] preset={preset} variant={variant}")
    print(" ".join(cmd))
    if dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=repo_root(), env=env)
    return completed.returncode


def summarize_results(results_root: Path, run_label: str, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "summarize_latest_matrix_results.py"),
        "--root",
        str((results_root / run_label).resolve()),
    ]
    print("[matrix_easy] summarize")
    print(" ".join(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=repo_root())
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the configured matrix_easy variants for one model profile.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=DEFAULT_VARIANTS,
        help="Subset of variants to run. Defaults to the reduced default matrix.",
    )
    parser.add_argument(
        "--preset",
        default="qwen_local_35b_9b",
        help="Named model preset from config/custom_configs/matrix_easy/model_presets.yaml.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Output subdirectory label. Defaults to the preset name.",
    )
    parser.add_argument(
        "--instance-slice",
        default=None,
        help="Override the instance slice in the generated config, e.g. ':1' or '5:6'.",
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
        help="Set run-batch num_workers for each generated variant config.",
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
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not run the matrix summary script after the batch completes.",
    )
    parser.add_argument("--list-presets", action="store_true", help="List valid model presets and exit.")
    add_slot_args(parser, "small")
    add_slot_args(parser, "big")
    args = parser.parse_args()

    if args.list_presets:
        for preset in preset_names():
            print(preset)
        return 0
    if args.preset not in preset_names():
        raise SystemExit(f"Unknown preset '{args.preset}'. Valid presets: {', '.join(preset_names())}")

    run_label = args.run_label or args.preset
    slot_overrides = slot_overrides_from_args(args)
    exit_code = 0
    for variant in args.variants:
        if variant not in ALL_VARIANTS:
            raise SystemExit(f"Unknown variant '{variant}'. Valid variants: {', '.join(ALL_VARIANTS)}")
        rc = run_variant(
            args.sweagent_bin,
            args.results_root.resolve(),
            run_label,
            variant,
            args.preset,
            slot_overrides,
            args.instance_slice,
            args.num_workers,
            args.dry_run,
        )
        if rc != 0:
            exit_code = rc
            if args.stop_on_error:
                break

    if exit_code == 0 and not args.skip_summary:
        summary_rc = summarize_results(args.results_root.resolve(), run_label, args.dry_run)
        if summary_rc != 0:
            exit_code = summary_rc

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
