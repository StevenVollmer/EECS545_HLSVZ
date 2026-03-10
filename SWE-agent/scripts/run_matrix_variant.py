#!/usr/bin/env python3
"""Print or run one generated matrix_easy variant command for a chosen model profile."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from matrix_easy_common import (
    ALL_VARIANTS,
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


def build_command(sweagent_bin: Path, generated_config: Path) -> list[str]:
    return [str(sweagent_bin), "run-batch", "--config", str(generated_config)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Print or run a matrix_easy command for one variant/profile.")
    parser.add_argument("variant", nargs="?", help="Variant name. Use --list to see valid names.")
    parser.add_argument("--list", action="store_true", help="List valid variants and exit.")
    parser.add_argument("--list-presets", action="store_true", help="List valid model presets and exit.")
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
        "--sweagent-bin",
        type=Path,
        default=default_sweagent_bin(),
        help="Path to the sweagent binary.",
    )
    parser.add_argument("--run", action="store_true", help="Execute the command instead of only printing it.")
    add_slot_args(parser, "small")
    add_slot_args(parser, "big")
    args = parser.parse_args()

    if args.list:
        for variant in ALL_VARIANTS:
            print(variant)
        return 0

    if args.list_presets:
        for preset in preset_names():
            print(preset)
        return 0

    if not args.variant:
        raise SystemExit("Provide a variant name or use --list.")
    if args.variant not in ALL_VARIANTS:
        raise SystemExit(f"Unknown variant '{args.variant}'. Valid variants: {', '.join(ALL_VARIANTS)}")
    if args.preset not in preset_names():
        raise SystemExit(f"Unknown preset '{args.preset}'. Valid presets: {', '.join(preset_names())}")

    run_label = args.run_label or args.preset
    results_root = args.results_root.resolve()
    config = build_variant_config(
        args.variant,
        args.preset,
        results_root / run_label,
        slot_overrides_from_args(args),
        instance_slice=args.instance_slice,
    )
    generated_config = generated_config_path(results_root, run_label, args.variant)
    write_yaml(generated_config, config)

    cmd = build_command(args.sweagent_bin, generated_config)
    print(" ".join(shlex.quote(part) for part in cmd))

    if not args.run:
        return 0

    env = os.environ.copy()
    env["PATH"] = f"{args.sweagent_bin.parent}:{env.get('PATH', '')}"
    completed = subprocess.run(cmd, cwd=repo_root(), env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
