#!/usr/bin/env python3
"""Launcher for run_tree_search.py with named presets.

Eliminates the need to type multi-line CLI config each time.  Choose a preset
and add any overrides; the effective configuration is printed before the run
starts so it is always transparent.

Usage:
  python run_mcts.py --preset standard \
      --instances-type file \
      --instances-path SWE-agent/custom_cases/simple_mean_bug \
      --output-dir SWE-agent/tree_search_runs/my_run

  python run_mcts.py --preset quick --model deepseek-r1:7b \\
      --instances-type file \\
      --instances-path SWE-agent/custom_cases/simple_mean_bug \\
      --output-dir SWE-agent/tree_search_runs/my_run

  # List available presets
  python run_mcts.py --list-presets

Presets
-------
quick     : Fast smoke-test.  7b coder, 10 iterations, 3 vote samples.
standard  : Default balanced run.  9b model, 20 iterations, 5 vote samples.  [DEFAULT]
thorough  : High-coverage run.  9b model, 40 iterations, 7 vote samples, 3 edit candidates.
debug     : Minimal run for quick iteration/debugging.  5 iterations, 1 vote sample, no planner.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "model": "qwen2.5-coder:7b-instruct",
        "planner_model": "qwen3.5:9b",
        "iterations": 10,
        "expansion_candidates": 2,
        "edit_vote_samples": 3,
        "max_node_depth": 12,
        "agent_architecture": "planner_coder",
    },
    "standard": {
        "model": "qwen3.5:9b",
        "planner_model": "qwen3.5:9b",
        "iterations": 20,
        "expansion_candidates": 2,
        "edit_vote_samples": 5,
        "max_node_depth": 20,
        "agent_architecture": "planner_coder_reviewer",
        "reviewer_rounds": 1,
    },
    "thorough": {
        "model": "qwen3.5:9b",
        "planner_model": "qwen3.5:9b",
        "iterations": 40,
        "expansion_candidates": 3,
        "edit_vote_samples": 7,
        "max_node_depth": 25,
        "agent_architecture": "planner_coder_reviewer",
        "reviewer_rounds": 2,
    },
    "debug": {
        "model": "qwen2.5-coder:7b-instruct",
        "planner_model": "qwen3.5:9b",
        "iterations": 5,
        "expansion_candidates": 1,
        "edit_vote_samples": 1,
        "max_node_depth": 10,
        "agent_architecture": "single",
    },
}

DEFAULT_PRESET = "standard"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[str, dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )

    # Meta
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS),
                        help=f"Named preset (default: {DEFAULT_PRESET})")
    parser.add_argument("--list-presets", action="store_true",
                        help="Print all preset configurations and exit")

    # Required by run_tree_search.py
    parser.add_argument("--instances-type", choices=["swe_bench", "file"], default="file")
    parser.add_argument("--instances-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    # Overridable preset keys
    parser.add_argument("--model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--reviewer-model", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--expansion-candidates", type=int, default=None)
    parser.add_argument("--edit-vote-samples", type=int, default=None)
    parser.add_argument("--max-node-depth", type=int, default=None)
    parser.add_argument("--agent-architecture", default=None,
                        choices=["single", "planner_coder", "planner_coder_reviewer"])
    parser.add_argument("--reviewer-rounds", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--slice", default=None)
    parser.add_argument("--shuffle", action="store_true", default=None)
    parser.add_argument("--post-startup-command", action="append", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)

    args = parser.parse_args()

    if args.list_presets:
        _print_presets()
        sys.exit(0)

    if args.output_dir is None:
        parser.error("--output-dir is required")

    # Build effective config: preset defaults + CLI overrides
    effective: dict[str, Any] = dict(PRESETS[args.preset])

    # Map CLI args to their run_tree_search.py flag names
    override_map = {
        "model": args.model,
        "planner_model": args.planner_model,
        "reviewer_model": args.reviewer_model,
        "iterations": args.iterations,
        "expansion_candidates": args.expansion_candidates,
        "edit_vote_samples": args.edit_vote_samples,
        "max_node_depth": args.max_node_depth,
        "agent_architecture": args.agent_architecture,
        "reviewer_rounds": args.reviewer_rounds,
        "num_ctx": args.num_ctx,
        "max_tokens": args.max_tokens,
        "api_base": args.api_base,
        "api_key": args.api_key,
        "run_name": args.run_name,
        "filter": args.filter,
        "slice": args.slice,
        "shuffle": args.shuffle,
        "post_startup_command": args.post_startup_command,
        "subset": args.subset,
        "split": args.split,
        "instances_type": args.instances_type,
        "instances_path": args.instances_path,
        "output_dir": args.output_dir,
    }
    for k, v in override_map.items():
        if v is not None:
            effective[k] = v

    # Ensure required fields
    if "instances_path" not in effective and args.instances_type == "file":
        parser.error("--instances-path is required when --instances-type=file")

    return args.preset, effective


# ---------------------------------------------------------------------------
# Preset display
# ---------------------------------------------------------------------------

def _print_presets() -> None:
    for name, cfg in PRESETS.items():
        marker = " [DEFAULT]" if name == DEFAULT_PRESET else ""
        print(f"\n{name}{marker}:")
        for k, v in cfg.items():
            print(f"  {k} = {v!r}")


# ---------------------------------------------------------------------------
# Build argv for run_tree_search.py
# ---------------------------------------------------------------------------

def _cfg_to_argv(cfg: dict[str, Any]) -> list[str]:
    """Convert an effective config dict into a sys.argv list for run_tree_search.py."""
    argv = ["run_tree_search.py"]

    bool_flags = {"shuffle"}
    list_flags = {"post_startup_command"}
    path_flags = {"instances_path", "output_dir"}

    # Key → CLI flag name mapping (underscores → hyphens)
    for key, value in cfg.items():
        flag = "--" + key.replace("_", "-")
        if key in bool_flags:
            if value:
                argv.append(flag)
        elif key in list_flags:
            for item in (value or []):
                argv.extend([flag, str(item)])
        elif key in path_flags:
            if value is not None:
                argv.extend([flag, str(value)])
        else:
            if value is not None:
                argv.extend([flag, str(value)])

    return argv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    preset_name, effective_cfg = _parse_args()

    print(f"Preset: {preset_name}")
    print("Effective configuration:")
    for k, v in effective_cfg.items():
        print(f"  {k} = {v!r}")
    print()

    argv = _cfg_to_argv(effective_cfg)
    print("Invoking run_tree_search.py with:")
    print("  " + " ".join(argv[1:]))
    print()

    # Run in-process using the same Python environment
    _this_dir = Path(__file__).resolve().parent
    run_script = _this_dir / "run_tree_search.py"

    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(run_script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
