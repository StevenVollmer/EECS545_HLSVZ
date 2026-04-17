#!/usr/bin/env python3
"""Launcher for run_tree_search.py with named presets.

Eliminates the need to type multi-line CLI config each time.  Choose a preset
and add any overrides; the effective configuration is printed before the run
starts so it is always transparent.

Usage — custom cases
---------------------
  # Single case
  python SWE-agent/scripts/tree_search_custom/run_mcts.py --preset standard \
      --instances-type file \
      --instances-path SWE-agent/custom_cases/simple_mean_bug \
      --output-dir SWE-agent/tree_search_runs/my_run

  # All 20 custom cases (pass the parent directory)
  python SWE-agent/scripts/tree_search_custom/run_mcts.py --preset standard \
      --instances-type file \
      --instances-path SWE-agent/custom_cases \
      --output-dir SWE-agent/tree_search_runs/all_custom_run_v5

Usage — SWE-bench subset
-------------------------
  # 3 SWE-bench Lite dev cases
  python run_mcts.py --preset standard \
      --instances-type swe_bench --subset lite --split dev --slice 0:3 \
      --output-dir SWE-agent/tree_search_runs/swelite_run

  # Specific instance by filter regex
  python run_mcts.py --preset standard \
      --instances-type swe_bench --subset lite --filter pylint-dev__astroid-1866 \
      --output-dir SWE-agent/tree_search_runs/astroid_run

  # Verbose: show model prompts/responses during execution
  python run_mcts.py --preset debug --verbose \
      --instances-type file \
      --instances-path SWE-agent/custom_cases/simple_mean_bug \
      --output-dir SWE-agent/tree_search_runs/debug_run

  # List available presets
  python run_mcts.py --list-presets

Presets
-------
quick     : Fast smoke-test.  7b coder, 10 iterations, 3 vote samples.
standard  : Default balanced run.  9b model, 20 iterations, 5 vote samples, reviewer.  [DEFAULT]
thorough  : High-coverage run.  9b model, 40 iterations, 7 vote samples, 2 reviewer rounds.
debug     : Minimal run for quick iteration/debugging.  5 iterations, 1 vote sample, no planner.
"""

from __future__ import annotations

import argparse
import json
import re
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
        "max_node_depth": 10,   # equal to iterations
        "agent_architecture": "planner_coder",
    },
    "standard": {
        "model": "qwen3.5:9b",
        "planner_model": "qwen3.5:9b",
        "iterations": 18,       # max turns on a linear path; branching spends budget across paths
        "expansion_candidates": 1,
        "edit_vote_samples": 5,
        "max_node_depth": 18,   # equal to iterations: depth is the binding constraint
        "agent_architecture": "planner_coder_reviewer",
        "reviewer_rounds": 2,   # one retry after reviewer revise feedback
        "reviewer_gate_mode": "soft",  # ablation confirmed +1 case vs strict, no regressions
    },
    "thorough": {
        "model": "qwen3.5:9b",
        "planner_model": "qwen3.5:9b",
        "iterations": 40,       # extra budget forces branching once any path hits depth 20
        "expansion_candidates": 3,
        "edit_vote_samples": 7,
        "max_node_depth": 20,   # < iterations: encourages broad search over deep grinding
        "agent_architecture": "planner_coder_reviewer",
        "reviewer_rounds": 2,
    },
    "hard_case_phase2": {
        "model": "qwen3.5:9b",
        "planner_model": "qwen3.5:9b",
        "iterations": 18,
        "expansion_candidates": 1,
        "edit_vote_samples": 5,
        "max_node_depth": 18,
        "agent_architecture": "planner_coder_reviewer",
        "reviewer_rounds": 2,
        "reviewer_gate_mode": "soft",
        "adaptive_branching": False,  # ablation showed no benefit; adds cost and instability
    },
    "debug": {
        "model": "qwen2.5-coder:7b-instruct",
        "planner_model": "qwen3.5:9b",
        "iterations": 8,
        "expansion_candidates": 1,
        "edit_vote_samples": 1,
        "max_node_depth": 8,
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
    parser.add_argument("--reviewer-gate-mode", default=None, choices=["strict", "soft"])
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
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print model prompts/responses to stdout during execution")
    parser.add_argument("--adaptive-branching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip instances that already have a .traj file in output-dir")
    parser.add_argument("--failures-of", type=Path, default=None,
                        help="Run only the instances that did not submit in this run directory")

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
        "reviewer_gate_mode": args.reviewer_gate_mode,
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
        "verbose": args.verbose if args.verbose else None,
        "adaptive_branching": args.adaptive_branching,
        "resume": args.resume if args.resume else None,
    }
    for k, v in override_map.items():
        if v is not None:
            effective[k] = v

    # --failures-of: build a filter regex from non-submitted trajs in a prior run
    if args.failures_of is not None:
        if not args.failures_of.is_dir():
            parser.error(f"--failures-of: not a directory: {args.failures_of}")
        failed_ids: list[str] = []
        for traj_path in sorted(args.failures_of.rglob("*.traj")):
            try:
                traj = json.loads(traj_path.read_text())
            except Exception:
                continue
            submitted = bool(traj.get("submitted", traj.get("info", {}).get("submitted", False)))
            if not submitted:
                iid = traj.get("instance_id", traj_path.stem)
                failed_ids.append(iid)
        if not failed_ids:
            print(f"[failures-of] No failures found in {args.failures_of} — nothing to run.")
            sys.exit(0)
        filter_regex = "(" + "|".join(re.escape(iid) for iid in failed_ids) + ")"
        effective["filter"] = filter_regex
        print(f"[failures-of] {len(failed_ids)} failures: {failed_ids}")
        print(f"[failures-of] filter = {filter_regex}")

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

    bool_flags = {"shuffle", "verbose", "resume", "adaptive_branching"}
    list_flags = {"post_startup_command"}
    path_flags = {"instances_path", "output_dir"}

    # Key → CLI flag name mapping (underscores → hyphens)
    for key, value in cfg.items():
        flag = "--" + key.replace("_", "-")
        if key in bool_flags:
            if key == "adaptive_branching":
                if value is True:
                    argv.append("--adaptive-branching")
                elif value is False:
                    argv.append("--no-adaptive-branching")
            elif value:
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
