#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


MODES = [
    ("coder_only", False, False),
    ("big_coder_only", False, False),
    ("planner_coder", True, False),
    ("planner_coder_reviewer", True, True),
]

DEFAULT_CONFIGS = [
    "SWE-agent/config/custom_configs/repair_pipeline_lmstudio.yaml",
    "SWE-agent/config/custom_configs/repair_pipeline_lmstudio_marshmallow.yaml",
    "SWE-agent/config/custom_configs/repair_pipeline_lmstudio_pydicom.yaml",
    "SWE-agent/config/custom_configs/repair_pipeline_lmstudio_p5js.yaml",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _slug_from_config(config: dict) -> str:
    return str(config["problem_statement"]["id"])


def _traj_path_for(output_dir: Path, problem_id: str) -> Path:
    return output_dir / problem_id / f"{problem_id}.traj"


def _extract_summary(traj_path: Path) -> dict:
    if not traj_path.exists():
        return {
            "traj_exists": False,
            "exit_status": "missing_traj",
            "submission_present": False,
            "instance_cost": None,
            "tokens_sent": None,
            "tokens_received": None,
            "api_calls": None,
            "role_model_stats": {},
            "validation_commands_run": 0,
            "validation_pass_count": 0,
            "validation_fail_count": 0,
            "validation_status": "missing_traj",
            "validation_examples": [],
        }

    data = json.loads(traj_path.read_text())
    info = data.get("info", {})
    stats = info.get("model_stats", {})
    submission = info.get("submission")
    validation_summary = _extract_validation_summary(data.get("trajectory", []))
    return {
        "traj_exists": True,
        "exit_status": info.get("exit_status"),
        "submission_present": bool(submission),
        "instance_cost": stats.get("instance_cost"),
        "tokens_sent": stats.get("tokens_sent"),
        "tokens_received": stats.get("tokens_received"),
        "api_calls": stats.get("api_calls"),
        "role_model_stats": info.get("role_model_stats", {}),
        **validation_summary,
    }


def _model_size_billions(model_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)b\b", model_name.lower())
    if not match:
        return None
    return float(match.group(1))


def _estimate_size_weighted_cost(role_model_stats: dict, role_models: dict[str, str]) -> tuple[float | None, dict[str, float]]:
    total = 0.0
    seen = False
    per_role: dict[str, float] = {}
    for role_name, stats in role_model_stats.items():
        size_b = _model_size_billions(role_models.get(role_name, ""))
        if size_b is None:
            continue
        tokens_sent = float(stats.get("tokens_sent", 0) or 0)
        tokens_received = float(stats.get("tokens_received", 0) or 0)
        # Output tokens are generally more expensive to generate than prompt tokens.
        proxy = size_b * (tokens_sent + 3.0 * tokens_received)
        per_role[role_name] = proxy
        total += proxy
        seen = True
    return (total if seen else None), per_role


def _flatten_role_stats(role_model_stats: dict) -> dict[str, int | float | None]:
    flat: dict[str, int | float | None] = {}
    for role in ("planner", "coder", "reviewer"):
        stats = role_model_stats.get(role, {}) or {}
        flat[f"{role}_tokens_sent"] = stats.get("tokens_sent")
        flat[f"{role}_tokens_received"] = stats.get("tokens_received")
        flat[f"{role}_api_calls"] = stats.get("api_calls")
        flat[f"{role}_instance_cost"] = stats.get("instance_cost")
    return flat


def _is_validation_command(action: str) -> bool:
    action = (action or "").lower()
    validation_markers = [
        "pytest",
        "py_compile",
        "python -m pytest",
        "python reproduce",
        "python reproduce_bug.py",
        "python tests/",
        "tox",
        "unittest",
        "nose",
        "npm test",
        "yarn test",
        "pnpm test",
        "jest",
        "vitest",
        "cargo test",
        "go test",
        "mvn test",
        "gradle test",
    ]
    return any(marker in action for marker in validation_markers)


def _observation_looks_like_failure(observation: str) -> bool:
    obs = (observation or "").lower()
    failure_markers = [
        "traceback",
        "failed",
        "error:",
        "assertionerror",
        "exception",
        "command not found",
        "no such file or directory",
        "collected 0 items / 1 error",
    ]
    return any(marker in obs for marker in failure_markers)


def _observation_looks_like_success(observation: str) -> bool:
    obs = (observation or "").lower()
    success_markers = [
        "passed",
        "ok",
        "script completed successfully",
        "result: true",
        "no errors",
    ]
    return any(marker in obs for marker in success_markers)


def _extract_validation_summary(trajectory: list[dict]) -> dict:
    examples: list[str] = []
    commands_run = 0
    pass_count = 0
    fail_count = 0

    for step in trajectory:
        action = str(step.get("action", "") or "")
        observation = str(step.get("observation", "") or "")
        if _is_validation_command(action):
            commands_run += 1
            if len(examples) < 5:
                examples.append(action.strip())
            if _observation_looks_like_failure(observation):
                fail_count += 1
            elif _observation_looks_like_success(observation):
                pass_count += 1

        if "handoff" in action and ("tests_run" in action or "test_results" in action):
            lowered = action.lower()
            if "passed" in lowered:
                pass_count += 1
            if "failed" in lowered or "error" in lowered:
                fail_count += 1

    if commands_run == 0 and pass_count == 0 and fail_count == 0:
        status = "not_run"
    elif fail_count > 0:
        status = "failed"
    elif pass_count > 0:
        status = "passed"
    else:
        status = "unknown"

    return {
        "validation_commands_run": commands_run,
        "validation_pass_count": pass_count,
        "validation_fail_count": fail_count,
        "validation_status": status,
        "validation_examples": examples,
    }


def _with_relative_costs(rows: list[dict]) -> list[dict]:
    baselines: dict[str, float] = {}
    big_baselines: dict[str, float] = {}
    for row in rows:
        if row.get("mode") == "coder_only":
            proxy = row.get("estimated_size_cost_proxy")
            if proxy not in (None, 0):
                baselines[str(row.get("problem_id"))] = float(proxy)
        if row.get("mode") == "big_coder_only":
            proxy = row.get("estimated_size_cost_proxy")
            if proxy not in (None, 0):
                big_baselines[str(row.get("problem_id"))] = float(proxy)

    updated: list[dict] = []
    for row in rows:
        row = dict(row)
        baseline = baselines.get(str(row.get("problem_id")))
        big_baseline = big_baselines.get(str(row.get("problem_id")))
        proxy = row.get("estimated_size_cost_proxy")
        if baseline and proxy is not None:
            row["relative_cost_vs_coder_only"] = float(proxy) / baseline
        else:
            row["relative_cost_vs_coder_only"] = None
        if big_baseline and proxy is not None:
            row["relative_cost_vs_big_coder_only"] = float(proxy) / big_baseline
        else:
            row["relative_cost_vs_big_coder_only"] = None
        updated.append(row)
    return updated


def _write_indexes(experiment_root: Path, rows: list[dict]) -> None:
    rows = _with_relative_costs(rows)
    json_path = experiment_root / "results_index.json"
    csv_path = experiment_root / "results_index.csv"
    md_path = experiment_root / "README.md"

    json_path.write_text(json.dumps(rows, indent=2))

    fieldnames = [
        "problem_id",
        "mode",
        "planner_enabled",
        "reviewer_enabled",
        "run_output_dir",
        "traj_path",
        "patch_path",
        "pred_path",
        "exit_status",
        "submission_present",
        "validation_status",
        "validation_commands_run",
        "validation_pass_count",
        "validation_fail_count",
        "tokens_sent",
        "tokens_received",
        "api_calls",
        "instance_cost",
        "planner_tokens_sent",
        "planner_tokens_received",
        "planner_api_calls",
        "coder_tokens_sent",
        "coder_tokens_received",
        "coder_api_calls",
        "reviewer_tokens_sent",
        "reviewer_tokens_received",
        "reviewer_api_calls",
        "estimated_size_cost_proxy",
        "relative_cost_vs_coder_only",
        "relative_cost_vs_big_coder_only",
        "returncode",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "# Repair Ablation Runs",
        "",
        f"- Total runs: {len(rows)}",
        f"- JSON index: `{json_path}`",
        f"- CSV index: `{csv_path}`",
        "- `relative_cost_vs_coder_only` is normalized within each problem, where coder-only = 1.0",
        "- `relative_cost_vs_big_coder_only` is normalized within each problem, where big_coder_only = 1.0",
        "- `validation_status` is inferred from trajectory/log evidence and is heuristic, not a full benchmark evaluator",
        "",
        "| Problem | Mode | Exit | Validation | Patch | Total Proxy | Rel Cost | P/C/R Proxy | P/C/R Calls | Run Dir |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        per_role_proxy = row.get("per_role_size_cost_proxy", {}) or {}
        per_role_calls = row.get("role_model_stats", {}) or {}
        lines.append(
            "| {problem_id} | {mode} | {exit_status} | {validation_status} ({validation_commands_run}) | {submission_present} | {estimated_size_cost_proxy} | {relative_cost_vs_coder_only} | p={planner_proxy}, c={coder_proxy}, r={reviewer_proxy} | p={planner_calls}, c={coder_calls}, r={reviewer_calls} | `{run_output_dir}` |".format(
                problem_id=row.get("problem_id"),
                mode=row.get("mode"),
                exit_status=row.get("exit_status"),
                validation_status=row.get("validation_status"),
                validation_commands_run=row.get("validation_commands_run"),
                submission_present=row.get("submission_present"),
                estimated_size_cost_proxy=row.get("estimated_size_cost_proxy"),
                relative_cost_vs_coder_only=row.get("relative_cost_vs_coder_only"),
                planner_proxy=per_role_proxy.get("planner"),
                coder_proxy=per_role_proxy.get("coder"),
                reviewer_proxy=per_role_proxy.get("reviewer"),
                planner_calls=(per_role_calls.get("planner", {}) or {}).get("api_calls"),
                coder_calls=(per_role_calls.get("coder", {}) or {}).get("api_calls"),
                reviewer_calls=(per_role_calls.get("reviewer", {}) or {}).get("api_calls"),
                run_output_dir=row.get("run_output_dir"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repair ablations across multiple problem configs.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Config files to use as base problem definitions.",
    )
    parser.add_argument(
        "--experiment-root",
        default="trajectories/repair_ablation_matrix",
        help="Directory where all runs, generated configs, and indexes are saved.",
    )
    parser.add_argument(
        "--sweagent-bin",
        default=None,
        help="Path to the sweagent executable. Defaults to env/bin/sweagent under the repo root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and indexes without executing sweagent.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    sweagent_bin = Path(args.sweagent_bin) if args.sweagent_bin else repo_root / "env" / "bin" / "sweagent"
    experiment_root = (repo_root / args.experiment_root).resolve()
    generated_dir = experiment_root / "generated_configs"
    runs_dir = experiment_root / "runs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for config_str in args.configs:
        config_path = (repo_root / config_str).resolve()
        base = _load_yaml(config_path)
        problem_id = _slug_from_config(base)

        for mode_name, enable_planner, enable_reviewer in MODES:
            config = json.loads(json.dumps(base))
            config["agent"]["enable_planner"] = enable_planner
            config["agent"]["enable_reviewer"] = enable_reviewer
            config["agent"]["model"]["per_instance_call_limit"] = 100
            if mode_name == "big_coder_only":
                config["agent"]["coder"] = config["agent"]["planner"]

            run_output_dir = runs_dir / problem_id / mode_name
            config["output_dir"] = str(run_output_dir)

            generated_config = generated_dir / f"{problem_id}__{mode_name}.yaml"
            _write_yaml(generated_config, config)

            command = [str(sweagent_bin), "run", "--config", str(generated_config)]
            result = None
            if not args.dry_run:
                result = subprocess.run(command, cwd=repo_root)

            traj_path = _traj_path_for(run_output_dir, problem_id)
            summary = _extract_summary(traj_path)
            role_models = {
                "planner": config["agent"].get("planner", ""),
                "coder": config["agent"].get("coder", ""),
                "reviewer": config["agent"].get("reviewer", ""),
            }
            estimated_size_cost_proxy, per_role_size_proxy = _estimate_size_weighted_cost(
                summary.get("role_model_stats", {}),
                role_models,
            )
            flat_role_stats = _flatten_role_stats(summary.get("role_model_stats", {}))
            rows.append(
                {
                    "problem_id": problem_id,
                    "mode": mode_name,
                    "planner_enabled": enable_planner,
                    "reviewer_enabled": enable_reviewer,
                    "planner_model": role_models["planner"],
                    "coder_model": role_models["coder"],
                    "reviewer_model": role_models["reviewer"],
                    "run_output_dir": str(run_output_dir),
                    "generated_config": str(generated_config),
                    "traj_path": str(traj_path),
                    "patch_path": str(run_output_dir / problem_id / f"{problem_id}.patch"),
                    "pred_path": str(run_output_dir / problem_id / f"{problem_id}.pred"),
                    "returncode": None if result is None else result.returncode,
                    "estimated_size_cost_proxy": estimated_size_cost_proxy,
                    "per_role_size_cost_proxy": per_role_size_proxy,
                    **flat_role_stats,
                    **summary,
                }
            )
            _write_indexes(experiment_root, rows)

    _write_indexes(experiment_root, rows)
    print(f"Wrote results to {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
