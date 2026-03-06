#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


MODES = [
    ("coder_only", False, False),
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
        }

    data = json.loads(traj_path.read_text())
    info = data.get("info", {})
    stats = info.get("model_stats", {})
    submission = info.get("submission")
    return {
        "traj_exists": True,
        "exit_status": info.get("exit_status"),
        "submission_present": bool(submission),
        "instance_cost": stats.get("instance_cost"),
        "tokens_sent": stats.get("tokens_sent"),
        "tokens_received": stats.get("tokens_received"),
        "api_calls": stats.get("api_calls"),
    }


def _write_indexes(experiment_root: Path, rows: list[dict]) -> None:
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
        "tokens_sent",
        "tokens_received",
        "api_calls",
        "instance_cost",
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
        "",
        "| Problem | Mode | Exit | Patch | Tokens Sent | Tokens Recv | API Calls | Run Dir |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {problem_id} | {mode} | {exit_status} | {submission_present} | {tokens_sent} | {tokens_received} | {api_calls} | `{run_output_dir}` |".format(
                problem_id=row.get("problem_id"),
                mode=row.get("mode"),
                exit_status=row.get("exit_status"),
                submission_present=row.get("submission_present"),
                tokens_sent=row.get("tokens_sent"),
                tokens_received=row.get("tokens_received"),
                api_calls=row.get("api_calls"),
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
            rows.append(
                {
                    "problem_id": problem_id,
                    "mode": mode_name,
                    "planner_enabled": enable_planner,
                    "reviewer_enabled": enable_reviewer,
                    "run_output_dir": str(run_output_dir),
                    "generated_config": str(generated_config),
                    "traj_path": str(traj_path),
                    "patch_path": str(run_output_dir / problem_id / f"{problem_id}.patch"),
                    "pred_path": str(run_output_dir / problem_id / f"{problem_id}.pred"),
                    "returncode": None if result is None else result.returncode,
                    **summary,
                }
            )
            _write_indexes(experiment_root, rows)

    _write_indexes(experiment_root, rows)
    print(f"Wrote results to {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
