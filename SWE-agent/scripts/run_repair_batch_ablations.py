#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import yaml


MODES = [
    ("coder_only", False, False, False),
    ("big_coder_only", False, False, True),
    ("planner_coder", True, False, False),
    ("planner_coder_reviewer", True, True, False),
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


def _load_results_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _result_sets(results: dict) -> tuple[set[str], set[str]]:
    submitted = set(results.get("submitted_ids", []))
    resolved = set(results.get("resolved_ids", results.get("resolved", [])))
    return submitted, resolved


def _collect_rows(run_root: Path, mode: str) -> list[dict]:
    rows_by_id: dict[str, dict] = {}
    results_path = run_root / "results.json"
    results = _load_results_json(results_path)
    submitted_ids, resolved_ids = _result_sets(results)

    for traj_path in sorted(run_root.rglob("*.traj")):
        try:
            data = json.loads(traj_path.read_text())
        except Exception:
            continue
        info = data.get("info", {})
        row = {
            "mode": mode,
            "instance_id": traj_path.stem,
            "traj_path": str(traj_path),
            "patch_path": str(traj_path.with_suffix(".patch")),
            "pred_path": str(traj_path.with_suffix(".pred")),
            "exit_status": info.get("exit_status"),
            "submission_present": bool(info.get("submission")),
            "tokens_sent": (info.get("model_stats") or {}).get("tokens_sent"),
            "tokens_received": (info.get("model_stats") or {}).get("tokens_received"),
            "api_calls": (info.get("model_stats") or {}).get("api_calls"),
            "official_evaluated": traj_path.stem in submitted_ids,
            "official_resolved": traj_path.stem in resolved_ids,
            "results_json_path": str(results_path) if results_path.exists() else "",
        }
        rows_by_id[row["instance_id"]] = row

    for instance_id in sorted(submitted_ids | resolved_ids):
        if instance_id in rows_by_id:
            continue
        rows_by_id[instance_id] = {
            "mode": mode,
            "instance_id": instance_id,
            "traj_path": "",
            "patch_path": "",
            "pred_path": "",
            "exit_status": "missing_traj",
            "submission_present": instance_id in submitted_ids,
            "tokens_sent": None,
            "tokens_received": None,
            "api_calls": None,
            "official_evaluated": instance_id in submitted_ids,
            "official_resolved": instance_id in resolved_ids,
            "results_json_path": str(results_path) if results_path.exists() else "",
        }

    return [rows_by_id[key] for key in sorted(rows_by_id)]


def _write_indexes(experiment_root: Path, rows: list[dict]) -> None:
    json_path = experiment_root / "batch_results_index.json"
    csv_path = experiment_root / "batch_results_index.csv"
    md_path = experiment_root / "BATCH_README.md"

    json_path.write_text(json.dumps(rows, indent=2))

    fieldnames = [
        "mode",
        "instance_id",
        "exit_status",
        "submission_present",
        "tokens_sent",
        "tokens_received",
        "api_calls",
        "official_evaluated",
        "official_resolved",
        "traj_path",
        "patch_path",
        "pred_path",
        "results_json_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "# Official Batch Eval Runs",
        "",
        f"- JSON index: `{json_path}`",
        f"- CSV index: `{csv_path}`",
        "- `results.json` is the official SWE-bench evaluation artifact when present.",
        "- `official_resolved` comes from `results.json`, not trajectory heuristics.",
        "- `test-repo` is not part of SWE-bench batch evaluation and is intentionally excluded here.",
        "",
        "| Mode | Instance | Exit | Submitted | Evaluated | Resolved | Tokens In | Tokens Out | Calls | Traj |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {instance_id} | {exit_status} | {submission_present} | {official_evaluated} | {official_resolved} | {tokens_sent} | {tokens_received} | {api_calls} | `{traj_path}` |".format(
                mode=row.get("mode"),
                instance_id=row.get("instance_id"),
                exit_status=row.get("exit_status"),
                submission_present=row.get("submission_present"),
                official_evaluated=row.get("official_evaluated"),
                official_resolved=row.get("official_resolved"),
                tokens_sent=row.get("tokens_sent"),
                tokens_received=row.get("tokens_received"),
                api_calls=row.get("api_calls"),
                traj_path=row.get("traj_path"),
            )
        )
    lines.extend(["", "## Mode Totals", "", "| Mode | Evaluated | Resolved | Success Rate |", "| --- | ---: | ---: | ---: |"])
    for mode in sorted({str(row.get("mode")) for row in rows}):
        mode_rows = [row for row in rows if row.get("mode") == mode]
        evaluated = sum(bool(row.get("official_evaluated")) for row in mode_rows)
        resolved = sum(bool(row.get("official_resolved")) for row in mode_rows)
        success_rate = f"{(resolved / evaluated):.2f}" if evaluated else "n/a"
        lines.append(f"| {mode} | {evaluated} | {resolved} | {success_rate} |")
    md_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official SWE-bench batch evaluation for multi-agent ablations.")
    parser.add_argument(
        "--base-config",
        default="SWE-agent/config/custom_configs/repair_pipeline_lmstudio_batch_eval.yaml",
        help="Base run-batch config to expand into ablation modes.",
    )
    parser.add_argument(
        "--experiment-root",
        default="trajectories/repair_batch_eval_matrix",
        help="Directory for generated configs, run outputs, and summary indexes.",
    )
    parser.add_argument(
        "--sweagent-bin",
        default=None,
        help="Path to sweagent binary. Defaults to env/bin/sweagent under repo root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root()
    sweagent_bin = Path(args.sweagent_bin) if args.sweagent_bin else repo_root / "env" / "bin" / "sweagent"
    experiment_root = (repo_root / args.experiment_root).resolve()
    generated_dir = experiment_root / "generated_configs"
    runs_dir = experiment_root / "runs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_yaml((repo_root / args.base_config).resolve())
    all_rows: list[dict] = []

    for mode_name, enable_planner, enable_reviewer, big_coder in MODES:
        config = json.loads(json.dumps(base_config))
        config["agent"]["enable_planner"] = enable_planner
        config["agent"]["enable_reviewer"] = enable_reviewer
        config["agent"]["model"]["per_instance_call_limit"] = 50
        if big_coder:
            config["agent"]["coder"] = config["agent"]["planner"]

        run_output_dir = runs_dir / mode_name
        config["output_dir"] = str(run_output_dir)
        generated_config = generated_dir / f"{mode_name}.yaml"
        _write_yaml(generated_config, config)

        if not args.dry_run:
            subprocess.run([str(sweagent_bin), "run-batch", "--config", str(generated_config)], cwd=repo_root)

        all_rows.extend(_collect_rows(run_output_dir, mode_name))
        _write_indexes(experiment_root, all_rows)

    _write_indexes(experiment_root, all_rows)
    print(f"Wrote batch evaluation outputs to {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
