#!/usr/bin/env python3
"""Run a matrix of custom-runner experiments and analyze the results."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_SCRIPT = REPO_ROOT / "SWE-agent" / "scripts" / "run_custom_swebench.py"
ANALYZER_SCRIPT = REPO_ROOT / "SWE-agent" / "scripts" / "analyze_custom_runs.py"
PRESET_FILE = REPO_ROOT / "SWE-agent" / "config" / "custom_configs" / "custom_runner_model_presets.yaml"
CUSTOM_CASES_ROOT = REPO_ROOT / "SWE-agent" / "custom_cases"


@dataclass
class MatrixJob:
    preset: str
    architecture: str
    case_name: str
    case_path: Path
    instance_id: str
    output_dir: Path

    @property
    def run_name(self) -> str:
        return f"{self.preset}__{self.architecture}__{self.case_name}"


def _load_presets() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(PRESET_FILE.read_text()) or {}
    presets = raw.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError("Preset file must contain a top-level 'presets' mapping")
    return presets


def _load_cases(root: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case_file in sorted(root.glob("*/case.json")):
        raw = yaml.safe_load(case_file.read_text())
        if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
            raise ValueError(f"{case_file} must contain a one-item list")
        item = raw[0]
        case_name = case_file.parent.name
        cases[case_name] = {
            "case_path": case_file.parent,
            "instance_id": str(item["instance_id"]),
            "problem_statement": str(item.get("problem_statement", "")),
        }
    return cases


def _split_csv(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _job_manifest(job: MatrixJob) -> dict[str, Any]:
    return {
        "preset": job.preset,
        "architecture": job.architecture,
        "case_name": job.case_name,
        "case_path": str(job.case_path),
        "instance_id": job.instance_id,
        "output_dir": str(job.output_dir),
        "run_name": job.run_name,
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _build_runner_cmd(job: MatrixJob, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--preset",
        job.preset,
        "--agent-architecture",
        job.architecture,
        "--instances-type",
        "file",
        "--instances-path",
        str(job.case_path),
        "--filter",
        job.instance_id,
        "--output-dir",
        str(job.output_dir),
        "--run-name",
        job.run_name,
    ]
    if args.max_turns is not None:
        cmd.extend(["--max-turns", str(args.max_turns)])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.reviewer_rounds is not None:
        cmd.extend(["--reviewer-rounds", str(args.reviewer_rounds)])
    return cmd


def _build_analyzer_cmd(job: MatrixJob) -> list[str]:
    return [
        sys.executable,
        str(ANALYZER_SCRIPT),
        str(job.output_dir),
        "--json",
        "--write-json",
        str(job.output_dir / "analysis.json"),
    ]


def _execute_job(job: MatrixJob, args: argparse.Namespace) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(job.output_dir / "job_manifest.json", _job_manifest(job))

    result: dict[str, Any] = {
        **_job_manifest(job),
        "runner_exit_code": None,
        "analyzer_exit_code": None,
        "skipped": False,
    }
    traj_path = job.output_dir / job.instance_id / f"{job.instance_id}.traj"
    runner_cmd = _build_runner_cmd(job, args)
    analyzer_cmd = _build_analyzer_cmd(job)
    result["runner_cmd"] = runner_cmd
    result["analyzer_cmd"] = analyzer_cmd

    if args.skip_existing and traj_path.exists():
        result["runner_exit_code"] = 0
        result["skipped"] = True
    elif not args.analyze_only and not args.dry_run:
        runner_proc = subprocess.run(
            runner_cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
        )
        result["runner_exit_code"] = runner_proc.returncode
        _write_text(job.output_dir / "runner.stdout.log", runner_proc.stdout)
        _write_text(job.output_dir / "runner.stderr.log", runner_proc.stderr)
    elif not args.analyze_only:
        result["runner_exit_code"] = 0

    if not args.dry_run:
        analyzer_proc = subprocess.run(
            analyzer_cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
        )
        result["analyzer_exit_code"] = analyzer_proc.returncode
        _write_text(job.output_dir / "analyzer.stdout.log", analyzer_proc.stdout)
        _write_text(job.output_dir / "analyzer.stderr.log", analyzer_proc.stderr)
        preview = analyzer_proc.stdout.splitlines()[:12]
        result["analyzer_preview"] = preview
    else:
        result["analyzer_exit_code"] = 0
        result["analyzer_preview"] = []

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "SWE-agent" / "custom_matrix_runs" / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--presets", help="Comma-separated preset names. Default: all presets.")
    parser.add_argument(
        "--architectures",
        default="single,planner_coder,planner_coder_reviewer",
        help="Comma-separated architectures.",
    )
    parser.add_argument("--cases", help="Comma-separated case directory names. Default: all cases.")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--reviewer-rounds", type=int)
    parser.add_argument("--parallel", type=int, default=1, help="Number of jobs to run concurrently.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    presets = _load_presets()
    cases = _load_cases(CUSTOM_CASES_ROOT)

    selected_presets = _split_csv(args.presets) or sorted(presets.keys())
    selected_architectures = _split_csv(args.architectures)
    selected_cases = _split_csv(args.cases) or sorted(cases.keys())

    unknown_presets = sorted(set(selected_presets) - set(presets))
    unknown_cases = sorted(set(selected_cases) - set(cases))
    if unknown_presets:
        raise SystemExit(f"Unknown presets: {', '.join(unknown_presets)}")
    if unknown_cases:
        raise SystemExit(f"Unknown cases: {', '.join(unknown_cases)}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[MatrixJob] = []
    for preset in selected_presets:
        for architecture in selected_architectures:
            for case_name in selected_cases:
                case_meta = cases[case_name]
                jobs.append(
                    MatrixJob(
                        preset=preset,
                        architecture=architecture,
                        case_name=case_name,
                        case_path=Path(case_meta["case_path"]),
                        instance_id=str(case_meta["instance_id"]),
                        output_dir=output_root / preset / architecture / case_name,
                    )
                )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "python": sys.executable,
        "output_root": str(output_root),
        "jobs": [_job_manifest(job) for job in jobs],
    }
    _write_json(output_root / "matrix_manifest.json", manifest)

    print(f"output_root: {output_root}")
    print(f"jobs: {len(jobs)}")

    job_results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"[plan {index}/{len(jobs)}] {job.run_name}", flush=True)
        print(f"  runner: {' '.join(_build_runner_cmd(job, args))}", flush=True)
        print(f"  analyzer: {' '.join(_build_analyzer_cmd(job))}", flush=True)

    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1")

    if args.parallel == 1 or args.dry_run:
        for index, job in enumerate(jobs, start=1):
            print(f"\n[{index}/{len(jobs)}] {job.run_name}", flush=True)
            result = _execute_job(job, args)
            job_results.append(result)
            print(
                f"  finished: runner_exit={result['runner_exit_code']} analyzer_exit={result['analyzer_exit_code']}",
                flush=True,
            )
            for line in result.get("analyzer_preview", []):
                print(f"    {line}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_map = {executor.submit(_execute_job, job, args): job for job in jobs}
            completed = 0
            for future in concurrent.futures.as_completed(future_map):
                completed += 1
                job = future_map[future]
                result = future.result()
                job_results.append(result)
                print(
                    f"\n[{completed}/{len(jobs)}] {job.run_name} finished:"
                    f" runner_exit={result['runner_exit_code']}"
                    f" analyzer_exit={result['analyzer_exit_code']}",
                    flush=True,
                )
                for line in result.get("analyzer_preview", []):
                    print(f"    {line}")

    _write_json(output_root / "matrix_jobs.json", job_results)

    final_analyzer_cmd = [
        sys.executable,
        str(ANALYZER_SCRIPT),
        str(output_root),
        "--json",
        "--write-json",
        str(output_root / "analysis.summary.json"),
    ]
    print(f"\nfinal analyzer: {' '.join(final_analyzer_cmd)}", flush=True)
    if not args.dry_run:
        summary_result = subprocess.run(
            final_analyzer_cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
        )
        if summary_result.returncode == 0:
            print(summary_result.stdout)
        else:
            print(summary_result.stdout)
            print(summary_result.stderr, file=sys.stderr)


if __name__ == "__main__":
    main()
