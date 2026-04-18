#!/usr/bin/env python3
"""Run a matrix of custom-runner experiments and analyze the results."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SCRIPT = REPO_ROOT / "SWE-agent" / "scripts" / "custom" / "run_custom_swebench.py"
ANALYZER_SCRIPT = REPO_ROOT / "SWE-agent" / "scripts" / "custom" / "analyze_custom_runs.py"
PRESET_FILE = REPO_ROOT / "SWE-agent" / "config" / "custom_configs" / "custom_runner_model_presets.yaml"
DEFAULT_CASE_ROOTS = [
    REPO_ROOT / "SWE-agent" / "custom_cases",
    REPO_ROOT / "SWE-agent" / "custom_cases_2",
    REPO_ROOT / "SWE-agent" / "custom_cases_3",
]
TRANSIENT_RUNNER_ERROR_MARKERS = (
    "Runtime did not start within timeout",
    "ClientConnectorError:",
    "Cannot connect to host 127.0.0.1:",
    "swerex-remote: not found",
    "DockerPullError:",
)


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


def _load_cases(roots: list[Path]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        for case_file in sorted(root.glob("*/case.json")):
            raw = yaml.safe_load(case_file.read_text())
            if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
                raise ValueError(f"{case_file} must contain a one-item list")
            item = raw[0]
            case_name = case_file.parent.name
            if case_name in cases:
                existing = cases[case_name]["case_path"]
                raise ValueError(f"Duplicate case name '{case_name}' in {existing} and {case_file.parent}")
            cases[case_name] = {
                "case_path": case_file.parent,
                "instance_id": str(item["instance_id"]),
                "problem_statement": str(item.get("problem_statement", "")),
            }
    return cases


def _credential_ready(preset_name: str, preset: dict[str, Any]) -> tuple[bool, str | None]:
    if preset.get("api_key"):
        return True, None
    backend = str(preset.get("backend", ""))
    if backend == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            return True, None
        return False, "missing OPENAI_API_KEY"
    if backend == "umich":
        env_name = "UMICH_API_KEY"
        if os.environ.get(env_name):
            return True, None
        # UMich presets usually carry explicit api_key in the preset, so falling
        # back to env-only would be unexpected but still worth checking.
        return False, f"missing {env_name} and preset has no api_key"
    if backend == "ollama":
        return True, None
    if backend == "lmstudio":
        return True, None
    return True, None


def _is_split_preset(preset: dict[str, Any]) -> bool:
    return bool(preset.get("planner_model") or preset.get("reviewer_model"))


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


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def _runner_failed_before_agent_loop(stdout_text: str, stderr_text: str) -> bool:
    combined = f"{stdout_text}\n{stderr_text}"
    if "[turn 1] calling model" in combined:
        return False
    return any(marker in combined for marker in TRANSIENT_RUNNER_ERROR_MARKERS)


def _is_completed_job(job: MatrixJob) -> bool:
    instance_dir = job.output_dir / job.instance_id
    traj_path = instance_dir / f"{job.instance_id}.traj"
    analysis_path = job.output_dir / "analysis.json"
    if not traj_path.exists() or not analysis_path.exists():
        return False
    try:
        traj = json.loads(traj_path.read_text())
        analysis = json.loads(analysis_path.read_text())
    except Exception:
        return False
    if not isinstance(traj, dict) or not isinstance(analysis, dict):
        return False
    stopped_reason = str(traj.get("stopped_reason", "") or "")
    if not stopped_reason:
        return False
    results = analysis.get("results", [])
    if not isinstance(results, list) or not results:
        return False
    return True


def _cleanup_partial_job(job: MatrixJob) -> None:
    if not job.output_dir.exists():
        return
    for child in job.output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _execute_job(job: MatrixJob, args: argparse.Namespace) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(job.output_dir / "job_manifest.json", _job_manifest(job))

    result: dict[str, Any] = {
        **_job_manifest(job),
        "runner_exit_code": None,
        "analyzer_exit_code": None,
        "skipped": False,
        "skip_reason": None,
        "runner_attempts": 0,
    }
    runner_cmd = _build_runner_cmd(job, args)
    analyzer_cmd = _build_analyzer_cmd(job)
    result["runner_cmd"] = runner_cmd
    result["analyzer_cmd"] = analyzer_cmd

    partial_cleanup = False
    if args.skip_existing and _is_completed_job(job):
        result["runner_exit_code"] = 0
        result["skipped"] = True
        result["skip_reason"] = "existing completed job"
    elif args.skip_existing and job.output_dir.exists() and any(job.output_dir.iterdir()):
        _cleanup_partial_job(job)
        partial_cleanup = True
        result["partial_cleanup"] = True
    if args.skip_presets and job.preset in args.skip_presets and not result["skipped"]:
        result["runner_exit_code"] = 0
        result["skipped"] = True
        result["skip_reason"] = f"preset preflight failed: {args.skip_presets[job.preset]}"
    elif not result["skipped"] and not args.analyze_only and not args.dry_run:
        max_attempts = max(1, int(args.runner_retries) + 1)
        runner_proc: subprocess.CompletedProcess[str] | None = None
        runner_stdout = ""
        runner_stderr = ""
        for attempt in range(1, max_attempts + 1):
            result["runner_attempts"] = attempt
            runner_proc = subprocess.run(
                runner_cmd,
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
            )
            runner_stdout = runner_proc.stdout
            runner_stderr = runner_proc.stderr
            if (
                runner_proc.returncode == 0
                or attempt >= max_attempts
                or not _runner_failed_before_agent_loop(runner_stdout, runner_stderr)
            ):
                break
            time_marker = f"\n\n[retry {attempt}/{max_attempts - 1}] transient runner/bootstrap failure detected, retrying job\n"
            runner_stdout += time_marker
        result["runner_exit_code"] = runner_proc.returncode if runner_proc else 1
        _write_text(job.output_dir / "runner.stdout.log", runner_stdout)
        _write_text(job.output_dir / "runner.stderr.log", runner_stderr)
    elif not result["skipped"] and not args.analyze_only:
        result["runner_exit_code"] = 0

    should_analyze = not result["skipped"]
    if not args.dry_run and should_analyze:
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
    elif result["skipped"]:
        result["analyzer_exit_code"] = 0
        result["analyzer_preview"] = [f"skipped analysis: {result['skip_reason']}"]
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
    parser.add_argument("--runner-retries", type=int, default=1, help="Retries for transient runner/bootstrap failures.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-single-for-split-presets",
        action="store_true",
        default=True,
        help="Do not schedule `single` architecture jobs for presets that already define planner/reviewer split models.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    presets = _load_presets()
    cases = _load_cases(DEFAULT_CASE_ROOTS)

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
        preset_cfg = presets[preset]
        for architecture in selected_architectures:
            if args.skip_single_for_split_presets and architecture == "single" and _is_split_preset(preset_cfg):
                continue
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
    skip_presets: dict[str, str] = {}
    for preset_name in selected_presets:
        ok, reason = _credential_ready(preset_name, presets[preset_name])
        if not ok and reason:
            skip_presets[preset_name] = reason
    args.skip_presets = skip_presets
    if skip_presets:
        manifest["skipped_presets"] = skip_presets
    _write_json(output_root / "matrix_manifest.json", manifest)

    print(f"output_root: {output_root}")
    print(f"jobs: {len(jobs)}")
    if skip_presets:
        print("skipping presets due to missing credentials:", flush=True)
        for preset_name, reason in sorted(skip_presets.items()):
            print(f"  - {preset_name}: {reason}", flush=True)

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
