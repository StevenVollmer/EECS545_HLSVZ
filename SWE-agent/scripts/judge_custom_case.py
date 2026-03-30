#!/usr/bin/env python3
"""Judge custom case fixtures and run outputs against case-defined checks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CommandResult:
    name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    passed: bool
    failures: list[str]


def _resolve_case_file(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_dir():
        for name in ("case.json", "case.yaml", "case.yml"):
            nested = candidate / name
            if nested.exists():
                return nested
        raise ValueError(f"No case metadata found in {candidate}")
    return candidate


def _load_case(case_path: Path) -> tuple[dict[str, Any], Path]:
    case_file = _resolve_case_file(case_path)
    raw = yaml.safe_load(case_file.read_text())
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError(f"{case_file} must contain a one-item list of case objects")
    item = raw[0]
    repo_path_raw = item.get("repo_path", "repo")
    repo_path = Path(repo_path_raw)
    if not repo_path.is_absolute():
        repo_path = (case_file.parent / repo_path).resolve()
    else:
        repo_path = repo_path.resolve()
    return item, repo_path


def _run_shell(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(subprocess.os.environ)
    python_bin_dir = str(Path(sys.executable).parent)
    env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"
    return subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
        executable="/bin/zsh",
        env=env,
    )


def _run_check(check: dict[str, Any], cwd: Path) -> CommandResult:
    command = str(check["command"])
    result = _run_shell(command, cwd)
    failures: list[str] = []
    expected_exit = int(check.get("expect_exit_code", 0))
    if result.returncode != expected_exit:
        failures.append(f"expected exit code {expected_exit}, got {result.returncode}")
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    haystack = stdout + ("\n" + stderr if stderr else "")
    for text in check.get("stdout_contains", []):
        if text not in haystack:
            failures.append(f"missing expected text: {text!r}")
    for text in check.get("stdout_not_contains", []):
        if text in haystack:
            failures.append(f"unexpected text present: {text!r}")
    return CommandResult(
        name=str(check.get("name", command)),
        command=command,
        exit_code=result.returncode,
        stdout=stdout,
        stderr=stderr,
        passed=not failures,
        failures=failures,
    )


def _copy_repo_to_temp(repo_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="custom-case-judge-"))
    target = temp_dir / "repo"
    shutil.copytree(repo_path, target)
    return target


def _apply_patch(repo_path: Path, patch_path: Path) -> None:
    result = subprocess.run(
        ["git", "apply", str(patch_path.resolve())],
        cwd=str(repo_path),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to apply patch: {result.stderr.strip() or result.stdout.strip()}")


def _run_commands(commands: list[str], cwd: Path) -> list[CommandResult]:
    results: list[CommandResult] = []
    for command in commands:
        proc = _run_shell(command, cwd)
        failures: list[str] = []
        if proc.returncode != 0:
            failures.append(f"setup command failed with exit code {proc.returncode}")
        results.append(
            CommandResult(
                name=f"setup:{command}",
                command=command,
                exit_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                passed=not failures,
                failures=failures,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", dest="case_path", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "success", "patch"], default="baseline")
    parser.add_argument("--patch-file", type=Path)
    parser.add_argument("--run-install", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_item, repo_path = _load_case(args.case_path)
    evaluation = case_item.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")

    if args.mode == "baseline":
        checks = evaluation.get("baseline_checks", [])
        work_repo = repo_path
    else:
        checks = evaluation.get("success_checks", [])
        work_repo = repo_path

    cleanup_path: Path | None = None
    if args.mode == "patch":
        if args.patch_file is None:
            raise ValueError("--patch-file is required for --mode patch")
        cleanup_path = _copy_repo_to_temp(repo_path).parent
        work_repo = cleanup_path / "repo"
        _apply_patch(work_repo, args.patch_file)

    if not isinstance(checks, list):
        raise ValueError("Checks must be a list")

    setup_results: list[CommandResult] = []
    if args.run_install:
        install_commands = [str(x) for x in case_item.get("install_commands", [])]
        setup_commands = [str(x) for x in case_item.get("setup_commands", [])]
        setup_results.extend(_run_commands(install_commands + setup_commands, work_repo))

    check_results = [_run_check(check, work_repo) for check in checks]
    all_results = setup_results + check_results
    passed = all(result.passed for result in all_results)

    output = {
        "case": str(_resolve_case_file(args.case_path)),
        "repo_path": str(work_repo),
        "mode": args.mode,
        "passed": passed,
        "results": [
            {
                "name": result.name,
                "command": result.command,
                "exit_code": result.exit_code,
                "passed": result.passed,
                "failures": result.failures,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            for result in all_results
        ],
    }

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"case: {output['case']}")
        print(f"mode: {args.mode}")
        print(f"passed: {passed}")
        for result in all_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"\n[{status}] {result.name}")
            print(f"command: {result.command}")
            print(f"exit_code: {result.exit_code}")
            if result.failures:
                for failure in result.failures:
                    print(f"failure: {failure}")
            if result.stdout.strip():
                print("stdout:")
                print(result.stdout.rstrip())
            if result.stderr.strip():
                print("stderr:")
                print(result.stderr.rstrip())

    if cleanup_path is not None:
        shutil.rmtree(cleanup_path, ignore_errors=True)

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
