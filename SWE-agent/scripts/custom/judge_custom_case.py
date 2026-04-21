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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "failures": self.failures,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


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
        timeout=60,
        executable="/bin/zsh",
        env=env,
    )


def _run_check(check: dict[str, Any], cwd: Path) -> CommandResult:
    command = str(check["command"])
    try:
        result = _run_shell(command, cwd)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=str(check.get("name", command)),
            command=command,
            exit_code=-1,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + "\n[JUDGE] command timed out after 60s",
            passed=False,
            failures=["command timed out after 60s"],
        )
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


def _filtered_patch_text(patch_text: str) -> str:
    if not patch_text.strip():
        return patch_text

    keep_blocks: list[str] = []
    for block in patch_text.split("diff --git "):
        if not block:
            continue
        full_block = "diff --git " + block
        header = full_block.splitlines()[0] if full_block.splitlines() else ""
        lowered = header.lower()
        if "__pycache__" in lowered or lowered.endswith(".pyc") or ".pyc " in lowered:
            continue
        keep_blocks.append(full_block)
    return "".join(keep_blocks)


def _apply_patch(repo_path: Path, patch_path: Path) -> None:
    filtered_patch = _filtered_patch_text(patch_path.read_text())
    if not filtered_patch.strip():
        return
    temp_patch = repo_path / ".judge_filtered.patch"
    temp_patch.write_text(filtered_patch)
    result = subprocess.run(
        ["git", "apply", str(temp_patch.resolve())],
        cwd=str(repo_path),
        text=True,
        capture_output=True,
    )
    temp_patch.unlink(missing_ok=True)
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


def evaluate_case(
    *,
    case_path: Path,
    mode: str,
    patch_file: Path | None = None,
    run_install: bool = False,
) -> dict[str, Any]:
    case_item, repo_path = _load_case(case_path)
    evaluation = case_item.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")

    if mode == "baseline":
        checks = evaluation.get("baseline_checks", [])
        work_repo = repo_path
    elif mode in {"success", "patch"}:
        checks = evaluation.get("success_checks", [])
        work_repo = repo_path
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cleanup_path: Path | None = None
    if mode == "patch":
        if patch_file is None:
            raise ValueError("patch_file is required for mode='patch'")
        cleanup_path = _copy_repo_to_temp(repo_path).parent
        work_repo = cleanup_path / "repo"
        _apply_patch(work_repo, patch_file)

    if not isinstance(checks, list):
        raise ValueError("Checks must be a list")

    setup_results: list[CommandResult] = []
    if run_install:
        install_commands = [str(x) for x in case_item.get("install_commands", [])]
        setup_commands = [str(x) for x in case_item.get("setup_commands", [])]
        setup_results.extend(_run_commands(install_commands + setup_commands, work_repo))

    check_results = [_run_check(check, work_repo) for check in checks]
    all_results = setup_results + check_results
    passed = all(result.passed for result in all_results)

    output = {
        "case": str(_resolve_case_file(case_path)),
        "repo_path": str(work_repo),
        "mode": mode,
        "passed": passed,
        "results": [result.to_dict() for result in all_results],
    }

    if cleanup_path is not None:
        shutil.rmtree(cleanup_path, ignore_errors=True)

    return output


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
    output = evaluate_case(
        case_path=args.case_path,
        mode=args.mode,
        patch_file=args.patch_file,
        run_install=args.run_install,
    )
    all_results = output["results"]
    passed = bool(output["passed"])

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"case: {output['case']}")
        print(f"mode: {args.mode}")
        print(f"passed: {passed}")
        for result in all_results:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"\n[{status}] {result['name']}")
            print(f"command: {result['command']}")
            print(f"exit_code: {result['exit_code']}")
            if result["failures"]:
                for failure in result["failures"]:
                    print(f"failure: {failure}")
            if result["stdout"].strip():
                print("stdout:")
                print(result["stdout"].rstrip())
            if result["stderr"].strip():
                print("stderr:")
                print(result["stderr"].rstrip())

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
