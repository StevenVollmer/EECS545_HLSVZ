#!/usr/bin/env python3
"""Evaluate patches from .pred files against case.json success_checks.

Usage:
    python eval_patches.py <run_dir> [<cases_root>]

run_dir:    a directory tree containing .pred files with "model_patch" fields
cases_root: directory of case folders (each with case.json + repo/), default: SWE-agent/custom_cases
"""
import json, pathlib, subprocess, shutil, sys, tempfile

DEFAULT_CASES_ROOT = pathlib.Path(__file__).parent.parent.parent / "custom_cases"


def apply_and_check(patch: str, success_checks: list, repo_path: pathlib.Path) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_repo = pathlib.Path(tmpdir) / "repo"
        shutil.copytree(repo_path, tmp_repo)
        subprocess.run(["git", "init"], cwd=tmp_repo, capture_output=True)
        patch_file = pathlib.Path(tmpdir) / "changes.patch"
        patch_file.write_text(patch + "\n")
        r = subprocess.run(
            ["git", "apply", "--recount", "--ignore-whitespace", str(patch_file)],
            capture_output=True, text=True, cwd=tmp_repo,
        )
        if r.returncode != 0:
            return "patch_fail"
        for check in success_checks:
            r = subprocess.run(
                check["command"], shell=True, capture_output=True, text=True, cwd=tmp_repo
            )
            out = r.stdout + r.stderr
            if r.returncode != check.get("expect_exit_code", 0): return "fail"
            if any(s not in out for s in check.get("stdout_contains", [])): return "fail"
            if any(s in out for s in check.get("stdout_not_contains", [])): return "fail"
        return "pass"


def eval_dir(run_dir: pathlib.Path, cases_root: pathlib.Path) -> dict:
    results = {}
    for pred_file in sorted(run_dir.rglob("*.pred")):
        pred = json.loads(pred_file.read_text())
        patch = pred.get("model_patch", "").strip()
        instance_id = pred.get("instance_id", pred_file.stem)
        # Derive case name: strip trailing _NNN
        parts = instance_id.rsplit("_", 1)
        case_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else instance_id
        case_json = cases_root / case_name / "case.json"
        if not case_json.exists():
            results[instance_id] = "no_case"
            continue
        if not patch:
            results[instance_id] = "fail"
            continue
        case_data = json.loads(case_json.read_text())[0]
        repo_path = cases_root / case_name / case_data["repo_path"]
        results[instance_id] = apply_and_check(patch, case_data["evaluation"]["success_checks"], repo_path)
    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = pathlib.Path(sys.argv[1])
    cases_root = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CASES_ROOT

    results = eval_dir(run_dir, cases_root)
    passes = sum(1 for v in results.values() if v == "pass")
    total = len(results)
    pct = f"{100*passes//total}%" if total else "—"

    for inst, status in sorted(results.items()):
        mark = "✓" if status == "pass" else "✗"
        print(f"  {mark} {inst}: {status}")
    print(f"\n{run_dir.name}: {passes}/{total} = {pct}")


if __name__ == "__main__":
    main()
