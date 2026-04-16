#!/usr/bin/env python3
"""Verify submitted patches against each case's test suite.

run_tree_search.py considers an instance "submitted" when the agent called
submit().  That does NOT guarantee the patch is correct — the agent may have
submitted a broken or empty fix.  This script applies each .patch file inside
a Docker container (reusing the same image as run_tree_search.py) and runs the
case's own test suite to produce a ground-truth pass/fail verdict.

Usage:
  # Evaluate all submissions in a batch run
  python eval_patches.py SWE-agent/tree_search_runs/all_custom_run_v5

  # Only evaluate instances that were marked submitted
  python eval_patches.py run_dir --submitted-only

  # Write a verdict report
  python eval_patches.py run_dir --output verdicts.txt

  # Verbose: show test output for each case
  python eval_patches.py run_dir --verbose

Output (per instance):
  ✓ PASS  simple_mean_bug_001   (patch: 4 lines, 2 tests passed)
  ✗ FAIL  budget_snapshot_001   (patch: 0 lines — empty submission)
  ✗ FAIL  board_rollup_001      (3 tests, 1 failed: test_board_service.py::test_rollup_total)
  —       contact_card_001      (no patch — run errored)

Requires:
  - Docker available and running
  - Custom case directories at the path stored in the .traj file
    (same structure used by run_tree_search.py)
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DOCKER_IMAGE = "swe-agent"   # same image used by run_tree_search.py; override with --image
TIMEOUT_SECONDS = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_trajs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    found = sorted(root.rglob("*.traj"))
    if not found:
        print(f"No .traj files found under {root}", file=sys.stderr)
        sys.exit(1)
    return found


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def _patch_path(traj_path: Path) -> Path | None:
    stem = traj_path.stem
    candidate = traj_path.with_name(f"{stem}.patch")
    return candidate if candidate.exists() else None


def _find_case_dir(traj: dict[str, Any], traj_path: Path) -> Path | None:
    """Locate the custom case directory that the traj was run against."""
    # traj may store the case path explicitly
    case_dir = traj.get("case_dir") or traj.get("mcts_meta", {}).get("case_dir")
    if case_dir and Path(case_dir).is_dir():
        return Path(case_dir)

    # Fall back: scan sibling and parent dirs for case.json / case.yaml
    for candidate in [traj_path.parent, traj_path.parent.parent]:
        for name in ("case.json", "case.yaml", "case.yml"):
            f = candidate / name
            if f.exists():
                return candidate
    return None


def _read_case(case_dir: Path) -> dict[str, Any] | None:
    for name in ("case.json", "case.yaml", "case.yml"):
        f = case_dir / name
        if f.exists():
            import yaml
            return yaml.safe_load(f.read_text())
    return None


def _docker_available() -> bool:
    return shutil.which("docker") is not None


# ---------------------------------------------------------------------------
# Patch application + test run inside Docker
# ---------------------------------------------------------------------------

def _run_in_docker(
    image: str,
    repo_url: str,
    patch_text: str,
    test_command: str,
    post_startup: list[str],
    verbose: bool,
) -> dict[str, Any]:
    """
    Spin up a fresh container, apply the patch, run the test command.
    Returns {exit_code, stdout, stderr, error}.
    """
    if not _docker_available():
        return {"exit_code": -1, "stdout": "", "stderr": "", "error": "docker not found"}

    # Write patch to a temp file so we can bind-mount it
    with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as tf:
        tf.write(patch_text)
        patch_file = tf.name

    # Build shell script to run inside the container
    apply_patch = "git apply --whitespace=fix /patch.patch 2>&1 || patch -p1 < /patch.patch 2>&1"
    if not patch_text.strip():
        apply_patch = "echo '(empty patch — skipping apply)'"

    startup_cmds = "\n".join(post_startup) if post_startup else ""
    script = f"""
set -e
cd /repo
{startup_cmds}
{apply_patch}
{test_command}
"""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{patch_file}:/patch.patch:ro",
        "--network", "none",
        image,
        "bash", "-c", script,
    ]
    if verbose:
        print(f"  [docker] {' '.join(shlex.quote(c) for c in cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": "",
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "", "error": f"timeout after {TIMEOUT_SECONDS}s"}
    except Exception as exc:
        return {"exit_code": -1, "stdout": "", "stderr": "", "error": str(exc)}
    finally:
        Path(patch_file).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    traj_path: Path,
    image: str,
    submitted_only: bool,
    verbose: bool,
) -> dict[str, Any]:
    traj = _load(traj_path)
    iid = traj.get("instance_id", traj_path.stem)

    submitted = bool(traj.get("submitted", traj.get("info", {}).get("submitted", False)))
    error_msg = traj.get("error", "")

    if error_msg:
        return {"iid": iid, "verdict": "error", "detail": f"run errored: {error_msg[:80]}",
                "submitted": False}

    if not submitted:
        if submitted_only:
            return {"iid": iid, "verdict": "skip", "detail": "not submitted", "submitted": False}
        return {"iid": iid, "verdict": "not_submitted",
                "detail": f"stopped={traj.get('stopped_reason', '?')}", "submitted": False}

    # Load patch
    patch_path = _patch_path(traj_path)
    patch_text = patch_path.read_text() if patch_path else traj.get("patch", "")
    patch_lines = sum(1 for ln in patch_text.splitlines()
                      if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))

    if not patch_text.strip():
        return {"iid": iid, "verdict": "fail", "detail": "empty patch (submitted nothing)",
                "submitted": True, "patch_lines": 0}

    # Find case dir + metadata
    case_dir = _find_case_dir(traj, traj_path)
    if case_dir is None:
        return {"iid": iid, "verdict": "skip",
                "detail": "case directory not found — cannot run tests", "submitted": True,
                "patch_lines": patch_lines}

    case_data = _read_case(case_dir)
    items = case_data if isinstance(case_data, list) else [case_data]
    # Find the right item by instance_id
    item = next((i for i in items if isinstance(i, dict) and i.get("instance_id") == iid),
                items[0] if items else {})

    repo_url        = item.get("repo_url", item.get("repo", ""))
    test_command    = item.get("test_command", item.get("eval_command", "python -m pytest -q"))
    post_startup    = item.get("post_startup_commands", [])

    if not repo_url:
        return {"iid": iid, "verdict": "skip",
                "detail": "no repo_url in case — cannot verify",
                "submitted": True, "patch_lines": patch_lines}

    # Run inside Docker
    docker_result = _run_in_docker(
        image=image,
        repo_url=repo_url,
        patch_text=patch_text,
        test_command=test_command,
        post_startup=post_startup,
        verbose=verbose,
    )

    if docker_result["error"]:
        return {"iid": iid, "verdict": "error",
                "detail": docker_result["error"],
                "submitted": True, "patch_lines": patch_lines}

    passed = docker_result["exit_code"] == 0
    stdout = docker_result["stdout"]

    # Extract pytest summary line if present
    detail_lines = [ln for ln in stdout.splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
    detail = detail_lines[-1].strip() if detail_lines else f"exit={docker_result['exit_code']}"
    if verbose:
        print(f"  [output] {stdout[-500:]}")

    return {
        "iid": iid,
        "verdict": "pass" if passed else "fail",
        "detail": detail,
        "submitted": True,
        "patch_lines": patch_lines,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _icon(verdict: str) -> str:
    return {"pass": "✓ PASS", "fail": "✗ FAIL", "error": "✗ ERR ", "skip": "—     ",
            "not_submitted": "—     "}.get(verdict, "?     ")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path,
                        help="Batch run directory")
    parser.add_argument("--submitted-only", action="store_true",
                        help="Skip instances that were not submitted")
    parser.add_argument("--image", default=DOCKER_IMAGE,
                        help=f"Docker image to use (default: {DOCKER_IMAGE})")
    parser.add_argument("--verbose", action="store_true",
                        help="Show test output for each case")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write verdicts to this file")
    args = parser.parse_args()

    if not _docker_available():
        print("WARNING: docker not found — patch application will fail. "
              "Install Docker or run on a machine that has it.", file=sys.stderr)

    trajs = _find_trajs(args.run_dir)
    verdicts: list[dict] = []

    for traj_path in trajs:
        print(f"  evaluating {traj_path.stem}...", end=" ", flush=True)
        v = _evaluate(traj_path, args.image, args.submitted_only, args.verbose)
        verdicts.append(v)
        print(_icon(v["verdict"]))

    # Summary table
    lines: list[str] = ["", "─" * 80]
    lines.append(f"  {'Instance':<30}  {'Verdict':<8}  {'Patch':>6}  Detail")
    lines.append("─" * 80)
    for v in verdicts:
        patch_str = f"{v.get('patch_lines', 0):>4}L" if v.get("patch_lines") is not None else "    —"
        lines.append(f"  {v['iid']:<30}  {_icon(v['verdict']):<8}  {patch_str}  {v['detail'][:50]}")
    lines.append("─" * 80)

    n_pass  = sum(1 for v in verdicts if v["verdict"] == "pass")
    n_fail  = sum(1 for v in verdicts if v["verdict"] == "fail")
    n_err   = sum(1 for v in verdicts if v["verdict"] == "error")
    n_skip  = sum(1 for v in verdicts if v["verdict"] in ("skip", "not_submitted"))
    total   = len(verdicts)
    lines.append(f"  PASS={n_pass}  FAIL={n_fail}  ERR={n_err}  SKIP={n_skip}  "
                 f"TOTAL={total}  accuracy={100*n_pass/max(total-n_skip,1):.0f}%")
    lines.append("─" * 80)

    output = "\n".join(lines)
    print(output)
    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
