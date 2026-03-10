#!/usr/bin/env python3
"""Show high-signal progress for a matrix_easy results root."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def _candidate_roots(repo_root: Path) -> list[Path]:
    patterns = ["matrix_easy_runs_*", "matrix_easy_runs_full_*"]
    roots: list[Path] = []
    for pattern in patterns:
        roots.extend(path for path in repo_root.glob(pattern) if path.is_dir())
    return sorted(roots, key=lambda path: path.stat().st_mtime, reverse=True)


def _resolve_results_root(repo_root: Path, results_root: str | None) -> Path:
    if results_root is not None:
        path = Path(results_root)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        return path

    roots = _candidate_roots(repo_root)
    if not roots:
        raise SystemExit("No matrix results roots found.")
    return roots[0]


def _latest_activity(root: Path) -> tuple[Path | None, float | None]:
    latest_path: Path | None = None
    latest_mtime: float | None = None
    for pattern in ("*.traj", "*.debug.log", "*.info.log", "*.trace.log", "run_batch.log", "run_batch_exit_statuses.yaml"):
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
            if latest_mtime is None or mtime > latest_mtime:
                latest_path = path
                latest_mtime = mtime
    return latest_path, latest_mtime


def _count_incomplete_trajs(root: Path) -> int:
    incomplete = 0
    for traj_path in root.rglob("*.traj"):
        try:
            data = json.loads(traj_path.read_text())
        except Exception:
            incomplete += 1
            continue
        exit_status = data.get("info", {}).get("exit_status")
        if exit_status in (None, "early_exit"):
            incomplete += 1
    return incomplete


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Show progress for a matrix_easy results root.")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Results root to inspect. Defaults to the most recently modified matrix root.",
    )
    parser.add_argument(
        "--active-seconds",
        type=int,
        default=900,
        help="Treat the run as active if the newest tracked file is newer than this many seconds.",
    )
    args = parser.parse_args()

    results_root = _resolve_results_root(repo_root, args.results_root)
    latest_path, latest_mtime = _latest_activity(results_root)
    now = time.time()

    print(f"results_root: {results_root}")
    if latest_path is None or latest_mtime is None:
        print("status: no tracked activity files yet")
    else:
        idle_seconds = int(now - latest_mtime)
        state = "active" if idle_seconds <= args.active_seconds else "idle"
        print(f"status: {state} (idle {idle_seconds}s)")
        print(f"latest_file: {latest_path}")

    preset_dirs = sorted(path for path in results_root.iterdir() if path.is_dir() and not path.name.startswith("_"))
    print(f"presets_started: {len(preset_dirs)}")

    total_variant_dirs = 0
    total_status_files = 0
    total_counted_instances = 0
    for preset_dir in preset_dirs:
        variant_dirs = sorted(path for path in preset_dir.iterdir() if path.is_dir() and not path.name.startswith("_"))
        total_variant_dirs += len(variant_dirs)
        print(f"preset: {preset_dir.name} variants_started={len(variant_dirs)}")
        for variant_dir in variant_dirs:
            status_path = variant_dir / "run_batch_exit_statuses.yaml"
            if not status_path.exists():
                print(f"  {variant_dir.name}: in_progress")
                continue
            total_status_files += 1
            data = yaml.safe_load(status_path.read_text()) or {}
            statuses = data.get("instances_by_exit_status", {})
            counted = sum(len(items) for items in statuses.values())
            total_counted_instances += counted
            summary = ", ".join(f"{name}={len(items)}" for name, items in sorted(statuses.items()))
            print(f"  {variant_dir.name}: counted_instances={counted} [{summary}]")

    print(f"variant_dirs_started: {total_variant_dirs}")
    print(f"variant_status_files: {total_status_files}")
    print(f"counted_instances: {total_counted_instances}")
    print(f"incomplete_trajs: {_count_incomplete_trajs(results_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
