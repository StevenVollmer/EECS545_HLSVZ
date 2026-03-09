#!/usr/bin/env python3
"""Build a compact index for latest_matrix_easy_results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from analyze_traj_quality import score_traj


VARIANTS = [
    "small_coder_only",
    "big_coder_only",
    "big_planner_small_coder",
    "big_planner_big_coder",
    "big_planner_small_coder_small_reviewer",
    "big_planner_small_coder_big_reviewer",
    "all_3_big",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def latest_root() -> Path:
    return repo_root() / "latest_matrix_easy_results"


def read_exit_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text().splitlines()
    current = None
    for line in text:
        stripped = line.strip()
        if stripped.endswith(":") and stripped not in {"instances_by_exit_status:", "total_cost:"}:
            current = stripped[:-1]
    return current or "unknown"


def read_traj_score(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "quality_score": "n/a",
            "grounding_score": "n/a",
            "completion_score": "n/a",
            "efficiency_score": "n/a",
            "progress_score": "n/a",
            "penalty_score": "n/a",
            "steps": "n/a",
            "validation_runs": "n/a",
            "successful_edit_steps": "n/a",
            "failed_edit_steps": "n/a",
            "submitted": "n/a",
            "planner_phase_enabled": "n/a",
        }
    data = json.loads(path.read_text())
    return score_traj(data)


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = latest_root()
    for variant in VARIANTS:
        variant_dir = root / variant
        traj = variant_dir / "pydicom__pydicom-1458" / "pydicom__pydicom-1458.traj"
        patch = variant_dir / "pydicom__pydicom-1458" / "pydicom__pydicom-1458.patch"
        info_log = variant_dir / "pydicom__pydicom-1458" / "pydicom__pydicom-1458.info.log"
        debug_log = variant_dir / "pydicom__pydicom-1458" / "pydicom__pydicom-1458.debug.log"
        exit_yaml = variant_dir / "run_batch_exit_statuses.yaml"
        preds = variant_dir / "preds.json"
        score = read_traj_score(traj)
        rows.append(
            {
                "variant": variant,
                "exit_status": read_exit_status(exit_yaml),
                "quality_score": score["quality_score"],
                "grounding_score": score["grounding_score"],
                "completion_score": score["completion_score"],
                "efficiency_score": score["efficiency_score"],
                "progress_score": score["progress_score"],
                "penalty_score": score["penalty_score"],
                "steps": score["steps"],
                "validation_runs": score["validation_runs"],
                "successful_edit_steps": score["successful_edit_steps"],
                "failed_edit_steps": score["failed_edit_steps"],
                "submitted": score["submitted"],
                "planner_phase_enabled": score["planner_phase_enabled"],
                "traj": str(traj),
                "patch": str(patch) if patch.exists() else "",
                "info_log": str(info_log) if info_log.exists() else "",
                "debug_log": str(debug_log) if debug_log.exists() else "",
                "preds": str(preds) if preds.exists() else "",
                "exit_yaml": str(exit_yaml) if exit_yaml.exists() else "",
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, object]], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n")


def write_md(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Latest Matrix Easy Results",
        "",
        "| Variant | Exit | Quality | Completion | Efficiency | Grounding | Validations | Good Edits | Failed Edits | Submitted | Planner | Steps |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {exit_status} | {quality_score} | {completion_score} | {efficiency_score} | {grounding_score} | {validation_runs} | {successful_edit_steps} | {failed_edit_steps} | {submitted} | {planner_phase_enabled} | {steps} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['variant']}",
                f"- exit: `{row['exit_status']}`",
                f"- traj: `{row['traj']}`",
                f"- patch: `{row['patch'] or 'none'}`",
                f"- info log: `{row['info_log'] or 'none'}`",
                f"- debug log: `{row['debug_log'] or 'none'}`",
                f"- preds: `{row['preds'] or 'none'}`",
                f"- exit yaml: `{row['exit_yaml'] or 'none'}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    root = latest_root()
    root.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows, root / "summary.csv")
    write_json(rows, root / "summary.json")
    write_md(rows, root / "README.md")
    print(f"Wrote summary files to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
