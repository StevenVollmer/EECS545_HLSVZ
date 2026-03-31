#!/usr/bin/env python3
"""Render a markdown report for a custom experiment matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZER_SCRIPT = REPO_ROOT / "SWE-agent" / "scripts" / "analyze_custom_runs.py"


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _collect_run_results(matrix_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(matrix_root.rglob("analysis.json")):
        data = _load_json(path)
        for result in data.get("results", []):
            if isinstance(result, dict):
                results.append(result)
    return results


def _maybe_generate_summary(matrix_root: Path) -> dict[str, Any]:
    summary_path = matrix_root / "analysis.summary.json"
    if summary_path.exists():
        return _load_json(summary_path)

    cmd = [
        sys.executable,
        str(ANALYZER_SCRIPT),
        str(matrix_root),
        "--json",
        "--write-json",
        str(summary_path),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "failed to generate summary")
    return _load_json(summary_path)


def _fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _group_key(result: dict[str, Any]) -> tuple[str, str]:
    return str(result.get("model", "")), str(result.get("architecture", ""))


def _preset_architecture_summary(results: list[dict[str, Any]]) -> list[list[str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[_group_key(result)].append(result)

    rows: list[list[str]] = []
    for (model, architecture), group in sorted(grouped.items()):
        rows.append(
            [
                model,
                architecture,
                str(len(group)),
                _fmt_float(sum(1 for r in group if r.get("success_passed")) / len(group), 3),
                _fmt_float(sum(1 for r in group if r.get("observed_success_passed")) / len(group), 3),
                _fmt_float(_mean([float(r.get("total_score", 0)) for r in group])),
                _fmt_float(_mean([float(r.get("relative_cost_to_4o_mini", 0)) for r in group]), 3),
                _fmt_float(_mean([float(r.get("turns", 0)) for r in group]), 1),
                _fmt_float(_mean([float(r.get("parse_errors", 0)) for r in group]), 1),
                _fmt_float(_mean([float(r.get("tool_error_count", 0)) for r in group]), 1),
            ]
        )
    return rows


def _case_summary(results: list[dict[str, Any]]) -> list[list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("instance_id", ""))].append(result)

    rows: list[list[str]] = []
    for case_id, group in sorted(grouped.items()):
        best = max(group, key=lambda item: (float(item.get("total_score", 0)), float(item.get("observed_success_passed", False))))
        rows.append(
            [
                case_id,
                str(len(group)),
                _fmt_float(sum(1 for r in group if r.get("success_passed")) / len(group), 3),
                _fmt_float(sum(1 for r in group if r.get("observed_success_passed")) / len(group), 3),
                _fmt_float(_mean([float(r.get("total_score", 0)) for r in group])),
                str(best.get("model", "")),
                str(best.get("architecture", "")),
                str(best.get("total_score", "")),
            ]
        )
    return rows


def _per_case_detail(results: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("instance_id", ""))].append(result)

    blocks: list[str] = []
    for case_id, group in sorted(grouped.items()):
        blocks.append(f"### {case_id}")
        rows: list[list[str]] = []
        for result in sorted(
            group,
            key=lambda item: (
                str(item.get("model", "")),
                str(item.get("architecture", "")),
            ),
        ):
            run_dir = Path(str(result.get("run_dir", "")))
            rows.append(
                [
                    str(result.get("model", "")),
                    str(result.get("architecture", "")),
                    str(result.get("total_score", "")),
                    "yes" if result.get("success_passed") else "no",
                    "yes" if result.get("observed_success_passed") else "no",
                    _fmt_float(float(result.get("relative_cost_to_4o_mini", 0)), 3),
                    str(result.get("turns", "")),
                    str(result.get("parse_errors", "")),
                    str(result.get("tool_error_count", "")),
                    str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_absolute() and REPO_ROOT in run_dir.parents else str(run_dir),
                ]
            )
        blocks.append(
            _markdown_table(
                [
                    "Model",
                    "Architecture",
                    "Score",
                    "Strict Success",
                    "Observed Success",
                    "Rel Cost",
                    "Turns",
                    "Parse Err",
                    "Tool Err",
                    "Run Dir",
                ],
                rows,
            )
        )
        blocks.append("")
    return "\n".join(blocks).rstrip()


def _interesting_failures(results: list[dict[str, Any]], limit: int) -> str:
    failures = [
        result
        for result in results
        if not result.get("success_passed")
    ]
    failures.sort(
        key=lambda item: (
            -float(item.get("total_score", 0)),
            float(item.get("relative_cost_to_4o_mini", 0)),
        )
    )
    lines: list[str] = []
    for result in failures[:limit]:
        notes = "; ".join(str(note) for note in result.get("notes", [])[:4])
        lines.append(
            f"- `{result.get('instance_id')}` | `{result.get('model')}` | `{result.get('architecture')}` | "
            f"score `{result.get('total_score')}` | cost `{_fmt_float(float(result.get('relative_cost_to_4o_mini', 0)), 3)}`"
            + (f" | {notes}" if notes else "")
        )
    return "\n".join(lines) if lines else "- None"


def _top_runs(results: list[dict[str, Any]], limit: int) -> str:
    top = sorted(
        results,
        key=lambda item: (
            bool(item.get("success_passed")),
            float(item.get("total_score", 0)),
            -float(item.get("relative_cost_to_4o_mini", 0)),
        ),
        reverse=True,
    )
    lines: list[str] = []
    for result in top[:limit]:
        lines.append(
            f"- `{result.get('instance_id')}` | `{result.get('model')}` | `{result.get('architecture')}` | "
            f"score `{result.get('total_score')}` | strict `{result.get('success_passed')}` | "
            f"observed `{result.get('observed_success_passed')}` | cost `{_fmt_float(float(result.get('relative_cost_to_4o_mini', 0)), 3)}`"
        )
    return "\n".join(lines) if lines else "- None"


def render_report(matrix_root: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    aggregate = summary.get("aggregate", {}) if isinstance(summary.get("aggregate"), dict) else {}
    lines: list[str] = []
    lines.append(f"# Custom Matrix Report: {matrix_root.name}")
    lines.append("")
    lines.append(f"- Matrix root: `{matrix_root}`")
    lines.append(f"- Runs: `{aggregate.get('runs', len(results))}`")
    lines.append(f"- Strict resolved rate: `{aggregate.get('resolved_rate', 0)}`")
    lines.append(f"- Observed resolved rate: `{aggregate.get('observed_resolved_rate', 0)}`")
    lines.append(f"- Avg total score: `{aggregate.get('avg_total_score', 0)}`")
    lines.append(f"- Avg relative cost to 4o-mini: `{aggregate.get('avg_relative_cost_to_4o_mini', 0)}`")
    lines.append("")
    lines.append("## By Model And Architecture")
    lines.append("")
    lines.append(
        _markdown_table(
            [
                "Model",
                "Architecture",
                "Runs",
                "Strict Resolve",
                "Observed Resolve",
                "Avg Score",
                "Avg Rel Cost",
                "Avg Turns",
                "Avg Parse Err",
                "Avg Tool Err",
            ],
            _preset_architecture_summary(results),
        )
    )
    lines.append("")
    lines.append("## By Case")
    lines.append("")
    lines.append(
        _markdown_table(
            [
                "Case",
                "Runs",
                "Strict Resolve",
                "Observed Resolve",
                "Avg Score",
                "Best Model",
                "Best Architecture",
                "Best Score",
            ],
            _case_summary(results),
        )
    )
    lines.append("")
    lines.append("## Top Runs")
    lines.append("")
    lines.append(_top_runs(results, limit=12))
    lines.append("")
    lines.append("## Best Failures")
    lines.append("")
    lines.append(_interesting_failures(results, limit=12))
    lines.append("")
    lines.append("## Per-Case Comparison")
    lines.append("")
    lines.append(_per_case_detail(results))
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_root = args.matrix_root.resolve()
    summary = _maybe_generate_summary(matrix_root)
    results = _collect_run_results(matrix_root)
    report = render_report(matrix_root, summary, results)
    output_path = args.output or (matrix_root / "README.md")
    output_path.write_text(report)
    print(output_path)


if __name__ == "__main__":
    main()
