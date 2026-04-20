#!/usr/bin/env python3
"""Build summaries for matrix batch results with issue-focused metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from analyze_traj_quality import score_traj


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_results_root() -> Path:
    return repo_root() / "latest_matrix_easy_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=default_results_root(),
        help="Matrix results root containing one directory per variant.",
    )
    return parser.parse_args()


def discover_variants(root: Path) -> list[Path]:
    variants = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name == "projects":
            continue
        if (path / "run_batch.config.yaml").exists() or (path / "run_batch_exit_statuses.yaml").exists():
            variants.append(path)
    return variants


def discover_preset_roots(root: Path) -> list[Path]:
    presets = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name == "projects":
            continue
        if discover_variants(path):
            presets.append(path)
    return presets


def display_config_name(row: dict[str, object]) -> str:
    preset = str(row.get("preset", ""))
    variant = str(row.get("variant", ""))
    return f"{preset}/{variant}" if preset else variant


def project_id_from_instance(instance_id: str) -> str:
    match = re.match(r"^(?P<project>.+)-(?P<issue>\d+)$", instance_id)
    return match.group("project") if match else instance_id


def read_exit_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    statuses: dict[str, str] = {}
    current: str | None = None
    inside = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "instances_by_exit_status:":
            inside = True
            continue
        if not inside:
            continue
        if stripped.startswith("total_cost:"):
            break
        if stripped.endswith(":") and not stripped.startswith("-"):
            current = stripped[:-1]
            continue
        if stripped.startswith("- ") and current:
            statuses[stripped[2:].strip()] = current
    return statuses


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
            "manual_submit": "n/a",
            "clean_exit": "n/a",
            "planner_phase_enabled": "n/a",
            "tokens_in": "n/a",
            "tokens_out": "n/a",
            "token_total": "n/a",
            "tokens_per_step": "n/a",
            "api_calls": "n/a",
            "relative_cost_estimate": "n/a",
            "estimated_cost_usd": "n/a",
            "issue_alignment_score": "n/a",
            "solution_focus_score": "n/a",
            "workflow_score": "n/a",
            "stability_score": "n/a",
            "analysis_score": "n/a",
            "aligned_files_inspected": "n/a",
            "aligned_files_edited": "n/a",
            "validation_after_edit": "n/a",
            "edited_file_alignment": "n/a",
            "inspected_file_alignment": "n/a",
            "edited_files": "",
            "aligned_files": "",
        }
    data = json.loads(path.read_text())
    return score_traj(data)


def list_variant_instances(variant_dir: Path) -> set[str]:
    return {
        child.name
        for child in variant_dir.iterdir()
        if child.is_dir() and any(child.glob("*.traj"))
    }


def build_rows_for_preset(root: Path, preset_name: str) -> list[dict[str, object]]:
    variants = discover_variants(root)
    rows: list[dict[str, object]] = []

    for variant_dir in variants:
        variant = variant_dir.name
        exit_statuses = read_exit_statuses(variant_dir / "run_batch_exit_statuses.yaml")
        instance_ids = sorted(set(list_variant_instances(variant_dir)) | set(exit_statuses))
        preds = variant_dir / "preds.json"
        for instance_id in instance_ids:
            instance_dir = variant_dir / instance_id
            traj = instance_dir / f"{instance_id}.traj"
            patch = instance_dir / f"{instance_id}.patch"
            info_log = instance_dir / f"{instance_id}.info.log"
            debug_log = instance_dir / f"{instance_id}.debug.log"
            exit_yaml = variant_dir / "run_batch_exit_statuses.yaml"
            score = read_traj_score(traj)
            rows.append(
                {
                    "preset": preset_name,
                    "project_id": project_id_from_instance(instance_id),
                    "instance_id": instance_id,
                    "variant": variant,
                    "exit_status": exit_statuses.get(instance_id, "missing"),
                    "analysis_score": score["analysis_score"],
                    "issue_alignment_score": score["issue_alignment_score"],
                    "solution_focus_score": score["solution_focus_score"],
                    "workflow_score": score["workflow_score"],
                    "stability_score": score["stability_score"],
                    "aligned_files_inspected": score["aligned_files_inspected"],
                    "aligned_files_edited": score["aligned_files_edited"],
                    "validation_after_edit": score["validation_after_edit"],
                    "edited_file_alignment": score["edited_file_alignment"],
                    "inspected_file_alignment": score["inspected_file_alignment"],
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
                    "manual_submit": score["manual_submit"],
                    "clean_exit": score["clean_exit"],
                    "planner_phase_enabled": score["planner_phase_enabled"],
                    "tokens_in": score["tokens_in"],
                    "tokens_out": score["tokens_out"],
                    "token_total": score["token_total"],
                    "tokens_per_step": score["tokens_per_step"],
                    "api_calls": score["api_calls"],
                    "relative_cost_estimate": score["relative_cost_estimate"],
                    "estimated_cost_usd": score["estimated_cost_usd"],
                    "edited_files": score["edited_files"],
                    "aligned_files": score["aligned_files"],
                    "traj": str(traj) if traj.exists() else "",
                    "patch": str(patch) if patch.exists() else "",
                    "info_log": str(info_log) if info_log.exists() else "",
                    "debug_log": str(debug_log) if debug_log.exists() else "",
                    "preds": str(preds) if preds.exists() else "",
                    "exit_yaml": str(exit_yaml) if exit_yaml.exists() else "",
                }
            )
    return rows


def build_rows(root: Path) -> list[dict[str, object]]:
    variants = discover_variants(root)
    if variants:
        return build_rows_for_preset(root, "")
    preset_roots = discover_preset_roots(root)
    rows: list[dict[str, object]] = []
    for preset_root in preset_roots:
        rows.extend(build_rows_for_preset(preset_root, preset_root.name))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, object]], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n")


def score_fraction(value: object) -> tuple[float, int] | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, denominator = value.split("/", 1)
    try:
        return float(numerator), int(denominator)
    except ValueError:
        return None


def average_fraction(rows: list[dict[str, object]], key: str) -> str:
    values = [score_fraction(row[key]) for row in rows]
    parsed = [value for value in values if value is not None]
    if not parsed:
        return "n/a"
    total = sum(value for value, _ in parsed)
    scale = parsed[0][1]
    return f"{total / len(parsed):.1f}/{scale}"


def average_int(rows: list[dict[str, object]], key: str) -> str:
    values = [row[key] for row in rows if isinstance(row[key], int) and not isinstance(row[key], bool)]
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.1f}"


def average_float(rows: list[dict[str, object]], key: str) -> str:
    values = [float(row[key]) for row in rows if isinstance(row[key], (int, float)) and not isinstance(row[key], bool)]
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.2f}"


def average_usd(rows: list[dict[str, object]], key: str) -> str:
    """Average USD values, skipping None (unknown pricing) and non-numeric entries.

    Returns 'n/a' when no rows have known pricing, or a dollar-formatted string
    marked with '~' to indicate it is an estimate (e.g. '~$0.0124').
    """
    values = [float(row[key]) for row in rows if isinstance(row[key], (int, float)) and not isinstance(row[key], bool)]
    if not values:
        return "n/a"
    return f"~${sum(values) / len(values):.4f}"


def true_count(rows: list[dict[str, object]], key: str) -> str:
    values = [row[key] for row in rows if isinstance(row[key], bool)]
    if not values:
        return "n/a"
    return f"{sum(values)}/{len(values)}"


def positive_count(rows: list[dict[str, object]], key: str) -> str:
    values = [row[key] for row in rows if isinstance(row[key], int) and not isinstance(row[key], bool)]
    if not values:
        return "n/a"
    positives = sum(1 for value in values if value > 0)
    return f"{positives}/{len(values)}"


def group_by(rows: list[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(sorted(grouped.items()))


def average_score_value(rows: list[dict[str, object]], key: str) -> float:
    parsed = [score_fraction(row[key]) for row in rows]
    usable = [value for value in parsed if value is not None]
    if not usable:
        return -1.0
    return sum(value for value, _ in usable) / len(usable)


def build_variant_rollup(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rollup = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[display_config_name(row)].append(row)
    for config_name, variant_rows in sorted(grouped.items()):
        rollup.append(
            {
                "config": config_name,
                "instances": len(variant_rows),
                "submitted": true_count(variant_rows, "submitted"),
                "manual_submit": true_count(variant_rows, "manual_submit"),
                "validated_after_edit": true_count(variant_rows, "validation_after_edit"),
                "aligned_edits": positive_count(variant_rows, "aligned_files_edited"),
                "avg_analysis": average_fraction(variant_rows, "analysis_score"),
                "avg_issue_alignment": average_fraction(variant_rows, "issue_alignment_score"),
                "avg_focus": average_fraction(variant_rows, "solution_focus_score"),
                "avg_workflow": average_fraction(variant_rows, "workflow_score"),
                "avg_stability": average_fraction(variant_rows, "stability_score"),
                "avg_steps": average_int(variant_rows, "steps"),
                "avg_tokens_in": average_int(variant_rows, "tokens_in"),
                "avg_tokens_out": average_int(variant_rows, "tokens_out"),
                "avg_rel_cost": average_float(variant_rows, "relative_cost_estimate"),
                "avg_cost_usd": average_usd(variant_rows, "estimated_cost_usd"),
                "avg_legacy_quality": average_fraction(variant_rows, "quality_score"),
            }
        )
    return rollup


def build_project_rollup(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rollup = []
    for project_id, project_rows in group_by(rows, "project_id").items():
        by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in project_rows:
            by_config[display_config_name(row)].append(row)
        best_variant = max(by_config.items(), key=lambda item: average_score_value(item[1], "analysis_score"))[0]
        rollup.append(
            {
                "project_id": project_id,
                "issues": len({str(row['instance_id']) for row in project_rows}),
                "configs": len({display_config_name(row) for row in project_rows}),
                "avg_analysis": average_fraction(project_rows, "analysis_score"),
                "avg_issue_alignment": average_fraction(project_rows, "issue_alignment_score"),
                "avg_focus": average_fraction(project_rows, "solution_focus_score"),
                "avg_workflow": average_fraction(project_rows, "workflow_score"),
                "avg_stability": average_fraction(project_rows, "stability_score"),
                "avg_tokens_in": average_int(project_rows, "tokens_in"),
                "avg_tokens_out": average_int(project_rows, "tokens_out"),
                "avg_rel_cost": average_float(project_rows, "relative_cost_estimate"),
                "avg_cost_usd": average_usd(project_rows, "estimated_cost_usd"),
                "best_variant": best_variant,
            }
        )
    return rollup


def build_instance_rollup(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rollup = []
    for instance_id, instance_rows in group_by(rows, "instance_id").items():
        by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in instance_rows:
            by_config[display_config_name(row)].append(row)
        best_variant = max(by_config.items(), key=lambda item: average_score_value(item[1], "analysis_score"))[0]
        best_analysis = average_fraction(by_config[best_variant], "analysis_score")
        exit_counts = defaultdict(int)
        for row in instance_rows:
            exit_counts[str(row["exit_status"])] += 1
        exit_summary = ", ".join(f"{status}={count}" for status, count in sorted(exit_counts.items()))
        rollup.append(
            {
                "project_id": str(instance_rows[0]["project_id"]),
                "instance_id": instance_id,
                "variants_run": len(instance_rows),
                "submitted": true_count(instance_rows, "submitted"),
                "validated_after_edit": true_count(instance_rows, "validation_after_edit"),
                "aligned_edits": positive_count(instance_rows, "aligned_files_edited"),
                "avg_analysis": average_fraction(instance_rows, "analysis_score"),
                "avg_issue_alignment": average_fraction(instance_rows, "issue_alignment_score"),
                "avg_focus": average_fraction(instance_rows, "solution_focus_score"),
                "avg_workflow": average_fraction(instance_rows, "workflow_score"),
                "avg_stability": average_fraction(instance_rows, "stability_score"),
                "avg_tokens_in": average_int(instance_rows, "tokens_in"),
                "avg_tokens_out": average_int(instance_rows, "tokens_out"),
                "avg_rel_cost": average_float(instance_rows, "relative_cost_estimate"),
                "avg_cost_usd": average_usd(instance_rows, "estimated_cost_usd"),
                "best_variant": best_variant,
                "best_analysis": best_analysis,
                "exit_summary": exit_summary,
            }
        )
    return sorted(rollup, key=lambda row: (str(row["project_id"]), str(row["instance_id"])))


def write_root_readme(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("# Matrix Batch Results\n\nNo rows found.\n")
        return

    variant_rollup = build_variant_rollup(rows)
    project_rollup = build_project_rollup(rows)
    instance_rollup = build_instance_rollup(rows)
    lines = [
        "# Matrix Batch Results",
        "",
        f"- presets: `{len({row['preset'] for row in rows if row['preset']})}`",
        f"- variants: `{len({row['variant'] for row in rows})}`",
        f"- configs: `{len({display_config_name(row) for row in rows})}`",
        f"- projects: `{len({row['project_id'] for row in rows})}`",
        f"- issues: `{len({row['instance_id'] for row in rows})}`",
        f"- observed runs: `{len(rows)}`",
        "",
        "## Primary Aggregate",
        "",
        "These metrics prioritize issue alignment, focused editing, validation after edits, and execution stability.",
        "",
        "| Config | Instances | Submitted | Manual Submit | Validated After Edit | Aligned Edits | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Cost USD | Avg Steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in variant_rollup:
        lines.append(
            "| {config} | {instances} | {submitted} | {manual_submit} | {validated_after_edit} | {aligned_edits} | {avg_analysis} | {avg_issue_alignment} | {avg_focus} | {avg_workflow} | {avg_stability} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {avg_cost_usd} | {avg_steps} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Issue Index",
            "",
            "| Issue | Project | Configs Run | Submitted | Validated After Edit | Aligned Edits | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Cost USD | Best Variant | Best Analysis | Exit Mix |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in instance_rollup:
        lines.append(
            "| {instance_id} | {project_id} | {variants_run} | {submitted} | {validated_after_edit} | {aligned_edits} | {avg_analysis} | {avg_issue_alignment} | {avg_focus} | {avg_workflow} | {avg_stability} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {avg_cost_usd} | {best_variant} | {best_analysis} | {exit_summary} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Project Index",
            "",
            "| Project | Issues | Configs | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Cost USD | Best Variant | Report |",
            "| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in project_rollup:
        report_path = f"./projects/{row['project_id']}/README.md"
        lines.append(
            "| {project_id} | {issues} | {configs} | {avg_analysis} | {avg_issue_alignment} | {avg_focus} | {avg_workflow} | {avg_stability} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {avg_cost_usd} | {best_variant} | [{project_id}]({report_path}) |".format(
                report_path=report_path,
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: one row per `(variant, instance)` pair",
            "- `summary.json`: JSON version of the same table",
            "- `projects/<project>/README.md`: per-project comparisons across all variants",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_project_reports(rows: list[dict[str, object]], root: Path) -> None:
    projects_root = root / "projects"
    projects_root.mkdir(parents=True, exist_ok=True)
    for project_id, project_rows in group_by(rows, "project_id").items():
        project_dir = projects_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        write_csv(project_rows, project_dir / "summary.csv")
        write_json(project_rows, project_dir / "summary.json")
        variant_rollup = build_variant_rollup(project_rows)
        instance_rollup = build_instance_rollup(project_rows)
        lines = [
            f"# {project_id}",
            "",
            f"- issues: `{len({row['instance_id'] for row in project_rows})}`",
            f"- presets: `{len({row['preset'] for row in project_rows if row['preset']})}`",
            f"- variants: `{len({row['variant'] for row in project_rows})}`",
            f"- configs: `{len({display_config_name(row) for row in project_rows})}`",
            f"- observed runs: `{len(project_rows)}`",
            "",
            "## Variant Aggregate",
            "",
            "| Config | Instances | Submitted | Manual Submit | Validated After Edit | Aligned Edits | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Cost USD | Avg Steps | Legacy Quality |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in variant_rollup:
            lines.append(
                "| {config} | {instances} | {submitted} | {manual_submit} | {validated_after_edit} | {aligned_edits} | {avg_analysis} | {avg_issue_alignment} | {avg_focus} | {avg_workflow} | {avg_stability} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {avg_cost_usd} | {avg_steps} | {avg_legacy_quality} |".format(
                    **row
                )
            )
        lines.extend(
            [
                "",
                "## Issue Aggregate",
                "",
                "| Issue | Configs Run | Submitted | Validated After Edit | Aligned Edits | Avg Analysis | Avg Issue Alignment | Avg Focus | Avg Workflow | Avg Stability | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Cost USD | Best Variant | Best Analysis | Exit Mix |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in instance_rollup:
            lines.append(
                "| {instance_id} | {variants_run} | {submitted} | {validated_after_edit} | {aligned_edits} | {avg_analysis} | {avg_issue_alignment} | {avg_focus} | {avg_workflow} | {avg_stability} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {avg_cost_usd} | {best_variant} | {best_analysis} | {exit_summary} |".format(
                    **row
                )
            )
        lines.extend(
            [
                "",
                "## Instance Details",
                "",
                "| Instance | Preset | Variant | Exit | Analysis | Issue Align | Focus | Workflow | Stability | Validated After Edit | Aligned Files Edited | Edited Files | Legacy Quality | Completion | Grounding | In Tok | Out Tok | Rel Cost | Cost USD | Steps |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(project_rows, key=lambda item: (str(item["instance_id"]), str(item["variant"]))):
            lines.append(
                "| {instance_id} | {preset} | {variant} | {exit_status} | {analysis_score} | {issue_alignment_score} | {solution_focus_score} | {workflow_score} | {stability_score} | {validation_after_edit} | {aligned_files_edited} | {edited_files} | {quality_score} | {completion_score} | {grounding_score} | {tokens_in} | {tokens_out} | {relative_cost_estimate} | {estimated_cost_usd} | {steps} |".format(
                    **row
                )
            )
        lines.append("")
        (project_dir / "README.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = build_rows(root)
    write_csv(rows, root / "summary.csv")
    write_json(rows, root / "summary.json")
    write_root_readme(rows, root / "README.md")
    write_project_reports(rows, root)
    print(f"Wrote summary files to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
