#!/usr/bin/env python3
"""Build per-instance, per-project, and overall summaries for matrix batch results."""

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
            "planner_phase_enabled": "n/a",
            "tokens_in": "n/a",
            "tokens_out": "n/a",
            "token_total": "n/a",
            "tokens_per_step": "n/a",
            "api_calls": "n/a",
            "relative_cost_estimate": "n/a",
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
                    "tokens_in": score["tokens_in"],
                    "tokens_out": score["tokens_out"],
                    "token_total": score["token_total"],
                    "tokens_per_step": score["tokens_per_step"],
                    "api_calls": score["api_calls"],
                    "relative_cost_estimate": score["relative_cost_estimate"],
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
    values = [row[key] for row in rows if isinstance(row[key], int)]
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.1f}"


def average_float(rows: list[dict[str, object]], key: str) -> str:
    values = [float(row[key]) for row in rows if isinstance(row[key], (int, float))]
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.1f}"


def true_count(rows: list[dict[str, object]], key: str) -> str:
    values = [row[key] for row in rows if isinstance(row[key], bool)]
    if not values:
        return "n/a"
    return f"{sum(values)}/{len(values)}"


def group_by(rows: list[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(sorted(grouped.items()))


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
                "avg_quality": average_fraction(variant_rows, "quality_score"),
                "avg_completion": average_fraction(variant_rows, "completion_score"),
                "avg_efficiency": average_fraction(variant_rows, "efficiency_score"),
                "avg_grounding": average_fraction(variant_rows, "grounding_score"),
                "avg_steps": average_int(variant_rows, "steps"),
                "avg_tokens_in": average_int(variant_rows, "tokens_in"),
                "avg_tokens_out": average_int(variant_rows, "tokens_out"),
                "avg_rel_cost": average_float(variant_rows, "relative_cost_estimate"),
            }
        )
    return rollup


def build_project_rollup(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rollup = []
    for project_id, project_rows in group_by(rows, "project_id").items():
        by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in project_rows:
            by_config[display_config_name(row)].append(row)
        best_variant = max(
            by_config.items(),
            key=lambda item: (
                sum(score_fraction(row["quality_score"])[0] for row in item[1] if score_fraction(row["quality_score"]) is not None)
                / max(1, len([row for row in item[1] if score_fraction(row["quality_score"]) is not None]))
            ),
        )[0]
        rollup.append(
            {
                "project_id": project_id,
                "issues": len({str(row["instance_id"]) for row in project_rows}),
                "configs": len({display_config_name(row) for row in project_rows}),
                "avg_quality": average_fraction(project_rows, "quality_score"),
                "avg_completion": average_fraction(project_rows, "completion_score"),
                "avg_efficiency": average_fraction(project_rows, "efficiency_score"),
                "avg_tokens_in": average_int(project_rows, "tokens_in"),
                "avg_tokens_out": average_int(project_rows, "tokens_out"),
                "avg_rel_cost": average_float(project_rows, "relative_cost_estimate"),
                "best_variant": best_variant,
            }
        )
    return rollup


def average_quality_value(rows: list[dict[str, object]]) -> float:
    parsed = [score_fraction(row["quality_score"]) for row in rows]
    usable = [value for value in parsed if value is not None]
    if not usable:
        return -1.0
    return sum(value for value, _ in usable) / len(usable)


def build_instance_rollup(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rollup = []
    for instance_id, instance_rows in group_by(rows, "instance_id").items():
        by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in instance_rows:
            by_config[display_config_name(row)].append(row)
        best_variant = max(by_config.items(), key=lambda item: average_quality_value(item[1]))[0]
        best_quality = average_fraction(by_config[best_variant], "quality_score")
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
                "avg_quality": average_fraction(instance_rows, "quality_score"),
                "avg_completion": average_fraction(instance_rows, "completion_score"),
                "avg_efficiency": average_fraction(instance_rows, "efficiency_score"),
                "avg_tokens_in": average_int(instance_rows, "tokens_in"),
                "avg_tokens_out": average_int(instance_rows, "tokens_out"),
                "avg_rel_cost": average_float(instance_rows, "relative_cost_estimate"),
                "best_variant": best_variant,
                "best_quality": best_quality,
                "exit_summary": exit_summary,
            }
        )
    return sorted(rollup, key=lambda row: (str(row["project_id"]), str(row["instance_id"])))


def write_root_readme(rows: list[dict[str, object]], path: Path) -> None:
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
        "## Variant Aggregate",
        "",
        "| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |",
        "| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in variant_rollup:
        lines.append(
            "| {config} | {instances} | {submitted} | {avg_quality} | {avg_completion} | {avg_efficiency} | {avg_grounding} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {avg_steps} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Issue Index",
            "",
            "| Issue | Project | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |",
            "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in instance_rollup:
        lines.append(
            "| {instance_id} | {project_id} | {variants_run} | {submitted} | {avg_quality} | {avg_completion} | {avg_efficiency} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {best_variant} | {best_quality} | {exit_summary} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Project Index",
            "",
            "| Project | Issues | Configs | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Report |",
            "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in project_rollup:
        report_path = f"./projects/{row['project_id']}/README.md"
        lines.append(
            "| {project_id} | {issues} | {configs} | {avg_quality} | {avg_completion} | {avg_efficiency} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {best_variant} | [{project_id}]({report_path}) |".format(
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
            "| Config | Instances | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg Grounding | Avg In Tok | Avg Out Tok | Avg Rel Cost | Avg Steps |",
            "| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in variant_rollup:
            lines.append(
                "| {config} | {instances} | {submitted} | {avg_quality} | {avg_completion} | {avg_efficiency} | {avg_grounding} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {avg_steps} |".format(
                    **row
                )
            )
        lines.extend(
            [
                "",
                "## Issue Aggregate",
                "",
                "| Issue | Configs Run | Submitted | Avg Quality | Avg Completion | Avg Efficiency | Avg In Tok | Avg Out Tok | Avg Rel Cost | Best Variant | Best Quality | Exit Mix |",
                "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in instance_rollup:
            lines.append(
                "| {instance_id} | {variants_run} | {submitted} | {avg_quality} | {avg_completion} | {avg_efficiency} | {avg_tokens_in} | {avg_tokens_out} | {avg_rel_cost} | {best_variant} | {best_quality} | {exit_summary} |".format(
                    **row
                )
            )
        lines.extend(
            [
                "",
                "## Instance Details",
                "",
                "| Instance | Preset | Variant | Exit | Quality | Completion | Efficiency | Grounding | In Tok | Out Tok | Rel Cost | Validations | Good Edits | Failed Edits | Submitted | Steps |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in sorted(project_rows, key=lambda item: (str(item["instance_id"]), str(item["variant"]))):
            lines.append(
                "| {instance_id} | {preset} | {variant} | {exit_status} | {quality_score} | {completion_score} | {efficiency_score} | {grounding_score} | {tokens_in} | {tokens_out} | {relative_cost_estimate} | {validation_runs} | {successful_edit_steps} | {failed_edit_steps} | {submitted} | {steps} |".format(
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
