#!/usr/bin/env python3
"""Analyze custom-run outputs with deterministic case evaluation and relative compute-burden scoring."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from judge_custom_case import evaluate_case


VALIDATION_HINTS = ("pytest", "unittest", "python -m", "python3 -m", "tox", "nox")
SEARCH_HINTS = ("find ", "rg ", "grep ", "ls ", "git diff", "cat ", "python scripts/")
PATH_RE = re.compile(r"(/repo/[^\s'\"`]+|[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)")
MODEL_SIZE_RE = re.compile(r"(?P<size>\d+(?:\.\d+)?)b\b", re.IGNORECASE)
OPENAI_COST_MULTIPLIERS = {
    "gpt-4o-mini": {"input": 1.0, "output": 4.0},
    "gpt-4o": {"input": 16.67, "output": 16.67},
}
PRESET_FILE = Path(__file__).resolve().parents[2] / "config" / "custom_configs" / "custom_runner_model_presets.yaml"


@dataclass
class RunArtifact:
    run_dir: Path
    instance_dir: Path
    instance_id: str
    traj_path: Path
    patch_path: Path
    info_log_path: Path | None
    run_config_path: Path


def _load_presets() -> dict[str, dict[str, Any]]:
    if not PRESET_FILE.exists():
        return {}
    raw = yaml.safe_load(PRESET_FILE.read_text()) or {}
    presets = raw.get("presets", {})
    return presets if isinstance(presets, dict) else {}


def _infer_preset_name(run_dir: Path, presets: dict[str, dict[str, Any]]) -> str:
    for parent in run_dir.parents:
        if parent.name in presets:
            return parent.name
    return ""


def _infer_model_size_rank(model_name: str) -> int:
    lowered = (model_name or "").lower()
    if "gpt-4o-mini" in lowered:
        return 2
    match = MODEL_SIZE_RE.search(lowered)
    if not match:
        return 0
    size = float(match.group("size"))
    if size >= 100:
        return 5
    if size >= 30:
        return 4
    if size >= 10:
        return 3
    if size > 0:
        return 1
    return 0


def _role_size_ranks(
    *,
    preset_name: str,
    presets: dict[str, dict[str, Any]],
    architecture: str,
    model_name: str,
    planner_model_name: str,
    reviewer_model_name: str,
) -> tuple[int, int, int]:
    preset = presets.get(preset_name, {})
    preset_rank = int(preset.get("size_rank", 0) or 0)
    coder_rank = int(preset.get("coder_size_rank", 0) or 0)
    planner_rank = int(preset.get("planner_size_rank", 0) or 0)
    reviewer_rank = int(preset.get("reviewer_size_rank", 0) or 0)

    inferred_coder = _infer_model_size_rank(model_name)
    inferred_planner = _infer_model_size_rank(planner_model_name or model_name)
    inferred_reviewer = _infer_model_size_rank(reviewer_model_name or planner_model_name or model_name)

    if architecture == "single":
        coder_rank = coder_rank or preset_rank or inferred_coder
        planner_rank = 0
        reviewer_rank = 0
    elif architecture == "planner_coder":
        coder_rank = coder_rank or preset_rank or inferred_coder
        planner_rank = planner_rank or preset_rank or inferred_planner
        reviewer_rank = 0
    else:
        coder_rank = coder_rank or inferred_coder or preset_rank
        planner_rank = planner_rank or preset_rank or inferred_planner
        reviewer_rank = reviewer_rank or planner_rank or inferred_reviewer

    if architecture != "single" and planner_rank == 0:
        planner_rank = inferred_planner or coder_rank
    if architecture == "planner_coder_reviewer" and reviewer_rank == 0:
        reviewer_rank = inferred_reviewer or planner_rank or coder_rank
    return planner_rank, coder_rank, reviewer_rank


def _resolve_cases_root(path: Path) -> Path:
    return path.resolve()


def _load_case_map(cases_root: Path) -> dict[str, dict[str, Any]]:
    case_map: dict[str, dict[str, Any]] = {}
    for case_file in sorted(cases_root.glob("*/case.json")):
        raw = yaml.safe_load(case_file.read_text())
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            instance_id = str(item.get("instance_id", "")).strip()
            if not instance_id:
                continue
            case_map[instance_id] = {
                "case_file": case_file.resolve(),
                "case_dir": case_file.parent.resolve(),
                "item": item,
            }
    return case_map


def _collect_artifacts(target: Path) -> list[RunArtifact]:
    target = target.resolve()
    artifacts: list[RunArtifact] = []
    if target.is_file() and target.name.endswith(".traj"):
        instance_dir = target.parent
        run_dir = instance_dir.parent
        artifacts.append(
            RunArtifact(
                run_dir=run_dir,
                instance_dir=instance_dir,
                instance_id=target.stem,
                traj_path=target,
                patch_path=instance_dir / f"{target.stem}.patch",
                info_log_path=(instance_dir / f"{target.stem}.info.log"),
                run_config_path=run_dir / "run_batch.config.yaml",
            )
        )
        return artifacts

    candidate_runs: list[Path]
    if (target / "run_batch.config.yaml").exists():
        candidate_runs = [target]
    else:
        candidate_runs = sorted({path.parent for path in target.rglob("run_batch.config.yaml")})

    for run_dir in sorted(candidate_runs):
        for traj_path in sorted(run_dir.glob("*/*.traj")):
            instance_id = traj_path.stem
            instance_dir = traj_path.parent
            artifacts.append(
                RunArtifact(
                    run_dir=run_dir,
                    instance_dir=instance_dir,
                    instance_id=instance_id,
                    traj_path=traj_path,
                    patch_path=instance_dir / f"{instance_id}.patch",
                    info_log_path=(instance_dir / f"{instance_id}.info.log"),
                    run_config_path=run_dir / "run_batch.config.yaml",
                )
            )
    return artifacts


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _iter_turns(traj: dict[str, Any]) -> list[dict[str, Any]]:
    turns = traj.get("turns", [])
    return turns if isinstance(turns, list) else []


def _tool_results(turn: dict[str, Any]) -> list[dict[str, Any]]:
    results = turn.get("tool_results", [])
    return results if isinstance(results, list) else []


def _tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    calls = turn.get("tool_calls", [])
    return calls if isinstance(calls, list) else []


def _normalize_path(path: str) -> str:
    value = path.strip().strip("`").rstrip(".,:")
    if value.startswith("/repo/"):
        return value[len("/repo/") :]
    if value.startswith("/"):
        return value.lstrip("/")
    return value


def _extract_paths_from_text(text: str) -> set[str]:
    return {_normalize_path(match.group(1)) for match in PATH_RE.finditer(text)}


def _is_validation_command(command: str) -> bool:
    lowered = command.lower()
    return any(hint in lowered for hint in VALIDATION_HINTS)


def _is_search_command(command: str) -> bool:
    lowered = command.lower()
    return any(hint in lowered for hint in SEARCH_HINTS)


def _extract_changed_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/"):
                files.append(parts[2][2:])
    return files


def _count_patch_hunks(patch_text: str) -> int:
    return sum(1 for line in patch_text.splitlines() if line.startswith("@@"))


def _count_patch_changed_lines(patch_text: str) -> int:
    count = 0
    for line in patch_text.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def _extract_model_size(model_name: str) -> float | None:
    match = MODEL_SIZE_RE.search(model_name)
    if not match:
        return None
    return float(match.group("size"))


def _estimate_relative_compute(tokens_in: int, tokens_out: int, model_name: str) -> tuple[float, dict[str, float]]:
    base_input = 1.0
    base_output = 4.0
    lowered = model_name.lower()
    for known, multipliers in OPENAI_COST_MULTIPLIERS.items():
        if known in lowered:
            input_mult = multipliers["input"]
            output_mult = multipliers["output"]
            total = tokens_in * input_mult + tokens_out * output_mult
            return total, {"input_multiplier": input_mult, "output_multiplier": output_mult}

    size_b = _extract_model_size(model_name)
    if size_b is None:
        input_mult = 3.0
        output_mult = 9.0
    else:
        # Local / self-hosted proxy: output is materially more expensive, and larger models scale non-linearly.
        scale = max(1.0, size_b / 8.0)
        input_mult = round(scale, 3)
        output_mult = round(scale * 3.0, 3)
    total = tokens_in * input_mult + tokens_out * output_mult
    return total, {"input_multiplier": input_mult, "output_multiplier": output_mult}


def _estimate_missing_tokens(turns: list[dict[str, Any]]) -> tuple[int, int]:
    assistant_chars = 0
    tool_output_chars = 0
    tool_call_count = 0
    for turn in turns:
        assistant_chars += len(str(turn.get("assistant_text", "")))
        for call in _tool_calls(turn):
            tool_call_count += 1
            tool_output_chars += len(json.dumps(call))
        for result in _tool_results(turn):
            tool_output_chars += len(str(result.get("output", "")))
    estimated_output = max(1, round(assistant_chars / 4))
    estimated_input = max(1, round((assistant_chars / 2) + (tool_output_chars / 8) + (tool_call_count * 180)))
    return estimated_input, estimated_output


def _case_analysis_metadata(case_item: dict[str, Any]) -> dict[str, Any]:
    value = case_item.get("analysis", {})
    return value if isinstance(value, dict) else {}


def _fraction_passed(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    passed = sum(1 for result in results if result.get("passed"))
    return passed / len(results)


def _is_environment_blocked_result(result: dict[str, Any]) -> bool:
    stderr = str(result.get("stderr", ""))
    stdout = str(result.get("stdout", ""))
    combined = f"{stdout}\n{stderr}"
    return "No module named pytest" in combined or "No module named" in combined and "pytest" in combined


def _effective_fraction(results: list[dict[str, Any]]) -> tuple[float, int]:
    considered = [result for result in results if not _is_environment_blocked_result(result)]
    if not considered:
        return 0.0, 0
    return _fraction_passed(considered), len(considered)


def _note(notes: list[str], message: str) -> None:
    if message not in notes:
        notes.append(message)


def analyze_artifact(artifact: RunArtifact, case_entry: dict[str, Any], run_install: bool) -> dict[str, Any]:
    traj = _load_json(artifact.traj_path)
    run_config = _load_yaml(artifact.run_config_path)
    presets = _load_presets()
    case_item = case_entry["item"]
    case_file = case_entry["case_file"]
    analysis_meta = _case_analysis_metadata(case_item)
    likely_fix_paths = {_normalize_path(path) for path in analysis_meta.get("likely_fix_paths", [])}
    showcase = str(analysis_meta.get("showcase", "") or "")
    difficulty = str(analysis_meta.get("difficulty", "") or "")
    case_policy = case_item.get("policy", {}) if isinstance(case_item.get("policy", {}), dict) else {}
    allow_test_edits = bool(case_policy.get("allow_test_edits", False))

    patch_text = artifact.patch_path.read_text() if artifact.patch_path.exists() else ""
    changed_files = _extract_changed_files(patch_text)
    changed_files_set = set(changed_files)
    changed_lines = _count_patch_changed_lines(patch_text)
    patch_hunks = _count_patch_hunks(patch_text)

    baseline_eval = evaluate_case(case_path=case_file, mode="baseline", run_install=run_install)
    if patch_text.strip():
        success_eval = evaluate_case(case_path=case_file, mode="patch", patch_file=artifact.patch_path, run_install=run_install)
    else:
        success_eval = {
            "case": str(case_file),
            "repo_path": "",
            "mode": "patch",
            "passed": False,
            "results": [],
        }

    turns = _iter_turns(traj)
    stats = traj.get("stats", {}) if isinstance(traj.get("stats"), dict) else {}
    stopped_reason = str(traj.get("stopped_reason", ""))
    submitted = bool(traj.get("submitted"))

    parse_errors = 0
    empty_responses = 0
    duplicate_action_recoveries = 0
    tool_error_count = 0
    missing_path_errors = 0
    successful_edit_count = 0
    failed_edit_count = 0
    validation_count = 0
    search_count = 0
    view_count = 0
    repeated_success_loop_hints = 0
    repeated_failure_stops = 0
    inspected_files: set[str] = set()
    edited_files: set[str] = set()
    validation_commands: list[str] = []
    search_commands: list[str] = []
    turn_summaries: list[dict[str, Any]] = []
    first_edit_turn: int | None = None
    validations_after_edit = 0

    for turn in turns:
        parse_error = turn.get("parse_error")
        if parse_error:
            parse_errors += 1
            if "empty response" in str(parse_error).lower():
                empty_responses += 1
            if "recovered the first action only" in str(parse_error).lower():
                duplicate_action_recoveries += 1

        for call in _tool_calls(turn):
            name = str(call.get("name", ""))
            args = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
            if name == "view":
                view_count += 1
                path = args.get("path")
                if isinstance(path, str):
                    inspected_files.add(_normalize_path(path))
            elif name in {"str_replace", "insert", "undo_edit"}:
                path = args.get("path")
                if isinstance(path, str):
                    edited_files.add(_normalize_path(path))
                if first_edit_turn is None:
                    first_edit_turn = int(turn.get("turn", 0) or 0)
            elif name == "bash":
                command = str(args.get("command", ""))
                if _is_validation_command(command):
                    validation_count += 1
                    validation_commands.append(command)
                    if first_edit_turn is not None:
                        validations_after_edit += 1
                if _is_search_command(command):
                    search_count += 1
                    search_commands.append(command)

        for result in _tool_results(turn):
            output = str(result.get("output", ""))
            name = str(result.get("name", ""))
            is_error = bool(result.get("is_error"))
            if is_error:
                tool_error_count += 1
                if "does not exist" in output.lower() or "filenotfounderror" in output.lower():
                    missing_path_errors += 1
                if name in {"str_replace", "insert"}:
                    failed_edit_count += 1
            else:
                if name in {"str_replace", "insert", "undo_edit"}:
                    successful_edit_count += 1
            if output:
                inspected_files.update(_extract_paths_from_text(output))

        assistant_text = str(turn.get("assistant_text", ""))
        turn_summaries.append(
            {
                "turn": int(turn.get("turn", 0) or 0),
                "parse_error": parse_error or "",
                "assistant_preview": assistant_text[:180],
            }
        )

    if artifact.info_log_path and artifact.info_log_path.exists():
        info_text = artifact.info_log_path.read_text()
        repeated_success_loop_hints = info_text.count("repeated successful tool call threshold")
        repeated_failure_stops = info_text.count("stopping after repeated tool failure threshold")
    else:
        info_text = ""

    baseline_fraction = _fraction_passed(baseline_eval["results"])
    success_fraction = _fraction_passed(success_eval["results"])
    baseline_effective_fraction, baseline_effective_checks = _effective_fraction(baseline_eval["results"])
    success_effective_fraction, success_effective_checks = _effective_fraction(success_eval["results"])
    evaluation_blocked = any(_is_environment_blocked_result(result) for result in baseline_eval["results"] + success_eval["results"])

    model_name = str(run_config.get("model", ""))
    input_tokens = int(stats.get("input_tokens", 0) or 0)
    output_tokens = int(stats.get("output_tokens", 0) or 0)
    compute_estimated = False
    if input_tokens == 0 and output_tokens == 0 and turns:
        input_tokens, output_tokens = _estimate_missing_tokens(turns)
        compute_estimated = True
    relative_compute_units, compute_detail = _estimate_relative_compute(input_tokens, output_tokens, model_name)
    reference_compute_units = (input_tokens * 1.0) + (output_tokens * 4.0)
    relative_to_4o_mini = 0.0 if reference_compute_units == 0 else relative_compute_units / reference_compute_units

    functional_correctness = round((baseline_effective_fraction * 5.0) + (success_effective_fraction * 45.0))
    repair_precision = 0
    regression_safety = 0
    search_grounding = 0
    efficiency_control = 0
    disallowed_test_edit_penalty = False
    notes: list[str] = []

    if changed_files:
        repair_precision += 4
    if likely_fix_paths and changed_files_set & likely_fix_paths:
        repair_precision += 8
    elif likely_fix_paths and changed_files_set:
        _note(notes, f"edited files missed likely fix paths: expected overlap with {sorted(likely_fix_paths)}")
    if changed_files and len(changed_files) <= 2:
        repair_precision += 4
    elif len(changed_files) > 3:
        _note(notes, f"patch touched many files ({len(changed_files)})")
    if changed_lines and changed_lines <= 40:
        repair_precision += 2
    elif changed_lines > 120:
        _note(notes, f"patch is broad ({changed_lines} changed lines)")
    if all(not path.startswith(("tests/", "docs/")) for path in changed_files_set) and changed_files_set:
        repair_precision += 2
    elif changed_files_set:
        _note(notes, "patch includes tests/docs changes")
    if changed_files_set and not allow_test_edits:
        changed_test_files = [path for path in changed_files if path.startswith("tests/") or "/tests/" in path]
        if changed_test_files:
            disallowed_test_edit_penalty = True
            _note(notes, f"case disallows test edits but patch changed tests: {changed_test_files[:3]}")

    if patch_text.strip():
        regression_safety += 3
    if success_effective_fraction >= 1.0 and success_effective_checks > 0:
        regression_safety += 8
    elif success_effective_fraction >= 0.5 and success_effective_checks > 0:
        regression_safety += 4
        _note(notes, "only part of the success validation passed")
    else:
        _note(notes, "success validation failed")
    if validation_count > 0:
        regression_safety += 2
    if validations_after_edit > 0:
        regression_safety += 2
    elif successful_edit_count > 0:
        _note(notes, "edited code without post-edit validation")

    if likely_fix_paths:
        inspected_overlap = inspected_files & likely_fix_paths
        edited_overlap = edited_files & likely_fix_paths
        if inspected_overlap:
            search_grounding += 5
        elif search_count > 0 or view_count > 0:
            _note(notes, "inspected repository but did not inspect the likely fix files")
        if edited_overlap:
            search_grounding += 3
        if first_edit_turn is not None and (validation_count > 0 or search_count > 0):
            search_grounding += 2
    else:
        if search_count > 0:
            search_grounding += 4
        if view_count > 0:
            search_grounding += 4
        if first_edit_turn is not None and validation_count > 0:
            search_grounding += 2

    efficiency_control = 5
    if parse_errors > 0:
        efficiency_control -= 1
        _note(notes, f"{parse_errors} protocol/parse errors")
    if tool_error_count > 0:
        efficiency_control -= 1
        _note(notes, f"{tool_error_count} tool errors")
    if missing_path_errors > 1:
        efficiency_control -= 1
        _note(notes, f"{missing_path_errors} missing-path lookups")
    if len(turns) > 35:
        efficiency_control -= 1
        _note(notes, f"long run ({len(turns)} turns)")
    if repeated_success_loop_hints > 0 or stopped_reason == "repeated_tool_failure":
        efficiency_control -= 1
        _note(notes, "loop control triggered")
    efficiency_control = max(0, efficiency_control)
    if disallowed_test_edit_penalty:
        repair_precision = max(0, repair_precision - 6)
        regression_safety = max(0, regression_safety - 4)
    observed_success = success_effective_checks > 0 and math.isclose(success_effective_fraction, 1.0)

    if not submitted:
        _note(notes, f"run ended without submit ({stopped_reason or 'unknown stop'})")
    if patch_text.strip() and not success_eval["passed"] and not (evaluation_blocked and observed_success):
        _note(notes, "patch exists but does not satisfy success checks")
    if evaluation_blocked:
        _note(notes, "host evaluation environment is missing some test dependencies")
    if parse_errors == 0 and tool_error_count == 0 and success_eval["passed"]:
        _note(notes, "clean run: no parse errors, no tool errors, success checks passed")

    total_score = functional_correctness + repair_precision + regression_safety + search_grounding + efficiency_control

    role_stats = traj.get("role_model_stats", {}) if isinstance(traj.get("role_model_stats"), dict) else {}
    if not role_stats:
        role_stats = {
            "coder": {
                "model": model_name,
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "api_calls": int(stats.get("turns", 0) or 0),
            }
        }

    architecture = run_config.get("agent_architecture", "single")
    planner_model_name = ""
    reviewer_model_name = ""
    if isinstance(role_stats.get("planner"), dict):
        planner_model_name = str(role_stats["planner"].get("model", "") or "")
    if isinstance(role_stats.get("reviewer"), dict):
        reviewer_model_name = str(role_stats["reviewer"].get("model", "") or "")
    if not planner_model_name:
        planner_model_name = str(run_config.get("planner_model", "") or "")
    if not reviewer_model_name:
        reviewer_model_name = str(run_config.get("reviewer_model", "") or "")
    if architecture == "single":
        config_label = f"single::{model_name}"
    elif architecture == "planner_coder":
        config_label = f"planner_coder::{planner_model_name or model_name}->{model_name}"
    else:
        config_label = f"planner_coder_reviewer::{planner_model_name or model_name}->{model_name}->{reviewer_model_name or model_name}"
    preset_name = _infer_preset_name(artifact.run_dir, presets)
    planner_size_rank, coder_size_rank, reviewer_size_rank = _role_size_ranks(
        preset_name=preset_name,
        presets=presets,
        architecture=architecture,
        model_name=model_name,
        planner_model_name=planner_model_name,
        reviewer_model_name=reviewer_model_name,
    )
    effective_compute = max(float(relative_to_4o_mini), 0.001)
    score_per_compute = round(float(total_score) / effective_compute, 3)
    resolved_per_compute = round((1.0 if success_eval["passed"] else 0.0) / effective_compute, 3)
    mixed_size = architecture != "single" and planner_size_rank > coder_size_rank

    return {
        "run_dir": str(artifact.run_dir),
        "run_name": artifact.run_dir.name,
        "instance_dir": str(artifact.instance_dir),
        "instance_id": artifact.instance_id,
        "case_file": str(case_file),
        "case_showcase": showcase,
        "case_difficulty": difficulty,
        "architecture": architecture,
        "preset_name": preset_name,
        "model": model_name,
        "planner_model": planner_model_name,
        "reviewer_model": reviewer_model_name,
        "config_label": config_label,
        "planner_size_rank": planner_size_rank,
        "coder_size_rank": coder_size_rank,
        "reviewer_size_rank": reviewer_size_rank,
        "mixed_size": mixed_size,
        "tool_call_mode": run_config.get("tool_call_mode", ""),
        "submitted": submitted,
        "stopped_reason": stopped_reason,
        "duration_seconds": traj.get("duration_seconds", 0),
        "turns": len(turns),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "relative_compute_units": round(relative_compute_units, 2),
        "relative_compute_to_4o_mini": round(relative_to_4o_mini, 3),
        "score_per_compute": score_per_compute,
        "resolved_per_compute": resolved_per_compute,
        "compute_model": {
            "baseline": "gpt-4o-mini-equivalent token compute units",
            **compute_detail,
        },
        "compute_estimated_from_trace": compute_estimated,
        "functional_correctness": functional_correctness,
        "repair_precision": repair_precision,
        "regression_safety": regression_safety,
        "search_grounding": search_grounding,
        "efficiency_control": efficiency_control,
        "total_score": total_score,
        "baseline_passed": baseline_eval["passed"],
        "success_passed": success_eval["passed"],
        "observed_success_passed": observed_success,
        "evaluation_blocked": evaluation_blocked,
        "baseline_check_fraction": round(baseline_fraction, 3),
        "success_check_fraction": round(success_fraction, 3),
        "baseline_effective_fraction": round(baseline_effective_fraction, 3),
        "success_effective_fraction": round(success_effective_fraction, 3),
        "baseline_effective_checks": baseline_effective_checks,
        "success_effective_checks": success_effective_checks,
        "baseline_results": baseline_eval["results"],
        "success_results": success_eval["results"],
        "changed_files": changed_files,
        "changed_line_count": changed_lines,
        "patch_hunks": patch_hunks,
        "inspected_files": sorted(inspected_files),
        "edited_files": sorted(edited_files),
        "successful_edit_count": successful_edit_count,
        "failed_edit_count": failed_edit_count,
        "validation_count": validation_count,
        "validations_after_edit": validations_after_edit,
        "search_count": search_count,
        "view_count": view_count,
        "parse_errors": parse_errors,
        "empty_responses": empty_responses,
        "duplicate_action_recoveries": duplicate_action_recoveries,
        "tool_error_count": tool_error_count,
        "missing_path_errors": missing_path_errors,
        "repeated_success_loop_hints": repeated_success_loop_hints,
        "repeated_failure_stops": repeated_failure_stops,
        "validation_commands": validation_commands,
        "search_commands": search_commands[:12],
        "role_model_stats": role_stats,
        "notes": notes,
        "turn_summaries": turn_summaries[:12],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "runs": 0,
            "avg_total_score": 0.0,
            "resolved_rate": 0.0,
            "avg_relative_compute_to_4o_mini": 0.0,
        }
    runs = len(results)
    resolved = sum(1 for result in results if result["success_passed"])
    observed_resolved = sum(1 for result in results if result["observed_success_passed"])
    blocked = sum(1 for result in results if result["evaluation_blocked"])
    return {
        "runs": runs,
        "resolved_rate": round(resolved / runs, 3),
        "observed_resolved_rate": round(observed_resolved / runs, 3),
        "evaluation_blocked_runs": blocked,
        "avg_total_score": round(sum(result["total_score"] for result in results) / runs, 2),
        "avg_relative_compute_to_4o_mini": round(sum(result["relative_compute_to_4o_mini"] for result in results) / runs, 3),
        "avg_turns": round(sum(result["turns"] for result in results) / runs, 2),
        "avg_parse_errors": round(sum(result["parse_errors"] for result in results) / runs, 2),
        "avg_tool_errors": round(sum(result["tool_error_count"] for result in results) / runs, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Run root, single run dir, or single .traj file.")
    parser.add_argument("--cases-root", type=Path, default=Path("SWE-agent/custom_cases"))
    parser.add_argument("--run-install", action="store_true", help="Run case install/setup commands before evaluation checks.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_root = _resolve_cases_root(args.cases_root)
    case_map = _load_case_map(cases_root)
    artifacts = _collect_artifacts(args.target)
    if not artifacts:
        raise SystemExit(f"No custom-run artifacts found under {args.target}")

    results: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    for artifact in artifacts:
        case_entry = case_map.get(artifact.instance_id)
        if case_entry is None:
            missing_cases.append(artifact.instance_id)
            continue
        results.append(analyze_artifact(artifact, case_entry, args.run_install))

    payload = {
        "target": str(args.target.resolve()),
        "cases_root": str(cases_root),
        "missing_cases": sorted(set(missing_cases)),
        "aggregate": _aggregate(results),
        "results": results,
    }

    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(payload, indent=2))

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"target: {payload['target']}")
    print(f"runs: {payload['aggregate']['runs']}")
    print(f"resolved_rate: {payload['aggregate']['resolved_rate']}")
    print(f"observed_resolved_rate: {payload['aggregate']['observed_resolved_rate']}")
    print(f"evaluation_blocked_runs: {payload['aggregate']['evaluation_blocked_runs']}")
    print(f"avg_total_score: {payload['aggregate']['avg_total_score']}")
    print(f"avg_relative_compute_to_4o_mini: {payload['aggregate']['avg_relative_compute_to_4o_mini']}")
    if payload["missing_cases"]:
        print(f"missing_cases: {', '.join(payload['missing_cases'])}")

    for result in results:
        print()
        print(
            f"[{result['run_name']}/{result['instance_id']}] total={result['total_score']}"
            f" success={result['success_passed']}"
            f" observed_success={result['observed_success_passed']}"
            f" submitted={result['submitted']}"
        )
        print(
            "  scores:"
            f" functional={result['functional_correctness']}"
            f" precision={result['repair_precision']}"
            f" safety={result['regression_safety']}"
            f" grounding={result['search_grounding']}"
            f" efficiency={result['efficiency_control']}"
        )
        print(
            "  compute:"
            f" relative_to_4o_mini={result['relative_compute_to_4o_mini']}"
            f" input_tokens={result['input_tokens']}"
            f" output_tokens={result['output_tokens']}"
        )
        print(
            "  behavior:"
            f" turns={result['turns']}"
            f" parse_errors={result['parse_errors']}"
            f" tool_errors={result['tool_error_count']}"
            f" changed_files={len(result['changed_files'])}"
        )
        for note in result["notes"][:6]:
            print(f"  note: {note}")


if __name__ == "__main__":
    main()
