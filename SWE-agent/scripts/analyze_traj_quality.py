#!/usr/bin/env python3
"""Summarize generic trajectory quality signals for SWE-agent runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


QUALITY_SCALE = 10
EFFICIENCY_SCALE = 5
COMPLETION_SCALE = 5
ABS_PATH_RE = re.compile(r"(/(?:testbed|workspace|repo)[^\s'\"`]+)")
EDIT_FAILURE_HINTS = (
    "no replacement was performed",
    "usage: str_replace_editor",
    "invalid `view_range`",
    "traceback",
    "error:",
)
INSPECTION_PREFIXES = (
    "str_replace_editor view ",
    "grep ",
    "sed ",
    "cat ",
)
MODEL_SIZE_RE = re.compile(r"(?P<size>\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def _iter_query_texts(step: dict) -> list[str]:
    texts: list[str] = []
    for message in step.get("query", []):
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    texts.append(block["text"])
    return texts


def planner_phase_enabled(data: dict) -> bool:
    trajectory = data.get("trajectory", [])
    for step in trajectory:
        action = step.get("action", "") or ""
        if "handoff " in action or "cat /testbed/handoff.txt" in action:
            return True

        for message in step.get("query", []):
            if message.get("role_name") == "planner":
                return True

        for text in _iter_query_texts(step):
            lowered = text.lower()
            if "the planner is disabled" in lowered:
                return False
            if "planner handoff" in lowered or "planner contract" in lowered:
                return True

    return False


def load_traj(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_replay_config(data: dict) -> dict:
    config = data.get("replay_config")
    if isinstance(config, dict):
        return config
    if isinstance(config, str) and config:
        try:
            return json.loads(config)
        except json.JSONDecodeError:
            return {}
    return {}


def model_size_weight(model_name: str) -> float:
    match = MODEL_SIZE_RE.search(model_name)
    if not match:
        return 1.0
    return float(match.group("size"))


def estimate_relative_cost(tokens_in: int, tokens_out: int, model_name: str) -> float:
    size_weight = model_size_weight(model_name)
    return (tokens_in * size_weight) + (tokens_out * size_weight * 2.0)


def _extract_paths(text: str) -> list[str]:
    return [match.rstrip(".,:") for match in ABS_PATH_RE.findall(text)]


def _is_edit_action(action: str) -> bool:
    return action.startswith("str_replace_editor str_replace ")


def _is_inspection_action(action: str) -> bool:
    return action.startswith(INSPECTION_PREFIXES)


def _is_validation_action(action: str) -> bool:
    stripped = action.strip()
    return (
        "pytest" in stripped
        or " python -c " in f" {stripped} "
        or " python3 -c " in f" {stripped} "
        or stripped.startswith("python -c ")
        or stripped.startswith("python3 -c ")
        or " python -m " in f" {stripped} "
        or " python3 -m " in f" {stripped} "
    )


def _format_score(value: int | float, scale: int, precision: int = 0) -> str:
    if precision == 0:
        return f"{int(value)}/{scale}"
    return f"{value:.{precision}f}/{scale}"


def score_traj(data: dict) -> dict:
    trajectory = data.get("trajectory", [])
    info = data.get("info", {}) if isinstance(data.get("info"), dict) else {}
    replay_config = parse_replay_config(data)
    agent_cfg = replay_config.get("agent", {}) if isinstance(replay_config, dict) else {}
    role_model_names = {
        "planner": agent_cfg.get("planner", ""),
        "coder": agent_cfg.get("coder", ""),
        "reviewer": agent_cfg.get("reviewer", ""),
    }
    role_model_stats = info.get("role_model_stats", {}) if isinstance(info.get("role_model_stats"), dict) else {}
    total_model_stats = info.get("model_stats", {}) if isinstance(info.get("model_stats"), dict) else {}
    planner_enabled = planner_phase_enabled(data)
    progress_score = 0
    penalty_score = 0
    applicable_progress_max = 0
    inspected_files: set[str] = set()
    edited_files: set[str] = set()
    inspected_before_first_edit = False
    saw_edit_attempt = False
    summary: dict[str, object] = {
        "steps": len(trajectory),
        "planner_phase_enabled": planner_enabled,
        "planner_handoff": False,
        "coder_started": False,
        "inspection_steps": 0,
        "unique_files_inspected": 0,
        "unique_files_edited": 0,
        "inspected_before_first_edit": False,
        "edit_attempts": 0,
        "successful_edit_steps": 0,
        "failed_edit_steps": 0,
        "validation_runs": 0,
        "submitted": False,
        "autosubmitted": False,
        "empty_steps": 0,
        "command_error_steps": 0,
        "syntax_errors": 0,
        "tokens_in": int(total_model_stats.get("tokens_sent", 0) or 0),
        "tokens_out": int(total_model_stats.get("tokens_received", 0) or 0),
        "api_calls": int(total_model_stats.get("api_calls", 0) or 0),
        "token_total": 0,
        "tokens_per_step": "0.0",
        "relative_cost_estimate": 0.0,
        "planner_model": role_model_names["planner"],
        "coder_model": role_model_names["coder"],
        "reviewer_model": role_model_names["reviewer"],
        "planner_tokens_in": 0,
        "planner_tokens_out": 0,
        "coder_tokens_in": 0,
        "coder_tokens_out": 0,
        "reviewer_tokens_in": 0,
        "reviewer_tokens_out": 0,
        "applicable_progress_max": 0,
        "progress_score": 0,
        "penalty_score": 0,
        "quality_score": "0/10",
        "grounding_score": "0/5",
        "completion_score": "0/5",
        "efficiency_score": "0/5",
    }

    for step in trajectory:
        action = step.get("action", "") or ""
        observation = step.get("observation", "") or ""
        observation_lower = observation.lower()

        if not action.strip():
            summary["empty_steps"] = int(summary["empty_steps"]) + 1

        if action.startswith("handoff "):
            summary["planner_handoff"] = True
        if "cat /testbed/handoff.txt" in action:
            summary["coder_started"] = True

        if _is_inspection_action(action):
            summary["inspection_steps"] = int(summary["inspection_steps"]) + 1
            action_paths = _extract_paths(action)
            inspected_files.update(action_paths)
            if not saw_edit_attempt and action_paths:
                inspected_before_first_edit = True

        if _is_edit_action(action):
            saw_edit_attempt = True
            summary["edit_attempts"] = int(summary["edit_attempts"]) + 1
            action_paths = _extract_paths(action)
            edited_files.update(action_paths)
            if any(hint in observation_lower for hint in EDIT_FAILURE_HINTS):
                summary["failed_edit_steps"] = int(summary["failed_edit_steps"]) + 1
                summary["command_error_steps"] = int(summary["command_error_steps"]) + 1
            else:
                summary["successful_edit_steps"] = int(summary["successful_edit_steps"]) + 1

        if _is_validation_action(action):
            summary["validation_runs"] = int(summary["validation_runs"]) + 1
        if action.strip() == "submit":
            summary["submitted"] = True
        if "autosubmitted" in observation_lower:
            summary["autosubmitted"] = True
        if "syntax error" in observation_lower or "usage: str_replace_editor" in observation_lower:
            summary["syntax_errors"] = int(summary["syntax_errors"]) + 1
        if "invalid `view_range`" in observation_lower or "traceback" in observation_lower or "error:" in observation_lower:
            summary["command_error_steps"] = int(summary["command_error_steps"]) + 1

    summary["unique_files_inspected"] = len(inspected_files)
    summary["unique_files_edited"] = len(edited_files)
    summary["inspected_before_first_edit"] = inspected_before_first_edit
    summary["token_total"] = int(summary["tokens_in"]) + int(summary["tokens_out"])
    if trajectory:
        summary["tokens_per_step"] = f"{int(summary['token_total']) / len(trajectory):.1f}"

    relative_cost = 0.0
    for role in ("planner", "coder", "reviewer"):
        stats = role_model_stats.get(role, {}) if isinstance(role_model_stats.get(role), dict) else {}
        role_in = int(stats.get("tokens_sent", 0) or 0)
        role_out = int(stats.get("tokens_received", 0) or 0)
        summary[f"{role}_tokens_in"] = role_in
        summary[f"{role}_tokens_out"] = role_out
        relative_cost += estimate_relative_cost(role_in, role_out, str(role_model_names.get(role, "")))
    summary["relative_cost_estimate"] = round(relative_cost, 1)

    if planner_enabled:
        applicable_progress_max += 3
        if summary["planner_handoff"]:
            progress_score += 2
        if summary["coder_started"]:
            progress_score += 1

    applicable_progress_max += 8
    if int(summary["inspection_steps"]) > 0:
        progress_score += 2
    if int(summary["unique_files_inspected"]) > 0:
        progress_score += 1
    if int(summary["successful_edit_steps"]) > 0:
        progress_score += 2
    elif int(summary["edit_attempts"]) > 0:
        progress_score += 1
    if int(summary["validation_runs"]) > 0:
        progress_score += 1
    if summary["submitted"]:
        progress_score += 2

    if int(summary["command_error_steps"]) > 0:
        penalty_score += min(3, int(summary["command_error_steps"]))
    if int(summary["failed_edit_steps"]) > 1:
        penalty_score += 1
    if int(summary["empty_steps"]) > 0:
        penalty_score += 1

    raw_net_score = max(0, progress_score - penalty_score)
    normalized_score = 0
    if applicable_progress_max > 0:
        normalized_score = round(raw_net_score * QUALITY_SCALE / applicable_progress_max)
    net_score = max(0, min(QUALITY_SCALE, normalized_score))

    grounding = 0
    if int(summary["inspection_steps"]) > 0:
        grounding += 2
    if inspected_before_first_edit:
        grounding += 1
    if edited_files and edited_files.issubset(inspected_files):
        grounding += 1
    if int(summary["unique_files_edited"]) <= 2:
        grounding += 1

    completion = 0
    if int(summary["successful_edit_steps"]) > 0:
        completion += 2
    elif int(summary["edit_attempts"]) > 0:
        completion += 1
    if int(summary["validation_runs"]) > 0:
        completion += 1
    if summary["submitted"]:
        completion += 2
    elif summary["autosubmitted"]:
        completion += 1

    efficiency = EFFICIENCY_SCALE
    if int(summary["command_error_steps"]) > 0:
        efficiency -= 1
    if int(summary["failed_edit_steps"]) > 0:
        efficiency -= 1
    if int(summary["empty_steps"]) > 0:
        efficiency -= 1
    if int(summary["inspection_steps"]) > 10:
        efficiency -= 1
    if len(trajectory) > 25 or int(summary["token_total"]) > 100000:
        efficiency -= 1
    if trajectory and (int(summary["token_total"]) / len(trajectory)) > 4000:
        efficiency -= 1
    efficiency = max(0, min(EFFICIENCY_SCALE, efficiency))

    summary["applicable_progress_max"] = applicable_progress_max
    summary["progress_score"] = progress_score
    summary["penalty_score"] = penalty_score
    summary["quality_score"] = _format_score(net_score, QUALITY_SCALE)
    summary["grounding_score"] = _format_score(grounding, 5)
    summary["completion_score"] = _format_score(completion, COMPLETION_SCALE)
    summary["efficiency_score"] = _format_score(efficiency, EFFICIENCY_SCALE)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traj", type=Path)
    args = parser.parse_args()
    data = load_traj(args.traj)
    print(json.dumps(score_traj(data), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
