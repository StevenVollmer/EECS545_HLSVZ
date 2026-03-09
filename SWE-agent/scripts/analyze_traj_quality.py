#!/usr/bin/env python3
"""Summarize how close a trajectory got to the intended bug fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_FILE = "/testbed/pydicom/pixel_data_handlers/numpy_handler.py"
WRONG_FILE_HINTS = [
    "/testbed/pydicom/pixels/pixel_array.py",
]
TARGET_TOKENS = [
    "PixelRepresentation",
    "required_elements",
    "FloatPixelData",
    "DoubleFloatPixelData",
]
QUALITY_SCALE = 10
EFFICIENCY_SCALE = 5
COMPLETION_SCALE = 5
EDIT_FAILURE_HINTS = (
    "no replacement was performed",
    "usage: str_replace_editor",
    "invalid `view_range`",
    "error:",
)


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


def _is_edit_action(action: str) -> bool:
    return action.startswith("str_replace_editor str_replace ")


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


def score_traj(data: dict) -> dict:
    trajectory = data.get("trajectory", [])
    progress_score = 0
    penalty_score = 0
    planner_enabled = planner_phase_enabled(data)
    applicable_progress_max = 0
    summary: dict[str, object] = {
        "steps": len(trajectory),
        "planner_phase_enabled": planner_enabled,
        "planner_handoff": False,
        "coder_started": False,
        "target_file_reads": 0,
        "target_token_hits": 0,
        "wrong_file_edits": 0,
        "edit_attempts": 0,
        "successful_edit_steps": 0,
        "failed_edit_steps": 0,
        "validation_runs": 0,
        "submitted": False,
        "autosubmitted": False,
        "empty_steps": 0,
        "command_error_steps": 0,
        "syntax_errors": 0,
        "off_target_drift": False,
        "applicable_progress_max": 0,
        "progress_score": 0,
        "penalty_score": 0,
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

        if "handoff " in action:
            summary["planner_handoff"] = True
        if "cat /testbed/handoff.txt" in action:
            summary["coder_started"] = True
        if TARGET_FILE in action or TARGET_FILE in observation:
            summary["target_file_reads"] = int(summary["target_file_reads"]) + 1
        if any(token in action or token in observation for token in TARGET_TOKENS):
            summary["target_token_hits"] = int(summary["target_token_hits"]) + 1
        if _is_edit_action(action):
            summary["edit_attempts"] = int(summary["edit_attempts"]) + 1
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
        if any(hint in action for hint in WRONG_FILE_HINTS) and "str_replace_editor" in action:
            summary["wrong_file_edits"] = int(summary["wrong_file_edits"]) + 1
            summary["off_target_drift"] = True
        if "syntax error" in observation_lower or "usage: str_replace_editor" in observation:
            summary["syntax_errors"] = int(summary["syntax_errors"]) + 1
        if (
            "invalid `view_range`" in observation_lower
            or "traceback" in observation_lower
            or "error:" in observation_lower
        ):
            summary["command_error_steps"] = int(summary["command_error_steps"]) + 1

    if planner_enabled:
        applicable_progress_max += 3
        if summary["planner_handoff"]:
            progress_score += 2
        if summary["coder_started"]:
            progress_score += 1

    applicable_progress_max += 8
    if int(summary["target_file_reads"]) > 0:
        progress_score += 2
    if int(summary["target_token_hits"]) > 1:
        progress_score += 2
    if int(summary["successful_edit_steps"]) > 0:
        progress_score += 2
    elif int(summary["edit_attempts"]) > 0:
        progress_score += 1
    if int(summary["validation_runs"]) > 0:
        progress_score += 1
    if summary["submitted"]:
        progress_score += 1
    if int(summary["wrong_file_edits"]) > 0:
        penalty_score += min(3, int(summary["wrong_file_edits"]))
    if int(summary["command_error_steps"]) > 0:
        penalty_score += min(3, int(summary["command_error_steps"]))
    if summary["off_target_drift"]:
        penalty_score += 1
    if int(summary["empty_steps"]) > 0:
        penalty_score += 1

    raw_net_score = max(0, progress_score - penalty_score)
    normalized_score = 0
    if applicable_progress_max > 0:
        normalized_score = round(raw_net_score * QUALITY_SCALE / applicable_progress_max)
    net_score = max(0, min(QUALITY_SCALE, normalized_score))

    grounding = 0
    if int(summary["target_file_reads"]) > 0:
        grounding += 2
    if int(summary["wrong_file_edits"]) == 0:
        grounding += 2
    if not summary["off_target_drift"]:
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
    if int(summary["target_file_reads"]) > 8:
        efficiency -= 1
    if len(trajectory) > 25:
        efficiency -= 1
    efficiency = max(0, min(EFFICIENCY_SCALE, efficiency))

    summary["applicable_progress_max"] = applicable_progress_max
    summary["progress_score"] = progress_score
    summary["penalty_score"] = penalty_score
    summary["quality_score"] = f"{net_score}/{QUALITY_SCALE}"
    summary["grounding_score"] = f"{grounding}/5"
    summary["completion_score"] = f"{completion}/{COMPLETION_SCALE}"
    summary["efficiency_score"] = f"{efficiency}/{EFFICIENCY_SCALE}"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traj", type=Path)
    args = parser.parse_args()
    data = load_traj(args.traj)
    print(json.dumps(score_traj(data), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
