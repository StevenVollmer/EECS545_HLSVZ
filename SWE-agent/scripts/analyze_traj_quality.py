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


def load_traj(path: Path) -> dict:
    return json.loads(path.read_text())


def score_traj(data: dict) -> dict:
    trajectory = data.get("trajectory", [])
    progress_score = 0
    penalty_score = 0
    max_score = 10
    summary: dict[str, object] = {
        "steps": len(trajectory),
        "planner_handoff": False,
        "coder_started": False,
        "target_file_reads": 0,
        "target_token_hits": 0,
        "wrong_file_edits": 0,
        "edit_attempts": 0,
        "syntax_errors": 0,
        "off_target_drift": False,
        "progress_score": 0,
        "penalty_score": 0,
        "grounding_score": "0/5",
    }

    for step in trajectory:
        action = step.get("action", "") or ""
        observation = step.get("observation", "") or ""

        if "handoff " in action:
            summary["planner_handoff"] = True
        if "cat /testbed/handoff.txt" in action:
            summary["coder_started"] = True
        if TARGET_FILE in action or TARGET_FILE in observation:
            summary["target_file_reads"] = int(summary["target_file_reads"]) + 1
        if any(token in action or token in observation for token in TARGET_TOKENS):
            summary["target_token_hits"] = int(summary["target_token_hits"]) + 1
        if "str_replace_editor" in action:
            summary["edit_attempts"] = int(summary["edit_attempts"]) + 1
        if any(hint in action for hint in WRONG_FILE_HINTS) and "str_replace_editor" in action:
            summary["wrong_file_edits"] = int(summary["wrong_file_edits"]) + 1
            summary["off_target_drift"] = True
        if "syntax error" in observation.lower() or "usage: str_replace_editor" in observation:
            summary["syntax_errors"] = int(summary["syntax_errors"]) + 1

    if summary["planner_handoff"]:
        progress_score += 2
    if summary["coder_started"]:
        progress_score += 1
    if int(summary["target_file_reads"]) > 0:
        progress_score += 2
    if int(summary["target_token_hits"]) > 1:
        progress_score += 2
    if int(summary["edit_attempts"]) > 0:
        progress_score += 2
    if int(summary["wrong_file_edits"]) > 0:
        penalty_score += min(3, int(summary["wrong_file_edits"]))
    if int(summary["syntax_errors"]) > 0:
        penalty_score += min(3, int(summary["syntax_errors"]))
    if summary["off_target_drift"]:
        penalty_score += 1

    net_score = max(0, min(max_score, progress_score - penalty_score))
    grounding = 0
    if int(summary["target_file_reads"]) > 0:
        grounding += 2
    if int(summary["wrong_file_edits"]) == 0:
        grounding += 2
    if not summary["off_target_drift"]:
        grounding += 1
    summary["progress_score"] = progress_score
    summary["penalty_score"] = penalty_score
    summary["quality_score"] = f"{net_score}/{max_score}"
    summary["grounding_score"] = f"{grounding}/5"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traj", type=Path)
    args = parser.parse_args()
    data = load_traj(args.traj)
    print(json.dumps(score_traj(data), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
