#!/usr/bin/env python3
"""Summarize trajectory signals that are closer to issue-level progress."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANALYSIS_SCALE = 20
QUALITY_SCALE = 10
EFFICIENCY_SCALE = 5
COMPLETION_SCALE = 5
ABS_PATH_RE = re.compile(r"(/(?:testbed|workspace|repo)[^\s'\"`]+)")
ISSUE_PATH_RE = re.compile(r"(?P<path>(?:[\w.-]+/)+[\w.-]+\.\w+)")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
MODEL_SIZE_RE = re.compile(r"(?P<size>\d+(?:\.\d+)?)b\b", re.IGNORECASE)
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
    "rg ",
    "sed ",
    "cat ",
    "ls ",
    "find ",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "bug",
    "by",
    "code",
    "data",
    "describe",
    "do",
    "does",
    "edit",
    "expected",
    "file",
    "for",
    "from",
    "handler",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "issue",
    "it",
    "its",
    "of",
    "on",
    "or",
    "output",
    "perform",
    "pixel",
    "problem",
    "representation",
    "run",
    "should",
    "step",
    "steps",
    "test",
    "than",
    "that",
    "the",
    "this",
    "to",
    "use",
    "used",
    "using",
    "was",
    "were",
    "with",
}


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


def _normalize_runtime_path(path: str) -> str:
    normalized = path.strip().rstrip(".,:`")
    for prefix in ("/testbed/", "/workspace/", "/repo/"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized.lstrip("/")


def _normalize_issue_path(path: str) -> str:
    candidate = path.strip().rstrip(".,:)]}`")
    for marker in ("/blob/", "/raw/"):
        if marker in candidate:
            suffix = candidate.split(marker, 1)[1]
            parts = suffix.split("/")
            if len(parts) >= 2:
                candidate = "/".join(parts[1:])
    if "github.com/" in candidate:
        candidate = candidate.split("github.com/", 1)[1]
        parts = candidate.split("/")
        if len(parts) >= 5:
            candidate = "/".join(parts[4:])
    return candidate.lstrip("/")


def _path_tokens(path: str) -> set[str]:
    normalized = _normalize_runtime_path(path)
    raw = normalized.replace("/", " ").replace(".", " ").replace("-", " ").replace("_", " ")
    return {token.lower() for token in TOKEN_RE.findall(raw) if len(token) > 2}


def _extract_problem_statement(data: dict, replay_config: dict) -> str:
    for source in (data, replay_config):
        problem_statement = source.get("problem_statement")
        if isinstance(problem_statement, dict) and isinstance(problem_statement.get("text"), str):
            return problem_statement["text"]
        if isinstance(problem_statement, str):
            return problem_statement
    return ""


def _extract_issue_paths(problem_statement: str) -> list[str]:
    candidates = {
        _normalize_issue_path(match.group("path"))
        for match in ISSUE_PATH_RE.finditer(problem_statement)
    }
    return sorted(path for path in candidates if path)


def _extract_issue_keywords(problem_statement: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(problem_statement)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def _extract_issue_symbols(problem_statement: str) -> set[str]:
    symbols = {match.group(1) for match in re.finditer(r"`([^`]+)`", problem_statement)}
    return {symbol for symbol in symbols if len(symbol) > 2}


def _path_keyword_overlap(path: str, issue_keywords: set[str]) -> float:
    path_terms = {token for token in _path_tokens(path) if token not in STOPWORDS}
    if not path_terms or not issue_keywords:
        return 0.0
    return len(path_terms & issue_keywords) / len(path_terms)


def _path_matches_issue(path: str, issue_paths: list[str], issue_keywords: set[str]) -> bool:
    normalized_path = _normalize_runtime_path(path)
    basename = Path(normalized_path).name
    for issue_path in issue_paths:
        issue_basename = Path(issue_path).name
        if normalized_path.endswith(issue_path) or basename == issue_basename:
            return True
    return _path_keyword_overlap(path, issue_keywords) >= 0.34


def _is_edit_action(action: str) -> bool:
    return action.startswith("str_replace_editor str_replace ")


def _is_inspection_action(action: str) -> bool:
    return action.startswith(INSPECTION_PREFIXES)


def _is_validation_action(action: str) -> bool:
    stripped = action.strip()
    return (
        "pytest" in stripped
        or "tox" in stripped
        or "nox" in stripped
        or "unittest" in stripped
        or "make test" in stripped
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
    problem_statement = _extract_problem_statement(data, replay_config)
    issue_paths = _extract_issue_paths(problem_statement)
    issue_keywords = _extract_issue_keywords(problem_statement)
    issue_symbols = _extract_issue_symbols(problem_statement)
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
    aligned_inspected_files: set[str] = set()
    aligned_edited_files: set[str] = set()
    inspected_sequence: list[str] = []
    edited_sequence: list[str] = []
    inspected_before_first_edit = False
    saw_edit_attempt = False
    inspected_before_edit_count = 0
    first_edit_step: int | None = None
    first_aligned_edit_step: int | None = None
    validation_after_edit = False
    validation_after_edit_count = 0
    manual_submit = False
    issue_symbol_hits = 0

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
        "issue_file_hints": len(issue_paths),
        "issue_symbol_hints": len(issue_symbols),
        "aligned_inspection_steps": 0,
        "aligned_edit_steps": 0,
        "aligned_files_inspected": 0,
        "aligned_files_edited": 0,
        "inspected_before_edit_count": 0,
        "validation_after_edit": False,
        "validation_after_edit_count": 0,
        "first_edit_step": -1,
        "first_aligned_edit_step": -1,
        "manual_submit": False,
        "clean_exit": False,
        "edited_file_alignment": "0.00",
        "inspected_file_alignment": "0.00",
        "issue_alignment_score": "0/5",
        "solution_focus_score": "0/5",
        "workflow_score": "0/5",
        "stability_score": "0/5",
        "analysis_score": "0/20",
        "edited_files": "",
        "inspected_files": "",
        "aligned_files": "",
        "applicable_progress_max": 0,
        "progress_score": 0,
        "penalty_score": 0,
        "quality_score": "0/10",
        "grounding_score": "0/5",
        "completion_score": "0/5",
        "efficiency_score": "0/5",
    }

    for index, step in enumerate(trajectory, start=1):
        action = step.get("action", "") or ""
        observation = step.get("observation", "") or ""
        observation_lower = observation.lower()
        step_text = "\n".join([action, observation, *(_iter_query_texts(step))])

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
            inspected_sequence.extend(action_paths)
            aligned_paths = [path for path in action_paths if _path_matches_issue(path, issue_paths, issue_keywords)]
            if aligned_paths:
                summary["aligned_inspection_steps"] = int(summary["aligned_inspection_steps"]) + 1
                aligned_inspected_files.update(aligned_paths)
            if not saw_edit_attempt and action_paths:
                inspected_before_first_edit = True

        if _is_edit_action(action):
            saw_edit_attempt = True
            if first_edit_step is None:
                first_edit_step = index
            summary["edit_attempts"] = int(summary["edit_attempts"]) + 1
            action_paths = _extract_paths(action)
            edited_files.update(action_paths)
            edited_sequence.extend(action_paths)
            aligned_paths = [path for path in action_paths if _path_matches_issue(path, issue_paths, issue_keywords)]
            if aligned_paths:
                summary["aligned_edit_steps"] = int(summary["aligned_edit_steps"]) + 1
                aligned_edited_files.update(aligned_paths)
                if first_aligned_edit_step is None:
                    first_aligned_edit_step = index
            inspected_before_edit_count += sum(1 for path in action_paths if path in inspected_files)
            if any(hint in observation_lower for hint in EDIT_FAILURE_HINTS):
                summary["failed_edit_steps"] = int(summary["failed_edit_steps"]) + 1
                summary["command_error_steps"] = int(summary["command_error_steps"]) + 1
            else:
                summary["successful_edit_steps"] = int(summary["successful_edit_steps"]) + 1

        if _is_validation_action(action):
            summary["validation_runs"] = int(summary["validation_runs"]) + 1
            if saw_edit_attempt:
                validation_after_edit = True
                validation_after_edit_count += 1
        if action.strip() == "submit":
            summary["submitted"] = True
            manual_submit = "autosubmitted" not in observation_lower
        if "autosubmitted" in observation_lower:
            summary["autosubmitted"] = True
        if "syntax error" in observation_lower or "usage: str_replace_editor" in observation_lower:
            summary["syntax_errors"] = int(summary["syntax_errors"]) + 1
        if "invalid `view_range`" in observation_lower or "traceback" in observation_lower or "error:" in observation_lower:
            summary["command_error_steps"] = int(summary["command_error_steps"]) + 1
        if issue_symbols and any(symbol in step_text for symbol in issue_symbols):
            issue_symbol_hits += 1

    summary["unique_files_inspected"] = len(inspected_files)
    summary["unique_files_edited"] = len(edited_files)
    summary["inspected_before_first_edit"] = inspected_before_first_edit
    summary["aligned_files_inspected"] = len(aligned_inspected_files)
    summary["aligned_files_edited"] = len(aligned_edited_files)
    summary["inspected_before_edit_count"] = inspected_before_edit_count
    summary["validation_after_edit"] = validation_after_edit
    summary["validation_after_edit_count"] = validation_after_edit_count
    summary["first_edit_step"] = first_edit_step if first_edit_step is not None else -1
    summary["first_aligned_edit_step"] = first_aligned_edit_step if first_aligned_edit_step is not None else -1
    summary["manual_submit"] = manual_submit
    summary["clean_exit"] = info.get("exit_status") not in (None, "early_exit")
    summary["edited_files"] = ",".join(sorted(_normalize_runtime_path(path) for path in edited_files))
    summary["inspected_files"] = ",".join(sorted(_normalize_runtime_path(path) for path in inspected_files))
    summary["aligned_files"] = ",".join(
        sorted(_normalize_runtime_path(path) for path in (aligned_inspected_files | aligned_edited_files))
    )
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

    inspected_alignment = 0.0
    if inspected_sequence:
        inspected_alignment = sum(_path_keyword_overlap(path, issue_keywords) for path in inspected_sequence) / len(inspected_sequence)
    edited_alignment = 0.0
    if edited_sequence:
        edited_alignment = sum(_path_keyword_overlap(path, issue_keywords) for path in edited_sequence) / len(edited_sequence)

    issue_alignment = 0
    if aligned_inspected_files:
        issue_alignment += 2
    elif inspected_alignment >= 0.25:
        issue_alignment += 1
    if aligned_edited_files:
        issue_alignment += 2
    elif edited_alignment >= 0.25:
        issue_alignment += 1
    if issue_symbol_hits > 0:
        issue_alignment += 1

    focus = 0
    if first_edit_step is None or first_aligned_edit_step in (None, first_edit_step):
        focus += 1
    if inspected_before_edit_count >= len(edited_sequence) and edited_sequence:
        focus += 1
    if int(summary["unique_files_edited"]) <= 2:
        focus += 1
    if int(summary["unique_files_inspected"]) <= 5:
        focus += 1
    if not edited_sequence or int(summary["aligned_edit_steps"]) == int(summary["edit_attempts"]):
        focus += 1

    workflow = 0
    if int(summary["inspection_steps"]) > 0:
        workflow += 1
    if int(summary["successful_edit_steps"]) > 0:
        workflow += 1
    if validation_after_edit:
        workflow += 1
    if summary["submitted"]:
        workflow += 1
    if manual_submit or summary["clean_exit"]:
        workflow += 1

    stability = 5
    if int(summary["command_error_steps"]) > 0:
        stability -= min(2, int(summary["command_error_steps"]))
    if int(summary["failed_edit_steps"]) > 0:
        stability -= 1
    if int(summary["empty_steps"]) > 0:
        stability -= 1
    if len(trajectory) > 40 or int(summary["token_total"]) > 150000:
        stability -= 1
    if trajectory and (int(summary["token_total"]) / len(trajectory)) > 6000:
        stability -= 1
    stability = max(0, min(5, stability))

    summary["applicable_progress_max"] = applicable_progress_max
    summary["progress_score"] = progress_score
    summary["penalty_score"] = penalty_score
    summary["edited_file_alignment"] = f"{edited_alignment:.2f}"
    summary["inspected_file_alignment"] = f"{inspected_alignment:.2f}"
    summary["issue_alignment_score"] = _format_score(issue_alignment, 5)
    summary["solution_focus_score"] = _format_score(focus, 5)
    summary["workflow_score"] = _format_score(workflow, 5)
    summary["stability_score"] = _format_score(stability, 5)
    summary["analysis_score"] = _format_score(issue_alignment + focus + workflow + stability, ANALYSIS_SCALE)
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
