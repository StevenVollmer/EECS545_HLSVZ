#!/usr/bin/env python3
"""Standalone custom SWE-bench runner with a simple OpenAI-compatible tool loop.

This is intentionally separate from the existing SWE-agent agent stack so we can
experiment with our own prompting and control logic while still reusing:
- SWE-bench instance loading
- SWE-ReX/SWEEnv environment setup
- Docker testbed execution

Supported backends:
- OpenAI API
- Ollama
- LM Studio
- UMich OpenAI-compatible endpoint
- Any OpenAI-compatible endpoint via --api-base
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import shlex
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.environment.repo import LocalRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv
from sweagent.run.batch_instances import BatchInstance, SWEBenchInstances
from swerex.deployment.config import DockerDeploymentConfig, get_deployment
from swerex.runtime.abstract import Command, UploadRequest

BACKEND_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "api_base": "http://localhost:11434",
        "api_key_env": "OLLAMA_API_KEY",
        "api_key_fallback": "ollama",
    },
    "lmstudio": {
        "api_base": "http://127.0.0.1:1234/v1",
        "api_key_env": "LMSTUDIO_API_KEY",
        "api_key_fallback": "lm-studio",
    },
    "umich": {
        "api_base": "http://promaxgb10-d473.eecs.umich.edu:8000/v1",
        "api_key_env": "UMICH_API_KEY",
        "api_key_fallback": "umich",
    },
    "custom": {},
}


def _preset_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "custom_configs" / "custom_runner_model_presets.yaml"


def _load_presets() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(_preset_config_path().read_text()) or {}
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError("Preset file must contain a top-level 'presets' mapping")
    return presets


def _preset_names() -> list[str]:
    return sorted(_load_presets().keys())


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run one bash command inside the testbed repository. Use this for grep, pytest, git diff, and small python repro commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 600, "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view",
            "description": "Read a whole file or a line range with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace one exact unique block of text in a file. old_str must match exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert",
            "description": "Insert text after a 1-based line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer", "minimum": 0},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "line", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_edit",
            "description": "Undo the last file edit made through str_replace or insert for a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Finish the run after you have validated the fix and inspected the diff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
            },
        },
    },
]


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _dump_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _resolve_api_key(backend: str, explicit: str | None) -> str | None:
    if explicit:
        if explicit.startswith("$"):
            return os.environ.get(explicit[1:])
        return explicit
    defaults = BACKEND_DEFAULTS.get(backend, {})
    env_name = defaults.get("api_key_env")
    if env_name:
        value = os.environ.get(env_name)
        if value:
            return value
    return defaults.get("api_key_fallback")


def _normalize_model_name(backend: str, model: str) -> str:
    if "/" in model:
        return model
    if backend == "ollama" and not model.startswith("ollama/"):
        return f"ollama/{model}"
    return model


def _build_system_prompt(repo_root: str) -> str:
    return f"""You are a software engineer solving a SWE-bench issue.

You are working inside a repository mounted at `{repo_root}`.

Rules:
- Do not think out loud or reveal reasoning. Take actions with tools instead.
- Reproduce the bug before the first source edit when feasible.
- Fix executable runtime logic before touching tests, docs, comments, or examples.
- Do not change tests to match an incorrect implementation unless the case explicitly allows test edits.
- Prefer targeted inspection and one deliberate block replacement over many tiny edits.
- If you are not sure about an exact file path, use `bash` with `find`, `rg --files`, or `ls` before calling `view` or editing tools.
- If a file tool reports that a path does not exist, stop guessing filenames and discover the real path first.
- For multi-line replacements, pass actual multi-line strings in tool arguments. Do not insert literal backslash-n text unless you truly want `\\n` in the file.
- Never replace a bare identifier when it can appear in multiple places. Replace the smallest unique surrounding block.
- After the first successful semantic code edit, validate immediately before making more edits.
- If the case provides exact success checks, those checks are the definition of done. Generic tests passing is not enough if a required success check still fails.
- If you corrupt formatting or insert literal `\\n` text by mistake, use undo_edit or replace the whole block cleanly.
- Before submit, inspect the diff and make sure you changed executable source code and ran a relevant validation.
- Assume standard tools and project dependencies are already present when possible. Only install packages after a command proves they are missing and the package is necessary for required validation or reproduction.
- Use tools instead of describing what you want to do.
"""


def _build_react_json_prompt() -> str:
    return """If native tool calling is unavailable, do not think out loud. Respond with JSON only and no surrounding prose.

Schema:
{"tool":"bash|view|str_replace|insert|undo_edit|submit","arguments":{...}}

Hard output contract:
- Do not reveal reasoning.
- Return exactly one action object per response.
- Do not return an array.
- Do not return multiple newline-separated objects.
- Do not include `<think>`, `</think>`, `<tool_call>`, XML tags, markdown commentary, bullet points, explanations, or any plain English outside the one JSON object.
- Do not ask what help is needed. You already have the task.
- Do not describe a plan. Pick exactly one next action and return only that action.
- On early turns, prefer one discovery action such as `bash` with `find`, `rg --files`, `ls`, or one targeted `view`.

Everything below is invalid and must never appear:
- Multiple JSON objects in one response
- Any text before the JSON object
- Any text after the JSON object
- `</think>` or similar wrapper tags
- Questions to the user
- Summaries of what you saw

If you violate the contract, the runner will ignore your response and ask again.

Use these exact argument names:
- str_replace: `path`, `old_str`, `new_str`
- view: `path`, `start_line`, `end_line`
- bash: `command`
- insert: `path`, `line`, `new_str`
- submit: `summary`

Do not output explanations before or after the JSON.
Do not output comments.
Do not use markdown unless it is a single ```json fence containing only the JSON payload.
Make sure every `{` and `[` is closed. A missing closing `}` will make the action unreadable.
Before responding, mentally check that the JSON parses and that arrays/objects are balanced.

Valid responses:
{"tool":"bash","arguments":{"command":"find /repo -type f -name '*.py' | head -30"}}
{"tool":"view","arguments":{"path":"README.md","start_line":1,"end_line":50}}
```json
{"tool":"submit","arguments":{"summary":"Fixed the denominator bug"}}
```

Invalid responses:
Here is my next step: {"tool":"bash","arguments":{"command":"ls"}}
{"tool":"bash","arguments":{"command":"ls"}} {"tool":"view","arguments":{"path":"README.md"}}
{"tool":"bash","arguments":{"command":"ls"}}
</think>
What would you like me to help you with?

When in doubt, output one simple discovery action only.
"""


def _build_openai_tools_prompt() -> str:
    return """Use native tool calling for every action.

Do not put tool arguments in normal assistant text.
Do not return raw JSON in message content.
Do not describe a plan instead of calling a tool.
If you want to inspect, edit, validate, or submit, do it via a native tool call.
Do not change tests to make an incorrect implementation look correct.
"""


def _build_task_prompt(problem_statement: str, repo_root: str) -> str:
    return f"""Solve this SWE-bench issue inside `{repo_root}`.

Do not think out loud. Do not reveal reasoning. Do not explain a plan. Return only one concrete next action at a time.

Problem statement:
<problem_statement>
{problem_statement}
</problem_statement>

Workflow:
1. Reproduce or run a direct targeted validation first.
2. Inspect the runtime code path.
3. Make the smallest correct code change.
4. Validate immediately.
5. Inspect `git diff`.
6. Call `submit`.

Important:
- Existing tests may only reflect the current buggy behavior. Do not assume a passing baseline test means the behavior is correct.
- Do not edit tests to fit a wrong implementation unless the case explicitly allows test edits.
"""


def _build_runtime_context_prompt(repo_root: str, runtime_context: str) -> str:
    return f"""Runtime context for this repository:

- Repository root: `{repo_root}`
- Use repo-relative paths like `calculator.py` or absolute paths under `{repo_root}`
- Current shell working directory is the repository root unless a command changes it

Startup observations:
<runtime_context>
{runtime_context}
</runtime_context>
"""


def _build_case_validation_prompt(problem_statement) -> str:
    extra_fields = getattr(problem_statement, "extra_fields", {}) or {}
    evaluation = extra_fields.get("evaluation", {})
    policy = extra_fields.get("policy", {})
    if not isinstance(evaluation, dict):
        return ""
    allow_test_edits = bool(policy.get("allow_test_edits", False)) if isinstance(policy, dict) else False
    baseline = evaluation.get("baseline_checks", [])
    success = evaluation.get("success_checks", [])
    baseline_cmds = [str(item.get("command", "")) for item in baseline if isinstance(item, dict) and item.get("command")]
    success_cmds = [str(item.get("command", "")) for item in success if isinstance(item, dict) and item.get("command")]
    if not baseline_cmds and not success_cmds:
        return ""
    lines = [
        "Case validation hints:",
        "- Use these exact commands when they fit the task. Do not invent narrower test names unless you have verified they exist.",
        "- Treat the success checks as the real definition of done, even if some older baseline tests still pass or still encode the buggy behavior.",
    ]
    if not allow_test_edits:
        lines.append("- Do not modify tests for this case. Fix runtime code so the required success checks pass.")
    if baseline_cmds:
        lines.append("- Reproduction commands:")
        lines.extend(f"  - {cmd}" for cmd in baseline_cmds[:3])
    if success_cmds:
        lines.append("- Success validation commands:")
        lines.extend(f"  - {cmd}" for cmd in success_cmds[:3])
    return "\n".join(lines)


def _extract_case_evaluation(problem_statement) -> dict[str, Any]:
    extra_fields = getattr(problem_statement, "extra_fields", {}) or {}
    evaluation = extra_fields.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return {}
    return evaluation


def _extract_case_success_commands(problem_statement) -> list[str]:
    evaluation = _extract_case_evaluation(problem_statement)
    success = evaluation.get("success_checks", [])
    commands: list[str] = []
    for item in success:
        if isinstance(item, dict) and item.get("command"):
            commands.append(str(item["command"]).strip())
    return commands


def _extract_case_success_checks(problem_statement) -> list[dict[str, Any]]:
    evaluation = _extract_case_evaluation(problem_statement)
    success = evaluation.get("success_checks", [])
    checks: list[dict[str, Any]] = []
    for item in success:
        if isinstance(item, dict) and item.get("command"):
            checks.append(dict(item))
    return checks


def _extract_case_policy(problem_statement) -> dict[str, Any]:
    extra_fields = getattr(problem_statement, "extra_fields", {}) or {}
    policy = extra_fields.get("policy", {})
    if not isinstance(policy, dict):
        policy = {}
    return {
        "allow_test_edits": bool(policy.get("allow_test_edits", False)),
        "allow_doc_edits": bool(policy.get("allow_doc_edits", False)),
    }


def _extract_case_analysis(problem_statement) -> dict[str, Any]:
    extra_fields = getattr(problem_statement, "extra_fields", {}) or {}
    analysis = extra_fields.get("analysis", {})
    if not isinstance(analysis, dict):
        return {}
    likely_fix_paths = [str(item).strip() for item in analysis.get("likely_fix_paths", []) if str(item).strip()]
    return {
        "likely_fix_paths": likely_fix_paths[:5],
        "showcase": str(analysis.get("showcase", "")).strip(),
        "difficulty": str(analysis.get("difficulty", "")).strip(),
    }


def _build_case_analysis_prompt(problem_statement) -> str:
    analysis = _extract_case_analysis(problem_statement)
    if not analysis:
        return ""
    lines = ["Case analysis hints:"]
    likely_fix_paths = analysis.get("likely_fix_paths") or []
    if likely_fix_paths:
        lines.append("- Likely fix paths:")
        lines.extend(f"  - {path}" for path in likely_fix_paths[:5])
        lines.append("- Treat these as high-signal hints, not proof. Use them to rank inspection order, not to skip reproduction or evidence.")
    showcase = analysis.get("showcase", "")
    if showcase:
        lines.append(f"- Case showcase emphasis: {showcase}")
    difficulty = analysis.get("difficulty", "")
    if difficulty:
        lines.append(f"- Case difficulty: {difficulty}")
    return "\n".join(lines)


def _command_output_satisfies_check(check: dict[str, Any], *, exit_code: int | None, output: str) -> bool:
    if exit_code is None:
        return False
    expected_exit = int(check.get("expect_exit_code", 0))
    if exit_code != expected_exit:
        return False
    haystack = output or ""
    for text in check.get("stdout_contains", []):
        if str(text) not in haystack:
            return False
    for text in check.get("stdout_not_contains", []):
        if str(text) in haystack:
            return False
    return True


def _command_output_failure_reasons(check: dict[str, Any], *, exit_code: int | None, output: str) -> list[str]:
    failures: list[str] = []
    expected_exit = int(check.get("expect_exit_code", 0))
    if exit_code is None or exit_code != expected_exit:
        failures.append(f"expected exit code {expected_exit}, got {exit_code}")
    haystack = output or ""
    for text in check.get("stdout_contains", []):
        value = str(text)
        if value not in haystack:
            failures.append(f"missing expected text {value!r}")
    for text in check.get("stdout_not_contains", []):
        value = str(text)
        if value in haystack:
            failures.append(f"unexpected text present {value!r}")
    return failures


def _build_planner_system_prompt(repo_root: str) -> str:
    return f"""You are the planner for a software repair task.

You do not edit files. You produce a concrete repair contract for the coder.

Rules:
- Do not think out loud.
- Output JSON only.
- Do not include markdown fences.
- Do not include patch text or exact replacement strings.
- Keep the plan focused on likely runtime files under `{repo_root}`.
- Prefer a conservative, high-signal handoff over a broad speculative dump.
- Do not invent exact file paths unless the runtime context strongly supports them.
- If uncertain, name modules/directories/functions and one discovery command instead of hallucinating filenames.
- Keep `files_likely_affected` to at most 4 entries, ordered from most likely to least likely.
- Prefer runtime code over tests. Do not list test files as likely fix locations unless the case explicitly allows test edits or the issue is clearly about missing coverage.
- Prefer robust reproduction and validation commands that already appear in the runtime context, README excerpt, or case metadata.
- Avoid brittle shell one-liners that require tricky quoting or escaping. Prefer stable commands like `pytest ...`, `python scripts/...`, `rg`, `find`, and `view`.
- If a repro requires a tricky quoted literal, describe the safer equivalent in `reproduction_notes` and keep the command itself simple.
- Treat your plan as ranked guidance, not certainty. If the bug could live in more than one layer, reflect that in the file ordering and hypothesis rather than overcommitting.
- Prefer one primary hypothesis and one secondary fallback over a broad list of speculative files.
- Do not include target symbols you cannot justify from the issue text or runtime context.

Return a JSON object with these keys:
- problem_summary: short string
- root_cause_hypothesis: short string
- files_likely_affected: array of file paths
- target_symbols: array of functions/classes/modules
- discovery_priority: array of 1-4 files or modules to inspect first
- first_actions: array of 1-3 concrete next steps for the coder
- safe_reproduction_steps: array of safe commands or checks
- reproduction_notes: array of short strings about edge cases or quoting hazards
- required_validations: array of safe commands or checks
- allowed_change_types: array of strings
- forbidden_edits: array of strings
- escalation_conditions: array of strings
"""


def _build_planner_task_prompt(problem_statement: str, repo_root: str, runtime_context: str) -> str:
    return f"""Create a repair plan for this issue in `{repo_root}`.

Problem statement:
<problem_statement>
{problem_statement}
</problem_statement>

Runtime context:
<runtime_context>
{runtime_context}
</runtime_context>

Constraints:
- Do not guess exact file paths unless the runtime context already supports them.
- Prefer module-level areas and symbols over invented filenames.
- If the code location is uncertain, say so in `root_cause_hypothesis` and keep `files_likely_affected` conservative.
- Use `first_actions` to tell the coder exactly how to start: discovery, repro, inspect.
- Use `safe_reproduction_steps` and `required_validations` to name the highest-signal checks only.
- Only name exact test commands or script paths if they already appear in the runtime context or the problem statement.
- If existing tests pass at baseline but the problem statement still describes a user-visible bug, do not treat the baseline tests as proof of correctness.
- Prefer a runtime fix plan over a test-adjustment plan. Existing tests may be lagging indicators of the intended behavior.
- Avoid fragile commands with nested quotes or apostrophe-heavy literals when a safer script/test command exists.
- Rank the most likely files and symbols so the coder can inspect quickly instead of re-searching the whole repo.
- Prefer at most 1 safe reproduction command and at most 2 required validation commands.
- Put the safest, highest-signal command first. Prefer `python scripts/...`, `pytest ...`, or direct file inspection over complex inline `python -c` commands.
- Use `reproduction_notes` to warn about quoting hazards, manual observations, or when the coder should prefer an equivalent script/test command.
- Use `files_likely_affected` and `discovery_priority` as a ranked shortlist. Do not pad them just to reach the limit.
- If the issue could plausibly be in a presenter, service, or shared utility, express that uncertainty instead of pretending one layer is certain.
"""


def _build_reviewer_system_prompt(repo_root: str) -> str:
    return f"""You are the reviewer for a software repair task.

You do not edit files directly. You review the planner contract, the coder's patch, and the observed validations.

Rules:
- Do not think out loud.
- Output JSON only.
- Do not include markdown fences.
- Decide whether the current patch is ready or should go back to the coder.
- Use the latest post-edit validation evidence as the primary review signal.
- Be conservative about rejection. Do not reject for style, naming preference, or hypothetical cleanup.
- Accept if the observed fix is behaviorally correct, validations are sufficient for the case, and the patch is reasonably focused.
- Reject only when there is concrete evidence of a remaining bug, missing required validation, wrong-file fix, or regression risk.
- If case-defined success checks exist, do not accept unless those checks were actually run after the final edit and their observed results support the fix.
- If a required success check failed, treat that as the primary rejection reason.
- If the patch looks good and all required success checks passed after the final edit, accept even if there were earlier parse or tool errors.
- Treat edits to tests as a rejection reason unless the case explicitly allows test edits or a targeted regression test accompanies a real runtime fix.
- If the issue is behavioral and existing checks are weak, consider requiring one targeted test or one stronger regression check before acceptance.
- When rejecting, name one primary failure reason, give at most 2 required changes, at most 2 files to revisit, and at most 2 validations to rerun.
- Tie every rejection to observed evidence from the patch, the latest failing validation output, or missing success checks.

Return a JSON object with these keys:
- decision: `accept` or `revise`
- summary: short string
- required_changes: array of strings
- files_to_revisit: array of file paths
- validations_to_rerun: array of commands or checks
- plan_adherence: short string
- risk_assessment: short string

The repository root is `{repo_root}`.
"""


def _is_validation_command(command: str) -> bool:
    lowered = command.lower()
    return any(
        marker in lowered
        for marker in (
            "pytest",
            "py.test",
            "python -m pytest",
            "python3 -m pytest",
            "python scripts/",
            "python3 scripts/",
            "python app/",
            "python3 app/",
            "demo",
            "preview",
            "render",
            "repr(",
        )
    )


def _summarize_validation_events(
    turns: list[dict[str, Any]],
    success_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    all_events: list[dict[str, Any]] = []
    relevant_events: list[dict[str, Any]] = []
    success_check_map: dict[str, dict[str, Any]] = {}
    for check in success_checks:
        command = str(check.get("command", "")).strip()
        if command:
            success_check_map[command] = check

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        turn_number = turn.get("turn")
        for result in turn.get("tool_results", []):
            if not isinstance(result, dict) or result.get("name") != "bash":
                continue
            arguments = result.get("arguments")
            command = ""
            if isinstance(arguments, dict):
                command = str(arguments.get("command", "")).strip()
            if not command:
                continue
            output = str(result.get("output", ""))
            output_short = output[:1200]
            exit_code = result.get("exit_code")
            matching_check = success_check_map.get(command)
            event = {
                "turn": turn_number,
                "command": command,
                "exit_code": exit_code,
                "is_error": bool(result.get("is_error")),
                "matched_success_check": str(matching_check.get("name", "")).strip() if matching_check else "",
                "check_passed": _command_output_satisfies_check(matching_check, exit_code=exit_code, output=output) if matching_check else None,
                "output": output_short,
            }
            all_events.append(event)
            if matching_check or _is_validation_command(command):
                relevant_events.append(event)

    latest_by_command: dict[str, dict[str, Any]] = {}
    for event in all_events:
        latest_by_command[event["command"]] = event

    passing_success_checks: list[dict[str, Any]] = []
    failing_success_checks: list[dict[str, Any]] = []
    missing_success_checks: list[dict[str, Any]] = []
    for check in success_checks:
        command = str(check.get("command", "")).strip()
        name = str(check.get("name", "")).strip() or command
        latest = latest_by_command.get(command)
        if latest is None:
            missing_success_checks.append({"name": name, "command": command})
            continue
        if bool(latest.get("check_passed")):
            passing_success_checks.append(
                {
                    "name": name,
                    "command": command,
                    "turn": latest.get("turn"),
                    "exit_code": latest.get("exit_code"),
                }
            )
        else:
            failing_success_checks.append(
                {
                    "name": name,
                    "command": command,
                    "turn": latest.get("turn"),
                    "exit_code": latest.get("exit_code"),
                    "output": str(latest.get("output", ""))[:800],
                }
            )

    relevant_tail = relevant_events[-12:]
    passing_validations = [
        {
            "turn": event.get("turn"),
            "command": event.get("command"),
            "matched_success_check": event.get("matched_success_check"),
            "exit_code": event.get("exit_code"),
        }
        for event in relevant_tail
        if not bool(event.get("is_error")) and (event.get("exit_code") in (0, None) or bool(event.get("check_passed")))
    ][-6:]
    failing_validations = [
        {
            "turn": event.get("turn"),
            "command": event.get("command"),
            "matched_success_check": event.get("matched_success_check"),
            "exit_code": event.get("exit_code"),
            "output": str(event.get("output", ""))[:800],
        }
        for event in relevant_tail
        if bool(event.get("is_error")) or (event.get("exit_code") not in (0, None)) or event.get("check_passed") is False
    ][-6:]

    return {
        "recent_commands": [
            {
                "turn": event.get("turn"),
                "command": event.get("command"),
                "matched_success_check": event.get("matched_success_check"),
                "exit_code": event.get("exit_code"),
                "output": str(event.get("output", ""))[:500],
            }
            for event in relevant_tail
        ],
        "passing_validations": passing_validations,
        "failing_validations": failing_validations,
        "passing_success_checks": passing_success_checks,
        "failing_success_checks": failing_success_checks,
        "missing_success_checks": missing_success_checks,
    }


def _build_reviewer_task_prompt(
    *,
    planner_handoff: dict[str, Any] | None,
    coder_result: dict[str, Any],
    patch_text: str,
    case_evaluation: dict[str, Any] | None,
    case_policy: dict[str, Any] | None,
) -> str:
    stats = coder_result.get("stats", {}) if isinstance(coder_result.get("stats"), dict) else {}
    success_checks = []
    if isinstance(case_evaluation, dict):
        success_checks = [dict(item) for item in case_evaluation.get("success_checks", []) if isinstance(item, dict)]
    validation_report = _summarize_validation_events(coder_result.get("turns", []), success_checks)
    loop_state = coder_result.get("loop_state", {}) if isinstance(coder_result.get("loop_state"), dict) else {}
    changed_files = [str(path) for path in loop_state.get("changed_files", []) if str(path).strip()]
    satisfied_checks = list(loop_state.get("satisfied_success_checks", []))
    condensed_turns = [
        {
            "turn": turn.get("turn"),
            "parse_error": turn.get("parse_error"),
            "tool_calls": [call.get("name") for call in turn.get("tool_calls", []) if isinstance(call, dict)],
        }
        for turn in coder_result.get("turns", [])[:20]
    ]
    return json.dumps(
        {
            "planner_handoff": planner_handoff or {},
            "case_evaluation": case_evaluation or {},
            "case_policy": case_policy or {},
            "review_focus": [
                "Reject solutions that modify tests to hide a behavior bug unless the case explicitly allows test edits.",
                "If baseline tests and required success checks disagree, trust the required success checks and the problem statement.",
                "Use the latest failing validation output and the actual patch as the primary evidence for rejection.",
                "Prefer one primary failure reason and at most two concrete next steps."
            ],
            "coder_summary": {
                "submitted": coder_result.get("submitted", False),
                "submission_summary": coder_result.get("submission_summary", ""),
                "stopped_reason": coder_result.get("stopped_reason", ""),
                "stats": stats,
                "changed_files": changed_files,
                "condensed_turns": condensed_turns,
                "satisfied_success_checks": satisfied_checks,
                "missing_success_check_names": [item.get("name", "") for item in validation_report.get("missing_success_checks", [])],
                "failed_success_check_names": [item.get("name", "") for item in validation_report.get("failing_success_checks", [])],
            },
            "validation_report": validation_report,
            "patch": patch_text,
        },
        indent=2,
    )


def _build_planner_handoff_prompt(planner_handoff: dict[str, Any]) -> str:
    return (
        "Planner handoff JSON:\n"
        + json.dumps(planner_handoff, indent=2)
        + "\nTreat this handoff as ranked guidance, not proof. "
        "Follow it unless repository evidence disproves it. "
        "Start with the listed first_actions or safe_reproduction_steps. "
        "Inspect discovery_priority and files_likely_affected before broad searching, but do not stay anchored to a wrong layer once repro or file evidence points elsewhere. "
        "If a planner repro command looks brittle or quote-heavy, use the safer equivalent implied by reproduction_notes or the runtime context instead of retrying broken quoting variants."
    )


def _build_reviewer_feedback_prompt(review_feedback: dict[str, Any]) -> str:
    return (
        "Reviewer feedback JSON:\n"
        + json.dumps(review_feedback, indent=2)
        + "\nAddress the required_changes first. "
        "Revisit only the listed files unless new evidence forces a broader search. "
        "Run the requested validations before submit. "
        "If the reviewer says `revise`, do not submit until the rejection is resolved. "
        "Do not change tests to satisfy the reviewer unless the case explicitly allows test edits."
    )


def _extract_first_json_dict(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        parts = text.split("```")
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped:
                    text = stripped
                    break
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : index + 1]
                parsed = json.loads(snippet)
                if not isinstance(parsed, dict):
                    raise ValueError("response JSON was not an object")
                return parsed
    raise ValueError("unterminated JSON object")


def _default_planner_handoff() -> dict[str, Any]:
    return {
        "problem_summary": "",
        "root_cause_hypothesis": "Planner output could not be parsed cleanly. Reproduce first and inspect the most likely runtime files from the repository context.",
        "files_likely_affected": [],
        "target_symbols": [],
        "discovery_priority": [],
        "first_actions": [
            "Run one high-signal reproduction or targeted validation command.",
            "Inspect the most likely runtime files from the startup context before editing.",
        ],
        "safe_reproduction_steps": [],
        "reproduction_notes": [],
        "required_validations": [],
        "allowed_change_types": ["targeted runtime code fixes"],
        "forbidden_edits": ["broad refactors without evidence"],
        "escalation_conditions": ["If the issue is still unclear after targeted inspection and reproduction."],
    }


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _is_brittle_command(text: str) -> bool:
    lowered = text.lower()
    if "python -c" in lowered or "python3 -c" in lowered:
        return True
    if text.count("'") >= 3 or text.count('"') >= 6:
        return True
    return False


def _normalize_planner_handoff(raw: dict[str, Any]) -> dict[str, Any]:
    handoff = dict(_default_planner_handoff())
    if not isinstance(raw, dict):
        return handoff

    for key in ("problem_summary", "root_cause_hypothesis"):
        value = str(raw.get(key, "")).strip()
        if value:
            handoff[key] = value

    files_likely = _string_list(raw.get("files_likely_affected"), limit=4)
    discovery_priority = _string_list(raw.get("discovery_priority"), limit=4)
    target_symbols = _string_list(raw.get("target_symbols"), limit=4)
    first_actions = _string_list(raw.get("first_actions"), limit=3)
    required_validations = _string_list(raw.get("required_validations"), limit=2)
    allowed_change_types = _string_list(raw.get("allowed_change_types"), limit=4)
    forbidden_edits = _string_list(raw.get("forbidden_edits"), limit=4)
    escalation_conditions = _string_list(raw.get("escalation_conditions"), limit=4)
    reproduction_notes = _string_list(raw.get("reproduction_notes"), limit=3)

    safe_repro = _string_list(raw.get("safe_reproduction_steps"), limit=3)
    if not safe_repro:
        safe_repro = _string_list(raw.get("reproduction_steps"), limit=3)
    safe_repro = [cmd for cmd in safe_repro if not _is_brittle_command(cmd)]
    if not safe_repro and _string_list(raw.get("safe_reproduction_steps"), limit=3):
        reproduction_notes.append("Planner suggested only brittle reproduction commands; prefer an equivalent script, pytest command, or direct file inspection first.")

    deduped_files: list[str] = []
    seen_files: set[str] = set()
    for item in files_likely:
        normalized = item.lstrip("./")
        if normalized in seen_files:
            continue
        seen_files.add(normalized)
        deduped_files.append(normalized)
    deduped_priority: list[str] = []
    seen_priority: set[str] = set()
    for item in discovery_priority or deduped_files[:3]:
        normalized = item.lstrip("./")
        if normalized in seen_priority:
            continue
        seen_priority.add(normalized)
        deduped_priority.append(normalized)

    handoff["files_likely_affected"] = deduped_files
    handoff["discovery_priority"] = deduped_priority
    handoff["target_symbols"] = target_symbols
    handoff["first_actions"] = first_actions or handoff["first_actions"]
    handoff["safe_reproduction_steps"] = safe_repro[:1]
    handoff["reproduction_notes"] = reproduction_notes
    handoff["required_validations"] = required_validations
    handoff["allowed_change_types"] = allowed_change_types or handoff["allowed_change_types"]
    handoff["forbidden_edits"] = forbidden_edits or handoff["forbidden_edits"]
    handoff["escalation_conditions"] = escalation_conditions or handoff["escalation_conditions"]

    if handoff["safe_reproduction_steps"]:
        first = handoff["safe_reproduction_steps"][0]
        if all(first not in action for action in handoff["first_actions"]):
            handoff["first_actions"] = [f"Run `{first}` first to confirm the behavior."] + handoff["first_actions"][:2]

    return handoff


def _default_reviewer_feedback() -> dict[str, Any]:
    return {
        "decision": "revise",
        "summary": "Reviewer output could not be parsed cleanly. Continue with focused reproduction, inspection, and validation.",
        "required_changes": [
            "Use one concrete next action tied to reproduction, inspection, editing, or validation.",
        ],
        "files_to_revisit": [],
        "validations_to_rerun": [],
        "plan_adherence": "Unable to assess due to malformed reviewer output.",
        "risk_assessment": "Proceed cautiously and validate before submit.",
    }


def _call_json_role(
    *,
    model: str,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    num_ctx: int | None,
    system_prompt: str,
    user_prompt: str,
    fallback_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int], str]:
    import litellm

    litellm.suppress_debug_info = True
    completion_kwargs: dict[str, Any] = {
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": min(max_tokens, 1400),
    }
    if model.startswith("ollama/"):
        completion_kwargs["reasoning_effort"] = "none"
        if num_ctx is not None:
            completion_kwargs["num_ctx"] = num_ctx
    messages = completion_kwargs["messages"]
    total_in = 0
    total_out = 0
    raw_content = ""

    for attempt in range(2):
        completion_kwargs["messages"] = messages
        response = litellm.completion(**completion_kwargs)
        usage = getattr(response, "usage", None)
        total_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        total_out += int(getattr(usage, "completion_tokens", 0) or 0)
        message = response.choices[0].message
        content = message.content or ""
        raw_content = str(content)
        try:
            parsed = _extract_first_json_dict(raw_content)
            return parsed, {"tokens_in": total_in, "tokens_out": total_out, "api_calls": attempt + 1}, raw_content
        except ValueError:
            if attempt == 0:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": raw_content},
                    {
                        "role": "user",
                        "content": "Return the same answer as exactly one valid JSON object and nothing else. Do not include explanations or markdown.",
                    },
                ]

    fallback = dict(fallback_payload)
    fallback["_parse_error"] = "role_json_parse_failed"
    fallback["_raw_response_preview"] = raw_content[:500]
    return fallback, {"tokens_in": total_in, "tokens_out": total_out, "api_calls": 2}, raw_content


@dataclass
class ToolExecutionRecord:
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool = False
    exit_code: int | None = None


@dataclass
class TurnRecord:
    turn: int
    assistant_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolExecutionRecord] = field(default_factory=list)
    finish_reason: str | None = None
    parse_error: str | None = None


@dataclass
class RunStats:
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    tool_calls: int = 0


@dataclass
class ParseOutcome:
    tool_calls: list[dict[str, Any]]
    error: str | None = None


@dataclass
class LoopState:
    executable_edit_made: bool = False
    validation_passed: bool = False
    validation_attempted_after_edit: bool = False
    diff_checked: bool = False
    successful_post_edit_commands: set[str] = field(default_factory=set)
    satisfied_success_checks: set[str] = field(default_factory=set)
    changed_files: set[str] = field(default_factory=set)


class ToolRuntime:
    def __init__(self, env: SWEEnv, repo_root: str):
        self.env = env
        self.repo_root = repo_root
        self.edit_history: dict[str, list[str]] = defaultdict(list)
        self.submitted = False
        self.submit_summary = ""

    def _validate_path(self, path: str) -> str:
        if not path.startswith("/"):
            path = str(Path(self.repo_root) / path)
        if not path.startswith(self.repo_root):
            raise ValueError(f"Path must stay under {self.repo_root}")
        return path

    def _read(self, path: str) -> str:
        return self.env.read_file(path)

    def _path_exists(self, path: str) -> bool:
        quoted = shlex.quote(path)
        output = self.env.communicate(
            input=f"if [ -e {quoted} ]; then printf 'exists'; else printf 'missing'; fi",
            timeout=10,
            check="ignore",
        )
        return output.strip() == "exists"

    def _path_is_file(self, path: str) -> bool:
        quoted = shlex.quote(path)
        output = self.env.communicate(
            input=f"if [ -f {quoted} ]; then printf 'file'; else printf 'not-file'; fi",
            timeout=10,
            check="ignore",
        )
        return output.strip() == "file"

    def _write(self, path: str, content: str) -> None:
        self.edit_history[path].append(self._read(path))
        self.env.write_file(path, content)

    def bash(self, command: str, timeout: int = 30) -> tuple[str, int]:
        sentinel = "__CODEX_EXIT_CODE__:"
        output = self.env.communicate(
            input=(
                f"cd {shlex.quote(self.repo_root)}\n"
                "{\n"
                f"{command}\n"
                "}\n"
                "status=$?\n"
                f"printf '\\n{sentinel}%s\\n' \"$status\"\n"
            ),
            timeout=timeout,
            check="ignore",
        )
        text = output or ""
        exit_code = 0
        marker_index = text.rfind(sentinel)
        if marker_index != -1:
            trailer = text[marker_index + len(sentinel):].strip().splitlines()
            if trailer:
                try:
                    exit_code = int(trailer[0].strip())
                except ValueError:
                    exit_code = 0
            text = text[:marker_index].rstrip()
        return (text or "(no output)", exit_code)

    def view(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        path = self._validate_path(path)
        if not self._path_exists(path):
            return (
                f"Path `{path}` does not exist. "
                "Use `bash` with `find`, `rg --files`, or `ls` to discover the correct path before trying `view` again."
            )
        if not self._path_is_file(path):
            return (
                f"Path `{path}` is not a regular file. "
                "Use `bash` to inspect the directory contents and then call `view` on a specific file."
            )
        text = self._read(path)
        lines = text.splitlines()
        start = 1 if start_line is None else start_line
        end = len(lines) if end_line is None else end_line
        start = max(1, start)
        end = max(start, min(end, len(lines)))
        body = "\n".join(f"{idx:6}\t{lines[idx - 1]}" for idx in range(start, end + 1))
        return f"Here's the result of running `cat -n` on {path}:\n{body}"

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        path = self._validate_path(path)
        text = self._read(path)
        count = text.count(old_str)
        if count == 0:
            return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
        if count > 1:
            return f"No replacement was performed. Multiple occurrences of old_str `{old_str}` were found. Please ensure it is unique."
        updated = text.replace(old_str, new_str, 1)
        self._write(path, updated)
        return self.view(path)

    def insert(self, path: str, line: int, new_str: str) -> str:
        path = self._validate_path(path)
        text = self._read(path)
        lines = text.splitlines(keepends=True)
        line = max(0, min(line, len(lines)))
        insert_text = new_str if new_str.endswith("\n") else f"{new_str}\n"
        lines.insert(line, insert_text)
        self._write(path, "".join(lines))
        start = max(1, line)
        return self.view(path, start_line=start, end_line=min(start + 8, len(lines) + 1))

    def undo_edit(self, path: str) -> str:
        path = self._validate_path(path)
        if not self.edit_history[path]:
            return f"No edit history available for {path}."
        previous = self.edit_history[path].pop()
        self.env.write_file(path, previous)
        return self.view(path)

    def submit(self, summary: str = "") -> str:
        self.submitted = True
        self.submit_summary = summary
        return f"Submission recorded. Summary: {summary or '(none)'}"

    @staticmethod
    def _output_indicates_tool_failure(name: str, output: str) -> bool:
        text = (output or "").strip().lower()
        if not text:
            return False
        if name == "view":
            return (
                "does not exist" in text
                or "is not a regular file" in text
            )
        if name == "str_replace":
            return (
                text.startswith("no replacement was performed")
                or "multiple occurrences of old_str" in text
            )
        if name == "undo_edit":
            return text.startswith("no edit history available")
        return False

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecutionRecord:
        try:
            if name == "bash":
                output, exit_code = self.bash(arguments["command"], int(arguments.get("timeout", 30)))
                return ToolExecutionRecord(name=name, arguments=arguments, output=output, is_error=exit_code != 0, exit_code=exit_code)
            elif name == "view":
                output = self.view(
                    arguments["path"],
                    arguments.get("start_line"),
                    arguments.get("end_line"),
                )
            elif name == "str_replace":
                output = self.str_replace(arguments["path"], arguments["old_str"], arguments["new_str"])
            elif name == "insert":
                output = self.insert(arguments["path"], int(arguments["line"]), arguments["new_str"])
            elif name == "undo_edit":
                output = self.undo_edit(arguments["path"])
            elif name == "submit":
                output = self.submit(arguments.get("summary", ""))
            else:
                raise ValueError(f"Unknown tool {name}")
            return ToolExecutionRecord(
                name=name,
                arguments=arguments,
                output=output,
                is_error=self._output_indicates_tool_failure(name, output),
                exit_code=0,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolExecutionRecord(name=name, arguments=arguments, output=f"{type(exc).__name__}: {exc}", is_error=True, exit_code=None)


class CustomAgentLoop:
    def __init__(
        self,
        *,
        model: str,
        api_base: str | None,
        api_key: str | None,
        temperature: float,
        max_turns: int,
        max_tokens: int,
        num_ctx: int | None,
        env: SWEEnv,
        repo_root: str,
        problem_statement: str,
        runtime_context: str,
        max_identical_tool_failures: int,
        tool_call_mode: str,
        success_validation_commands: list[str] | None = None,
        success_validation_checks: list[dict[str, Any]] | None = None,
        case_policy: dict[str, Any] | None = None,
        role_name: str = "coder",
        extra_user_prompts: list[str] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.env = env
        self.repo_root = repo_root
        self.problem_statement = problem_statement
        self.runtime_context = runtime_context
        self.max_identical_tool_failures = max_identical_tool_failures
        self.tool_call_mode = tool_call_mode
        self.success_validation_commands = [cmd.strip() for cmd in (success_validation_commands or []) if str(cmd).strip()]
        self.success_validation_checks = [dict(item) for item in (success_validation_checks or []) if isinstance(item, dict)]
        self.case_policy = dict(case_policy or {})
        self.role_name = role_name
        self.extra_user_prompts = extra_user_prompts or []
        self.tools = ToolRuntime(env, repo_root)
        self.turns: list[TurnRecord] = []
        self.stats = RunStats()
        self.repeat_hashes: list[str] = []
        self.log_fn = log_fn or (lambda _line: None)
        self.consecutive_failure_counts: dict[str, int] = defaultdict(int)
        self.consecutive_success_counts: dict[str, int] = defaultdict(int)
        self.state = LoopState()

    def _log(self, message: str) -> None:
        self.log_fn(message)

    @staticmethod
    def _clip(text: str, limit: int = 400) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "...<truncated>"

    def _messages(self) -> list[dict[str, Any]]:
        messages = [
            {"role": "system", "content": _build_system_prompt(self.repo_root)},
            {"role": "user", "content": _build_task_prompt(self.problem_statement, self.repo_root)},
            {"role": "user", "content": _build_runtime_context_prompt(self.repo_root, self.runtime_context)},
        ]
        for prompt in self.extra_user_prompts:
            messages.append({"role": "user", "content": prompt})
        if self.tool_call_mode == "react_json":
            messages.append({"role": "user", "content": _build_react_json_prompt()})
        elif self.tool_call_mode == "openai_tools":
            messages.append({"role": "user", "content": _build_openai_tools_prompt()})
        return messages

    @staticmethod
    def _normalize_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        if name == "str_replace":
            if "old_text" in normalized and "old_str" not in normalized:
                normalized["old_str"] = normalized.pop("old_text")
            if "new_text" in normalized and "new_str" not in normalized:
                normalized["new_str"] = normalized.pop("new_text")
            if "replace" in normalized and "old_str" not in normalized:
                normalized["old_str"] = normalized.pop("replace")
            if "with" in normalized and "new_str" not in normalized:
                normalized["new_str"] = normalized.pop("with")
        if name == "view":
            if "start" in normalized and "start_line" not in normalized:
                normalized["start_line"] = normalized.pop("start")
            if "end" in normalized and "end_line" not in normalized:
                normalized["end_line"] = normalized.pop("end")
            if "line_start" in normalized and "start_line" not in normalized:
                normalized["start_line"] = normalized.pop("line_start")
            if "line_end" in normalized and "end_line" not in normalized:
                normalized["end_line"] = normalized.pop("line_end")
        if name == "bash":
            if "cmd" in normalized and "command" not in normalized:
                normalized["command"] = normalized.pop("cmd")
        if name == "insert":
            if "insert_line" in normalized and "line" not in normalized:
                normalized["line"] = normalized.pop("insert_line")
            if "text" in normalized and "new_str" not in normalized:
                normalized["new_str"] = normalized.pop("text")
        return normalized

    @classmethod
    def _infer_tool_name(cls, payload: dict[str, Any], arguments: dict[str, Any]) -> str | None:
        explicit = payload.get("tool") or payload.get("action") or payload.get("name") or payload.get("tool_name")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        if "command" in arguments or "cmd" in arguments:
            return "bash"
        if "summary" in arguments:
            return "submit"
        if "path" in arguments:
            if "old_str" in arguments or "old_text" in arguments or "replace" in arguments:
                return "str_replace"
            if "line" in arguments or "insert_line" in arguments:
                return "insert"
            return "view"
        return None

    @classmethod
    def _sanitize_react_json_text(cls, content: str) -> str:
        text = content.strip()
        if not text:
            return text
        text = re.sub(r"<think\b[^>]*>.*?</think>", "\n", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</?think\b[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<tool_call\b[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</tool_call>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<analysis\b[^>]*>.*?</analysis>", "\n", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</?analysis\b[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _payload_to_tool_call(cls, payload: dict[str, Any]) -> dict[str, Any] | None:
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = {
                k: v
                for k, v in payload.items()
                if k not in {"tool", "action", "name", "tool_name", "type"}
            }
        tool = cls._infer_tool_name(payload, arguments if isinstance(arguments, dict) else {})
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            return None
        tool = tool.strip()
        return {
            "id": f"react-1-{tool}",
            "name": tool,
            "arguments": cls._normalize_tool_arguments(tool, arguments),
        }

    @classmethod
    def _parse_react_json(cls, content: str) -> ParseOutcome:
        text = cls._sanitize_react_json_text(content)
        if not text:
            return ParseOutcome(tool_calls=[], error="Empty response; expected JSON action object or array.")
        candidate_payloads: list[Any] = []
        parse_errors: list[str] = []

        def add_payload_from_snippet(snippet: str) -> None:
            snippet = snippet.strip()
            if not snippet:
                return
            try:
                candidate_payloads.append(json.loads(snippet))
                return
            except json.JSONDecodeError as exc:
                parse_errors.append(f"json.loads failed: {exc.msg} at line {exc.lineno} column {exc.colno}")
            try:
                candidate_payloads.append(ast.literal_eval(snippet))
                return
            except (SyntaxError, ValueError) as exc:
                parse_errors.append(f"ast.literal_eval failed: {exc}")
            decoder = json.JSONDecoder()
            idx = 0
            multi: list[Any] = []
            while idx < len(snippet):
                while idx < len(snippet) and snippet[idx].isspace():
                    idx += 1
                if idx >= len(snippet):
                    break
                try:
                    obj, next_idx = decoder.raw_decode(snippet, idx)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"multi-object decode failed: {exc.msg} at column {exc.colno}")
                    multi = []
                    break
                multi.append(obj)
                idx = next_idx
            if multi:
                candidate_payloads.append(multi)

            first_obj_start = snippet.find("{")
            if first_obj_start != -1:
                depth = 0
                in_string = False
                escape = False
                for idx in range(first_obj_start, len(snippet)):
                    ch = snippet[idx]
                    if in_string:
                        if escape:
                            escape = False
                        elif ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                        continue
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = snippet[first_obj_start : idx + 1]
                            try:
                                candidate_payloads.append(json.loads(candidate))
                            except json.JSONDecodeError:
                                pass
                            break

        if "```" in text:
            parts = text.split("```")
            for idx, part in enumerate(parts):
                if idx % 2 == 1:
                    stripped = part.strip()
                    if stripped.startswith("json"):
                        stripped = stripped[4:].strip()
                    add_payload_from_snippet(stripped)

        add_payload_from_snippet(text)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            add_payload_from_snippet(text[start : end + 1])

        tool_calls: list[dict[str, Any]] = []
        for payload in candidate_payloads:
            if isinstance(payload, list):
                parsed_calls = [call for item in payload if isinstance(item, dict) for call in [cls._payload_to_tool_call(item)] if call]
                if parsed_calls:
                    first_call = parsed_calls[0]
                    if all(
                        cls._hash_tool_call(call["name"], call["arguments"])
                        == cls._hash_tool_call(first_call["name"], first_call["arguments"])
                        for call in parsed_calls
                    ):
                        return ParseOutcome(
                            tool_calls=[first_call],
                            error="Returned the same action multiple times in one response; recovered the first action only.",
                        )
                    return ParseOutcome(
                        tool_calls=[first_call],
                        error="Returned multiple actions in one response; executed only the first action.",
                    )
                return ParseOutcome(tool_calls=[], error="Returned multiple actions, but react_json mode requires exactly one JSON object per response.")
            if not isinstance(payload, dict):
                continue
            parsed_call = cls._payload_to_tool_call(payload)
            if parsed_call is None:
                continue
            tool_calls.append(parsed_call)
            return ParseOutcome(tool_calls=tool_calls, error=None)

        error = "Could not parse model action JSON."
        if parse_errors:
            error += " " + " | ".join(parse_errors[:3])
        if "{" in text or "[" in text:
            open_braces = text.count("{")
            close_braces = text.count("}")
            open_brackets = text.count("[")
            close_brackets = text.count("]")
            if open_braces != close_braces or open_brackets != close_brackets:
                error += (
                    f" Unbalanced delimiters detected: braces {open_braces}/{close_braces}, "
                    f"brackets {open_brackets}/{close_brackets}."
                )
        return ParseOutcome(tool_calls=[], error=error)

    @staticmethod
    def _hash_tool_call(name: str, arguments: dict[str, Any]) -> str:
        return json.dumps({"name": name, "arguments": arguments}, sort_keys=True)

    def _append_loop_warning_if_needed(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        digest = self._hash_tool_call(tool_name, arguments)
        self.repeat_hashes.append(digest)
        recent = self.repeat_hashes[-4:]
        if len(recent) == 4 and len(set(recent)) == 1:
            self._log(f"[loop-warning] repeated tool call detected: {tool_name} {json.dumps(arguments, sort_keys=True)}")
            return "You are repeating the same tool call. Stop looping, inspect a larger unique block, validate, or undo the bad edit."
        return None

    def _update_state_from_tool_result(self, result: ToolExecutionRecord) -> None:
        if result.is_error:
            return
        if result.name in {"str_replace", "insert", "undo_edit"}:
            self.state.executable_edit_made = True
            path = str(result.arguments.get("path", "")).strip()
            if path:
                self.state.changed_files.add(path)
        if result.name == "bash":
            if self.state.executable_edit_made:
                self.state.validation_attempted_after_edit = True
                command = str(result.arguments.get("command", "")).strip()
                if command and (result.exit_code == 0):
                    self.state.successful_post_edit_commands.add(command)
                if command:
                    for check in self.success_validation_checks:
                        if str(check.get("command", "")).strip() != command:
                            continue
                        check_name = str(check.get("name", command))
                        if _command_output_satisfies_check(check, exit_code=result.exit_code, output=result.output):
                            self.state.satisfied_success_checks.add(check_name)
            lowered = result.output.lower()
            if "passed" in lowered and "failed" not in lowered:
                self.state.validation_passed = True
            if "diff --git" in result.output:
                self.state.diff_checked = True

    def _current_changed_files(self) -> list[str]:
        output = self.env.communicate(
            input=f"cd {shlex.quote(self.repo_root)}\ngit diff --name-only",
            timeout=20,
            check="ignore",
        )
        return [line.strip() for line in (output or "").splitlines() if line.strip()]

    def _submit_precheck(self) -> str | None:
        if not self.state.executable_edit_made:
            return "Do not submit yet. No code edit has been made."
        patch = self.env.communicate(
            input=f"cd {shlex.quote(self.repo_root)}\ngit diff --no-color",
            timeout=20,
            check="ignore",
        )
        if not patch.strip():
            return "Do not submit yet. There is no patch in git diff."
        changed_files = self._current_changed_files()
        if changed_files and not bool(self.case_policy.get("allow_test_edits", False)):
            test_like_paths = [path for path in changed_files if path.startswith("tests/") or "/tests/" in path]
            if test_like_paths:
                preview = ", ".join(test_like_paths[:3])
                return (
                    "Do not submit yet. This case does not allow changing tests to make the behavior pass. "
                    f"Revert the test edits and fix runtime code instead. Changed test files: {preview}"
                )
        if not self.state.validation_attempted_after_edit:
            return "Do not submit yet. Run at least one reproduction or validation command after your edit."
        if self.success_validation_checks:
            missing_names = []
            for check in self.success_validation_checks:
                check_name = str(check.get("name", check.get("command", "")))
                if check_name not in self.state.satisfied_success_checks:
                    missing_names.append(check_name)
            if missing_names:
                preview = "; ".join(missing_names[:2])
                return f"Do not submit yet. Required case success checks have not passed yet: {preview}"
        elif self.success_validation_commands:
            missing = [cmd for cmd in self.success_validation_commands if cmd not in self.state.successful_post_edit_commands]
            if missing:
                preview = "; ".join(missing[:2])
                return f"Do not submit yet. Run the required case success checks after your edit: {preview}"
        if not self.state.diff_checked:
            return "Do not submit yet. Inspect `git diff` first."
        return None

    def _case_check_feedback(self, result: ToolExecutionRecord) -> str | None:
        if result.name != "bash":
            return None
        command = str(result.arguments.get("command", "")).strip()
        if not command:
            return None
        matching = [check for check in self.success_validation_checks if str(check.get("command", "")).strip() == command]
        if not matching:
            return None
        check = matching[0]
        check_name = str(check.get("name", command))
        if _command_output_satisfies_check(check, exit_code=result.exit_code, output=result.output):
            return f"Required success check `{check_name}` now passes."
        reasons = _command_output_failure_reasons(check, exit_code=result.exit_code, output=result.output)
        preview = "; ".join(reasons[:3])
        return (
            f"Required success check `{check_name}` still fails: {preview}. "
            "Do not optimize for generic tests alone; fix the behavior that this check is measuring."
        )

    def run(self) -> dict[str, Any]:
        import litellm

        litellm.suppress_debug_info = True
        messages = self._messages()
        stopped_reason = "max_turns"
        self._log(f"[run] starting role={self.role_name} model={self.model} repo_root={self.repo_root}")

        for turn in range(1, self.max_turns + 1):
            self.stats.turns = turn
            self._log(f"[turn {turn}] calling model")
            completion_kwargs = {
                "model": self.model,
                "api_base": self.api_base,
                "api_key": self.api_key,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.tool_call_mode == "openai_tools":
                completion_kwargs["tools"] = TOOL_SCHEMAS
                completion_kwargs["tool_choice"] = "auto"
            elif self.tool_call_mode == "react_json":
                completion_kwargs["max_tokens"] = min(self.max_tokens, 512)
                if self.model.startswith("ollama/"):
                    completion_kwargs["reasoning_effort"] = "none"
                    if self.num_ctx is not None:
                        completion_kwargs["num_ctx"] = self.num_ctx
                else:
                    completion_kwargs["stop"] = ["</think>", "<tool_call>", "```"]
            response = litellm.completion(**completion_kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.stats.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.stats.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            tool_calls = []
            parse_error = None
            if getattr(message, "tool_calls", None):
                for call in message.tool_calls:
                    arguments = call.function.arguments
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    tool_calls.append(
                        {
                            "id": call.id,
                            "name": call.function.name,
                            "arguments": arguments,
                        }
                    )
            else:
                parse_outcome = self._parse_react_json(content if isinstance(content, str) else json.dumps(content))
                tool_calls = parse_outcome.tool_calls
                parse_error = parse_outcome.error

            if len(tool_calls) > 1:
                tool_calls = tool_calls[:1]
                extra = "Returned multiple tool calls in one response; executed only the first action."
                parse_error = f"{parse_error} {extra}".strip() if parse_error else extra

            turn_record = TurnRecord(
                turn=turn,
                assistant_text=content if isinstance(content, str) else json.dumps(content),
                tool_calls=tool_calls,
                finish_reason=getattr(choice, "finish_reason", None),
                parse_error=parse_error,
            )
            self.turns.append(turn_record)
            if content:
                self._log(f"[turn {turn}] assistant {self._clip(content)}")
            if parse_error:
                self._log(f"[turn {turn}] parse-error {parse_error}")

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content or "",
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_message)

            if not tool_calls:
                self._log(f"[turn {turn}] no tool call returned")
                if self.tool_call_mode == "react_json":
                    corrective_hint = (
                        "Your previous response was unusable. "
                        "Respond with exactly one JSON action object and nothing else. "
                        "No `</think>`, no extra JSON objects, no tool-call tags, no explanations, no questions, and no conversational text. "
                        + (f"Your last action payload was invalid: {parse_error}" if parse_error else "")
                    ).strip()
                else:
                    corrective_hint = (
                        "Your previous response did not contain a native tool call. "
                        "Use native tool calling only. "
                        "Do not return raw JSON in assistant text, and do not describe a plan in plain English. "
                        + (f"Your last response was: {parse_error}" if parse_error else "")
                    ).strip()
                if turn <= 3:
                    corrective_hint += (
                        " On early turns, choose exactly one discovery action such as "
                        "`bash` with `find /repo -type f -name '*.py' | head -30`, "
                        "`bash` with `ls -la /repo`, or one targeted `view`."
                    )
                if self.state.validation_passed and not self.state.diff_checked:
                    corrective_hint += " Tests already passed. Your next best step is usually `bash` with `git diff`."
                elif self.state.validation_passed and self.state.diff_checked:
                    corrective_hint += " Tests already passed and diff was inspected. If the patch is correct, the next best step is `submit`."
                messages.append(
                    {
                        "role": "user",
                        "content": corrective_hint,
                    }
                )
                continue

            pending_user_messages: list[str] = []
            if parse_error:
                pending_user_messages.append(
                    "You returned an invalid action payload. Only the first valid action was executed. Next response must contain exactly one action and no extra plan text."
                )
            for call in tool_calls:
                self.stats.tool_calls += 1
                self._log(f"[turn {turn}] tool {call['name']} {json.dumps(call['arguments'], sort_keys=True)}")
                loop_warning = self._append_loop_warning_if_needed(call["name"], call["arguments"])
                if loop_warning:
                    pending_user_messages.append(loop_warning)
                if call["name"] == "submit":
                    submit_block = self._submit_precheck()
                    if submit_block:
                        result = ToolExecutionRecord(
                            name="submit",
                            arguments=call["arguments"],
                            output=submit_block,
                            is_error=True,
                        )
                    else:
                        result = self.tools.execute(call["name"], call["arguments"])
                else:
                    result = self.tools.execute(call["name"], call["arguments"])
                turn_record.tool_results.append(result)
                self._update_state_from_tool_result(result)
                case_feedback = self._case_check_feedback(result)
                if case_feedback:
                    pending_user_messages.append(case_feedback)
                level = "tool-error" if result.is_error else "tool-output"
                self._log(f"[turn {turn}] {level} {result.name}: {self._clip(result.output)}")
                repeated_success_warning = False
                failure_key = json.dumps(
                    {
                        "tool": result.name,
                        "arguments": call["arguments"],
                        "output": result.output,
                        "is_error": result.is_error,
                    },
                    sort_keys=True,
                )
                if result.is_error:
                    self.consecutive_failure_counts[failure_key] += 1
                    self.consecutive_success_counts.clear()
                    pending_user_messages.append(
                        "That tool call failed. Use the runtime context, repo root, pwd/ls output, and recent tool output to correct the next step. Do not retry the same failing call unchanged."
                    )
                    if self.consecutive_failure_counts[failure_key] >= self.max_identical_tool_failures:
                        stopped_reason = "repeated_tool_failure"
                        self._log(
                            f"[run] stopping after repeated tool failure threshold={self.max_identical_tool_failures} "
                            f"tool={result.name}"
                        )
                        break
                else:
                    self.consecutive_failure_counts.clear()
                    self.consecutive_success_counts[failure_key] += 1
                    repeated_success_warning = False
                    if self.consecutive_success_counts[failure_key] >= self.max_identical_tool_failures:
                        self._log(
                            f"[run] repeated successful tool call threshold={self.max_identical_tool_failures} "
                            f"tool={result.name}"
                        )
                        pending_user_messages.append(
                            "You already ran that exact tool call and got the same result multiple times. Do not run it again. Use the observed output to take a different next step: inspect a different file, edit code, run a broader validation, inspect `git diff`, or `submit`."
                        )
                        repeated_success_warning = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": result.output,
                    }
                )
            for pending in pending_user_messages:
                messages.append({"role": "user", "content": pending})
            if stopped_reason == "repeated_tool_failure":
                break

            if self.tools.submitted:
                stopped_reason = "submitted"
                self._log(f"[run] submit called after turn {turn}")
                break

        patch = self.env.communicate(
            input=f"git -C {shlex.quote(self.repo_root)} diff --no-color",
            timeout=30,
            check="ignore",
        )
        self._log(f"[run] finished stopped_reason={stopped_reason} submitted={self.tools.submitted} patch_chars={len(patch or '')}")
        return {
            "turns": [asdict(turn) for turn in self.turns],
            "stats": asdict(self.stats),
            "loop_state": {
                **asdict(self.state),
                "successful_post_edit_commands": sorted(self.state.successful_post_edit_commands),
                "satisfied_success_checks": sorted(self.state.satisfied_success_checks),
                "changed_files": sorted(self.state.changed_files),
            },
            "stopped_reason": stopped_reason,
            "submitted": self.tools.submitted,
            "submission_summary": self.tools.submit_summary,
            "patch": patch or "",
            "info": {
                "submission": patch if self.tools.submitted else "",
                "submitted": self.tools.submitted,
                "submission_summary": self.tools.submit_summary,
                "stopped_reason": stopped_reason,
            },
        }


def _instance_repo_root(instance) -> str:
    repo = instance.env.repo
    if repo is None:
        return "/"
    repo_name = getattr(repo, "repo_name", None)
    if repo_name:
        return f"/{repo_name}"
    return "/testbed"


@dataclass
class SafeLocalRepo:
    path: Path
    base_commit: str = "HEAD"

    @property
    def repo_name(self) -> str:
        return self.path.resolve().name.replace(" ", "-").replace("'", "")

    def copy(self, deployment) -> None:
        LocalRepoConfig(path=self.path, base_commit=self.base_commit).copy(deployment)

    def get_reset_commands(self) -> list[str]:
        return [
            "git status",
            "git restore .",
            "git reset --hard",
            f"git checkout {shlex.quote(self.base_commit)}",
            "git clean -fdq",
        ]


@dataclass
class PlainLocalDirectoryRepo:
    path: Path
    base_commit: str = "HEAD"

    @property
    def repo_name(self) -> str:
        return self.path.resolve().name.replace(" ", "-").replace("'", "")

    def copy(self, deployment) -> None:
        asyncio.run(
            deployment.runtime.upload(
                UploadRequest(source_path=str(self.path), target_path=f"/{self.repo_name}")
            )
        )
        result = asyncio.run(
            deployment.runtime.execute(Command(command=f"chown -R root:root /{self.repo_name}", shell=True))
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to chown copied directory /{self.repo_name}: "
                f"{result.stdout} {result.stderr}".strip()
            )

    def get_reset_commands(self) -> list[str]:
        return []


def _resolve_instances_path(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_dir():
        for name in ("case.json", "case.yaml", "case.yml"):
            nested = candidate / name
            if nested.exists():
                return nested
        raise ValueError(f"No case metadata file found in {candidate}. Expected case.json, case.yaml, or case.yml")
    return candidate


def _resolve_case_repo_path(case_file: Path, raw_repo_path: str | None) -> Path:
    if raw_repo_path:
        candidate = Path(str(raw_repo_path))
        if not candidate.is_absolute():
            candidate = (case_file.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    default_repo = case_file.parent / "repo"
    if default_repo.exists():
        return default_repo.resolve()
    raise ValueError(
        f"Case {case_file} must define repo_path or include a sibling repo/ directory"
    )


def _load_custom_file_instances(path: Path, filter_regex: str, slice_spec: str, shuffle: bool) -> list[BatchInstance]:
    case_file = _resolve_instances_path(path)
    raw_items = yaml.safe_load(case_file.read_text())
    if not isinstance(raw_items, list):
        raise ValueError("Custom instance file must contain a list of instances")

    instances: list[BatchInstance] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Each custom instance entry must be an object")
        repo_path = _resolve_case_repo_path(case_file, item.get("repo_path") or item.get("repo_name"))
        problem_text = str(item["problem_statement"])
        instance_id = str(item["instance_id"])
        image_name = str(item.get("image_name", "python:3.11"))
        base_commit = str(item.get("base_commit", "HEAD"))
        setup_commands = list(item.get("setup_commands", []))
        install_commands = list(item.get("install_commands", []))
        extra_fields = dict(item.get("extra_fields", {}))
        extra_fields["setup_commands"] = setup_commands
        extra_fields["install_commands"] = install_commands
        extra_fields["case_file"] = str(case_file)
        extra_fields["repo_path"] = str(repo_path)

        instances.append(
            BatchInstance(
                env=EnvironmentConfig.model_construct(
                    deployment=DockerDeploymentConfig(image=image_name, pull="missing"),
                    repo=PlainLocalDirectoryRepo(path=repo_path, base_commit=base_commit),
                    post_startup_commands=[],
                    post_startup_command_timeout=500,
                    name="main",
                ),
                problem_statement=TextProblemStatement(
                    text=problem_text,
                    id=instance_id,
                    extra_fields=extra_fields,
                ),
            )
        )

    def _slice_spec_to_slice(local_slice_spec: str) -> slice:
        if local_slice_spec == "":
            return slice(None)
        parts = local_slice_spec.split(":")
        values = [None if p == "" else int(p) for p in parts]
        if len(parts) == 1:
            return slice(values[0])
        if len(parts) == 2:
            return slice(values[0], values[1])
        if len(parts) == 3:
            return slice(values[0], values[1], values[2])
        raise ValueError(f"Invalid slice specification: {local_slice_spec!r}")

    if shuffle:
        import random
        instances = sorted(instances, key=lambda x: x.problem_statement.id)
        random.seed(42)
        random.shuffle(instances)

    import re

    filtered = [instance for instance in instances if re.match(filter_regex, instance.problem_statement.id)]
    return filtered[_slice_spec_to_slice(slice_spec)]


def _build_env(instance) -> SWEEnv:
    repo = instance.env.repo
    if isinstance(repo, LocalRepoConfig):
        repo = SafeLocalRepo(path=repo.path, base_commit=repo.base_commit)
    deployment = get_deployment(instance.env.deployment.model_copy(deep=True))
    return SWEEnv(
        deployment=deployment,
        repo=repo,
        post_startup_commands=list(instance.env.post_startup_commands),
        post_startup_command_timeout=instance.env.post_startup_command_timeout,
        name=instance.env.name,
    )


def _build_instances(args: argparse.Namespace):
    if args.instances_type == "file":
        if args.instances_path is None:
            raise ValueError("--instances-path is required when --instances-type=file")
        return _load_custom_file_instances(args.instances_path, args.filter, args.slice, args.shuffle)
    else:
        source = SWEBenchInstances(
            subset=args.subset,
            split=args.split,
            filter=args.filter,
            slice=args.slice,
            shuffle=args.shuffle,
            evaluate=False,
        )
        return source.get_instance_configs()


def _save_pred(instance_dir: Path, instance_id: str, variant_name: str, model_patch: str) -> dict[str, Any]:
    pred = {
        "model_name_or_path": variant_name,
        "instance_id": instance_id,
        "model_patch": model_patch,
    }
    _dump_json(instance_dir / f"{instance_id}.pred", pred)
    return pred


def parse_args() -> argparse.Namespace:
    preset_names = _preset_names()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  OpenAI preset:
    ./env/bin/python SWE-agent/scripts/custom/run_custom_swebench.py --preset openai_gpt4o_mini --filter pydicom__pydicom-1458 --output-dir SWE-agent/custom_runs/openai_gpt4omini

  OpenAI:
    ./env/bin/python SWE-agent/scripts/custom/run_custom_swebench.py --backend openai --model gpt-4o-mini --filter pydicom__pydicom-1458

  Ollama:
    ./env/bin/python SWE-agent/scripts/custom/run_custom_swebench.py --backend ollama --model qwen2.5-coder:7b-instruct --filter pydicom__pydicom-1458

  LM Studio:
    ./env/bin/python SWE-agent/scripts/custom/run_custom_swebench.py --backend lmstudio --model openai/local-model --filter pydicom__pydicom-1458

  UMich:
    ./env/bin/python SWE-agent/scripts/custom/run_custom_swebench.py --backend umich --model openai/openai/gpt-oss-120b --filter pydicom__pydicom-1458
""",
    )
    parser.add_argument("--preset", choices=preset_names)
    parser.add_argument("--backend", choices=sorted(BACKEND_DEFAULTS.keys()))
    parser.add_argument("--model")
    parser.add_argument("--agent-architecture", choices=["single", "planner_coder", "planner_coder_reviewer"], default="single")
    parser.add_argument("--planner-model")
    parser.add_argument("--reviewer-model")
    parser.add_argument("--api-base", dest="api_base")
    parser.add_argument("--api-key")
    parser.add_argument("--planner-api-base")
    parser.add_argument("--planner-api-key")
    parser.add_argument("--reviewer-api-base")
    parser.add_argument("--reviewer-api-key")
    parser.add_argument("--tool-call-mode", choices=["openai_tools", "react_json"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--planner-temperature", type=float)
    parser.add_argument("--reviewer-temperature", type=float)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-ctx", type=int)
    parser.add_argument("--reviewer-rounds", type=int)
    parser.add_argument("--max-identical-tool-failures", type=int, default=3)
    parser.add_argument("--post-startup-command", action="append", default=[])
    parser.add_argument("--instances-type", choices=["swe_bench", "file"], default="swe_bench")
    parser.add_argument("--instances-path", type=Path)
    parser.add_argument("--subset", choices=["lite", "verified", "full", "multimodal", "multilingual"], default="full")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--filter", default=".*")
    parser.add_argument("--slice", default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="custom_runner")
    args = parser.parse_args()

    if args.preset:
        preset = _load_presets()[args.preset]
        if args.backend is None:
            args.backend = preset.get("backend")
        if args.model is None:
            args.model = preset.get("model")
        if args.planner_model is None and preset.get("planner_model") is not None:
            args.planner_model = preset.get("planner_model")
        if args.reviewer_model is None and preset.get("reviewer_model") is not None:
            args.reviewer_model = preset.get("reviewer_model")
        if args.api_base is None:
            args.api_base = preset.get("api_base")
        if args.api_key is None and preset.get("api_key") is not None:
            args.api_key = preset.get("api_key")
        if args.planner_api_base is None and preset.get("planner_api_base") is not None:
            args.planner_api_base = preset.get("planner_api_base")
        if args.planner_api_key is None and preset.get("planner_api_key") is not None:
            args.planner_api_key = preset.get("planner_api_key")
        if args.reviewer_api_base is None and preset.get("reviewer_api_base") is not None:
            args.reviewer_api_base = preset.get("reviewer_api_base")
        if args.reviewer_api_key is None and preset.get("reviewer_api_key") is not None:
            args.reviewer_api_key = preset.get("reviewer_api_key")
        if args.tool_call_mode is None and preset.get("tool_call_mode") is not None:
            args.tool_call_mode = preset.get("tool_call_mode")
        if args.max_tokens == 4096 and preset.get("max_tokens") is not None:
            args.max_tokens = int(preset.get("max_tokens"))
        if args.num_ctx is None and preset.get("num_ctx") is not None:
            args.num_ctx = int(preset.get("num_ctx"))
        if args.temperature == 0.0 and preset.get("temperature") is not None:
            args.temperature = float(preset.get("temperature"))
        if args.planner_temperature is None and preset.get("planner_temperature") is not None:
            args.planner_temperature = float(preset.get("planner_temperature"))
        if args.reviewer_temperature is None and preset.get("reviewer_temperature") is not None:
            args.reviewer_temperature = float(preset.get("reviewer_temperature"))
        if args.run_name == "custom_runner":
            args.run_name = args.preset

    if args.backend is None:
        args.backend = "openai"
    if args.model is None:
        parser.error("--model is required unless provided by --preset")
    if args.tool_call_mode is None:
        args.tool_call_mode = "openai_tools"
    if args.planner_temperature is None:
        args.planner_temperature = 0.0 if args.agent_architecture != "single" else args.temperature
    if args.reviewer_temperature is None:
        args.reviewer_temperature = 0.0
    if args.reviewer_rounds is None:
        args.reviewer_rounds = 2 if args.agent_architecture == "planner_coder_reviewer" else 1
    if args.agent_architecture == "planner_coder_reviewer" and args.reviewer_rounds < 1:
        parser.error("--reviewer-rounds must be at least 1")

    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    api_base = args.api_base or BACKEND_DEFAULTS[args.backend].get("api_base")
    api_key = _resolve_api_key(args.backend, args.api_key)
    model = _normalize_model_name(args.backend, args.model)
    planner_api_base = args.planner_api_base or api_base
    planner_api_key = _resolve_api_key(args.backend, args.planner_api_key) if args.planner_api_key else api_key
    reviewer_api_base = args.reviewer_api_base or api_base
    reviewer_api_key = _resolve_api_key(args.backend, args.reviewer_api_key) if args.reviewer_api_key else api_key
    planner_model = _normalize_model_name(args.backend, args.planner_model) if args.planner_model else model
    reviewer_model = _normalize_model_name(args.backend, args.reviewer_model) if args.reviewer_model else model

    run_config = {
        "backend": args.backend,
        "model": model,
        "planner_model": planner_model if args.agent_architecture != "single" else "",
        "reviewer_model": reviewer_model if args.agent_architecture == "planner_coder_reviewer" else "",
        "api_base": api_base,
        "api_key_source": "explicit" if args.api_key else BACKEND_DEFAULTS[args.backend].get("api_key_env", ""),
        "planner_api_base": planner_api_base,
        "reviewer_api_base": reviewer_api_base,
        "tool_call_mode": args.tool_call_mode,
        "agent_architecture": args.agent_architecture,
        "temperature": args.temperature,
        "planner_temperature": args.planner_temperature,
        "reviewer_temperature": args.reviewer_temperature,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "num_ctx": args.num_ctx,
        "reviewer_rounds": args.reviewer_rounds,
        "max_identical_tool_failures": args.max_identical_tool_failures,
        "post_startup_commands": list(args.post_startup_command),
        "subset": args.subset,
        "split": args.split,
        "instances_type": args.instances_type,
        "instances_path": str(args.instances_path) if args.instances_path else "",
        "filter": args.filter,
        "slice": args.slice,
        "shuffle": args.shuffle,
        "run_name": args.run_name,
    }
    _dump_yaml(output_dir / "run_batch.config.yaml", run_config)

    instances = _build_instances(args)
    all_preds: dict[str, Any] = {}

    for instance in instances:
        instance_id = instance.problem_statement.id
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        info_log_path = instance_dir / f"{instance_id}.info.log"

        def log_line(message: str, *, _path: Path = info_log_path, _iid: str = instance_id) -> None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{timestamp} {_iid} {message}"
            print(line, flush=True)
            with _path.open("a") as fh:
                fh.write(line + "\n")

        start = time.time()
        env = _build_env(instance)

        try:
            log_line("[env] starting")
            env.start()
            repo_root = _instance_repo_root(instance)
            env.communicate(f"cd {shlex.quote(repo_root)}", check="ignore", timeout=10)
            env.communicate(
                input=(
                    f"cd {shlex.quote(repo_root)}\n"
                    "if [ ! -d .git ]; then\n"
                    "  git init\n"
                    "  git config user.email 'custom-runner@example.com'\n"
                    "  git config user.name 'Custom Runner'\n"
                    "  git add .\n"
                    "  git commit -m 'Initial snapshot' >/dev/null 2>&1 || true\n"
                    "fi\n"
                ),
                check="ignore",
                timeout=60,
            )
            install_commands = instance.problem_statement.extra_fields.get("install_commands", [])  # type: ignore[attr-defined]
            setup_commands = instance.problem_statement.extra_fields.get("setup_commands", [])  # type: ignore[attr-defined]
            for command in [*install_commands, *setup_commands]:
                log_line(f"[env] case_setup_command {command}")
                startup_output = env.communicate(
                    input=f"cd {shlex.quote(repo_root)}\n{command}",
                    check="ignore",
                    timeout=300,
                )
                if startup_output.strip():
                    log_line(f"[env] case_setup_output {startup_output.strip().replace(chr(10), ' | ')}")
            for command in args.post_startup_command:
                log_line(f"[env] post_startup_command {command}")
                startup_output = env.communicate(
                    input=f"cd {shlex.quote(repo_root)}\n{command}",
                    check="ignore",
                    timeout=300,
                )
                if startup_output.strip():
                    log_line(f"[env] post_startup_output {startup_output.strip().replace(chr(10), ' | ')}")
            log_line(f"[env] ready repo_root={repo_root}")
            runtime_context = env.communicate(
                input=(
                    f"cd {shlex.quote(repo_root)}\n"
                    "printf 'PWD: '; pwd\n"
                    "printf '\\nFILES:\\n'\n"
                    "ls\n"
                    "printf '\\nPYTHON FILES (depth<=3):\\n'\n"
                    "find . -maxdepth 3 -type f -name '*.py' | sort\n"
                    "printf '\\nREADME HEAD:\\n'\n"
                    "if [ -f README.md ]; then sed -n '1,120p' README.md; fi\n"
                    "printf '\\nGIT STATUS:\\n'\n"
                    "git status --short || true\n"
                ),
                check="ignore",
                timeout=20,
            )
            log_line(f"[env] startup_context {runtime_context.strip().replace(chr(10), ' | ')}")
            role_model_stats: dict[str, dict[str, Any]] = {}
            planner_handoff: dict[str, Any] | None = None
            review_feedback: dict[str, Any] | None = None
            case_validation_prompt = _build_case_validation_prompt(instance.problem_statement)
            case_analysis_prompt = _build_case_analysis_prompt(instance.problem_statement)
            case_evaluation = _extract_case_evaluation(instance.problem_statement)
            case_policy = _extract_case_policy(instance.problem_statement)
            success_validation_commands = _extract_case_success_commands(instance.problem_statement)
            success_validation_checks = _extract_case_success_checks(instance.problem_statement)

            if args.agent_architecture in {"planner_coder", "planner_coder_reviewer"}:
                log_line(f"[planner] calling model={planner_model}")
                planner_handoff, planner_stats, planner_raw = _call_json_role(
                    model=planner_model,
                    api_base=planner_api_base,
                    api_key=planner_api_key,
                    temperature=args.planner_temperature,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    system_prompt=_build_planner_system_prompt(repo_root),
                    user_prompt=(
                        _build_planner_task_prompt(instance.problem_statement.text, repo_root, runtime_context)
                        + ("\n\n" + case_analysis_prompt if case_analysis_prompt else "")
                        + ("\n\n" + case_validation_prompt if case_validation_prompt else "")
                    ),
                    fallback_payload=_default_planner_handoff(),
                )
                planner_handoff = _normalize_planner_handoff(planner_handoff)
                role_model_stats["planner"] = {"model": planner_model, **planner_stats}
                log_line(f"[planner] handoff {json.dumps(planner_handoff, sort_keys=True)}")
                if planner_raw.strip() and planner_raw.strip() != json.dumps(planner_handoff):
                    log_line(f"[planner] raw {planner_raw.strip()[:500]}")

            coder_result: dict[str, Any] | None = None
            coder_accumulated_stats = {"tokens_in": 0, "tokens_out": 0, "api_calls": 0, "turns": 0, "tool_calls": 0}

            for reviewer_round in range(max(1, args.reviewer_rounds if args.agent_architecture == "planner_coder_reviewer" else 1)):
                extra_prompts: list[str] = []
                if planner_handoff:
                    extra_prompts.append(_build_planner_handoff_prompt(planner_handoff))
                if review_feedback:
                    extra_prompts.append(_build_reviewer_feedback_prompt(review_feedback))
                if case_validation_prompt:
                    extra_prompts.append(case_validation_prompt)

                loop = CustomAgentLoop(
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=args.temperature,
                    max_turns=args.max_turns,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    env=env,
                    repo_root=repo_root,
                    problem_statement=instance.problem_statement.text,
                    runtime_context=runtime_context,
                    max_identical_tool_failures=args.max_identical_tool_failures,
                    tool_call_mode=args.tool_call_mode,
                    success_validation_commands=success_validation_commands,
                    success_validation_checks=success_validation_checks,
                    case_policy=case_policy,
                    role_name="coder",
                    extra_user_prompts=extra_prompts,
                    log_fn=log_line,
                )
                coder_result = loop.run()
                coder_stats = coder_result.get("stats", {}) if isinstance(coder_result.get("stats"), dict) else {}
                for key in coder_accumulated_stats:
                    coder_accumulated_stats[key] += int(coder_stats.get(key, 0) or 0)

                if args.agent_architecture != "planner_coder_reviewer":
                    break

                assert coder_result is not None
                patch_text = str(coder_result.get("patch", ""))
                log_line(f"[reviewer] calling model={reviewer_model} round={reviewer_round + 1}")
                review_feedback, reviewer_stats, reviewer_raw = _call_json_role(
                    model=reviewer_model,
                    api_base=reviewer_api_base,
                    api_key=reviewer_api_key,
                    temperature=args.reviewer_temperature,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    system_prompt=_build_reviewer_system_prompt(repo_root),
                    user_prompt=_build_reviewer_task_prompt(
                        planner_handoff=planner_handoff,
                        coder_result=coder_result,
                        patch_text=patch_text,
                        case_evaluation=case_evaluation,
                        case_policy=case_policy,
                    ),
                    fallback_payload=_default_reviewer_feedback(),
                )
                prior = role_model_stats.get("reviewer", {"model": reviewer_model, "tokens_in": 0, "tokens_out": 0, "api_calls": 0})
                role_model_stats["reviewer"] = {
                    "model": reviewer_model,
                    "tokens_in": int(prior.get("tokens_in", 0)) + int(reviewer_stats["tokens_in"]),
                    "tokens_out": int(prior.get("tokens_out", 0)) + int(reviewer_stats["tokens_out"]),
                    "api_calls": int(prior.get("api_calls", 0)) + int(reviewer_stats["api_calls"]),
                }
                log_line(f"[reviewer] decision {json.dumps(review_feedback, sort_keys=True)}")
                if reviewer_raw.strip() and reviewer_raw.strip() != json.dumps(review_feedback):
                    log_line(f"[reviewer] raw {reviewer_raw.strip()[:500]}")

                decision = str(review_feedback.get("decision", "")).lower()
                if decision == "accept":
                    break
                log_line("[reviewer] patch rejected; returning control to coder")

            assert coder_result is not None
            result = coder_result
            if args.agent_architecture == "planner_coder_reviewer" and str((review_feedback or {}).get("decision", "")).lower() != "accept":
                result["submitted"] = False
                result["submission_summary"] = ""
                result["stopped_reason"] = "reviewer_rejected"
                log_line("[reviewer] final decision is revise; marking run as reviewer_rejected")
            role_model_stats["coder"] = {"model": model, **coder_accumulated_stats}
            result["agent_architecture"] = args.agent_architecture
            result["role_model_stats"] = role_model_stats
            result["planner_handoff"] = planner_handoff or {}
            result["review_feedback"] = review_feedback or {}
            total_tokens_in = sum(int(stats.get("tokens_in", 0) or 0) for stats in role_model_stats.values())
            total_tokens_out = sum(int(stats.get("tokens_out", 0) or 0) for stats in role_model_stats.values())
            total_api_calls = sum(int(stats.get("api_calls", 0) or 0) for stats in role_model_stats.values())
            result["stats"]["input_tokens"] = total_tokens_in
            result["stats"]["output_tokens"] = total_tokens_out
            result["stats"]["api_calls"] = total_api_calls
            result["instance_id"] = instance_id
            result["duration_seconds"] = round(time.time() - start, 2)
            _dump_json(instance_dir / f"{instance_id}.traj", result)
            (instance_dir / f"{instance_id}.patch").write_text(result["patch"])
            log_line(f"[result] stopped_reason={result['stopped_reason']} submitted={result['submitted']}")
            pred = _save_pred(
                instance_dir,
                instance_id,
                args.run_name,
                result["info"]["submission"],
            )
            all_preds[instance_id] = pred
        except Exception as exc:  # noqa: BLE001
            log_line(f"[error] {type(exc).__name__}: {exc}")
            failure = {
                "instance_id": instance_id,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": round(time.time() - start, 2),
                "info": {"submission": "", "submitted": False, "stopped_reason": "error"},
            }
            _dump_json(instance_dir / f"{instance_id}.traj", failure)
            (instance_dir / f"{instance_id}.patch").write_text("")
            pred = _save_pred(instance_dir, instance_id, args.run_name, "")
            all_preds[instance_id] = pred
        finally:
            log_line("[env] shutting down")
            env.close()

    _dump_json(output_dir / "preds.json", all_preds)


if __name__ == "__main__":
    main()
