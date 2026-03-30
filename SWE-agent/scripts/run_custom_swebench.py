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
    return Path(__file__).resolve().parents[1] / "config" / "custom_configs" / "custom_runner_model_presets.yaml"


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
    if backend == "ollama" and not model.startswith("ollama/"):
        return f"ollama/{model}"
    return model


def _build_system_prompt(repo_root: str) -> str:
    return f"""You are a software engineer solving a SWE-bench issue.

You are working inside a repository mounted at `{repo_root}`.

Rules:
- Reproduce the bug before the first source edit when feasible.
- Fix executable runtime logic before touching tests, docs, comments, or examples.
- Prefer targeted inspection and one deliberate block replacement over many tiny edits.
- If you are not sure about an exact file path, use `bash` with `find`, `rg --files`, or `ls` before calling `view` or editing tools.
- If a file tool reports that a path does not exist, stop guessing filenames and discover the real path first.
- For multi-line replacements, pass actual multi-line strings in tool arguments. Do not insert literal backslash-n text unless you truly want `\\n` in the file.
- Never replace a bare identifier when it can appear in multiple places. Replace the smallest unique surrounding block.
- After the first successful semantic code edit, validate immediately before making more edits.
- If you corrupt formatting or insert literal `\\n` text by mistake, use undo_edit or replace the whole block cleanly.
- Before submit, inspect the diff and make sure you changed executable source code and ran a relevant validation.
- Assume standard tools and project dependencies are already present when possible. Only install packages after a command proves they are missing and the package is necessary for required validation or reproduction.
- Use tools instead of describing what you want to do.
"""


def _build_react_json_prompt() -> str:
    return """If native tool calling is unavailable, respond with JSON only and no surrounding prose.

Schema:
{"tool":"bash|view|str_replace|insert|undo_edit|submit","arguments":{...}}

You must return exactly one action object per response.
Do not return an array.
Do not return multiple newline-separated objects.
You may use a single ```json fenced block, but it must contain exactly one JSON object.

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

Examples:
{"tool":"bash","arguments":{"command":"python3 -m pytest test_calculator.py"}}
{"tool":"view","arguments":{"path":"calculator.py","start_line":1,"end_line":20}}
{"tool":"submit","arguments":{"summary":"Fixed the denominator bug"}}
"""


def _build_task_prompt(problem_statement: str, repo_root: str) -> str:
    return f"""Solve this SWE-bench issue inside `{repo_root}`.

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


@dataclass
class ToolExecutionRecord:
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool = False


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
    diff_checked: bool = False


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

    def bash(self, command: str, timeout: int = 30) -> str:
        output = self.env.communicate(
            input=f"cd {shlex.quote(self.repo_root)}\n{command}",
            timeout=timeout,
            check="ignore",
        )
        return output or "(no output)"

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

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecutionRecord:
        try:
            if name == "bash":
                output = self.bash(arguments["command"], int(arguments.get("timeout", 30)))
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
            return ToolExecutionRecord(name=name, arguments=arguments, output=output, is_error=False)
        except Exception as exc:  # noqa: BLE001
            return ToolExecutionRecord(name=name, arguments=arguments, output=f"{type(exc).__name__}: {exc}", is_error=True)


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
        env: SWEEnv,
        repo_root: str,
        problem_statement: str,
        runtime_context: str,
        max_identical_tool_failures: int,
        tool_call_mode: str,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.env = env
        self.repo_root = repo_root
        self.problem_statement = problem_statement
        self.runtime_context = runtime_context
        self.max_identical_tool_failures = max_identical_tool_failures
        self.tool_call_mode = tool_call_mode
        self.tools = ToolRuntime(env, repo_root)
        self.turns: list[TurnRecord] = []
        self.stats = RunStats()
        self.repeat_hashes: list[str] = []
        self.log_fn = log_fn or (lambda _line: None)
        self.consecutive_failure_counts: dict[str, int] = defaultdict(int)
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
        if self.tool_call_mode == "react_json":
            messages.append({"role": "user", "content": _build_react_json_prompt()})
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
    def _parse_react_json(cls, content: str) -> ParseOutcome:
        text = content.replace("</think>", "\n").replace("<think>", "\n").strip()
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
                return ParseOutcome(
                    tool_calls=[],
                    error="Returned multiple actions, but react_json mode requires exactly one JSON object per response.",
                )
            if not isinstance(payload, dict):
                continue
            tool = payload.get("tool") or payload.get("action") or payload.get("name") or payload.get("tool_name")
            arguments = payload.get("arguments")
            if arguments is None:
                arguments = {
                    k: v
                    for k, v in payload.items()
                    if k not in {"tool", "action", "name", "tool_name", "type"}
                }
            if not isinstance(tool, str) or not isinstance(arguments, dict):
                continue
            tool = tool.strip()
            tool_calls.append(
                {
                    "id": f"react-1-{tool}",
                    "name": tool,
                    "arguments": cls._normalize_tool_arguments(tool, arguments),
                }
            )
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

    def _append_loop_warning_if_needed(self, messages: list[dict[str, Any]], tool_name: str, arguments: dict[str, Any]) -> None:
        digest = self._hash_tool_call(tool_name, arguments)
        self.repeat_hashes.append(digest)
        recent = self.repeat_hashes[-4:]
        if len(recent) == 4 and len(set(recent)) == 1:
            self._log(f"[loop-warning] repeated tool call detected: {tool_name} {json.dumps(arguments, sort_keys=True)}")
            messages.append(
                {
                    "role": "user",
                    "content": "You are repeating the same tool call. Stop looping, inspect a larger unique block, validate, or undo the bad edit.",
                }
            )

    def _update_state_from_tool_result(self, result: ToolExecutionRecord) -> None:
        if result.is_error:
            return
        if result.name in {"str_replace", "insert", "undo_edit"}:
            self.state.executable_edit_made = True
        if result.name == "bash":
            lowered = result.output.lower()
            if "passed" in lowered and "failed" not in lowered:
                self.state.validation_passed = True
            if "diff --git" in result.output:
                self.state.diff_checked = True

    def run(self) -> dict[str, Any]:
        import litellm

        litellm.suppress_debug_info = True
        messages = self._messages()
        stopped_reason = "max_turns"
        self._log(f"[run] starting model={self.model} repo_root={self.repo_root}")

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
            elif self.tool_call_mode == "react_json":
                parse_outcome = self._parse_react_json(content if isinstance(content, str) else json.dumps(content))
                tool_calls = parse_outcome.tool_calls
                parse_error = parse_outcome.error

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
                corrective_hint = (
                    "Use a tool call to inspect, edit, validate, or submit. "
                    "If native tool calling is unavailable, respond with valid JSON only and exactly one action object per response. "
                    + (f"Your last action payload was invalid: {parse_error}" if parse_error else "")
                ).strip()
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

            for call in tool_calls:
                self.stats.tool_calls += 1
                self._log(f"[turn {turn}] tool {call['name']} {json.dumps(call['arguments'], sort_keys=True)}")
                self._append_loop_warning_if_needed(messages, call["name"], call["arguments"])
                result = self.tools.execute(call["name"], call["arguments"])
                turn_record.tool_results.append(result)
                self._update_state_from_tool_result(result)
                level = "tool-error" if result.is_error else "tool-output"
                self._log(f"[turn {turn}] {level} {result.name}: {self._clip(result.output)}")
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
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That tool call failed. Use the runtime context, repo root, pwd/ls output, and recent tool output to correct the next step. "
                                "Do not retry the same failing call unchanged."
                            ),
                        }
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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": result.output,
                    }
                )
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
                    deployment=DockerDeploymentConfig(image=image_name),
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
    ./env/bin/python SWE-agent/scripts/run_custom_swebench.py --preset openai_gpt4o_mini --filter pydicom__pydicom-1458 --output-dir SWE-agent/custom_runs/openai_gpt4omini

  OpenAI:
    ./env/bin/python SWE-agent/scripts/run_custom_swebench.py --backend openai --model gpt-4o-mini --filter pydicom__pydicom-1458

  Ollama:
    ./env/bin/python SWE-agent/scripts/run_custom_swebench.py --backend ollama --model qwen2.5-coder:7b-instruct --filter pydicom__pydicom-1458

  LM Studio:
    ./env/bin/python SWE-agent/scripts/run_custom_swebench.py --backend lmstudio --model openai/local-model --filter pydicom__pydicom-1458

  UMich:
    ./env/bin/python SWE-agent/scripts/run_custom_swebench.py --backend umich --model openai/openai/gpt-oss-120b --filter pydicom__pydicom-1458
""",
    )
    parser.add_argument("--preset", choices=preset_names)
    parser.add_argument("--backend", choices=sorted(BACKEND_DEFAULTS.keys()))
    parser.add_argument("--model")
    parser.add_argument("--api-base", dest="api_base")
    parser.add_argument("--api-key")
    parser.add_argument("--tool-call-mode", choices=["openai_tools", "react_json"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=4096)
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
        if args.api_base is None:
            args.api_base = preset.get("api_base")
        if args.api_key is None and preset.get("api_key") is not None:
            args.api_key = preset.get("api_key")
        if args.tool_call_mode is None and preset.get("tool_call_mode") is not None:
            args.tool_call_mode = preset.get("tool_call_mode")
        if args.run_name == "custom_runner":
            args.run_name = args.preset

    if args.backend is None:
        args.backend = "openai"
    if args.model is None:
        parser.error("--model is required unless provided by --preset")
    if args.tool_call_mode is None:
        args.tool_call_mode = "openai_tools"

    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    api_base = args.api_base or BACKEND_DEFAULTS[args.backend].get("api_base")
    api_key = _resolve_api_key(args.backend, args.api_key)
    model = _normalize_model_name(args.backend, args.model)

    run_config = {
        "backend": args.backend,
        "model": model,
        "api_base": api_base,
        "api_key_source": "explicit" if args.api_key else BACKEND_DEFAULTS[args.backend].get("api_key_env", ""),
        "tool_call_mode": args.tool_call_mode,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
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
                    "printf '\\nGIT STATUS:\\n'\n"
                    "git status --short || true\n"
                ),
                check="ignore",
                timeout=20,
            )
            log_line(f"[env] startup_context {runtime_context.strip().replace(chr(10), ' | ')}")

            loop = CustomAgentLoop(
                model=model,
                api_base=api_base,
                api_key=api_key,
                temperature=args.temperature,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                env=env,
                repo_root=repo_root,
                problem_statement=instance.problem_statement.text,
                runtime_context=runtime_context,
                max_identical_tool_failures=args.max_identical_tool_failures,
                tool_call_mode=args.tool_call_mode,
                log_fn=log_line,
            )
            result = loop.run()
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
