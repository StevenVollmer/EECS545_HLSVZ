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

from sweagent.environment.repo import LocalRepoConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.run.batch_instances import InstancesFromFile, SWEBenchInstances
from swerex.deployment.config import get_deployment

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
- For multi-line replacements, pass actual multi-line strings in tool arguments. Do not insert literal backslash-n text unless you truly want `\\n` in the file.
- Never replace a bare identifier when it can appear in multiple places. Replace the smallest unique surrounding block.
- After the first successful semantic code edit, validate immediately before making more edits.
- If you corrupt formatting or insert literal `\\n` text by mistake, use undo_edit or replace the whole block cleanly.
- Before submit, inspect the diff and make sure you changed executable source code and ran a relevant validation.
- Assume standard tools and project dependencies are already present when possible. Only install packages after a command proves they are missing and the package is necessary for required validation or reproduction.
- Use tools instead of describing what you want to do.
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


@dataclass
class RunStats:
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    tool_calls: int = 0


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
        self.tools = ToolRuntime(env, repo_root)
        self.turns: list[TurnRecord] = []
        self.stats = RunStats()
        self.repeat_hashes: list[str] = []
        self.log_fn = log_fn or (lambda _line: None)
        self.consecutive_failure_counts: dict[str, int] = defaultdict(int)

    def _log(self, message: str) -> None:
        self.log_fn(message)

    @staticmethod
    def _clip(text: str, limit: int = 400) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "...<truncated>"

    def _messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": _build_system_prompt(self.repo_root)},
            {"role": "user", "content": _build_task_prompt(self.problem_statement, self.repo_root)},
            {"role": "user", "content": _build_runtime_context_prompt(self.repo_root, self.runtime_context)},
        ]

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

    def run(self) -> dict[str, Any]:
        import litellm

        litellm.suppress_debug_info = True
        messages = self._messages()
        stopped_reason = "max_turns"
        self._log(f"[run] starting model={self.model} repo_root={self.repo_root}")

        for turn in range(1, self.max_turns + 1):
            self.stats.turns = turn
            self._log(f"[turn {turn}] calling model")
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.stats.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.stats.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            tool_calls = []
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

            turn_record = TurnRecord(
                turn=turn,
                assistant_text=content if isinstance(content, str) else json.dumps(content),
                tool_calls=tool_calls,
                finish_reason=getattr(choice, "finish_reason", None),
            )
            self.turns.append(turn_record)
            if content:
                self._log(f"[turn {turn}] assistant {self._clip(content)}")

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
                messages.append(
                    {
                        "role": "user",
                        "content": "Use a tool call to inspect, edit, validate, or submit. Do not stop with prose only.",
                    }
                )
                continue

            for call in tool_calls:
                self.stats.tool_calls += 1
                self._log(f"[turn {turn}] tool {call['name']} {json.dumps(call['arguments'], sort_keys=True)}")
                self._append_loop_warning_if_needed(messages, call["name"], call["arguments"])
                result = self.tools.execute(call["name"], call["arguments"])
                turn_record.tool_results.append(result)
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
        source = InstancesFromFile(
            path=args.instances_path,
            filter=args.filter,
            slice=args.slice,
            shuffle=args.shuffle,
        )
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
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
    parser.add_argument("--backend", choices=sorted(BACKEND_DEFAULTS.keys()), default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", dest="api_base")
    parser.add_argument("--api-key")
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
    return parser.parse_args()


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
