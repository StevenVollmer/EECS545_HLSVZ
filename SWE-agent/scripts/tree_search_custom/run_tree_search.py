#!/usr/bin/env python3
"""
MCTS-inspired SWE-bench agent optimised for local Ollama small models.

Each node in the search tree represents the environment after one tool action.
Selection uses UCB1. Expansion samples the model with temperature > 0 to get
K candidate next actions. Value is derived from LoopState signals (success
checks, validation passing, edit quality). Backtracking restores git state
inside the Docker container.

Usage (single case):
  ./env/bin/python SWE-agent/scripts/tree_search_custom/run_tree_search.py \
      --model qwen2.5-coder:7b-instruct \
      --instances-type file \
      --instances-path SWE-agent/custom_cases/simple_mean_bug \
      --output-dir SWE-agent/tree_search_runs/simple_mean_bug_mcts

Usage (SWE-bench Lite dev slice):
  ./env/bin/python SWE-agent/scripts/tree_search_custom/run_tree_search.py \\
      --model qwen2.5-coder:7b-instruct \\
      --filter pylint-dev__astroid-1866 \\
      --output-dir SWE-agent/tree_search_runs/astroid_mcts

Available local models (ollama):
  qwen2.5-coder:7b-instruct   <- default coder
  qwen3.5:9b                  <- default planner
  deepseek-r1:7b
  granite-code:8b
  mistral:7b-instruct-v0.3-q4_K_M
  codellama:7b-instruct
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import re

import yaml

# ---------------------------------------------------------------------------
# Import shared utilities from the sibling custom/ directory
# ---------------------------------------------------------------------------
_CUSTOM_DIR = Path(__file__).resolve().parents[1] / "custom"
sys.path.insert(0, str(_CUSTOM_DIR))

from run_custom_swebench import (  # noqa: E402
    BACKEND_DEFAULTS,
    TOOL_SCHEMAS,
    ToolRuntime,
    LoopState,
    RunStats,
    TurnRecord,
    ToolExecutionRecord,
    ParseOutcome,
    _build_system_prompt,
    _build_task_prompt,
    _build_runtime_context_prompt,
    _build_react_json_prompt,
    _build_openai_tools_prompt,
    _call_json_role,
    _normalize_planner_handoff,
    _default_planner_handoff,
    _default_reviewer_feedback,
    _build_planner_system_prompt,
    _build_planner_task_prompt,
    _build_reviewer_system_prompt,
    _build_reviewer_task_prompt,
    _build_planner_handoff_prompt,
    _build_reviewer_feedback_prompt,
    _build_case_validation_prompt,
    _build_case_analysis_prompt,
    _extract_case_evaluation,
    _extract_case_policy,
    _extract_case_success_commands,
    _extract_case_success_checks,
    _command_output_satisfies_check,
    _command_output_failure_reasons,
    _load_custom_file_instances,
    _extract_first_json_dict,
    PlainLocalDirectoryRepo,
    _build_env,
    _instance_repo_root,
    _save_pred,
    _dump_json,
    _dump_yaml,
    _normalize_model_name,
    _resolve_api_key,
    _summarize_validation_events,
)
from sweagent.run.batch_instances import SWEBenchInstances  # noqa: E402

# ---------------------------------------------------------------------------
# Ollama defaults
# ---------------------------------------------------------------------------

OLLAMA_API_BASE = "http://localhost:11434"
OLLAMA_API_KEY = "ollama"
DEFAULT_CODER_MODEL = "qwen3.5:9b"
DEFAULT_PLANNER_MODEL = "qwen3.5:9b"

# ---------------------------------------------------------------------------
# Think-tag helpers (qwen3.5:9b and other reasoning models wrap output in
# <think>...</think> blocks that break JSON extraction)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).strip()


def _is_infra_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        needle in lowered
        for needle in (
            "ports are not available",
            "container process terminated",
            "docker: error response from daemon",
            "failed to create endpoint",
            "cannot connect to the docker daemon",
            "address already in use",
            "network is unreachable",
        )
    )


def _build_reviewer_constraints_prompt(review_feedback: dict[str, Any]) -> str:
    required_changes = [
        str(item).strip() for item in review_feedback.get("required_changes", []) if str(item).strip()
    ]
    files_to_revisit = [
        str(item).strip() for item in review_feedback.get("files_to_revisit", []) if str(item).strip()
    ]
    validations_to_rerun = [
        str(item).strip() for item in review_feedback.get("validations_to_rerun", []) if str(item).strip()
    ]
    lines = [
        "Reviewer hard constraints for this round (must satisfy before submit):",
    ]
    if required_changes:
        lines.append("- Required changes:")
        lines.extend(f"  - {item}" for item in required_changes[:6])
    if files_to_revisit:
        lines.append("- Allowed files to revisit (unless new evidence requires expansion):")
        lines.extend(f"  - {item}" for item in files_to_revisit[:8])
    if validations_to_rerun:
        lines.append("- Required validations to rerun (exact commands):")
        lines.extend(f"  - {cmd}" for cmd in validations_to_rerun[:8])
    # Escalate "no hardcoding" into a CRITICAL constraint if reviewer mentioned it
    if any("hardcod" in str(c).lower() for c in required_changes):
        lines.append(
            "- CRITICAL: Do NOT hardcode specific string values, names, or IDs. "
            "The fix must work for ANY input, not just the test case shown. "
            "A hardcoded replacement that only fixes the example will be rejected."
        )
    lines.append("- Do not submit until the above constraints are satisfied.")
    return "\n".join(lines)


def _start_env_with_retries(env, log_fn: Callable[[str], None], max_attempts: int = 3) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            env.start()
            return
        except Exception as exc:  # pragma: no cover - daemon/network dependent
            last_exc = exc
            transient = _is_infra_error_text(str(exc))
            if not transient or attempt >= max_attempts:
                raise
            wait_seconds = min(2 ** attempt, 8)
            log_fn(
                f"[env] transient start failure attempt {attempt}/{max_attempts}: "
                f"{str(exc)[:180]} — retrying in {wait_seconds}s"
            )
            try:
                env.close()
            except Exception:
                pass
            time.sleep(wait_seconds)
    if last_exc is not None:
        raise last_exc


def _call_json_role_local(
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
    """Like ``_call_json_role`` but strips <think> tags before JSON extraction."""
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
        # JSON mode: prevents the model from emitting prose around the JSON object
        completion_kwargs["response_format"] = {"type": "json_object"}
        if num_ctx is not None:
            completion_kwargs["num_ctx"] = num_ctx

    messages = list(completion_kwargs["messages"])
    total_in = 0
    total_out = 0
    raw_content = ""

    for attempt in range(2):
        completion_kwargs["messages"] = messages
        response = litellm.completion(**completion_kwargs)
        usage = getattr(response, "usage", None)
        total_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        total_out += int(getattr(usage, "completion_tokens", 0) or 0)
        content = response.choices[0].message.content or ""
        raw_content = str(content)
        stripped = _strip_think_tags(raw_content)
        try:
            parsed = _extract_first_json_dict(stripped)
            return parsed, {"tokens_in": total_in, "tokens_out": total_out, "api_calls": attempt + 1}, raw_content
        except ValueError:
            if attempt == 0:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": raw_content},
                    {
                        "role": "user",
                        "content": (
                            "Return the same answer as exactly one valid JSON object "
                            "and nothing else. Do not include explanations or markdown."
                        ),
                    },
                ]

    fallback = dict(fallback_payload)
    fallback["_parse_error"] = "role_json_parse_failed"
    fallback["_raw_response_preview"] = raw_content[:500]
    return fallback, {"tokens_in": total_in, "tokens_out": total_out, "api_calls": 2}, raw_content


# ---------------------------------------------------------------------------
# Slim planner prompts optimised for small (7-9B) local models.
#
# The full planner schema has 12 keys — too many for a small model to fill
# reliably in one shot.  This version asks for 5 high-signal keys and includes
# a concrete JSON example so the model knows the exact expected shape.
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT_SLIM = """\
You are a planner for a software repair task.
Your only job is to output a JSON object that tells the coder where to look and how to validate.

Rules:
- Output ONLY a JSON object. No prose, no markdown fences, no explanation.
- Keep values short (one sentence or a short list).
- Do not invent file paths you cannot justify from the context.

Required JSON keys:
  "root_cause_hypothesis" : one sentence naming the likely bug and its location
  "files_likely_affected"  : array of 1-4 file paths, most likely first
  "first_edit_target"      : one sentence — file, approximate line, and what expression to change
  "first_actions"          : array of 1-3 concrete steps (read, reproduce, inspect)
  "required_validations"   : array of 1-2 commands that must pass for the fix to be correct
  "forbidden_edits"        : array of things the coder must NOT do

Example (do NOT copy this literally — fill in real values from the issue):
{
  "root_cause_hypothesis": "mean() divides total by len(values)-1 instead of len(values)",
  "files_likely_affected": ["calculator.py"],
  "first_edit_target": "calculator.py line ~12: denominator expression len(values)-1 should be len(values)",
  "first_actions": ["Read calculator.py lines around mean()", "Run pytest test_calculator.py to reproduce the failure"],
  "required_validations": ["python -m pytest test_calculator.py"],
  "forbidden_edits": ["do not modify test files"]
}
"""


def _build_planner_system_prompt_slim(repo_root: str) -> str:
    """Slim planner system prompt with example — better for small local models."""
    return _PLANNER_SYSTEM_PROMPT_SLIM + f"\nThe repository root inside the container is `{repo_root}`.\n"


def _build_planner_task_prompt_slim(
    problem_statement: str,
    repo_root: str,
    runtime_context: str,
    case_analysis: str = "",
) -> str:
    """Slim planner task prompt — truncated context keeps token count low."""
    # Limit context to 1500 chars so the full prompt fits comfortably in 4K ctx
    ctx_snippet = runtime_context[:1500].strip()
    analysis_section = f"\n{case_analysis}\n" if case_analysis.strip() else ""
    return (
        f"Plan a fix for the issue below in `{repo_root}`. Return JSON only.\n\n"
        f"ISSUE:\n{problem_statement}\n"
        f"{analysis_section}"
        f"\nREPOSITORY CONTEXT:\n{ctx_snippet}\n"
    )


# ---------------------------------------------------------------------------
# Local case loader: merges top-level evaluation/analysis/policy into
# extra_fields so _extract_case_success_checks etc. can find them even when
# the case.json author did NOT nest those keys under "extra_fields"
# ---------------------------------------------------------------------------

def _load_custom_file_instances_local(path: Path, filter_regex: str, slice_spec: str, shuffle: bool):
    """Like _load_custom_file_instances but promotes top-level case keys.

    The parent loader only copies `item.get("extra_fields", {})`, so top-level
    keys like ``evaluation``, ``analysis``, and ``policy`` are silently dropped
    when case authors place them at the root of case.json.  This version also
    merges those keys into ``extra_fields`` so the extractor helpers can find
    them.
    """
    from sweagent.run.batch_instances import BatchInstance
    from sweagent.environment.swe_env import EnvironmentConfig
    from swerex.deployment.config import DockerDeploymentConfig
    from sweagent.agent.problem_statement import TextProblemStatement
    import re as _re

    def _resolve(p: Path) -> Path | None:
        for name in ("case.json", "case.yaml", "case.yml"):
            if (p / name).exists():
                return p / name
        import glob as _g
        matches = _g.glob(str(p / "*.json")) + _g.glob(str(p / "*.yaml")) + _g.glob(str(p / "*.yml"))
        if matches:
            return Path(matches[0])
        return None

    # Support passing a parent directory that contains one case-dir per subdirectory.
    # If path itself has no case file, scan immediate subdirectories for case files.
    case_files: list[Path] = []
    direct = _resolve(path)
    if direct is not None:
        case_files = [direct]
    else:
        for sub in sorted(path.iterdir()):
            if sub.is_dir():
                cf = _resolve(sub)
                if cf is not None:
                    case_files.append(cf)
        if not case_files:
            raise FileNotFoundError(f"No case files found under {path} or its subdirectories")

    # Build (item, source_case_file) pairs so repo resolution stays correct
    # even when items came from different subdirectory case files.
    item_pairs: list[tuple[Any, Path]] = []
    for case_file in case_files:
        loaded = yaml.safe_load(case_file.read_text())
        if isinstance(loaded, dict):
            loaded = [loaded]
        if isinstance(loaded, list):
            for entry in loaded:
                if isinstance(entry, dict):
                    item_pairs.append((entry, case_file))

    _filter = _re.compile(filter_regex or ".*")
    instances: list[Any] = []

    for item, case_file in item_pairs:
        instance_id = str(item["instance_id"])
        if not _filter.search(instance_id):
            continue

        image_name = str(item.get("image_name", "python:3.11"))
        base_commit = str(item.get("base_commit", "HEAD"))

        # Determine repo path (mirrors _resolve_case_repo_path from parent)
        repo_path_str = item.get("repo_path", "")
        if repo_path_str:
            candidate = Path(repo_path_str)
            repo_path = candidate if candidate.is_absolute() else (case_file.parent / candidate).resolve()
        else:
            repo_path = (case_file.parent / "repo").resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"Repo path not found: {repo_path}")

        problem_text = str(item.get("problem_statement", ""))

        # Start with explicit extra_fields if present
        extra_fields: dict[str, Any] = dict(item.get("extra_fields", {}))

        # Promote top-level structured keys that case authors may place at root
        for top_key in ("evaluation", "analysis", "policy"):
            if top_key in item and top_key not in extra_fields:
                extra_fields[top_key] = item[top_key]

        # Always inject these convenience fields
        extra_fields["setup_commands"] = list(item.get("setup_commands", []))
        extra_fields["install_commands"] = list(item.get("install_commands", []))
        extra_fields["case_file"] = str(case_file)
        extra_fields["repo_path"] = str(repo_path)

        instances.append(
            BatchInstance(
                env=EnvironmentConfig.model_construct(
                    deployment=DockerDeploymentConfig(image=image_name, pull="missing"),
                    # Use PlainLocalDirectoryRepo so _instance_repo_root returns
                    # "/<repo_name>" (e.g. "/repo") instead of "/"
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

    # Slice / shuffle
    if shuffle:
        import random
        random.shuffle(instances)
    if slice_spec:
        parts = slice_spec.split(":")
        s = slice(*[int(p) if p else None for p in parts])
        instances = instances[s]

    return instances


# ---------------------------------------------------------------------------
# MCTS hyper-parameters
# ---------------------------------------------------------------------------

UCB_C = 1.414          # exploration constant (√2)
EXPANSION_CANDIDATES = 1  # branch candidates at edit-intent turns (explore branch never helped in practice)
EXPLORE_TEMPERATURE = 0.7  # temperature for explore candidate
EXPLOIT_TEMPERATURE = 0.1  # temperature for greedy / linear turns
MAX_ITERATIONS = 20     # total MCTS iterations (tree expansions)
MAX_NODE_DEPTH = 20     # max turns per branch before treating as terminal
MAX_TOKENS_REACT = 384  # token limit for react_json mode (small models)

# Improvement 1/3: selective branching + majority vote
EDIT_VOTE_SAMPLES = 5   # majority-vote samples when an edit action is detected
EDIT_VOTE_TEMPERATURE = 0.4  # temperature used for vote samples

# Improvement 2: context compression
COMPRESS_AFTER_TURNS = 5    # start truncating old tool outputs beyond this depth
COMPRESS_OUTPUT_CHARS = 200  # max chars kept per old tool output

# ---------------------------------------------------------------------------
# Value constants
# ---------------------------------------------------------------------------

VALUE_SUCCESS_CHECK_EACH = 20.0  # per satisfied case success check
VALUE_NEW_CHECK_PROGRESS = 10.0  # incremental reward when a step satisfies new checks
VALUE_SUBMITTED = 25.0
VALUE_VALIDATION_PASSED_AFTER_EDIT = 12.0
VALUE_DIFF_CHECKED = 3.0
VALUE_EDIT_MADE = 3.0
PENALTY_PARSE_ERROR = -1.5
PENALTY_TOOL_ERROR = -1.0
PENALTY_NO_EDIT_DEEP_TURN = -1.2
PENALTY_POST_EDIT_STAGNATION = -2.0
PENALTY_EMPTY_PATCH_ENDPOINT = -8.0
DEPTH_DISCOUNT = 0.97   # per-depth multiplier; prefers shallower solutions


# ---------------------------------------------------------------------------
# SearchNode
# ---------------------------------------------------------------------------

@dataclass
class SearchNode:
    """One node in the MCTS tree.

    A node represents the environment state *after* ``action`` was executed
    at ``parent``.  The root node has no parent and no action.
    """

    # Tree structure
    parent: "SearchNode | None"
    children: list["SearchNode"] = field(default_factory=list)
    depth: int = 0

    # Environment state (restored when this node is selected for expansion)
    git_diff: str = ""            # `git diff --no-color` from base HEAD
    loop_state: LoopState = field(default_factory=LoopState)

    # Conversation history UP TO AND INCLUDING this node's observation
    messages: list[dict[str, Any]] = field(default_factory=list)

    # The action that created this node and its observed result
    action: dict[str, Any] | None = None  # {name, arguments}
    observation: str = ""

    # Statistics used by UCB1
    visits: int = 0
    total_value: float = 0.0

    # Terminal flags
    submitted: bool = False
    submit_summary: str = ""
    stopped_reason: str = ""

    # Counters carried from parent for failure detection
    consecutive_failure_counts: dict[str, int] = field(default_factory=dict)
    consecutive_success_counts: dict[str, int] = field(default_factory=dict)
    repeat_hashes: list[str] = field(default_factory=list)

    # Per-turn records for trajectory export
    turn_records: list[TurnRecord] = field(default_factory=list)
    parse_errors: int = 0

    # MCTS improvement metadata
    is_branch_point: bool = False          # True when this node was created at an edit branch
    vote_counts: dict[str, int] = field(default_factory=dict)  # majority-vote tallies for edit nodes
    vote_total_samples: int = 0
    auto_finalized: bool = False
    check_progress_gain: int = 0
    post_edit_no_progress_turns: int = 0
    has_nonempty_patch: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.submitted or self.stopped_reason != "" or self.depth >= MAX_NODE_DEPTH

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.mean_value
        explore = UCB_C * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore


# ---------------------------------------------------------------------------
# MCTSAgentLoop
# ---------------------------------------------------------------------------

class MCTSAgentLoop:
    """MCTS agent loop for a single SWE-bench instance.

    Replaces ``CustomAgentLoop`` with a tree-search strategy:
    - **Selection**: walk the tree from root using UCB1 until an unexpanded leaf.
    - **Expansion**: at the leaf, ask the model for ``EXPANSION_CANDIDATES``
      next actions (varied temperatures) and execute each one to create child
      nodes.
    - **Evaluation**: score each new child using ``_estimate_value``.
    - **Backpropagation**: propagate the value up to the root.
    - **Restore**: before expanding a node, restore the Docker container's git
      state to match that node.

    After ``MAX_ITERATIONS`` iterations the best complete trajectory is
    returned.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
        num_ctx: int | None,
        env,
        repo_root: str,
        problem_statement: str,
        runtime_context: str,
        max_identical_tool_failures: int,
        success_validation_commands: list[str] | None = None,
        success_validation_checks: list[dict[str, Any]] | None = None,
        case_policy: dict[str, Any] | None = None,
        extra_user_prompts: list[str] | None = None,
        log_fn: Callable[[str], None] | None = None,
        edit_vote_samples: int = EDIT_VOTE_SAMPLES,
        adaptive_branching: bool = True,
        verbose: bool = False,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.env = env
        self.repo_root = repo_root
        self.problem_statement = problem_statement
        self.runtime_context = runtime_context
        self.max_identical_tool_failures = max_identical_tool_failures
        self.success_validation_commands = [
            c.strip() for c in (success_validation_commands or []) if str(c).strip()
        ]
        self.success_validation_checks = [
            dict(i) for i in (success_validation_checks or []) if isinstance(i, dict)
        ]
        self.case_policy = dict(case_policy or {})
        self.extra_user_prompts = extra_user_prompts or []
        self.log_fn = log_fn or (lambda _: None)
        self.edit_vote_samples = max(1, edit_vote_samples)
        self.adaptive_branching = adaptive_branching
        self.verbose = verbose
        self._global_stats = RunStats()
        self._turn_counter = 0  # used to generate unique tool call IDs

    def _log(self, msg: str) -> None:
        self.log_fn(msg)

    def _vlog(self, msg: str) -> None:
        """Log only when verbose mode is enabled."""
        if self.verbose:
            self.log_fn(msg)

    # ------------------------------------------------------------------
    # Environment state helpers
    # ------------------------------------------------------------------

    def _capture_git_diff(self) -> str:
        """Return the current git diff inside the container."""
        return self.env.communicate(
            input=f"git -C {shlex.quote(self.repo_root)} diff --no-color",
            timeout=20,
            check="ignore",
        ) or ""

    def _restore_state(self, node: SearchNode) -> None:
        """Restore the container's git working tree to match ``node``."""
        reset_cmd = (
            f"cd {shlex.quote(self.repo_root)}\n"
            "git reset --hard HEAD\n"
            "git clean -fdq\n"
        )
        self.env.communicate(input=reset_cmd, timeout=30, check="ignore")
        if node.git_diff.strip():
            self.env.write_file("/tmp/mcts_restore.patch", node.git_diff)
            result = self.env.communicate(
                input=(
                    f"cd {shlex.quote(self.repo_root)}\n"
                    "git apply /tmp/mcts_restore.patch 2>&1 || "
                    "patch -p1 < /tmp/mcts_restore.patch 2>&1 || true"
                ),
                timeout=30,
                check="ignore",
            )
            if result and ("error" in result.lower() or "failed" in result.lower()):
                self._log(f"[restore] warning: {result.strip()[:200]}")

    # ------------------------------------------------------------------
    # Value estimation
    # ------------------------------------------------------------------

    def _estimate_value(self, node: SearchNode) -> float:
        """Score a node based on progress signals in its LoopState."""
        v = 0.0
        ls = node.loop_state
        total_checks = len(self.success_validation_checks) or 1
        v += VALUE_SUCCESS_CHECK_EACH * len(ls.satisfied_success_checks) / total_checks
        v += VALUE_NEW_CHECK_PROGRESS * node.check_progress_gain
        if node.submitted:
            v += VALUE_SUBMITTED
        if ls.validation_passed and ls.executable_edit_made:
            v += VALUE_VALIDATION_PASSED_AFTER_EDIT
        if ls.diff_checked:
            v += VALUE_DIFF_CHECKED
        if ls.executable_edit_made:
            v += VALUE_EDIT_MADE
        if not ls.executable_edit_made and node.depth >= 6:
            v += PENALTY_NO_EDIT_DEEP_TURN * (node.depth - 5)
        if node.post_edit_no_progress_turns >= 3:
            v += PENALTY_POST_EDIT_STAGNATION * (node.post_edit_no_progress_turns - 2)
        if node.depth >= 8 and ls.executable_edit_made and not node.has_nonempty_patch:
            v += PENALTY_EMPTY_PATCH_ENDPOINT
        if node.parse_errors >= 2:
            return float("-inf")  # permanently abandon high-parse-error branches
        v += PENALTY_PARSE_ERROR * node.parse_errors
        failure_total = sum(node.consecutive_failure_counts.values())
        v += PENALTY_TOOL_ERROR * failure_total
        # Depth discount — prefer compact solutions
        v *= DEPTH_DISCOUNT ** node.depth
        return max(v, 0.0)

    # ------------------------------------------------------------------
    # MCTS core operations
    # ------------------------------------------------------------------

    def _select(self, root: SearchNode) -> SearchNode:
        """Walk from root to a leaf using UCB1."""
        node = root
        while node.children:
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return unvisited[0]
            node = max(node.children, key=lambda c: c.ucb1(node.visits))
        return node

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        current: SearchNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _base_messages(self) -> list[dict[str, Any]]:
        msgs = [
            {"role": "system", "content": _build_system_prompt(self.repo_root)},
            {"role": "user", "content": _build_task_prompt(self.problem_statement, self.repo_root)},
            {"role": "user", "content": _build_runtime_context_prompt(self.repo_root, self.runtime_context)},
        ]
        for p in self.extra_user_prompts:
            msgs.append({"role": "user", "content": p})
        msgs.append({"role": "user", "content": _build_react_json_prompt()})
        return msgs

    # ------------------------------------------------------------------
    # Improvement helpers
    # ------------------------------------------------------------------

    def _is_edit_action(self, action: dict[str, Any]) -> bool:
        """Return True when the action will modify a file."""
        return action.get("name") in {"str_replace", "insert"}

    def _compress_messages(self, messages: list[dict[str, Any]], node_depth: int) -> list[dict[str, Any]]:
        """Truncate old tool outputs once history grows beyond COMPRESS_AFTER_TURNS.

        Keeps the full action sequence (assistant role messages) intact so the
        model still sees what was tried; only clips the verbose tool result text
        for turns that are far back in history.
        """
        if node_depth < COMPRESS_AFTER_TURNS:
            return messages

        # Count tool-result messages (role == "tool") and clip the older ones.
        tool_msg_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        # Keep the most recent COMPRESS_AFTER_TURNS tool results unclipped
        clip_up_to = len(tool_msg_indices) - COMPRESS_AFTER_TURNS
        clip_indices = set(tool_msg_indices[:max(0, clip_up_to)])

        compressed: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if i in clip_indices:
                content = str(msg.get("content", ""))
                if len(content) > COMPRESS_OUTPUT_CHARS:
                    content = content[:COMPRESS_OUTPUT_CHARS] + "…[truncated]"
                compressed.append({**msg, "content": content})
            else:
                compressed.append(msg)
        return compressed

    def _majority_vote_edit(
        self, messages: list[dict[str, Any]], n_samples: int,
        seed_call: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, int], int]:
        """Sample up to n_samples times (with early-exit) and return the most-voted edit.

        If *seed_call* is provided it is counted as the first vote without an
        extra model call (it was already the probe at EXPLOIT_TEMPERATURE).
        Sampling then stops as soon as all votes agree **and** at least 2
        samples have been counted — so a confident model costs 2 calls total
        (probe + 1 confirmation) while a genuinely inconsistent model uses up
        to n_samples calls.

        Returns (winner_call, vote_counts, used_samples) where vote_counts maps
        action-hash → vote count. Returns (None, {}, 0) if no parseable edit was
        produced.
        """
        from collections import Counter
        votes: Counter[str] = Counter()
        hash_to_call: dict[str, dict[str, Any]] = {}

        # Seed with probe result — counts as first vote, no extra model call
        if seed_call is not None and self._is_edit_action(seed_call):
            h = self._hash_action(seed_call["name"], seed_call.get("arguments", {}))
            votes[h] += 1
            hash_to_call[h] = seed_call

        remaining = (n_samples - 1) if seed_call is not None else n_samples
        for _ in range(remaining):
            try:
                content, tok_in, tok_out = self._call_model(messages, EDIT_VOTE_TEMPERATURE)
            except Exception:
                continue
            self._global_stats.input_tokens += tok_in
            self._global_stats.output_tokens += tok_out
            outcome = self._parse_action(content)
            if not outcome.tool_calls:
                continue
            call = outcome.tool_calls[0]
            if not self._is_edit_action(call):
                continue
            h = self._hash_action(call["name"], call.get("arguments", {}))
            votes[h] += 1
            hash_to_call.setdefault(h, call)
            # Early exit: unanimous after at least 2 samples
            if sum(votes.values()) >= 2 and len(votes) == 1:
                break

        if not votes:
            return None, {}, 0

        winner_hash = votes.most_common(1)[0][0]
        used_samples = sum(votes.values())
        return hash_to_call[winner_hash], dict(votes), used_samples

    def _sweep_success_checks_after_edit(self, child: SearchNode) -> None:
        """Run all unsatisfied success checks immediately after an edit node is created.

        This gives UCB1 empirical pass/fail signal on the edit within the same
        iteration rather than waiting for the model to run the check itself.
        Called only for edit-branch children, not for linear (non-edit) nodes.
        """
        if not child.loop_state.executable_edit_made:
            return
        for check in self.success_validation_checks:
            check_name = str(check.get("name", check.get("command", "")))
            if check_name in child.loop_state.satisfied_success_checks:
                continue
            cmd = str(check.get("command", "")).strip()
            if not cmd:
                continue
            try:
                chk_out = self.env.communicate(
                    input=f"cd {shlex.quote(self.repo_root)}\n{cmd}",
                    check="ignore", timeout=60,
                )
                exit_raw = self.env.communicate(
                    input="echo $?", check="ignore", timeout=5,
                ).strip()
                chk_exit = int(exit_raw) if exit_raw.isdigit() else 1
            except Exception:
                continue
            passed = _command_output_satisfies_check(check, exit_code=chk_exit, output=chk_out)
            if passed:
                child.loop_state.satisfied_success_checks.add(check_name)
                self._log(f"[mcts] post-edit check '{check_name}' PASSED")
            else:
                self._log(f"[mcts] post-edit check '{check_name}' failed")
            child.loop_state.validation_attempted_after_edit = True

    def _is_ready_for_finalization(self, node: SearchNode) -> bool:
        if not node.loop_state.executable_edit_made:
            return False
        if not node.loop_state.validation_attempted_after_edit:
            return False
        required = len(self.success_validation_checks)
        if required > 0 and len(node.loop_state.satisfied_success_checks) < required:
            return False
        # Don't auto-finalize if the patch touches test files (policy violation)
        if any(
            ln.startswith("+++ b/") and ("test" in ln.lower() or "/spec/" in ln)
            for ln in node.git_diff.splitlines()
        ):
            return False
        return True

    def _adaptive_edit_policy(self, node: SearchNode) -> tuple[int, int]:
        """Return (vote_samples, candidate_limit) for the current node.

        During post-edit stagnation, increase exploration to diversify edits.
        """
        vote_samples = self.edit_vote_samples
        candidate_limit = EXPANSION_CANDIDATES
        if not self.adaptive_branching:
            return vote_samples, candidate_limit

        stagnating = (
            node.loop_state.executable_edit_made
            and node.post_edit_no_progress_turns >= 2
        )
        if stagnating:
            vote_samples = min(max(self.edit_vote_samples + 2, 3), 9)
            candidate_limit = max(EXPANSION_CANDIDATES, 2)
        return vote_samples, candidate_limit

    def _serialize_tree(self, root: SearchNode, result_node: SearchNode) -> dict[str, Any]:
        """Walk the full tree and produce a JSON-serialisable dict for the traj.

        Marks the root→result_node path with ``on_result_path=True``.
        """
        # Build the result path set for quick lookup
        result_path: set[int] = set()
        cur: SearchNode | None = result_node
        while cur is not None:
            result_path.add(id(cur))
            cur = cur.parent

        nodes: list[dict[str, Any]] = []
        id_map: dict[int, str] = {}

        def walk(n: SearchNode, label: str, parent_label: str | None) -> None:
            id_map[id(n)] = label
            args = n.action.get("arguments", {}) if n.action else {}
            args_preview = json.dumps(args)[:80] if args else ""
            nodes.append({
                "id": label,
                "parent_id": parent_label,
                "depth": n.depth,
                "action_name": n.action["name"] if n.action else None,
                "action_args_preview": args_preview,
                "visits": n.visits,
                "mean_value": round(n.mean_value, 2),
                "submitted": n.submitted,
                "is_branch_point": n.is_branch_point,
                "edit_made": n.loop_state.executable_edit_made,
                "success_checks": len(n.loop_state.satisfied_success_checks),
                "check_progress_gain": n.check_progress_gain,
                "post_edit_no_progress_turns": n.post_edit_no_progress_turns,
                "has_nonempty_patch": n.has_nonempty_patch,
                "stopped_reason": n.stopped_reason,
                "vote_counts": n.vote_counts,
                "on_result_path": id(n) in result_path,
            })
            for i, child in enumerate(n.children):
                walk(child, f"{label}.{i}", label)

        walk(root, "0", None)
        return {
            "nodes": nodes,
            "result_node_id": id_map.get(id(result_node), "0"),
        }

    # ------------------------------------------------------------------
    # Model call
    # ------------------------------------------------------------------

    def _format_messages_verbose(self, messages: list[dict], last_n: int = 4) -> str:
        """Format the tail of a message list for verbose output."""
        lines = [f"  [context tail: last {min(last_n, len(messages))} of {len(messages)} messages]"]
        for msg in messages[-last_n:]:
            role = msg.get("role", "?")
            if role == "assistant":
                calls = msg.get("tool_calls", [])
                if calls:
                    fn = calls[0].get("function", {})
                    lines.append(f"  assistant → tool_call: {fn.get('name','?')}({fn.get('arguments','')[:120]})")
                else:
                    lines.append(f"  assistant: {(msg.get('content') or '')[:200]}")
            elif role == "tool":
                out = str(msg.get("content", ""))
                lines.append(f"  tool({msg.get('name','?')}): {out[:300]}{'…' if len(out) > 300 else ''}")
            elif role == "user":
                lines.append(f"  user: {str(msg.get('content',''))[:200]}")
            elif role == "system":
                lines.append(f"  system: <{len(str(msg.get('content','')))} chars>")
        return "\n".join(lines)

    def _call_model(self, messages: list[dict], temperature: float) -> tuple[str, int, int]:
        """Call the model and return (content, tokens_in, tokens_out)."""
        import litellm
        litellm.suppress_debug_info = True
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(self.max_tokens, MAX_TOKENS_REACT),
        }
        # reasoning_effort is only meaningful for Ollama reasoning models
        if self.model.startswith("ollama/"):
            kwargs["reasoning_effort"] = "none"
        if self.num_ctx is not None:
            kwargs["num_ctx"] = self.num_ctx
        response = litellm.completion(**kwargs)
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        content = (response.choices[0].message.content or "")
        return content, tokens_in, tokens_out

    def _parse_action(self, content: str) -> ParseOutcome:
        """Re-use the react_json parser from the parent module."""
        from run_custom_swebench import CustomAgentLoop
        return CustomAgentLoop._parse_react_json(content)

    def _hash_action(self, name: str, arguments: dict) -> str:
        return json.dumps({"name": name, "arguments": arguments}, sort_keys=True)

    def _execute_action(
        self,
        parent: SearchNode,
        action: dict[str, Any],
        temperature_label: str,
        tool_runtime: ToolRuntime,
    ) -> SearchNode:
        """Execute one action and return a new child SearchNode."""
        from run_custom_swebench import CustomAgentLoop

        name = action["name"]
        arguments = action.get("arguments", {})
        self._turn_counter += 1
        tool_call_id = f"ts-{name}-{self._turn_counter}"

        # Carry loop-detection state forward
        new_repeat_hashes = list(parent.repeat_hashes)
        new_repeat_hashes.append(self._hash_action(name, arguments))

        # Submit pre-check (mirror of CustomAgentLoop._submit_precheck)
        if name == "submit":
            # Auto-run any unsatisfied success checks before the precheck.
            # The agent may have run an equivalent command (e.g. pytest on one
            # file) that passed, but not the exact command string the check
            # requires for credit.  Running the check commands here means the
            # agent is never stuck in a "tests pass but submit blocked" loop.
            probe_ls = copy.deepcopy(parent.loop_state)
            for check in self.success_validation_checks:
                check_name = str(check.get("name", check.get("command", "")))
                if check_name in probe_ls.satisfied_success_checks:
                    continue
                cmd = str(check.get("command", "")).strip()
                if not cmd:
                    continue
                try:
                    chk_out = self.env.communicate(
                        input=f"cd {shlex.quote(self.repo_root)}\n{cmd}",
                        check="ignore",
                        timeout=60,
                    )
                    exit_raw = self.env.communicate(
                        input="echo $?", check="ignore", timeout=5
                    ).strip()
                    chk_exit = int(exit_raw) if exit_raw.isdigit() else 1
                except Exception:
                    continue
                if _command_output_satisfies_check(check, exit_code=chk_exit, output=chk_out):
                    probe_ls.satisfied_success_checks.add(check_name)
                    self._log(f"[mcts] auto-check '{check_name}' passed")
                else:
                    self._log(f"[mcts] auto-check '{check_name}' FAILED (exit={chk_exit})")

            if (
                parent.action
                and parent.action.get("name") == "submit"
                and len(probe_ls.satisfied_success_checks) <= len(parent.loop_state.satisfied_success_checks)
            ):
                result = ToolExecutionRecord(
                    name="submit",
                    arguments=arguments,
                    output=(
                        "ERROR: Repeated submit with no additional success-check progress. "
                        "Do not submit again yet; make a substantive fix and rerun required validations."
                    ),
                    is_error=True,
                )
            else:
                block = self._submit_precheck(probe_ls, tool_runtime)
                if block:
                    result = ToolExecutionRecord(
                        name="submit", arguments=arguments, output=block, is_error=True
                    )
                else:
                    result = tool_runtime.execute(name, arguments)
        else:
            probe_ls = parent.loop_state
            # Detect no-op str_replace before executing: old_str == new_str means
            # the model generated identical content and the file will not change.
            # Return a synthetic error immediately so the model is forced to
            # reconsider, and so executable_edit_made is NOT set True.
            if name in ("str_replace", "replace"):
                old_s = str(arguments.get("old_str") or arguments.get("old_string", "")).strip()
                new_s = str(arguments.get("new_str") or arguments.get("new_string", "")).strip()
                if old_s and old_s == new_s:
                    result = ToolExecutionRecord(
                        name=name, arguments=arguments, is_error=True,
                        output=(
                            "ERROR: No-op edit — new_str is identical to old_str. "
                            "The file was NOT changed. You must provide different content "
                            "in new_str to actually fix the bug. "
                            "Re-read the file, identify the exact line(s) to change, "
                            "and write a new_str that differs from old_str."
                        ),
                    )
                else:
                    result = tool_runtime.execute(name, arguments)
            else:
                result = tool_runtime.execute(name, arguments)

        # Loop warning
        recent = new_repeat_hashes[-4:]
        loop_warn: str | None = None
        if len(recent) == 4 and len(set(recent)) == 1:
            # Count how many consecutive times this exact action has been repeated
            total_repeat = 0
            for h in reversed(new_repeat_hashes):
                if h == recent[0]:
                    total_repeat += 1
                else:
                    break
            if name == "bash":
                cmd_preview = str(arguments.get("command", arguments.get("cmd", "")))[:80]
                loop_warn = (
                    f"STOP: You have run `{cmd_preview}` {total_repeat} times in a row "
                    f"and the output is identical every time. Running it again will not help. "
                    f"You must take a different action: read another file, make an edit, "
                    f"or run a different command."
                )
            elif name in ("str_replace", "insert", "replace"):
                loop_warn = (
                    f"STOP: You have attempted the same edit {total_repeat} times. "
                    f"Your edits are not changing the file or not fixing the issue. "
                    f"Re-read the file carefully and identify what specific line must change."
                )
            else:
                loop_warn = (
                    f"STOP: You have called `{name}` {total_repeat} times in a row "
                    f"with the same arguments. The result will not change. "
                    f"Try something completely different."
                )

        # Update loop state — start from probe_ls so auto-credited checks persist
        new_ls = copy.deepcopy(probe_ls)
        self._update_loop_state(new_ls, result)

        # Build new message history
        new_messages = list(parent.messages)
        # assistant turn
        new_messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        })
        # tool result turn
        new_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": result.output,
        })
        if loop_warn:
            new_messages.append({"role": "user", "content": loop_warn})
        elif (
            not new_ls.executable_edit_made
            and (parent.depth + 1) >= 6
            and name in {"view", "bash"}
        ):
            new_messages.append({
                "role": "user",
                "content": (
                    "STOP: You have spent several turns on discovery without making an edit. "
                    "Based on current evidence, propose and apply one concrete code edit now, "
                    "then run the required success validation command."
                ),
            })
        elif (
            name in {"str_replace", "insert"}
            and not parent.loop_state.executable_edit_made
            and self.success_validation_checks
        ):
            # First edit in this branch: remind the model to validate with the
            # exact check command so it observes actual vs expected output.
            cmds = [
                str(c.get("command", "")).strip()
                for c in self.success_validation_checks
                if str(c.get("command", "")).strip()
            ]
            if cmds:
                new_messages.append({
                    "role": "user",
                    "content": (
                        "Now run the validation to verify this edit produced the correct output: "
                        + "; ".join(cmds[:2])
                    ),
                })

        # Verbose: show tool execution result
        if self.verbose:
            args_preview = json.dumps(arguments)[:200]
            out_preview = result.output[:500] + ("…" if len(result.output) > 500 else "")
            err_tag = " [ERROR]" if result.is_error else ""
            self._vlog(
                f"[verbose] TOOL CALL: {name}({args_preview}){err_tag}\n"
                f"[verbose] TOOL RESULT:\n{out_preview}"
            )

        # Capture git state after edit
        git_diff = self._capture_git_diff()
        patch_line_count = sum(
            1
            for ln in git_diff.splitlines()
            if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
        )
        check_progress_gain = max(
            0,
            len(new_ls.satisfied_success_checks) - len(parent.loop_state.satisfied_success_checks),
        )
        post_edit_no_progress_turns = parent.post_edit_no_progress_turns
        if new_ls.executable_edit_made:
            if check_progress_gain > 0:
                post_edit_no_progress_turns = 0
            else:
                post_edit_no_progress_turns = parent.post_edit_no_progress_turns + 1

        # Update failure counters
        new_fail = dict(parent.consecutive_failure_counts)
        new_succ = dict(parent.consecutive_success_counts)
        fail_key = self._hash_action(name, arguments) + result.output[:80]
        if result.is_error:
            new_fail[fail_key] = new_fail.get(fail_key, 0) + 1
            new_succ.clear()
        else:
            new_fail.clear()
            new_succ[fail_key] = new_succ.get(fail_key, 0) + 1

        # Build turn record for trajectory export
        turn_rec = TurnRecord(
            turn=parent.depth + 1,
            assistant_text="",
            tool_calls=[{"name": name, "arguments": arguments}],
            tool_results=[result],
            finish_reason=None,
            parse_error=None,
        )

        child = SearchNode(
            parent=parent,
            depth=parent.depth + 1,
            git_diff=git_diff,
            loop_state=new_ls,
            messages=new_messages,
            action=action,
            observation=result.output,
            submitted=tool_runtime.submitted,
            submit_summary=tool_runtime.submit_summary,
            stopped_reason="submitted" if tool_runtime.submitted else "",
            consecutive_failure_counts=new_fail,
            consecutive_success_counts=new_succ,
            repeat_hashes=new_repeat_hashes,
            turn_records=list(parent.turn_records) + [turn_rec],
            parse_errors=parent.parse_errors,
            check_progress_gain=check_progress_gain,
            post_edit_no_progress_turns=post_edit_no_progress_turns,
            has_nonempty_patch=patch_line_count > 0,
        )
        return child

    def _expand(self, node: SearchNode, tool_runtime: ToolRuntime) -> list[SearchNode]:
        """Expand a node using selective branching.

        Strategy:
        - Apply context compression to the message history before calling the model.
        - Call the model once at EXPLOIT_TEMPERATURE to determine intent.
        - If the intended action is NOT an edit (bash, view, submit, …): execute
          linearly — one child, no branching.  This preserves the search budget.
        - If the intended action IS an edit (str_replace / insert): run majority
          vote across EDIT_VOTE_SAMPLES samples, then execute the top
          EXPANSION_CANDIDATES unique edits as separate children.
        """
        self._log(f"[mcts] expanding depth={node.depth} visits={node.visits}")

        # Apply context compression to avoid context-length degradation
        raw_messages = self._base_messages() if not node.messages else list(node.messages)
        if node.parse_errors > 0:
            raw_messages.append({
                "role": "user",
                "content": (
                    "Your previous response could not be parsed. "
                    "Respond with exactly one JSON action object and nothing else."
                ),
            })
        messages = self._compress_messages(raw_messages, node.depth)

        # ------------------------------------------------------------------
        # 1. Probe with one exploit-temperature call to detect intent
        # ------------------------------------------------------------------
        if self.verbose:
            self._vlog(
                f"[verbose] PROMPT (depth={node.depth}, temp={EXPLOIT_TEMPERATURE}):\n"
                + self._format_messages_verbose(messages)
            )
        try:
            probe_content, tok_in, tok_out = self._call_model(messages, EXPLOIT_TEMPERATURE)
        except Exception as exc:
            self._log(f"[mcts] probe call failed: {exc}")
            child = SearchNode(
                parent=node, depth=node.depth + 1, git_diff=node.git_diff,
                loop_state=copy.deepcopy(node.loop_state), messages=messages,
                action=None, observation="(model call failed)", stopped_reason="model_error",
                parse_errors=node.parse_errors + 1,
            )
            node.children.append(child)
            return [child]

        self._global_stats.input_tokens += tok_in
        self._global_stats.output_tokens += tok_out
        self._global_stats.tool_calls += 1

        if self.verbose:
            self._vlog(f"[verbose] RESPONSE:\n{probe_content[:600]}{'…' if len(probe_content) > 600 else ''}")

        probe_outcome = self._parse_action(probe_content)
        probe_call = probe_outcome.tool_calls[0] if probe_outcome.tool_calls else None

        if probe_call is None:
            self._log(f"[mcts] parse failed: {probe_outcome.error}")
            # Inline retry on first failure — the correction node in _expand()
            # is never reached because parse-fail nodes are terminal. Give the
            # model one chance to self-correct with the specific error.
            if node.parse_errors == 0:
                correction_messages = list(messages) + [{
                    "role": "user",
                    "content": (
                        f"Your response could not be parsed as JSON: {probe_outcome.error}\n"
                        "Respond with exactly one JSON action object and nothing else. "
                        "No prose, no markdown, no explanation — only the JSON object."
                    ),
                }]
                try:
                    retry_content, r_in, r_out = self._call_model(correction_messages, 0.5)
                    self._global_stats.input_tokens += r_in
                    self._global_stats.output_tokens += r_out
                    self._global_stats.tool_calls += 1
                    retry_outcome = self._parse_action(retry_content)
                    probe_call = retry_outcome.tool_calls[0] if retry_outcome.tool_calls else None
                    if probe_call is not None:
                        self._log("[mcts] parse retry succeeded")
                    else:
                        self._log(f"[mcts] parse retry also failed: {retry_outcome.error}")
                except Exception as exc:
                    self._log(f"[mcts] parse retry call failed: {exc}")

            if probe_call is None:
                child = SearchNode(
                    parent=node, depth=node.depth + 1, git_diff=node.git_diff,
                    loop_state=copy.deepcopy(node.loop_state), messages=messages,
                    action=None, observation="(no parseable action produced)",
                    stopped_reason="parse_failure", parse_errors=node.parse_errors + 1,
                )
                node.children.append(child)
                return [child]

        # ------------------------------------------------------------------
        # 2a. Non-edit action → linear execution (single child, no branching)
        # ------------------------------------------------------------------
        if not self._is_edit_action(probe_call):
            self._restore_state(node)
            tr = ToolRuntime(self.env, self.repo_root)
            child = self._execute_action(node, probe_call, "linear", tr)
            node.children.append(child)
            self._log(
                f"[mcts] linear  tool={probe_call['name']} "
                f"depth={child.depth} submitted={tr.submitted}"
            )
            return [child]

        # ------------------------------------------------------------------
        # 2b. Edit action → majority vote, then execute top-K unique edits
        # ------------------------------------------------------------------
        vote_samples, candidate_limit = self._adaptive_edit_policy(node)
        self._log(
            f"[mcts] edit detected — running majority vote (max={vote_samples}), "
            f"candidates={candidate_limit}"
        )
        winner, vote_counts, vote_used_samples = self._majority_vote_edit(
            messages, vote_samples, seed_call=probe_call
        )
        if winner is None:
            winner = probe_call

        # Build candidate list: winner first, then top runner-up (if EXPANSION_CANDIDATES > 1)
        # winner is first; add a runner-up explore sample if EXPANSION_CANDIDATES > 1
        candidates: list[dict[str, Any]] = [winner]
        seen_hashes: set[str] = {self._hash_action(winner["name"], winner.get("arguments", {}))}
        if candidate_limit > 1:
            try:
                explore_content, et_in, et_out = self._call_model(messages, EXPLORE_TEMPERATURE)
                self._global_stats.input_tokens += et_in
                self._global_stats.output_tokens += et_out
                explore_outcome = self._parse_action(explore_content)
                if explore_outcome.tool_calls:
                    ec = explore_outcome.tool_calls[0]
                    eh = self._hash_action(ec["name"], ec.get("arguments", {}))
                    if eh not in seen_hashes and self._is_edit_action(ec):
                        candidates.append(ec)
                        seen_hashes.add(eh)
            except Exception:
                pass
        candidates = candidates[:candidate_limit]

        children: list[SearchNode] = []
        for call in candidates:
            self._restore_state(node)
            tr = ToolRuntime(self.env, self.repo_root)
            child = self._execute_action(node, call, "edit_branch", tr)
            child.is_branch_point = True
            child.vote_counts = vote_counts if call is winner else {}
            child.vote_total_samples = vote_used_samples if call is winner else 0
            # Run success checks immediately so UCB1 gets empirical signal this
            # iteration rather than waiting for the model to run them itself.
            if self.success_validation_checks and not child.is_terminal:
                self._sweep_success_checks_after_edit(child)
            node.children.append(child)
            children.append(child)
            total_samples = vote_used_samples
            winner_votes = vote_counts.get(self._hash_action(call["name"], call.get("arguments", {})), "?")
            vote_info = f" votes={winner_votes}/{total_samples}" if call is winner else ""
            self._log(
                f"[mcts] edit    tool={call['name']} "
                f"depth={child.depth} submitted={tr.submitted}{vote_info}"
            )

        return children

    # ------------------------------------------------------------------
    # Submit pre-check (mirrors CustomAgentLoop._submit_precheck)
    # ------------------------------------------------------------------

    def _submit_precheck(self, ls: LoopState, tr: ToolRuntime) -> str | None:
        if not ls.executable_edit_made:
            return "Do not submit yet. No code edit has been made."
        patch = self.env.communicate(
            input=f"cd {shlex.quote(self.repo_root)}\ngit diff --no-color",
            timeout=20,
            check="ignore",
        )
        if not (patch or "").strip():
            return "Do not submit yet. There is no patch in git diff."
        if not ls.validation_attempted_after_edit:
            return "Do not submit yet. Run at least one validation command after your edit."
        if self.success_validation_checks:
            missing = [
                str(c.get("name", c.get("command", "")))
                for c in self.success_validation_checks
                if str(c.get("name", c.get("command", ""))) not in ls.satisfied_success_checks
            ]
            if missing:
                # Include the exact command(s) the agent must run so it is
                # never stuck wondering how to satisfy the check.
                missing_set = set(missing)
                cmds = [
                    str(c.get("command", "")).strip()
                    for c in self.success_validation_checks
                    if str(c.get("name", c.get("command", ""))) in missing_set
                    and c.get("command", "").strip()
                ]
                hint = f" — run: {'; '.join(cmds)}" if cmds else ""
                return (
                    f"Do not submit yet. Required success checks not yet passed: "
                    f"{'; '.join(missing[:2])}{hint}"
                )
        if not ls.diff_checked:
            return "Do not submit yet. Inspect `git diff` first."
        return None

    # ------------------------------------------------------------------
    # LoopState update (mirrors CustomAgentLoop._update_state_from_tool_result)
    # ------------------------------------------------------------------

    def _update_loop_state(self, ls: LoopState, result: ToolExecutionRecord) -> None:
        if result.is_error:
            return
        if result.name in {"str_replace", "insert", "undo_edit"}:
            ls.executable_edit_made = True
            path = str(result.arguments.get("path", "")).strip()
            if path:
                ls.changed_files.add(path)
        if result.name == "bash":
            if ls.executable_edit_made:
                ls.validation_attempted_after_edit = True
                cmd = str(result.arguments.get("command", "")).strip()
                if cmd and result.exit_code == 0:
                    ls.successful_post_edit_commands.add(cmd)
                if cmd:
                    for check in self.success_validation_checks:
                        if str(check.get("command", "")).strip() != cmd:
                            continue
                        check_name = str(check.get("name", cmd))
                        if _command_output_satisfies_check(check, exit_code=result.exit_code, output=result.output):
                            ls.satisfied_success_checks.add(check_name)
            lowered = result.output.lower()
            if "passed" in lowered and "failed" not in lowered:
                ls.validation_passed = True
            if "diff --git" in result.output:
                ls.diff_checked = True

    # ------------------------------------------------------------------
    # Best-path extraction
    # ------------------------------------------------------------------

    def _best_terminal(self, root: SearchNode) -> SearchNode | None:
        """Return the best terminal node (submitted, success checks satisfied)."""
        best: SearchNode | None = None
        best_val = -1.0

        def walk(n: SearchNode) -> None:
            nonlocal best, best_val
            if n.submitted:
                v = self._estimate_value(n)
                if v > best_val:
                    best_val = v
                    best = n
            for c in n.children:
                walk(c)

        walk(root)
        return best

    def _deepest_with_edit(self, root: SearchNode) -> SearchNode:
        """Fallback: return the most-visited leaf that at least made an edit."""
        candidates: list[SearchNode] = []

        def walk(n: SearchNode) -> None:
            if not n.children:
                candidates.append(n)
            for c in n.children:
                walk(c)

        walk(root)
        if not candidates:
            return root
        # Prefer nodes that made edits, break ties by value
        candidates.sort(key=lambda n: (n.loop_state.executable_edit_made, self._estimate_value(n)), reverse=True)
        return candidates[0]

    def _best_nonempty_patch_leaf(self, root: SearchNode) -> SearchNode | None:
        """Best leaf that preserves a non-empty patch and at least one edit."""
        candidates: list[SearchNode] = []

        def walk(n: SearchNode) -> None:
            if not n.children:
                if n.loop_state.executable_edit_made and n.has_nonempty_patch:
                    candidates.append(n)
            for c in n.children:
                walk(c)

        walk(root)
        if not candidates:
            return None
        candidates.sort(key=lambda n: (self._estimate_value(n), n.depth), reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Public run() entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        import litellm
        litellm.suppress_debug_info = True

        # Root node: clean state, just the base messages
        root = SearchNode(
            parent=None,
            depth=0,
            git_diff="",
            loop_state=LoopState(),
            messages=self._base_messages(),
        )

        self._log(f"[mcts] starting model={self.model} iterations={MAX_ITERATIONS} k={EXPANSION_CANDIDATES}")
        start = time.time()
        best_submitted: SearchNode | None = None
        best_ready: SearchNode | None = None

        for iteration in range(MAX_ITERATIONS):
            self._log(f"[mcts] iteration {iteration + 1}/{MAX_ITERATIONS}")

            # Selection
            leaf = self._select(root)
            if leaf.is_terminal:
                # Already terminal — update best if submitted
                if leaf.submitted and best_submitted is None:
                    best_submitted = leaf
                v = self._estimate_value(leaf)
                self._backpropagate(leaf, v)
                continue

            # Restore environment to the leaf's git state
            self._restore_state(leaf)

            # Expansion: create child nodes
            tr = ToolRuntime(self.env, self.repo_root)
            children = self._expand(leaf, tr)

            for child in children:
                v = self._estimate_value(child)
                self._backpropagate(child, v)
                self._log(
                    f"[mcts] child depth={child.depth} value={v:.2f} "
                    f"submitted={child.submitted} checks={len(child.loop_state.satisfied_success_checks)}"
                )
                if child.submitted and (
                    best_submitted is None
                    or self._estimate_value(child) > self._estimate_value(best_submitted)
                ):
                    best_submitted = child
                if self._is_ready_for_finalization(child) and (
                    best_ready is None
                    or self._estimate_value(child) > self._estimate_value(best_ready)
                ):
                    best_ready = child

            # Early exit if we found a fully-passing submission
            if best_submitted and len(best_submitted.loop_state.satisfied_success_checks) >= len(self.success_validation_checks):
                self._log(f"[mcts] all success checks satisfied at iteration {iteration + 1}; stopping early")
                break
            if best_ready is not None:
                self._log(
                    f"[mcts] all success checks satisfied without explicit submit at iteration {iteration + 1}; stopping early"
                )
                break

        elapsed = round(time.time() - start, 2)
        self._log(f"[mcts] search finished in {elapsed}s")

        # Choose the result node
        result_node = best_submitted or best_ready or self._best_terminal(root) or self._deepest_with_edit(root)
        if result_node is root:
            result_node.stopped_reason = "no_progress"
        if (
            not result_node.submitted
            and result_node.depth >= 8
            and not result_node.has_nonempty_patch
        ):
            alt = self._best_nonempty_patch_leaf(root)
            if alt is not None and alt is not result_node:
                self._log("[mcts] empty-patch guardrail: selecting best non-empty patch leaf")
                result_node = alt
                if not result_node.stopped_reason:
                    result_node.stopped_reason = "empty_patch_guardrail"

        # Restore the container to the result node's state for patch capture
        self._restore_state(result_node)
        patch = self.env.communicate(
            input=f"git -C {shlex.quote(self.repo_root)} diff --no-color",
            timeout=30,
            check="ignore",
        ) or ""
        if (
            not result_node.submitted
            and self._is_ready_for_finalization(result_node)
            and patch.strip()
        ):
            result_node.submitted = True
            result_node.auto_finalized = True
            result_node.submit_summary = (
                "Auto-finalized: required success checks were satisfied with a non-empty patch."
            )
            result_node.stopped_reason = "auto_finalized"
            self._log("[mcts] auto-finalized solved branch without explicit submit")

        # Build flattened turn list across the result path
        all_turns = [asdict(t) for t in result_node.turn_records]

        # vote_summary: one entry per edit branch point on the result path
        vote_summary = []
        cur: SearchNode | None = result_node
        path_nodes: list[SearchNode] = []
        while cur is not None:
            path_nodes.append(cur)
            cur = cur.parent
        path_nodes.reverse()
        for pn in path_nodes:
            if pn.is_branch_point and pn.vote_counts:
                vote_summary.append({
                    "depth": pn.depth,
                    "action": pn.action["name"] if pn.action else None,
                    "winner_votes": max(pn.vote_counts.values()),
                    "total_samples": pn.vote_total_samples or sum(pn.vote_counts.values()),
                    "unique_candidates": len(pn.vote_counts),
                })

        return {
            "turns": all_turns,
            "stats": {
                "input_tokens": self._global_stats.input_tokens,
                "output_tokens": self._global_stats.output_tokens,
                "turns": result_node.depth,
                "tool_calls": self._global_stats.tool_calls,
                "iterations": MAX_ITERATIONS,
                "tree_nodes_created": self._count_nodes(root),
                "best_node_depth": result_node.depth,
                "best_node_value": self._estimate_value(result_node),
            },
            "loop_state": {
                **asdict(result_node.loop_state),
                "successful_post_edit_commands": sorted(result_node.loop_state.successful_post_edit_commands),
                "satisfied_success_checks": sorted(result_node.loop_state.satisfied_success_checks),
                "changed_files": sorted(result_node.loop_state.changed_files),
            },
            "stopped_reason": result_node.stopped_reason or "max_iterations",
            "submitted": result_node.submitted,
            "submission_summary": result_node.submit_summary,
            "patch": patch,
            "info": {
                "submission": patch if result_node.submitted else "",
                "submitted": result_node.submitted,
                "submission_summary": result_node.submit_summary,
                "stopped_reason": result_node.stopped_reason or "max_iterations",
            },
            "auto_finalized": result_node.auto_finalized,
            "mcts_meta": {
                "model": self.model,
                "iterations": MAX_ITERATIONS,
                "expansion_candidates": EXPANSION_CANDIDATES,
                "edit_vote_samples": self.edit_vote_samples,
                "ucb_c": UCB_C,
                "root_visits": root.visits,
            },
            "vote_summary": vote_summary,
            "mcts_tree": self._serialize_tree(root, result_node),
        }

    def _count_nodes(self, node: SearchNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    global MAX_ITERATIONS, EXPANSION_CANDIDATES, EXPLORE_TEMPERATURE, UCB_C, MAX_NODE_DEPTH, EDIT_VOTE_SAMPLES
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    parser.add_argument("--model", default=DEFAULT_CODER_MODEL,
                        help=f"Ollama coder model (default: {DEFAULT_CODER_MODEL})")
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL,
                        help=f"Ollama planner model (default: {DEFAULT_PLANNER_MODEL})")
    parser.add_argument("--reviewer-model", default=None,
                        help="Ollama reviewer model (default: same as planner)")
    parser.add_argument("--api-base", default=OLLAMA_API_BASE)
    parser.add_argument("--api-key", default=OLLAMA_API_KEY)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=EXPLOIT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_REACT)

    # MCTS hyper-parameters
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS,
                        help="Total MCTS iterations (tree expansions)")
    parser.add_argument("--expansion-candidates", type=int, default=EXPANSION_CANDIDATES,
                        help="Branch candidates at edit-intent turns")
    parser.add_argument("--explore-temperature", type=float, default=EXPLORE_TEMPERATURE,
                        help="Temperature for exploration candidates")
    parser.add_argument("--ucb-c", type=float, default=UCB_C,
                        help="UCB1 exploration constant")
    parser.add_argument("--max-node-depth", type=int, default=MAX_NODE_DEPTH,
                        help="Max turns per branch before treating as terminal")
    parser.add_argument("--edit-vote-samples", type=int, default=EDIT_VOTE_SAMPLES,
                        help="Majority-vote samples per edit action (0 to disable)")

    # Architecture
    parser.add_argument("--agent-architecture",
                        choices=["single", "planner_coder", "planner_coder_reviewer"],
                        default="planner_coder")
    parser.add_argument("--reviewer-rounds", type=int, default=1)
    parser.add_argument("--reviewer-gate-mode", choices=["strict", "soft"], default="soft",
                        help="strict: any reviewer revise rejects; soft: only hard evidence blocks")
    parser.add_argument("--max-identical-tool-failures", type=int, default=3)
    parser.add_argument(
        "--adaptive-branching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive edit vote/candidate expansion during post-edit stagnation",
    )

    # Instances
    parser.add_argument("--instances-type", choices=["swe_bench", "file"], default="swe_bench")
    parser.add_argument("--instances-path", type=Path)
    parser.add_argument("--subset",
                        choices=["lite", "verified", "full", "multimodal", "multilingual"],
                        default="full")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--filter", default=".*")
    parser.add_argument("--slice", default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--post-startup-command", action="append", default=[])

    # Output
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="mcts_ollama")
    parser.add_argument("--resume", action="store_true",
                        help="Skip instances that already have a .traj file in output-dir")
    parser.add_argument("--verbose", action="store_true",
                        help="Print model prompts and responses to stdout during execution")

    args = parser.parse_args()

    # Apply global MCTS hyper-parameters from CLI
    MAX_ITERATIONS = args.iterations
    EXPANSION_CANDIDATES = args.expansion_candidates
    EXPLORE_TEMPERATURE = args.explore_temperature
    UCB_C = args.ucb_c
    MAX_NODE_DEPTH = args.max_node_depth
    EDIT_VOTE_SAMPLES = args.edit_vote_samples

    if args.reviewer_model is None:
        args.reviewer_model = args.planner_model

    return args


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _build_instances(args: argparse.Namespace):
    if args.instances_type == "file":
        if args.instances_path is None:
            raise ValueError("--instances-path is required when --instances-type=file")
        # Use local loader that promotes top-level evaluation/analysis/policy keys
        return _load_custom_file_instances_local(args.instances_path, args.filter, args.slice, args.shuffle)
    source = SWEBenchInstances(
        subset=args.subset,
        split=args.split,
        filter=args.filter,
        slice=args.slice,
        shuffle=args.shuffle,
        evaluate=False,
    )
    return source.get_instance_configs()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _normalize_model_name("ollama", args.model)
    planner_model = _normalize_model_name("ollama", args.planner_model)
    reviewer_model = _normalize_model_name("ollama", args.reviewer_model)

    run_config = {
        "runner": "mcts",
        "model": model,
        "planner_model": planner_model,
        "reviewer_model": reviewer_model,
        "api_base": args.api_base,
        "agent_architecture": args.agent_architecture,
        "reviewer_gate_mode": args.reviewer_gate_mode,
        "adaptive_branching": bool(args.adaptive_branching),
        "mcts": {
            "iterations": MAX_ITERATIONS,
            "expansion_candidates": EXPANSION_CANDIDATES,
            "explore_temperature": EXPLORE_TEMPERATURE,
            "exploit_temperature": EXPLOIT_TEMPERATURE,
            "ucb_c": UCB_C,
            "max_node_depth": MAX_NODE_DEPTH,
        },
        "filter": args.filter,
        "instances_type": args.instances_type,
        "instances_path": str(args.instances_path) if args.instances_path else "",
        "run_name": args.run_name,
    }
    _dump_yaml(output_dir / "run_batch.config.yaml", run_config)

    instances = _build_instances(args)
    all_preds: dict[str, Any] = {}

    for instance in instances:
        instance_id = instance.problem_statement.id
        instance_dir = output_dir / instance_id

        if args.resume and (instance_dir / f"{instance_id}.traj").exists():
            print(f"[resume] skipping {instance_id} — .traj already exists", flush=True)
            # Reload prediction so preds.json stays consistent when resuming
            pred_path = instance_dir / f"{instance_id}.pred"
            if pred_path.exists():
                try:
                    all_preds[instance_id] = json.loads(pred_path.read_text())
                except Exception:
                    pass
            continue

        instance_dir.mkdir(parents=True, exist_ok=True)
        info_log_path = instance_dir / f"{instance_id}.info.log"

        def log_line(msg: str, *, _path: Path = info_log_path, _iid: str = instance_id) -> None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{timestamp} {_iid} {msg}"
            print(line, flush=True)
            with _path.open("a") as fh:
                fh.write(line + "\n")

        start = time.time()
        env = _build_env(instance)

        try:
            log_line("[env] starting")
            _start_env_with_retries(env, log_line)
            repo_root = _instance_repo_root(instance)
            env.communicate(f"cd {shlex.quote(repo_root)}", check="ignore", timeout=10)
            # Ensure git repo exists inside container
            env.communicate(
                input=(
                    f"cd {shlex.quote(repo_root)}\n"
                    "if [ ! -d .git ]; then\n"
                    "  git init && git config user.email 'mcts@example.com'\n"
                    "  git config user.name 'MCTS Runner'\n"
                    "  git add . && git commit -m 'Initial snapshot' >/dev/null 2>&1 || true\n"
                    "fi\n"
                ),
                check="ignore",
                timeout=60,
            )
            # Case bootstrap
            install_cmds = instance.problem_statement.extra_fields.get("install_commands", [])  # type: ignore
            setup_cmds = instance.problem_statement.extra_fields.get("setup_commands", [])  # type: ignore
            for cmd in [*install_cmds, *setup_cmds, *args.post_startup_command]:
                log_line(f"[env] setup_command {cmd}")
                env.communicate(input=f"cd {shlex.quote(repo_root)}\n{cmd}", check="ignore", timeout=300)

            # Runtime context snapshot
            runtime_context = env.communicate(
                input=(
                    f"cd {shlex.quote(repo_root)}\n"
                    "printf 'PWD: '; pwd\n"
                    "printf '\\nFILES:\\n'; ls\n"
                    "printf '\\nPYTHON FILES (depth<=3):\\n'\n"
                    "find . -maxdepth 3 -type f -name '*.py' | sort\n"
                    "printf '\\nREADME HEAD:\\n'\n"
                    "if [ -f README.md ]; then sed -n '1,80p' README.md; fi\n"
                    "printf '\\nGIT STATUS:\\n'; git status --short || true\n"
                ),
                check="ignore",
                timeout=20,
            )
            log_line(f"[env] ready repo_root={repo_root}")

            case_validation_prompt = _build_case_validation_prompt(instance.problem_statement)
            case_analysis_prompt = _build_case_analysis_prompt(instance.problem_statement)
            case_evaluation = _extract_case_evaluation(instance.problem_statement)
            case_policy = _extract_case_policy(instance.problem_statement)
            success_cmds = _extract_case_success_commands(instance.problem_statement)
            success_checks = _extract_case_success_checks(instance.problem_statement)

            # ------------------------------------------------------------------
            # Planner (single-shot JSON call, same as custom runner)
            # ------------------------------------------------------------------
            role_model_stats: dict[str, Any] = {}
            planner_handoff: dict[str, Any] | None = None

            if args.agent_architecture in {"planner_coder", "planner_coder_reviewer"}:
                log_line(f"[planner] model={planner_model}")
                planner_handoff, planner_stats, _ = _call_json_role_local(
                    model=planner_model,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    # Use slim prompt: 5-key schema + example, fits small models
                    system_prompt=_build_planner_system_prompt_slim(repo_root),
                    user_prompt=_build_planner_task_prompt_slim(
                        instance.problem_statement.text, repo_root, runtime_context,
                        case_analysis=case_analysis_prompt,
                    ),
                    fallback_payload=_default_planner_handoff(),
                )
                planner_handoff = _normalize_planner_handoff(planner_handoff)
                role_model_stats["planner"] = {"model": planner_model, **planner_stats}
                log_line(f"[planner] handoff summary: {planner_handoff.get('root_cause_hypothesis', '')[:120]}")

            # ------------------------------------------------------------------
            # MCTS coder loop (possibly multiple reviewer rounds)
            # ------------------------------------------------------------------
            extra_prompts: list[str] = []
            if planner_handoff:
                extra_prompts.append(_build_planner_handoff_prompt(planner_handoff))
            if case_validation_prompt:
                extra_prompts.append(case_validation_prompt)

            coder_result: dict[str, Any] | None = None
            review_feedback: dict[str, Any] | None = None
            reviewer_rounds = args.reviewer_rounds if args.agent_architecture == "planner_coder_reviewer" else 1
            coder_rounds: list[dict[str, Any]] = []

            for rnd in range(reviewer_rounds):
                # Reset repo to HEAD before each retry so the second coder round
                # starts from a clean state, not on top of the rejected edits.
                if rnd > 0:
                    log_line(f"[mcts] resetting repo to HEAD for coder round {rnd + 1}")
                    env.communicate(
                        f"cd {shlex.quote(repo_root)}\ngit reset --hard HEAD\ngit clean -fdq",
                        check="ignore", timeout=30,
                    )

                round_extra = list(extra_prompts)
                if review_feedback:
                    round_extra.append(_build_reviewer_feedback_prompt(review_feedback))
                    round_extra.append(_build_reviewer_constraints_prompt(review_feedback))

                log_line(f"[mcts] coder round={rnd + 1} model={model}")
                mcts = MCTSAgentLoop(
                    model=model,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    env=env,
                    repo_root=repo_root,
                    problem_statement=instance.problem_statement.text,
                    runtime_context=runtime_context,
                    max_identical_tool_failures=args.max_identical_tool_failures,
                    success_validation_commands=success_cmds,
                    success_validation_checks=success_checks,
                    case_policy=case_policy,
                    extra_user_prompts=round_extra,
                    log_fn=log_line,
                    edit_vote_samples=args.edit_vote_samples,
                    adaptive_branching=bool(args.adaptive_branching),
                    verbose=args.verbose,
                )
                coder_result = mcts.run()
                role_model_stats["coder"] = {"model": model, **coder_result.get("stats", {})}

                if args.agent_architecture != "planner_coder_reviewer":
                    break

                assert coder_result is not None
                patch_text = str(coder_result.get("patch", ""))

                # --- Pre-run reviewer verification ----------------------------
                # Run success checks deterministically in the live container.
                # If all pass + patch is non-empty + patch doesn't touch test
                # files → skip the model reviewer call entirely.
                patch_touches_tests = any(
                    ln.startswith("+++ b/") and ("test" in ln.lower() or "/spec/" in ln)
                    for ln in patch_text.splitlines()
                )
                live_check_results: dict[str, dict] = {}
                if success_checks and patch_text.strip() and not patch_touches_tests:
                    for chk in success_checks:
                        chk_cmd = str(chk.get("command", "")).strip()
                        if not chk_cmd:
                            continue
                        chk_out = env.communicate(
                            f"cd {shlex.quote(repo_root)}\n{chk_cmd}",
                            check="ignore", timeout=60,
                        )
                        chk_exit_raw = env.communicate("echo $?", check="ignore", timeout=5).strip()
                        chk_exit = int(chk_exit_raw) if chk_exit_raw.isdigit() else 1
                        passed = _command_output_satisfies_check(chk, exit_code=chk_exit, output=chk_out)
                        live_check_results[str(chk.get("name", chk_cmd[:40]))] = {
                            "passed": passed,
                            "actual_output": chk_out[:300],
                            "expected_contains": chk.get("stdout_contains", []),
                            "must_not_contain": chk.get("stdout_not_contains", []),
                        }
                if (
                    live_check_results
                    and all(v["passed"] for v in live_check_results.values())
                    and patch_text.strip()
                    and not patch_touches_tests
                ):
                    review_feedback = {
                        "decision": "accept",
                        "primary_reason": "All success checks verified (pre-run); skipping model reviewer.",
                        "required_changes": [],
                        "validations_to_rerun": [],
                    }
                    log_line("[reviewer] auto-accept: all success checks passed pre-run")
                    coder_rounds.append({
                        "round": rnd + 1,
                        "turns": coder_result.get("turns", []),
                        "patch": coder_result.get("patch", ""),
                        "submitted": coder_result.get("submitted", False),
                        "stopped_reason": coder_result.get("stopped_reason", ""),
                        "stats": coder_result.get("stats", {}),
                        "mcts_tree": coder_result.get("mcts_tree", {}),
                        "vote_summary": coder_result.get("vote_summary", []),
                        "review_feedback": review_feedback,
                    })
                    if not coder_result.get("submitted", False):
                        coder_result["submitted"] = True
                        coder_result["stopped_reason"] = "reviewer_accepted_finalized"
                        coder_result["submission_summary"] = (
                            "Pre-run success checks all passed; reviewer auto-accepted."
                        )
                    break
                # --------------------------------------------------------------

                log_line(f"[reviewer] model={reviewer_model} round={rnd + 1}")
                review_feedback, rev_stats, _ = _call_json_role_local(
                    model=reviewer_model,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    num_ctx=args.num_ctx,
                    system_prompt=_build_reviewer_system_prompt(repo_root),
                    user_prompt=_build_reviewer_task_prompt(
                        planner_handoff=planner_handoff,
                        coder_result=coder_result,
                        patch_text=patch_text,
                        case_evaluation=case_evaluation,
                        case_policy=case_policy,
                        live_check_results=live_check_results or None,
                        prior_reviewer_feedback=review_feedback if rnd > 0 else None,
                    ),
                    fallback_payload=_default_reviewer_feedback(),
                )
                prior = role_model_stats.get("reviewer", {"tokens_in": 0, "tokens_out": 0, "api_calls": 0})
                role_model_stats["reviewer"] = {
                    "model": reviewer_model,
                    "tokens_in": int(prior.get("tokens_in", 0)) + rev_stats["tokens_in"],
                    "tokens_out": int(prior.get("tokens_out", 0)) + rev_stats["tokens_out"],
                    "api_calls": int(prior.get("api_calls", 0)) + rev_stats["api_calls"],
                }
                raw_decision = str(review_feedback.get("decision", "")).lower()
                reviewer_hard_reject = False
                if raw_decision == "revise":
                    required_check_count = len(success_checks)
                    satisfied_check_count = len((coder_result.get("loop_state") or {}).get("satisfied_success_checks", []))
                    missing_required_checks = required_check_count > 0 and satisfied_check_count < required_check_count
                    patch_empty = not patch_text.strip()
                    failed_live_checks = any(not v.get("passed", False) for v in (live_check_results or {}).values())
                    explicit_failing_evidence = bool(failed_live_checks or missing_required_checks or patch_empty)
                    risk_high = "high" in str(review_feedback.get("risk_assessment", "")).lower()
                    reviewer_hard_reject = (
                        args.reviewer_gate_mode == "strict"
                        or missing_required_checks
                        or patch_empty
                        or (risk_high and explicit_failing_evidence)
                    )
                    if args.reviewer_gate_mode == "soft" and not reviewer_hard_reject:
                        review_feedback["decision"] = "accept_soft_gate"
                        review_feedback["soft_gate_override"] = {
                            "reason": "reviewer_revise_without_hard_failure_evidence",
                            "missing_required_checks": missing_required_checks,
                            "failed_live_checks": failed_live_checks,
                            "patch_empty": patch_empty,
                        }
                log_line(
                    f"[reviewer] decision={review_feedback.get('decision', 'unknown')} "
                    f"(raw={raw_decision}, hard_reject={reviewer_hard_reject}, gate={args.reviewer_gate_mode})"
                )
                # Record this round's full history before potentially overwriting coder_result
                coder_rounds.append({
                    "round": rnd + 1,
                    "turns": coder_result.get("turns", []),
                    "patch": coder_result.get("patch", ""),
                    "submitted": coder_result.get("submitted", False),
                    "stopped_reason": coder_result.get("stopped_reason", ""),
                    "stats": coder_result.get("stats", {}),
                    "mcts_tree": coder_result.get("mcts_tree", {}),
                    "vote_summary": coder_result.get("vote_summary", []),
                    "review_feedback": review_feedback,
                })
                if str(review_feedback.get("decision", "")).lower() in {"accept", "accept_soft_gate"}:
                    if not coder_result.get("submitted", False) and str(coder_result.get("patch", "")).strip():
                        coder_result["submitted"] = True
                        coder_result["stopped_reason"] = "reviewer_accepted_finalized"
                        coder_result["submission_summary"] = (
                            "Reviewer accepted patch; finalized without explicit submit."
                            if str(review_feedback.get("decision", "")).lower() == "accept"
                            else "Soft-gate override accepted patch despite reviewer revise without hard failure evidence."
                        )
                    break
                log_line("[reviewer] revise — returning to MCTS coder")

            assert coder_result is not None
            result = coder_result
            if (
                args.agent_architecture == "planner_coder_reviewer"
                and str((review_feedback or {}).get("decision", "")).lower() not in {"accept", "accept_soft_gate"}
            ):
                result["submitted"] = False
                result["stopped_reason"] = "reviewer_rejected"

            result["agent_architecture"] = args.agent_architecture
            # "architecture" alias required by analyze_custom_runs.py
            result["architecture"] = args.agent_architecture
            result["role_model_stats"] = role_model_stats
            result["planner_handoff"] = planner_handoff or {}
            result["review_feedback"] = review_feedback or {}
            result["coder_rounds"] = coder_rounds  # full per-round history
            result["instance_id"] = instance_id
            result["duration_seconds"] = round(time.time() - start, 2)

            # Sum planner + reviewer tokens into top-level stats so
            # analyze_custom_runs.py sees the true total token spend
            planner_in = int((role_model_stats.get("planner") or {}).get("tokens_in", 0))
            planner_out = int((role_model_stats.get("planner") or {}).get("tokens_out", 0))
            reviewer_in = int((role_model_stats.get("reviewer") or {}).get("tokens_in", 0))
            reviewer_out = int((role_model_stats.get("reviewer") or {}).get("tokens_out", 0))
            result["stats"]["input_tokens"] += planner_in + reviewer_in
            result["stats"]["output_tokens"] += planner_out + reviewer_out

            _dump_json(instance_dir / f"{instance_id}.traj", result)
            (instance_dir / f"{instance_id}.patch").write_text(result["patch"])
            log_line(f"[result] stopped={result['stopped_reason']} submitted={result['submitted']}")

            pred = _save_pred(instance_dir, instance_id, args.run_name, result["info"]["submission"])
            all_preds[instance_id] = pred

        except BaseException as exc:
            tb_str = traceback.format_exc()
            try:
                log_line(f"[error] {type(exc).__name__}: {exc}\n{tb_str}")
            except Exception:
                pass  # log_line itself failed; traceback already printed to stderr below
            # Print to stderr so the crash is always visible in terminal output
            print(f"\n[run_tree_search ERROR] {instance_id}: {type(exc).__name__}: {exc}\n{tb_str}",
                  file=sys.stderr, flush=True)
            try:
                failure = {
                    "instance_id": instance_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_kind": "infra" if _is_infra_error_text(f"{exc}\n{tb_str}") else "agent",
                    "traceback": tb_str,
                    "duration_seconds": round(time.time() - start, 2),
                    "info": {"submission": "", "submitted": False, "stopped_reason": "error"},
                }
                _dump_json(instance_dir / f"{instance_id}.traj", failure)
                (instance_dir / f"{instance_id}.patch").write_text("")
                pred = _save_pred(instance_dir, instance_id, args.run_name, "")
                all_preds[instance_id] = pred
            except Exception:
                pass  # best-effort; don't let secondary failure hide the original crash
            # Re-raise non-Exception BaseExceptions (KeyboardInterrupt, SystemExit)
            # so the outer process can terminate cleanly.
            if not isinstance(exc, Exception):
                raise
        finally:
            log_line("[env] shutting down")
            try:
                env.close()
            except Exception:
                pass  # prevent env.close() errors from masking the original exception

    _dump_json(output_dir / "preds.json", all_preds)


if __name__ == "__main__":
    main()
