# Custom Runner Architecture Reference

Source: `scripts/custom/run_custom_swebench.py`

---

## What It Is

A fully standalone SWE-bench agent that bypasses the entire SWE-agent class hierarchy
(`DefaultAgent`, `MultiAgent`, YAML config, `ToolHandler`, `ModelConfig`).
It reuses only three things from the upstream packages:
- SWE-bench instance loading (`SWEBenchInstances`, `BatchInstance`)
- SWE-ReX Docker deployment (`SWEEnv`, `DockerDeploymentConfig`)
- The `SWEEnv.communicate()` / `read_file()` / `write_file()` primitives

Everything else — tool dispatch, LLM calls, prompts, planner/reviewer logic,
state tracking — is plain Python.

---

## Core Classes

### `ToolRuntime`

Thin shell over `SWEEnv`. Implements six tools:

| Tool | What it does |
|---|---|
| `bash` | Runs a command inside the container; captures stdout + exit code via sentinel |
| `view` | Reads a file with line numbers; validates path existence first |
| `str_replace` | Exact unique block replacement; enforces uniqueness |
| `insert` | Inserts text after a 1-based line number |
| `undo_edit` | Restores the previous file content from an in-memory stack |
| `submit` | Sets `self.submitted = True`; actual patch captured via `git diff` |

Each tool method returns a string observation. Errors (path missing, non-unique match,
no edit history) are returned as descriptive strings rather than exceptions, so the model
sees what went wrong.

### `LoopState`

Tracks semantic progress within a single coder run:

```python
executable_edit_made: bool          # any str_replace/insert/undo_edit succeeded
validation_passed: bool             # a bash command produced "passed" in output
validation_attempted_after_edit: bool
diff_checked: bool                  # model ran git diff
successful_post_edit_commands: set  # bash commands that exited 0 after first edit
satisfied_success_checks: set       # case-defined checks that passed
changed_files: set                  # file paths touched by edits
```

### `CustomAgentLoop`

The main model driver. Owns the message list and runs a turn loop:

1. Call `litellm.completion()` directly (no SWE-agent model abstraction)
2. Parse tool calls — two modes:
   - **`openai_tools`**: model uses native function calling; parse `message.tool_calls`
   - **`react_json`**: model outputs raw JSON; `_parse_react_json()` recovers it with
     multiple fallback strategies (fence stripping, brace-depth scanner, `ast.literal_eval`)
3. Enforce argument normalization (`old_text` → `old_str`, `start` → `start_line`, etc.)
4. Execute via `ToolRuntime.execute()`
5. Update `LoopState` from every tool result
6. Inject targeted feedback messages for:
   - Parse errors
   - Loop detection (4 identical consecutive calls)
   - Consecutive repeated tool failures
   - Case success check pass/fail inline
7. Gate `submit` via `_submit_precheck()` — blocks if: no edit made, no post-edit
   validation, test files changed without policy permission, required success checks
   not yet satisfied, `git diff` not inspected

---

## Three Agent Architectures

Controlled by `--agent-architecture`:

### `single`

`CustomAgentLoop` runs directly. No other roles.

### `planner_coder`

Before the coder loop, the **planner** is called as a **single non-interactive JSON API call**
(not a tool loop). It receives the problem statement + startup runtime context (files, README
head, git status) and returns a structured JSON object:

```
problem_summary, root_cause_hypothesis, files_likely_affected,
target_symbols, discovery_priority, first_actions,
safe_reproduction_steps, reproduction_notes, required_validations,
allowed_change_types, forbidden_edits, escalation_conditions
```

This is normalized (`_normalize_planner_handoff`) — brittle `python -c` commands are
stripped, file paths deduplicated, fallback defaults filled if parse fails — then injected
as an extra `user` message at the start of the coder's context.

### `planner_coder_reviewer`

Adds a **reviewer** after each coder loop. Also a single non-interactive JSON call.
Receives: planner handoff, full validation event history, git patch, case success check
results. Returns `{ decision: "accept" | "revise", required_changes, files_to_revisit,
validations_to_rerun }`.

If `revise`: reviewer feedback is injected into the coder's context, git state is **not**
reset (the coder continues from where it left off), and the loop repeats up to
`--reviewer-rounds` times. If the final decision is not `accept`, the run is marked
`reviewer_rejected`.

**Key design**: planner and reviewer are one-shot — no tool loop, no backtracking, no
history. Only the coder iterates.

---

## Case System

Custom local fixtures live in `SWE-agent/custom_cases/<name>/`.

Each directory contains:
- `case.json` — list of instance objects, each with:
  - `problem_statement`, `instance_id`, `image_name`, `base_commit`
  - `install_commands`, `setup_commands` (run before the agent starts)
  - `extra_fields.evaluation.baseline_checks` — commands that prove the bug exists
  - `extra_fields.evaluation.success_checks` — commands + expected output that prove the fix
  - `extra_fields.analysis.likely_fix_paths` — ranked hints for scoring + planner
  - `extra_fields.policy.allow_test_edits`
- `repo/` — plain directory (not a git repo on disk); the runner initializes `git init`
  inside the container at startup

The runner checks each success check inline after every `bash` call and informs the model
immediately via a feedback message.

---

## Evaluation Pipeline

| Script | Purpose |
|---|---|
| `judge_custom_case.py` | Runs `baseline_checks` / `success_checks` locally against a repo copy to confirm ground-truth pass/fail outside Docker |
| `analyze_custom_runs.py` | Offline scorer: reads `.traj` + `.patch` + `case.json`, outputs composite score |
| `run_custom_experiment_matrix.py` | Sweeps `(preset × architecture × case)` in parallel; writes `analysis.summary.json` |
| `render_custom_matrix_report.py` | Renders results into a human-readable comparison table |

### Scoring breakdown (`analyze_custom_runs.py`)

| Component | Max | Signal |
|---|---|---|
| `functional_correctness` | 50 | Case-defined success check pass rate |
| `repair_precision` | ~20 | Patch size, overlap with `likely_fix_paths`, test-edit penalty |
| `regression_safety` | ~15 | Validation after edit, success check coverage |
| `search_grounding` | ~10 | Inspected likely-fix files before editing |
| `efficiency_control` | 5 | Parse errors, tool errors, loop triggers |

Additional outputs: `success_passed` (strict), `relative_compute_to_4o_mini`,
`score_per_compute`, `resolved_per_compute`.

---

## Output Files Per Run

```
<output-dir>/
  run_batch.config.yaml          # full run config snapshot
  preds.json                     # batch-level predictions for SWE-bench eval
  <instance_id>/
    <instance_id>.traj           # full turn-by-turn JSON trajectory
    <instance_id>.patch          # git diff of the final patch
    <instance_id>.pred           # SWE-bench prediction format
    <instance_id>.info.log       # timestamped run log
```

---

## Local Model Defaults (Ollama)

Available on this machine:

| Model | Best for |
|---|---|
| `qwen2.5-coder:7b-instruct` | Coder role (code-specialized) |
| `qwen3.5:9b` | Planner/reviewer role (largest, strongest reasoning) |
| `deepseek-r1:7b` | Reasoning-heavy tasks |
| `granite-code:8b` | Alternative coder |
| `mistral:7b-instruct-v0.3-q4_K_M` | General fallback |
| `codellama:7b-instruct` | Legacy code model |

All models served at `http://localhost:11434`. Use `react_json` tool-call mode (small
models do not reliably support native function calling).
