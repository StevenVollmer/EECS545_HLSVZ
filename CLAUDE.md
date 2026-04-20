# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EECS545 ML course project. The goal is to evaluate whether a multi-role agent structure (planner → coder → reviewer) can improve SWE-bench performance relative to a single-role coder baseline while reducing compute burden. The project modifies SWE-agent to support role-local histories and per-role model assignment.

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt   # installs SWE-bench and SWE-agent in editable mode
```

Both `SWE-bench/` and `SWE-agent/` are git submodules installed as editable packages.

## Running Experiments

**Single issue, quick test:**
```bash
sweagent run \
    --config SWE-agent/config/custom_configs/multi_agent.yaml \
    --env.repo.github_url=https://github.com/SWE-agent/test-repo \
    --problem_statement.text="Add a simple hello world function to a new file named hello.py."
```

**Matrix sweep (batch, from `SWE-agent/scripts/`):**
```bash
cd SWE-agent/scripts
python run_matrix_easy.py --preset <preset_name> --run-label <label>
python run_matrix_sweep.py --sweep default           # run all presets in a named sweep
bash run_matrix_all_configs.sh                       # convenience wrapper with watchdog
```
Matrix scripts must be run from `SWE-agent/scripts/` because they import `matrix_easy_common` as a local module.

**Summarize results:**
```bash
cd SWE-agent/scripts
python summarize_latest_matrix_results.py --root ../latest_matrix_easy_results
```

**Analyze token stats from a run log:**
```bash
python sweagent_stats.py SWE-agent/trajectories/.../debug.log
```

**Run tests (SWE-agent):**
```bash
cd SWE-agent && pytest tests/
```

## Architecture

### Multi-Agent Implementation

The core custom code lives in `SWE-agent/sweagent/agent/custom/`:

- `multi_agent.py` — `MultiAgent` (type `multi`): single shared model, roles defined in YAML config
- `multi_agent_with_2_models.py` — `MultiAgent` (type `multi_2_models`): separate models for planner vs. coder
- `multi_agent_with_3_models.py` — `MultiAgent` (type `multi_3_models`): separate models for planner, coder, and reviewer; primary implementation used in benchmark runs

All three extend `DefaultAgent` from `SWE-agent/sweagent/agent/agents.py` and override `setup()`/`step()` to maintain role-local histories instead of a single shared history.

Config classes (`MultiAgentConfig`, `MultiAgentConfigMultiModel`, `MultiAgentConfigThreeModels`) are defined in `agents.py` and registered in the `AgentConfig` discriminated union. `MultiAgentConfigThreeModels` has `enable_planner` and `enable_reviewer` flags, which lets the same config class produce coder-only, planner+coder, or full planner+coder+reviewer runs.

### Config / Prompt Structure

- `SWE-agent/config/custom_configs/matrix_easy/` — production benchmark configs
  - `model_presets.yaml` — named model profiles with `big` and `small` slots (model name, api_base, token limits)
  - `role_prompts.yaml` — role system/instance templates shared across all matrix variants
  - `*.yaml` — per-variant base configs (e.g., `big_planner_small_coder.yaml`)
- `SWE-agent/config/custom_configs/` — development/one-off configs and archived attempts

### Matrix Easy Script Layer

`SWE-agent/scripts/matrix_easy_common.py` is the shared library for the matrix sweep scripts. It defines:
- `VariantSpec` — which roles are enabled and which model size slot each role uses
- `build_variant_config()` — merges a base YAML template + prompt bundle + model profile → a complete runnable config
- `INSTANCE_SETS` — named subsets of SWE-bench Lite (e.g., `4omini_easy_pair`, `astroid_only`) for smoke tests

`run_matrix_easy.py` runs one preset across multiple variants. `run_matrix_sweep.py` iterates presets from a named sweep. `run_matrix_sweep_watchdog.py` adds retry logic for flaky runs.

### Role Handoff Mechanism

Roles communicate via a `handoff` tool (defined in `SWE-agent/tools/handoff/`). The planner calls `handoff` when it has written `plan.txt`; the coder reads `plan.txt` and calls `submit` when done. The reviewer (if enabled) analyzes the patch and either accepts or re-invokes the coder.

### Key Files

| File | Purpose |
|---|---|
| `SWE-agent/sweagent/agent/agents.py` | `AgentConfig` union + all config classes including the three custom ones |
| `SWE-agent/sweagent/agent/custom/multi_agent_with_3_models.py` | Current primary multi-agent implementation |
| `SWE-agent/scripts/matrix_easy_common.py` | Shared helpers for variant/preset config generation |
| `SWE-agent/config/custom_configs/matrix_easy/model_presets.yaml` | Model profiles for benchmark runs |
| `SWE-agent/scripts/custom/run_custom_swebench.py` | Standalone agent (bypasses SWE-agent class stack); source of shared utilities |
| `SWE-agent/scripts/tree_search_custom/run_tree_search.py` | MCTS-inspired variant of the custom runner, optimised for local Ollama models |
| `SWE-agent/scripts/tree_search_custom/ARCHITECTURE.md` | Architecture reference for the custom runner |
| `sweagent_stats.py` | Token/cost stats parser for a debug.log |

## Model Access

- **Local (LM Studio):** `http://127.0.0.1:1234/v1`, key `lm-studio`
- **UMich cluster (gpt-oss-120b):** `http://promaxgb10-d473.eecs.umich.edu:8000/v1`
- **Ollama:** `http://localhost:11434`
- **OpenAI API:** standard endpoint; key via `$OPENAI_API_KEY`

All model-specific settings are captured in `model_presets.yaml` profiles. Use the `openai_api_env` or `openai_api_all_big` profiles with `$MATRIX_EASY_OPENAI_BIG_MODEL` / `$MATRIX_EASY_OPENAI_SMALL_MODEL` env vars for OpenAI.