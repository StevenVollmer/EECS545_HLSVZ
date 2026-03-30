# Custom SWE-bench Runner

`run_custom_swebench.py` is a standalone experiment path for SWE-bench runs.

It reuses:
- SWE-bench instance loading
- SWE-ReX Docker environments
- repository reset/setup through `SWEEnv`

It does not reuse the existing SWE-agent planner/coder/reviewer loop.

## Simple local smoke test

There is a tiny custom case at:

- [simple_mean_bug_instance.json](/Users/rafe/classes/eecs545/project/SWE-agent/custom_cases/simple_mean_bug_instance.json)
- [calculator.py](/Users/rafe/classes/eecs545/project/SWE-agent/custom_cases/simple_mean_bug_repo/calculator.py)

Custom cases are intended to be plain directories in the main repo, not nested git repos.
The runner uploads the directory, initializes a git repo inside the container if needed,
and reads case-local bootstrap commands from the JSON.

Run it with:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --preset openai_gpt4o_mini \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/simple_mean_bug_instance.json \
  --filter simple_mean_bug_001 \
  --output-dir SWE-agent/custom_runs/simple_mean_bug_openai
```

That case should be easy:
- reproduce with a tiny Python command or `pytest`
- fix one denominator bug in `calculator.py`
- run `pytest test_calculator.py`
- inspect `git diff`
- submit

## Additional custom cases

There are two harder local fixtures as well:

- [label_formatter_issue.json](/Users/rafe/classes/eecs545/project/SWE-agent/custom_cases/label_formatter_issue.json)
- [nested_app_issue.json](/Users/rafe/classes/eecs545/project/SWE-agent/custom_cases/nested_app_issue.json)

`label_formatter_001` is still small, but the problem statement is less explicit.
The repo is:

- [label_formatter_repo](/Users/rafe/classes/eecs545/project/SWE-agent/custom_cases/label_formatter_repo)

Run it with:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --preset openai_gpt4o_mini \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/label_formatter_issue.json \
  --filter label_formatter_001 \
  --output-dir SWE-agent/custom_runs/label_formatter_openai
```

`nested_app_001` is a larger app-shaped fixture with 17 files across nested directories.
The bug is somewhere in the display path, not directly at the top-level entrypoint.
The repo is:

- [nested_app_repo](/Users/rafe/classes/eecs545/project/SWE-agent/custom_cases/nested_app_repo)

Run it with:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --preset openai_gpt4o_mini \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/nested_app_issue.json \
  --filter nested_app_001 \
  --output-dir SWE-agent/custom_runs/nested_app_openai
```

## What it does

- uses a direct OpenAI-compatible tool-calling loop through `litellm`
- exposes its own small tool set: `bash`, `view`, `str_replace`, `insert`, `undo_edit`, `submit`
- writes per-instance:
  - `.traj`
  - `.patch`
  - `.pred`
- writes batch-level:
  - `run_batch.config.yaml`
  - `preds.json`

## Backends

Presets live in [custom_runner_model_presets.yaml](/Users/rafe/classes/eecs545/project/SWE-agent/config/custom_configs/custom_runner_model_presets.yaml).

Preset examples:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --preset openai_gpt4o_mini \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/simple_mean_bug_instance.json \
  --filter simple_mean_bug_001 \
  --output-dir SWE-agent/custom_runs/openai_gpt4omini
```

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --preset lmstudio_local \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/simple_mean_bug_instance.json \
  --filter simple_mean_bug_001 \
  --output-dir SWE-agent/custom_runs/lmstudio_local
```

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --preset umich_gptoss_120b \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/simple_mean_bug_instance.json \
  --filter simple_mean_bug_001 \
  --output-dir SWE-agent/custom_runs/umich_gptoss_120b
```

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --preset ollama_qwen35_9b \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/simple_mean_bug_instance.json \
  --filter simple_mean_bug_001 \
  --output-dir SWE-agent/custom_runs/ollama_qwen35_9b
```

OpenAI:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --backend openai \
  --model gpt-4o-mini \
  --filter pydicom__pydicom-1458 \
  --output-dir SWE-agent/custom_runs/openai_4o_mini_pydicom
```

Ollama:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --backend ollama \
  --model qwen2.5-coder:7b-instruct \
  --filter pydicom__pydicom-1458 \
  --output-dir SWE-agent/custom_runs/ollama_qwen_pydicom
```

LM Studio:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --backend lmstudio \
  --model openai/local-model \
  --filter pydicom__pydicom-1458 \
  --output-dir SWE-agent/custom_runs/lmstudio_pydicom
```

UMich endpoint:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --backend umich \
  --model openai/openai/gpt-oss-120b \
  --filter pydicom__pydicom-1458 \
  --output-dir SWE-agent/custom_runs/umich_gptoss_pydicom
```

## Notes

- `--api-base` overrides the backend default.
- `--api-key` overrides environment lookup. If it starts with `$`, the remainder is treated as an environment variable name.
- `--preset <name>` loads a model/backend preset, and explicit CLI flags still override the preset values.
- `--post-startup-command` can be repeated to bootstrap dependencies before the model starts.
- `--instances-type file --instances-path <json>` lets you run custom local cases instead of SWE-bench.
- In custom case JSON, use `install_commands` and `setup_commands` to define per-case bootstrap steps.
- Ollama defaults to `http://localhost:11434`.
- LM Studio defaults to `http://127.0.0.1:1234/v1`.
- UMich defaults to `http://promaxgb10-d668.eecs.umich.edu:8000/v1`.
- The runner is sequential right now. It is intended for debugging loop behavior first, not maximizing throughput.
