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

Run it with:

```bash
./env/bin/python SWE-agent/scripts/run_custom_swebench.py \
  --backend openai \
  --model gpt-4o-mini \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/simple_mean_bug_instance.json \
  --filter simple_mean_bug_001 \
  --post-startup-command "pip install pytest" \
  --output-dir SWE-agent/custom_runs/simple_mean_bug_openai
```

That case should be easy:
- reproduce with a tiny Python command or `pytest`
- fix one denominator bug in `calculator.py`
- run `pytest test_calculator.py`
- inspect `git diff`
- submit

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
- `--post-startup-command` can be repeated to bootstrap dependencies before the model starts.
- `--instances-type file --instances-path <json>` lets you run custom local cases instead of SWE-bench.
- Ollama defaults to `http://localhost:11434`.
- LM Studio defaults to `http://127.0.0.1:1234/v1`.
- UMich defaults to `http://promaxgb10-d473.eecs.umich.edu:8000/v1`.
- The runner is sequential right now. It is intended for debugging loop behavior first, not maximizing throughput.
