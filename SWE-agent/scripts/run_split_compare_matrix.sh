#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-SWE-agent/custom_matrix_runs/benchmark_round_split_compare}"
PARALLEL="${PARALLEL:-2}"
RETRIES="${RUNNER_RETRIES:-1}"
CASES="${CASES:-$(find SWE-agent/custom_cases -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | paste -sd, -)}"
SPLIT_PRESETS="${SPLIT_PRESETS:-umich_gptoss_120b,umich_qwen,umich_gptoss_planner_umich_qwen_coder,umich_qwen_planner_ollama_qwen_coder}"
BASELINE_PRESETS="${BASELINE_PRESETS:-openai_gpt4o_mini,umich_gptoss_120b,umich_qwen,ollama_qwen35_9b}"

echo "Cases: ${CASES}"
echo "Split presets: ${SPLIT_PRESETS}"
echo "Baseline presets: ${BASELINE_PRESETS}"

echo "Running split architectures into ${ROOT}"
./env/bin/python SWE-agent/scripts/run_custom_experiment_matrix.py \
    --presets "${SPLIT_PRESETS}" \
    --architectures planner_coder,planner_coder_reviewer \
    --cases "${CASES}" \
    --parallel "${PARALLEL}" \
    --runner-retries "${RETRIES}" \
    --skip-existing \
    --output-root "${ROOT}"

echo
echo "Running single-model baselines into ${ROOT}"
./env/bin/python SWE-agent/scripts/run_custom_experiment_matrix.py \
    --presets "${BASELINE_PRESETS}" \
    --architectures single \
    --cases "${CASES}" \
    --parallel "${PARALLEL}" \
    --runner-retries "${RETRIES}" \
    --skip-existing \
    --output-root "${ROOT}"

echo
echo "Finished matrix runs for ${ROOT}"
