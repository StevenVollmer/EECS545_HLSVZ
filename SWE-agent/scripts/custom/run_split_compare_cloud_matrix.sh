#!/usr/bin/env bash
set -euo pipefail

# Cloud-only split comparison.
# This excludes local backends like Ollama and LM Studio so architecture results
# are not dominated by local protocol/runtime instability.
PARALLEL=10
RUNNER_RETRIES=1

ROOT="${1:-SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud}"
CASES="${CASES:-$(find SWE-agent/custom_cases -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | paste -sd, -)}"
SPLIT_PRESETS="${SPLIT_PRESETS:-umich_gptoss_120b,umich_qwen,umich_gptoss_planner_umich_qwen_coder}"
BASELINE_PRESETS="${BASELINE_PRESETS:-openai_gpt4o_mini,umich_gptoss_120b,umich_qwen}"

echo "Cases: ${CASES}"
echo "Split presets: ${SPLIT_PRESETS}"
echo "Baseline presets: ${BASELINE_PRESETS}"
echo "Parallel workers: ${PARALLEL}"

echo "Running cloud split architectures into ${ROOT}"
python SWE-agent/scripts/custom/run_custom_experiment_matrix.py \
    --presets "${SPLIT_PRESETS}" \
    --architectures planner_coder,planner_coder_reviewer \
    --cases "${CASES}" \
    --parallel "${PARALLEL}" \
    --runner-retries "${RUNNER_RETRIES}" \
    --skip-existing \
    --output-root "${ROOT}"

echo
echo "Running cloud single-model baselines into ${ROOT}"
python SWE-agent/scripts/custom/run_custom_experiment_matrix.py \
    --presets "${BASELINE_PRESETS}" \
    --architectures single \
    --cases "${CASES}" \
    --parallel "${PARALLEL}" \
    --runner-retries "${RUNNER_RETRIES}" \
    --skip-existing \
    --output-root "${ROOT}"

echo
echo "Finished cloud matrix runs for ${ROOT}"
