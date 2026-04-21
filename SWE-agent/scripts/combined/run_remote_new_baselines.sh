#!/usr/bin/env bash
# Remote linear-baseline runs (no local GPU required).
# Split by model so two terminals can run in parallel:
#
#   Terminal 1:  bash run_remote_new_baselines.sh --group 30b
#   Terminal 2:  bash run_remote_new_baselines.sh --group 120b
#
# Usage (from project root):
#   bash SWE-agent/scripts/combined/run_remote_new_baselines.sh --group 30b [--resume]

set -euo pipefail

GROUP=""
RESUME_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)  GROUP="$2"; shift 2 ;;
        --resume) RESUME_FLAG="--resume"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$GROUP" ]]; then
    echo "Usage: $0 --group 30b|120b [--resume]"
    exit 1
fi

RUNNER="python3 SWE-agent/scripts/combined/run_combined.py"
OUT="combined_results"

C1="SWE-agent/custom_cases"
C2="SWE-agent/custom_cases_2"
C3="SWE-agent/custom_cases_3"

LINEAR_FLAGS=(
    --agent-architecture single
    --iterations 18 --expansion-candidates 1 --edit-vote-samples 1 --max-node-depth 18
    --reviewer-gate-mode soft   # unused by single-arch; required by argparse
    --no-adaptive-branching --no-failure-surfacing --no-hindsight-feedback
    --no-auto-finalize
)

run() {
    local run_id="$1"; local cases="$2"; shift 2
    echo ""
    echo "========================================"
    echo "  START: $run_id  $(date)"
    echo "========================================"
    $RUNNER \
        --instances-type file \
        --instances-path "$cases" \
        --output-dir "$OUT/$run_id" \
        --run-name "$run_id" \
        "$@" \
        $RESUME_FLAG
    echo "  DONE:  $run_id  $(date)"
}

if [[ "$GROUP" == "30b" ]]; then
    MODEL="openai/Qwen/Qwen3-VL-30B-A3B-Instruct"
    API_BASE="http://promaxgb10-d668.eecs.umich.edu:8000/v1"
    API_KEY="api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy"
    MODEL_FLAGS=(--model "$MODEL" --api-base "$API_BASE" --api-key "$API_KEY" --max-tokens 1024)

    run M_c1_30b_linear "$C1" "${MODEL_FLAGS[@]}" "${LINEAR_FLAGS[@]}"
    run M_c2_30b_linear "$C2" "${MODEL_FLAGS[@]}" "${LINEAR_FLAGS[@]}"
    run M_c3_30b_linear "$C3" "${MODEL_FLAGS[@]}" "${LINEAR_FLAGS[@]}"

elif [[ "$GROUP" == "120b" ]]; then
    MODEL="openai/openai/gpt-oss-120b"
    API_BASE="http://promaxgb10-d473.eecs.umich.edu:8000/v1"
    API_KEY="api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"
    MODEL_FLAGS=(--model "$MODEL" --api-base "$API_BASE" --api-key "$API_KEY" --max-tokens 4096)

    run N_c1_120b_linear "$C1" "${MODEL_FLAGS[@]}" "${LINEAR_FLAGS[@]}"
    run N_c2_120b_linear "$C2" "${MODEL_FLAGS[@]}" "${LINEAR_FLAGS[@]}"
    run N_c3_120b_linear "$C3" "${MODEL_FLAGS[@]}" "${LINEAR_FLAGS[@]}"

else
    echo "Unknown group: $GROUP (must be 30b or 120b)"
    exit 1
fi

echo ""
echo "Group '$GROUP' complete."
