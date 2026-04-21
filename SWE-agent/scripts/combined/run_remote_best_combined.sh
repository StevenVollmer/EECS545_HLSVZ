#!/usr/bin/env bash
# "Best combined" variant: 120b planner/reviewer + 30b coder MCTS
# + hindsight feedback + strict 120b reviewer. No LLM value function.
#
# This is C_strict + hindsight — the highest-accuracy configuration.
# C already showed hindsight helps (E_c1=85%), but E was confounded by the
# LLM value function which hurt c2/c3. This isolates hindsight alone.
#
# Soft-gate reference points for comparison:
#   C (no hindsight, soft):  c1=85%, c2=70%, c3=75%
#   E (hindsight+value, soft): c1=85%, c2=60%
#
# Usage (from project root):
#   bash SWE-agent/scripts/combined/run_remote_best_combined.sh [--resume]

set -euo pipefail

RESUME_FLAG=""
[[ "${1:-}" == "--resume" ]] && RESUME_FLAG="--resume"

RUNNER="python3 SWE-agent/scripts/combined/run_combined.py"
OUT="combined_results"

C1="SWE-agent/custom_cases"
C2="SWE-agent/custom_cases_2"
C3="SWE-agent/custom_cases_3"

_30B_MODEL="openai/Qwen/Qwen3-VL-30B-A3B-Instruct"
_30B_API_BASE="http://promaxgb10-d668.eecs.umich.edu:8000/v1"
_30B_API_KEY="api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy"

_120B_MODEL="openai/openai/gpt-oss-120b"
_120B_API_BASE="http://promaxgb10-d473.eecs.umich.edu:8000/v1"
_120B_API_KEY="api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"

# P: mixed MCTS + hindsight + strict 120b reviewer (no LLM value function)
P_FLAGS=(
    --model "$_30B_MODEL" --api-base "$_30B_API_BASE" --api-key "$_30B_API_KEY"
    --planner-model "$_120B_MODEL" --planner-api-base "$_120B_API_BASE" --planner-api-key "$_120B_API_KEY"
    --reviewer-model "$_120B_MODEL" --reviewer-api-base "$_120B_API_BASE" --reviewer-api-key "$_120B_API_KEY"
    --max-tokens 1024
    --iterations 18 --expansion-candidates 1 --edit-vote-samples 5 --max-node-depth 18
    --agent-architecture planner_coder_reviewer --reviewer-rounds 2
    --reviewer-gate-mode strict
    --failure-surfacing --hindsight-feedback
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

run P_c1_best_combined "$C1" "${P_FLAGS[@]}"
run P_c2_best_combined "$C2" "${P_FLAGS[@]}"
run P_c3_best_combined "$C3" "${P_FLAGS[@]}"

echo ""
echo "Best-combined variant complete."
