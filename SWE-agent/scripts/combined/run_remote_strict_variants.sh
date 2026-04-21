#!/usr/bin/env bash
# Strict-reviewer variants of B (Rafe linear) and C (mixed MCTS) on all case sets.
# Both use the 120b reviewer with hard gate (--reviewer-gate-mode strict).
# Compares directly against the already-completed soft-gate runs:
#   B_soft: c1=80%, c2=60%, c3=80%
#   C_soft: c1=85%, c2=70%, c3=75%
#
# Usage (from project root):
#   bash SWE-agent/scripts/combined/run_remote_strict_variants.sh [--resume]

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

# B strict: Rafe linear (120b planner/reviewer + 30b coder), strict 120b reviewer
B_STRICT_FLAGS=(
    --model "$_30B_MODEL" --api-base "$_30B_API_BASE" --api-key "$_30B_API_KEY"
    --planner-model "$_120B_MODEL" --planner-api-base "$_120B_API_BASE" --planner-api-key "$_120B_API_KEY"
    --reviewer-model "$_120B_MODEL" --reviewer-api-base "$_120B_API_BASE" --reviewer-api-key "$_120B_API_KEY"
    --max-tokens 1024
    --iterations 18 --expansion-candidates 1 --edit-vote-samples 1 --max-node-depth 18
    --agent-architecture planner_coder_reviewer --reviewer-rounds 2
    --reviewer-gate-mode strict
    --no-adaptive-branching --no-failure-surfacing --no-hindsight-feedback
)

# C strict: mixed MCTS (120b planner/reviewer + 30b coder MCTS), strict 120b reviewer
C_STRICT_FLAGS=(
    --model "$_30B_MODEL" --api-base "$_30B_API_BASE" --api-key "$_30B_API_KEY"
    --planner-model "$_120B_MODEL" --planner-api-base "$_120B_API_BASE" --planner-api-key "$_120B_API_KEY"
    --reviewer-model "$_120B_MODEL" --reviewer-api-base "$_120B_API_BASE" --reviewer-api-key "$_120B_API_KEY"
    --max-tokens 1024
    --iterations 18 --expansion-candidates 1 --edit-vote-samples 5 --max-node-depth 18
    --agent-architecture planner_coder_reviewer --reviewer-rounds 2
    --reviewer-gate-mode strict
    --failure-surfacing --no-hindsight-feedback
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

run B_strict_c1_rafe_linear  "$C1" "${B_STRICT_FLAGS[@]}"
run B_strict_c2_rafe_linear  "$C2" "${B_STRICT_FLAGS[@]}"
run B_strict_c3_rafe_linear  "$C3" "${B_STRICT_FLAGS[@]}"

run C_strict_c1_mixed_mcts   "$C1" "${C_STRICT_FLAGS[@]}"
run C_strict_c2_mixed_mcts   "$C2" "${C_STRICT_FLAGS[@]}"
run C_strict_c3_mixed_mcts   "$C3" "${C_STRICT_FLAGS[@]}"

echo ""
echo "All strict-reviewer variants complete."
