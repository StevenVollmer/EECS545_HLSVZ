#!/usr/bin/env bash
# Run all new local-GPU variants sequentially (single terminal, safe for one GPU).
# Variants: F_strict (9b coder + 120b planner/reviewer, strict — uses remote LLM, runs first),
#           L (9b single-role linear), A_strict (9b MCTS strict reviewer),
#           G_strict (9b MCTS + hindsight + strict reviewer)
# Each variant runs on c1, c2, and c3.
#
# Usage (from project root):
#   bash SWE-agent/scripts/combined/run_local_new_variants.sh
#   bash SWE-agent/scripts/combined/run_local_new_variants.sh --resume

set -euo pipefail

RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
fi

RUNNER="python3 SWE-agent/scripts/combined/run_combined.py"
OUT="combined_results"

C1="SWE-agent/custom_cases"
C2="SWE-agent/custom_cases_2"
C3="SWE-agent/custom_cases_3"

MODEL="qwen3.5:9b"
API_BASE="http://localhost:11434"
API_KEY="ollama"

LINEAR_FLAGS=(
    --model "$MODEL" --api-base "$API_BASE" --api-key "$API_KEY"
    --num-ctx 32768 --max-tokens 384
    --agent-architecture single
    --iterations 1 --expansion-candidates 1 --edit-vote-samples 1 --max-node-depth 5
    --reviewer-gate-mode soft   # unused by single-arch; required by argparse
    --no-adaptive-branching --no-failure-surfacing --no-hindsight-feedback
)

MCTS_STRICT_BASE=(
    --model "$MODEL" --api-base "$API_BASE" --api-key "$API_KEY"
    --num-ctx 32768 --max-tokens 384
    --planner-model "$MODEL"
    --agent-architecture planner_coder_reviewer
    --iterations 18 --expansion-candidates 1 --edit-vote-samples 5 --max-node-depth 18
    --reviewer-rounds 2 --reviewer-gate-mode strict --failure-surfacing
)

_120B_MODEL="openai/openai/gpt-oss-120b"
_120B_API_BASE="http://promaxgb10-d473.eecs.umich.edu:8000/v1"
_120B_API_KEY="api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"

# F_strict: 9b coder (local) + 120b planner/reviewer (remote), strict 120b reviewer
# Isolates coder-size contribution: compare A_strict (all 9b) vs F_strict (9b+120b) vs C_strict (30b+120b)
F_STRICT_FLAGS=(
    --model "$MODEL" --api-base "$API_BASE" --api-key "$API_KEY"
    --num-ctx 32768 --max-tokens 512
    --planner-model "$_120B_MODEL" --planner-api-base "$_120B_API_BASE" --planner-api-key "$_120B_API_KEY"
    --reviewer-model "$_120B_MODEL" --reviewer-api-base "$_120B_API_BASE" --reviewer-api-key "$_120B_API_KEY"
    --agent-architecture planner_coder_reviewer
    --iterations 18 --expansion-candidates 1 --edit-vote-samples 5 --max-node-depth 18
    --reviewer-rounds 2 --reviewer-gate-mode strict --failure-surfacing --no-hindsight-feedback
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

# ── F_strict: 9b coder + 120b planner/reviewer — remote LLM, runs first ─────
run F_strict_c1_9b_120b_planner "$C1" "${F_STRICT_FLAGS[@]}"
run F_strict_c2_9b_120b_planner "$C2" "${F_STRICT_FLAGS[@]}"
run F_strict_c3_9b_120b_planner "$C3" "${F_STRICT_FLAGS[@]}"

# ── L: 9b single-role linear (true no-search, no-roles baseline) ─────────────
run L_c1_9b_linear          "$C1" "${LINEAR_FLAGS[@]}"
run L_c2_9b_linear          "$C2" "${LINEAR_FLAGS[@]}"
run L_c3_9b_linear          "$C3" "${LINEAR_FLAGS[@]}"

# ── A_strict: 9b MCTS, strict 9b reviewer, no hindsight ─────────────────────
run A_strict_c1_9b_mcts     "$C1" "${MCTS_STRICT_BASE[@]}" --no-hindsight-feedback
run A_strict_c2_9b_mcts     "$C2" "${MCTS_STRICT_BASE[@]}" --no-hindsight-feedback
run A_strict_c3_9b_mcts     "$C3" "${MCTS_STRICT_BASE[@]}" --no-hindsight-feedback

# ── G_strict: 9b MCTS, strict 9b reviewer, with hindsight ───────────────────
run G_strict_c1_9b_hindsight "$C1" "${MCTS_STRICT_BASE[@]}" --hindsight-feedback
run G_strict_c2_9b_hindsight "$C2" "${MCTS_STRICT_BASE[@]}" --hindsight-feedback
run G_strict_c3_9b_hindsight "$C3" "${MCTS_STRICT_BASE[@]}" --hindsight-feedback

echo ""
echo "All local variants complete."
