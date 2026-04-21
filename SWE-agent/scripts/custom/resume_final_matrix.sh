#!/bin/bash
# Resumes the final matrix — only runs MCTS configs that are incomplete.
# Linear configs (all 5) are already done.
# Uses --resume to skip cases with existing .traj files.
set -u

PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
MCTS_RUNNER="$PROJECT_ROOT/SWE-agent/scripts/combined/run_combined.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/custom_matrix_runs/final_matrix"

CASES="board_rollup budget_snapshot contact_card digest_preview incident_brief invoice_footer label_formatter median_window milestone_rollup nested_app owner_recap owner_sort priority_snapshot renewal_preview risk_score shipment_preview simple_mean_bug status_slug team_digest workspace_digest numeric_drift_sum pagination_drift path_normalizer_cache retry_cap search_hit_localize stable_ranking weighted_median"

BATCH_SIZE=2

MCTS_COMMON=(
    --model "openai/Qwen/Qwen3-VL-30B-A3B-Instruct"
    --api-base "http://promaxgb10-d668.eecs.umich.edu:8000/v1"
    --api-key "api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy"
    --planner-model "openai/openai/gpt-oss-120b"
    --planner-api-base "http://promaxgb10-d473.eecs.umich.edu:8000/v1"
    --planner-api-key "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"
    --reviewer-model "openai/openai/gpt-oss-120b"
    --reviewer-api-base "http://promaxgb10-d473.eecs.umich.edu:8000/v1"
    --reviewer-api-key "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"
    --agent-architecture planner_coder
    --iterations 15
    --max-node-depth 15
    --edit-vote-samples 3
    --max-tokens 4096
    --resume
)

run_mcts_phase() {
    local label=$1
    shift
    local extra=("$@")
    local out="$OUTROOT/$label"
    mkdir -p "$out"
    local count=0
    for case in $CASES; do
        "$PY" "$MCTS_RUNNER" \
            "${MCTS_COMMON[@]}" \
            --instances-type file \
            --instances-path "$CASES_ROOT/$case" \
            --filter "${case}_001" \
            --output-dir "$out" \
            --run-name "$label" \
            ${extra[@]+"${extra[@]}"} \
            > "$out/${case}.log" 2>&1 &
        count=$((count + 1))
        if [ $((count % BATCH_SIZE)) -eq 0 ]; then
            wait
        fi
    done
    wait
    echo "[$(date +%H:%M:%S)] MCTS $label done"
}

echo "[$(date +%H:%M:%S)] Resuming MCTS configs (linear already complete)"

# mcts_baseline: 14/27 done, resume picks up remaining 13
run_mcts_phase mcts_baseline

# mcts_plan_critic: not started yet
run_mcts_phase mcts_plan_critic --plan-critic

# mcts_critic_gate: not started yet
run_mcts_phase mcts_critic_gate --critic-gate

echo "[$(date +%H:%M:%S)] ALL DONE"
echo "Output: $OUTROOT"
