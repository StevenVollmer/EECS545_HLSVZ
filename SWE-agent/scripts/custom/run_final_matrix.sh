#!/bin/bash
# FINAL MATRIX: All configs (linear + MCTS) on all 27 cases.
# One clean session, consistent results.
#
# Linear configs (5):  qwen, gpt→qw, gpt→qw+C, gpt→qw+R, gpt
# MCTS configs (3):    C_baseline, C_plan_critic, C_critic_gate
# Total: 8 configs × 27 cases = 216 runs
set -u

PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
LINEAR_RUNNER="$PROJECT_ROOT/SWE-agent/scripts/custom/run_custom_swebench.py"
MCTS_RUNNER="$PROJECT_ROOT/SWE-agent/scripts/combined/run_combined.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/custom_matrix_runs/final_matrix"

CASES="board_rollup budget_snapshot contact_card digest_preview incident_brief invoice_footer label_formatter median_window milestone_rollup nested_app owner_recap owner_sort priority_snapshot renewal_preview risk_score shipment_preview simple_mean_bug status_slug team_digest workspace_digest numeric_drift_sum pagination_drift path_normalizer_cache retry_cap search_hit_localize stable_ranking weighted_median"

BATCH_SIZE=2

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

# --- Linear runner helper ---
run_linear_phase() {
    local label=$1
    local preset=$2
    local arch=$3
    shift 3
    local extra=("$@")
    local count=0
    for case in $CASES; do
        local out="$OUTROOT/$label/$arch/$case"
        mkdir -p "$out"
        "$PY" "$LINEAR_RUNNER" \
            --preset "$preset" \
            --agent-architecture "$arch" \
            --instances-type file \
            --instances-path "$CASES_ROOT/$case" \
            --filter "${case}_001" \
            --output-dir "$out" \
            --max-turns 40 \
            ${extra[@]+"${extra[@]}"} \
            > "$out/runner.log" 2>&1 &
        count=$((count + 1))
        if [ $((count % BATCH_SIZE)) -eq 0 ]; then
            wait
        fi
    done
    wait
    echo "[$(date +%H:%M:%S)] LINEAR $label done"
}

# --- MCTS runner helper ---
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
            --resume \
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

echo "[$(date +%H:%M:%S)] FINAL MATRIX: 8 configs x 27 cases = 216 runs (batch=$BATCH_SIZE)"

# === LINEAR CONFIGS ===
run_linear_phase umich_qwen umich_qwen single
run_linear_phase umich_gptoss_planner_umich_qwen_coder umich_gptoss_planner_umich_qwen_coder planner_coder
run_linear_phase umich_gptoss_planner_critic_qwen_coder umich_gptoss_planner_critic_qwen_coder planner_coder --plan-critic
run_linear_phase umich_gptoss_planner_umich_qwen_coder_reviewer umich_gptoss_planner_umich_qwen_coder planner_coder_reviewer --reviewer-rounds 2
run_linear_phase umich_gptoss_120b umich_gptoss_120b single

# === MCTS CONFIGS ===
run_mcts_phase mcts_baseline
run_mcts_phase mcts_plan_critic --plan-critic
run_mcts_phase mcts_critic_gate --critic-gate

echo "[$(date +%H:%M:%S)] ALL DONE"
echo "Output: $OUTROOT"
