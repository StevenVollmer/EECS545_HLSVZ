#!/bin/bash
# Runs critic ablation variants on custom_cases (c1, 20 cases).
# Compares: C (baseline MCTS), C+critic (plan audit), C+gate (submission gate), C+both
# Batches 5 cases at a time to avoid overloading endpoints.
set -u

PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
RUNNER="$PROJECT_ROOT/SWE-agent/scripts/combined/run_combined.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/tree_search_runs/critic_ablation"

# Use c1 (original 20 cases)
CASES="board_rollup budget_snapshot contact_card digest_preview incident_brief invoice_footer label_formatter median_window milestone_rollup nested_app owner_recap owner_sort priority_snapshot renewal_preview risk_score shipment_preview simple_mean_bug status_slug team_digest workspace_digest"

BATCH_SIZE=2

# Common args for mixed MCTS (120b planner/reviewer + 30b coder)
COMMON_ARGS=(
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
    --instances-type file
    --iterations 15
    --max-node-depth 15
    --edit-vote-samples 3
    --max-tokens 4096
)

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

# Args: label extra_flags...
run_variant() {
    local label=$1
    shift
    local extra=("$@")
    local out="$OUTROOT/$label"
    mkdir -p "$out"
    local count=0
    for case in $CASES; do
        "$PY" "$RUNNER" \
            "${COMMON_ARGS[@]}" \
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
    echo "[$(date +%H:%M:%S)] $label done"
}

echo "[$(date +%H:%M:%S)] starting critic ablation: 4 variants x 20 cases"

# Variant C: baseline MCTS (no critic, auto-accept on)
run_variant "C_baseline"

# Variant C + plan critic (pre-coder audit)
run_variant "C_plan_critic" --plan-critic

# Variant C + critic gate (submission gate, no auto-accept)
run_variant "C_critic_gate" --critic-gate

# Variant C + both
run_variant "C_both" --plan-critic --critic-gate

echo "[$(date +%H:%M:%S)] ALL DONE"
echo "Output: $OUTROOT"
