#!/bin/bash
# Full 10-config x 27-case matrix.
# All 20 original cases + 7 new cases (excluding date_parse_locale,
# dep_cycle_detect, schema_migration_check).
# Runs all 27 cases per phase in parallel, phases sequential.
set -u

PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
RUNNER="$PROJECT_ROOT/SWE-agent/scripts/custom/run_custom_swebench.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/custom_matrix_runs/full_27_matrix"

# Original 20
CASES_OLD="board_rollup budget_snapshot contact_card digest_preview incident_brief invoice_footer label_formatter median_window milestone_rollup nested_app owner_recap owner_sort priority_snapshot renewal_preview risk_score shipment_preview simple_mean_bug status_slug team_digest workspace_digest"

# New 7 (10 minus 3 dropped)
CASES_NEW="numeric_drift_sum pagination_drift path_normalizer_cache retry_cap search_hit_localize stable_ranking weighted_median"

CASES="$CASES_OLD $CASES_NEW"

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

BATCH_SIZE=2

# Args: label preset arch [extra...]
run_phase() {
    local label=$1
    local preset=$2
    local arch=$3
    shift 3
    local extra=("$@")
    local count=0
    for case in $CASES; do
        local out="$OUTROOT/$label/$arch/$case"
        mkdir -p "$out"
        "$PY" "$RUNNER" \
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
    echo "[$(date +%H:%M:%S)] $label done"
}

echo "[$(date +%H:%M:%S)] starting: 10 configs x 27 cases = 270 jobs"

# 1. qwen solo (floor)
run_phase umich_qwen umich_qwen single

# 2. qwen -> qwen
run_phase umich_qwen_planner_qwen_coder umich_qwen planner_coder \
    --planner-model "openai/Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --planner-api-base "http://promaxgb10-d668.eecs.umich.edu:8000/v1" \
    --planner-api-key "api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy"

# 3. qwen -> qwen + critic
run_phase umich_qwen_planner_critic_qwen_coder umich_qwen planner_coder \
    --planner-model "openai/Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --planner-api-base "http://promaxgb10-d668.eecs.umich.edu:8000/v1" \
    --planner-api-key "api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy" \
    --plan-critic

# 4. gpt -> qwen
run_phase umich_gptoss_planner_umich_qwen_coder umich_gptoss_planner_umich_qwen_coder planner_coder

# 5. gpt -> qwen + reviewer
run_phase umich_gptoss_planner_umich_qwen_coder_reviewer umich_gptoss_planner_umich_qwen_coder planner_coder_reviewer \
    --reviewer-rounds 2

# 6. gpt -> qwen + critic
run_phase umich_gptoss_planner_critic_qwen_coder umich_gptoss_planner_critic_qwen_coder planner_coder \
    --plan-critic

# 7. gpt -> qwen + critic + reviewer
run_phase umich_gptoss_planner_critic_qwen_coder_reviewer umich_gptoss_planner_critic_qwen_coder planner_coder_reviewer \
    --plan-critic --reviewer-rounds 2

# 8. gpt -> gpt
run_phase umich_gptoss_planner_gptoss_coder umich_gptoss_120b planner_coder \
    --planner-model "openai/openai/gpt-oss-120b" \
    --planner-api-base "http://promaxgb10-d473.eecs.umich.edu:8000/v1" \
    --planner-api-key "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"

# 9. gpt -> gpt + critic
run_phase umich_gptoss_planner_critic_gptoss_coder umich_gptoss_120b planner_coder \
    --planner-model "openai/openai/gpt-oss-120b" \
    --planner-api-base "http://promaxgb10-d473.eecs.umich.edu:8000/v1" \
    --planner-api-key "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a" \
    --plan-critic

# 10. gpt solo (ceiling)
run_phase umich_gptoss_120b umich_gptoss_120b single

echo "ALL DONE at $(date +%H:%M:%S)"
echo "Output: $OUTROOT"
