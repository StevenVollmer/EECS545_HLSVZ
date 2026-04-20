#!/bin/bash
# Runs the full 6-config x 10-case matrix from scratch.
#
# Configs (run sequentially, 10 cases parallel per phase):
#   1. qwen              — small coder, no planner (floor)
#   2. gpt->qwen         — large planner + small coder (baseline)
#   3. reviewer          — large planner + small coder + post-coder review
#   4. critic            — large planner + pre-coder audit + small coder
#   5. crit+rev          — large planner + pre-coder audit + small coder + post-coder review
#   6. gpt               — large coder, no planner (ceiling)
#
# Usage: bash SWE-agent/scripts/custom/run_full_matrix.sh

set -u

PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
RUNNER="$PROJECT_ROOT/SWE-agent/scripts/custom/run_custom_swebench.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/custom_matrix_runs/routed_matrix"

CASES="date_parse_locale dep_cycle_detect numeric_drift_sum pagination_drift path_normalizer_cache retry_cap schema_migration_check search_hit_localize stable_ranking weighted_median"

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

# Args: label preset arch [extra...]
run_phase() {
    local label=$1
    local preset=$2
    local arch=$3
    shift 3
    local extra=("$@")
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
    done
    wait
    echo "[$(date +%H:%M:%S)] $label done"
}

echo "[$(date +%H:%M:%S)] starting: 6 configs x 10 cases = 60 jobs (sequential phases)"

# 1. qwen solo (floor)
run_phase umich_qwen umich_qwen single

# 2. gpt -> qwen (planner baseline)
run_phase umich_gptoss_planner_umich_qwen_coder umich_gptoss_planner_umich_qwen_coder planner_coder

# 3. reviewer (planner + coder + post-coder review)
run_phase umich_gptoss_planner_umich_qwen_coder_reviewer umich_gptoss_planner_umich_qwen_coder planner_coder_reviewer --reviewer-rounds 2

# 4. critic (planner + pre-coder audit + coder)
run_phase umich_gptoss_planner_critic_qwen_coder umich_gptoss_planner_critic_qwen_coder planner_coder --plan-critic

# 5. critic+reviewer (planner + pre-coder audit + coder + post-coder review)
run_phase umich_gptoss_planner_critic_qwen_coder_reviewer umich_gptoss_planner_critic_qwen_coder planner_coder_reviewer --plan-critic --reviewer-rounds 2

# 6. gpt solo (ceiling)
run_phase umich_gptoss_120b umich_gptoss_120b single

echo "ALL DONE"
echo "Output: $OUTROOT"
echo "Analyze with: $PY $PROJECT_ROOT/SWE-agent/scripts/custom/analyze_bucket_matrix.py $OUTROOT"
