#!/bin/bash
# Runs the 4-config x 10-case matrix for the planner-as-router experiment.
#
# Configs:
#   qwen        — small coder, no planner      (floor)
#   gpt->qwen   — large planner + small coder  (current best fixed pipeline)
#   routed      — large planner routes coder   (new: this paper's contribution)
#   gpt         — large coder, no planner      (ceiling)
#
# Each phase runs 10 cases in parallel via bash job control.
# Output root: SWE-agent/custom_matrix_runs/routed_matrix
#
# Usage: bash SWE-agent/scripts/custom/run_routed_matrix.sh

set -u

PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
RUNNER="$PROJECT_ROOT/SWE-agent/scripts/custom/run_custom_swebench.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/custom_matrix_runs/routed_matrix"

CASES="date_parse_locale dep_cycle_detect numeric_drift_sum pagination_drift path_normalizer_cache retry_cap schema_migration_check search_hit_localize stable_ranking weighted_median"

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT"

# Args: preset arch [extra...]
run_phase() {
    local preset=$1
    local arch=$2
    shift 2
    local extra=("$@")
    for case in $CASES; do
        local out="$OUTROOT/$preset/$arch/$case"
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
    echo "[$(date +%H:%M:%S)] $preset/$arch done"
}

echo "[$(date +%H:%M:%S)] starting: 4 configs x 10 cases = 40 jobs"

run_phase umich_qwen single
run_phase umich_gptoss_planner_umich_qwen_coder planner_coder
run_phase umich_gptoss_planner_routed planner_coder --route-coder-by-difficulty
run_phase umich_gptoss_120b single

echo "ALL DONE"
echo "Output: $OUTROOT"
echo "Analyze with: $PY $PROJECT_ROOT/SWE-agent/scripts/custom/analyze_bucket_matrix.py $OUTROOT"
