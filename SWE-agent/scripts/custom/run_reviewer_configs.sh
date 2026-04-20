#!/bin/bash
# Runs reviewer and critic+reviewer configs against the 10 cases.
# Baselines (qwen, gpt->qwen, routed, critic, gpt) are left intact.
set -u
PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
RUNNER="$PROJECT_ROOT/SWE-agent/scripts/custom/run_custom_swebench.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/custom_matrix_runs/routed_matrix"

CASES="date_parse_locale dep_cycle_detect numeric_drift_sum pagination_drift path_normalizer_cache retry_cap schema_migration_check search_hit_localize stable_ranking weighted_median"

# --- Config 1: reviewer (planner + coder + reviewer, no critic) ---
preset1=umich_gptoss_planner_umich_qwen_coder
arch1=planner_coder_reviewer
label1="${preset1}_reviewer"

rm -rf "$OUTROOT/$label1"
mkdir -p "$OUTROOT/$label1/$arch1"

for case in $CASES; do
    out="$OUTROOT/$label1/$arch1/$case"
    mkdir -p "$out"
    "$PY" "$RUNNER" \
        --preset "$preset1" \
        --agent-architecture "$arch1" \
        --reviewer-rounds 2 \
        --instances-type file \
        --instances-path "$CASES_ROOT/$case" \
        --filter "${case}_001" \
        --output-dir "$out" \
        --max-turns 40 \
        > "$out/runner.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] $label1/$arch1 done"

# --- Config 2: critic+reviewer (planner + critic + coder + reviewer) ---
preset2=umich_gptoss_planner_critic_qwen_coder
arch2=planner_coder_reviewer
label2="${preset2}_reviewer"

rm -rf "$OUTROOT/$label2"
mkdir -p "$OUTROOT/$label2/$arch2"

for case in $CASES; do
    out="$OUTROOT/$label2/$arch2/$case"
    mkdir -p "$out"
    "$PY" "$RUNNER" \
        --preset "$preset2" \
        --agent-architecture "$arch2" \
        --plan-critic \
        --reviewer-rounds 2 \
        --instances-type file \
        --instances-path "$CASES_ROOT/$case" \
        --filter "${case}_001" \
        --output-dir "$out" \
        --max-turns 40 \
        > "$out/runner.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] $label2/$arch2 done"

echo "ALL DONE"
echo "Analyze with: $PY $PROJECT_ROOT/SWE-agent/scripts/custom/analyze_bucket_matrix.py $OUTROOT"
