#!/bin/bash
# Runs the two missing configs: qwen->qwen and gpt->gpt
# Does NOT touch existing results in the output dir.
set -u
PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
RUNNER="$PROJECT_ROOT/SWE-agent/scripts/custom/run_custom_swebench.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/custom_matrix_runs/routed_matrix"

CASES="date_parse_locale dep_cycle_detect numeric_drift_sum pagination_drift path_normalizer_cache retry_cap schema_migration_check search_hit_localize stable_ranking weighted_median"

# --- qwen -> qwen (qwen planner + qwen coder) ---
label1=umich_qwen_planner_qwen_coder
arch1=planner_coder

rm -rf "$OUTROOT/$label1"
mkdir -p "$OUTROOT/$label1/$arch1"

for case in $CASES; do
    out="$OUTROOT/$label1/$arch1/$case"
    mkdir -p "$out"
    "$PY" "$RUNNER" \
        --preset umich_qwen \
        --agent-architecture "$arch1" \
        --planner-model "openai/Qwen/Qwen3-VL-30B-A3B-Instruct" \
        --planner-api-base "http://promaxgb10-d668.eecs.umich.edu:8000/v1" \
        --planner-api-key "api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy" \
        --instances-type file \
        --instances-path "$CASES_ROOT/$case" \
        --filter "${case}_001" \
        --output-dir "$out" \
        --max-turns 40 \
        > "$out/runner.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] $label1 done"

# --- gpt -> gpt (gpt planner + gpt coder) ---
label2=umich_gptoss_planner_gptoss_coder
arch2=planner_coder

rm -rf "$OUTROOT/$label2"
mkdir -p "$OUTROOT/$label2/$arch2"

for case in $CASES; do
    out="$OUTROOT/$label2/$arch2/$case"
    mkdir -p "$out"
    "$PY" "$RUNNER" \
        --preset umich_gptoss_120b \
        --agent-architecture "$arch2" \
        --planner-model "openai/openai/gpt-oss-120b" \
        --planner-api-base "http://promaxgb10-d473.eecs.umich.edu:8000/v1" \
        --planner-api-key "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a" \
        --instances-type file \
        --instances-path "$CASES_ROOT/$case" \
        --filter "${case}_001" \
        --output-dir "$out" \
        --max-turns 40 \
        > "$out/runner.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] $label2 done"

# --- qwen -> qwen + critic (qwen planner + critic + qwen coder) ---
label3=umich_qwen_planner_critic_qwen_coder
arch3=planner_coder

rm -rf "$OUTROOT/$label3"
mkdir -p "$OUTROOT/$label3/$arch3"

for case in $CASES; do
    out="$OUTROOT/$label3/$arch3/$case"
    mkdir -p "$out"
    "$PY" "$RUNNER" \
        --preset umich_qwen \
        --agent-architecture "$arch3" \
        --planner-model "openai/Qwen/Qwen3-VL-30B-A3B-Instruct" \
        --planner-api-base "http://promaxgb10-d668.eecs.umich.edu:8000/v1" \
        --planner-api-key "api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy" \
        --plan-critic \
        --instances-type file \
        --instances-path "$CASES_ROOT/$case" \
        --filter "${case}_001" \
        --output-dir "$out" \
        --max-turns 40 \
        > "$out/runner.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] $label3 done"

# --- gpt -> gpt + critic (gpt planner + critic + gpt coder) ---
label4=umich_gptoss_planner_critic_gptoss_coder
arch4=planner_coder

rm -rf "$OUTROOT/$label4"
mkdir -p "$OUTROOT/$label4/$arch4"

for case in $CASES; do
    out="$OUTROOT/$label4/$arch4/$case"
    mkdir -p "$out"
    "$PY" "$RUNNER" \
        --preset umich_gptoss_120b \
        --agent-architecture "$arch4" \
        --planner-model "openai/openai/gpt-oss-120b" \
        --planner-api-base "http://promaxgb10-d473.eecs.umich.edu:8000/v1" \
        --planner-api-key "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a" \
        --plan-critic \
        --instances-type file \
        --instances-path "$CASES_ROOT/$case" \
        --filter "${case}_001" \
        --output-dir "$out" \
        --max-turns 40 \
        > "$out/runner.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] $label4 done"

echo "ALL DONE"
