#!/bin/bash
# Re-runs just the routed config (10 jobs) against the existing baselines
# in the routed_matrix output dir. Use this to iterate on the planner prompt
# without re-running the baselines.
set -u
PROJECT_ROOT="/Users/rafe/classes/eecs545/project"
PY="$PROJECT_ROOT/env/bin/python"
RUNNER="$PROJECT_ROOT/SWE-agent/scripts/custom/run_custom_swebench.py"
CASES_ROOT="$PROJECT_ROOT/SWE-agent/custom_cases"
OUTROOT="$PROJECT_ROOT/SWE-agent/custom_matrix_runs/routed_matrix"

CASES="date_parse_locale dep_cycle_detect numeric_drift_sum pagination_drift path_normalizer_cache retry_cap schema_migration_check search_hit_localize stable_ranking weighted_median"

preset=umich_gptoss_planner_routed
arch=planner_coder

# Clean only the routed subtree; leave baselines intact.
rm -rf "$OUTROOT/$preset"
mkdir -p "$OUTROOT/$preset/$arch"

for case in $CASES; do
    out="$OUTROOT/$preset/$arch/$case"
    mkdir -p "$out"
    "$PY" "$RUNNER" \
        --preset "$preset" \
        --agent-architecture "$arch" \
        --route-coder-by-difficulty \
        --instances-type file \
        --instances-path "$CASES_ROOT/$case" \
        --filter "${case}_001" \
        --output-dir "$out" \
        --max-turns 40 \
        > "$out/runner.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] $preset/$arch done"
echo "Analyze with: $PY $PROJECT_ROOT/SWE-agent/scripts/custom/analyze_bucket_matrix.py $OUTROOT"
