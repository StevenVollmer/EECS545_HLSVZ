#!/usr/bin/env bash
# Re-run 9b fair linear baseline (L runs) with 18 iterations.
# Run from project root: bash SWE-agent/scripts/combined/run_local_linear_rerun.sh

set -euo pipefail

RUNNER="python3 SWE-agent/scripts/combined/run_combined.py"
OUT="combined_results"

rm -rf "$OUT/L_c1_9b_linear" "$OUT/L_c2_9b_linear" "$OUT/L_c3_9b_linear"

BASE=(
    --model qwen3.5:9b --api-base http://localhost:11434 --api-key ollama
    --num-ctx 32768 --max-tokens 384
    --agent-architecture single
    --iterations 18 --expansion-candidates 1 --edit-vote-samples 1 --max-node-depth 18
    --reviewer-gate-mode soft
    --no-adaptive-branching --no-failure-surfacing --no-hindsight-feedback
    --no-auto-finalize
    --instances-type file
)

for cs in c1 c2 c3; do
    case "$cs" in
        c1) CASES="SWE-agent/custom_cases"   ;;
        c2) CASES="SWE-agent/custom_cases_2" ;;
        c3) CASES="SWE-agent/custom_cases_3" ;;
    esac
    RUN_ID="L_${cs}_9b_linear"
    echo ""
    echo "========================================"
    echo "  START: $RUN_ID  $(date)"
    echo "========================================"
    $RUNNER "${BASE[@]}" \
        --instances-path "$CASES" \
        --output-dir "$OUT/$RUN_ID" \
        --run-name "$RUN_ID"
    echo "  DONE:  $RUN_ID  $(date)"
done

echo ""
echo "All L runs complete."
