#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-SWE-agent/custom_matrix_runs/benchmark_round_split_compare}"

./env/bin/python SWE-agent/scripts/custom/analyze_custom_runs.py \
  "${ROOT}" \
  --json \
  --write-json "${ROOT}/analysis.summary.json"

./env/bin/python SWE-agent/scripts/custom/render_custom_matrix_report.py \
  "${ROOT}" \
  --output "${ROOT}/README.md"

echo
echo "Wrote:"
echo "  ${ROOT}/analysis.summary.json"
echo "  ${ROOT}/README.md"
