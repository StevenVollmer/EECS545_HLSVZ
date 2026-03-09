#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR%/SWE-agent/scripts}/env/bin/python"
"$PYTHON_BIN" "$SCRIPT_DIR/run_matrix_easy.py" \
  --preset qwen_local_35b_9b \
  --run-label qwen_local_35b_9b \
  "$@"
