#!/usr/bin/env bash
set -euo pipefail

find SWE-agent/config -name "*.yaml" -type f -exec sed -i '/^agent:/a\  use_dspy: true' {} \;
