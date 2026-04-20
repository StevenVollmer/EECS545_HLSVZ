#!/usr/bin/env python3
"""Render a bucket-analysis table from a routed_matrix output root.

Usage:
    python analyze_bucket_matrix.py /path/to/routed_matrix

Prints a per-case table of pass/fail across configs, then classifies each case
into a bucket (trivial / plan-rescuable / needs-large / anti-plan / impossible)
and summarizes compute burden for the routed preset.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/rafe/classes/eecs545/project")
ANALYZER = PROJECT_ROOT / "SWE-agent" / "scripts" / "custom" / "analyze_custom_runs.py"
PY = PROJECT_ROOT / "env" / "bin" / "python"
CASES_ROOT = PROJECT_ROOT / "SWE-agent" / "custom_cases"

CASES = [
    # Original 20
    "board_rollup",
    "budget_snapshot",
    "contact_card",
    "digest_preview",
    "incident_brief",
    "invoice_footer",
    "label_formatter",
    "median_window",
    "milestone_rollup",
    "nested_app",
    "owner_recap",
    "owner_sort",
    "priority_snapshot",
    "renewal_preview",
    "risk_score",
    "shipment_preview",
    "simple_mean_bug",
    "status_slug",
    "team_digest",
    "workspace_digest",
    # New 7
    "numeric_drift_sum",
    "pagination_drift",
    "path_normalizer_cache",
    "retry_cap",
    "search_hit_localize",
    "stable_ranking",
    "weighted_median",
]

# (label, preset, architecture)
CONFIGS = [
    ("qwen", "umich_qwen", "single"),
    ("qw->qw", "umich_qwen_planner_qwen_coder", "planner_coder"),
    ("qw->qw+C", "umich_qwen_planner_critic_qwen_coder", "planner_coder"),
    ("gpt->qw", "umich_gptoss_planner_umich_qwen_coder", "planner_coder"),
    ("gpt->qw+R", "umich_gptoss_planner_umich_qwen_coder_reviewer", "planner_coder_reviewer"),
    ("gpt->qw+C", "umich_gptoss_planner_critic_qwen_coder", "planner_coder"),
    ("gpt->qw+CR", "umich_gptoss_planner_critic_qwen_coder_reviewer", "planner_coder_reviewer"),
    ("gpt->gpt", "umich_gptoss_planner_gptoss_coder", "planner_coder"),
    ("gpt->gpt+C", "umich_gptoss_planner_critic_gptoss_coder", "planner_coder"),
    ("gpt", "umich_gptoss_120b", "single"),
]


def analyze_run(run_dir: Path) -> dict:
    if not run_dir.is_dir():
        return {}
    proc = subprocess.run(
        [str(PY), str(ANALYZER), str(run_dir), "--cases-root", str(CASES_ROOT), "--json"],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    if not data.get("results"):
        return {}
    return data["results"][0]


def bucket_label(passed: dict[str, bool]) -> str:
    q = passed.get("qwen", False)
    gq = passed.get("gpt->qwen", False)
    g = passed.get("gpt", False)
    if q and g:
        return "trivial"
    if not q and gq and g:
        return "plan-rescuable"
    if not q and not gq and g:
        return "needs-large"
    if q and not gq:
        return "anti-plan"
    if not any([q, gq, g]):
        return "impossible"
    return "other"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <routed_matrix_output_root>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    rows: dict[str, dict[str, dict]] = {}
    for case in CASES:
        rows[case] = {}
        for label, preset, arch in CONFIGS:
            d = root / preset / arch / case
            rows[case][label] = analyze_run(d)

    labels = [c[0] for c in CONFIGS]
    col_w = 18

    # Pass/fail + score grid
    header = f"{'case':<25}" + "".join(f"{l:<{col_w}}" for l in labels) + "bucket"
    print(header)
    print("-" * len(header))
    buckets: dict[str, str] = {}
    for case, res in rows.items():
        passed = {l: bool(res[l].get("success_passed")) for l in labels}
        bkt = bucket_label(passed)
        buckets[case] = bkt
        cells = []
        for l in labels:
            r = res[l]
            p = passed[l]
            s = r.get("total_score", "-")
            cells.append(f"{'PASS' if p else 'FAIL'}({s})")
        print(f"{case:<25}" + "".join(f"{c:<{col_w}}" for c in cells) + bkt)

    # Bucket counts
    print("\nBucket counts:")
    from collections import Counter
    for k, v in Counter(buckets.values()).most_common():
        print(f"  {k:<20} {v}")

    # Resolved rate per config
    print("\nResolved rate per config:")
    for l in labels:
        passed_ct = sum(1 for c in CASES if bool(rows[c][l].get("success_passed")))
        print(f"  {l:<18} {passed_ct}/{len(CASES)}  ({passed_ct/len(CASES):.2f})")

    # Compute burden per config
    print("\nAvg relative compute burden (gpt-4o-mini equivalent):")
    for l in labels:
        vals = [rows[c][l].get("relative_compute_to_4o_mini") for c in CASES]
        vals = [v for v in vals if v is not None]
        if vals:
            print(f"  {l:<18} {sum(vals)/len(vals):.2f}")

    # Routing diagnostics
    print("\nRouter decisions (from routed config):")
    router_col = rows
    for case in CASES:
        r = rows[case].get("routed", {})
        # The routed_difficulty is stored in role_model_stats.coder.routed_difficulty
        role_stats = r.get("role_model_stats", {})
        coder_stats = role_stats.get("coder", {}) if isinstance(role_stats, dict) else {}
        rd = coder_stats.get("routed_difficulty", "?")
        used_model = coder_stats.get("model", "?")
        # compact model name
        short = used_model.split("/")[-1] if isinstance(used_model, str) else "?"
        print(f"  {case:<25} difficulty={rd:<8} coder={short}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
