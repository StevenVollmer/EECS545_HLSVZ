#!/usr/bin/env python3
"""Regression evaluation harness for tree_search_custom runs.

Runs a repeatable protocol on a run directory:
1) Submission/token summary from trajectories
2) Patch verification via eval_patches.py internals
3) Tier-3 verified pass-rate check
4) Optional run-to-run comparison command
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_patches import DOCKER_IMAGE, _evaluate, _find_trajs, _load


def _load_hard_case_ids(custom_cases_roots: list[Path]) -> set[str]:
    hard_ids: set[str] = set()
    for custom_cases_root in custom_cases_roots:
        if not custom_cases_root.exists():
            continue
        for case_file in custom_cases_root.glob("*/case.json"):
            try:
                loaded = json.loads(case_file.read_text())
            except Exception:
                continue
            items = loaded if isinstance(loaded, list) else [loaded]
            for item in items:
                if not isinstance(item, dict):
                    continue
                iid = str(item.get("instance_id", "")).strip()
                analysis = item.get("analysis", {}) if isinstance(item.get("analysis"), dict) else {}
                if iid and str(analysis.get("difficulty", "")).lower() == "hard":
                    hard_ids.add(iid)
    # Fallback to canonical Tier-3 custom set when difficulty tags are missing.
    if not hard_ids:
        hard_ids = {
            "budget_snapshot_001",
            "contact_card_001",
            "digest_preview_001",
            "nested_app_001",
            "owner_recap_001",
            "renewal_preview_001",
            "shipment_preview_001",
        }
    return hard_ids


def _parse_case_roots(value: str) -> list[Path]:
    roots: list[Path] = []
    for chunk in value.split(","):
        cleaned = chunk.strip()
        if cleaned:
            roots.append(Path(cleaned))
    return roots


def _traj_summary(run_dir: Path) -> dict[str, Any]:
    trajs = _find_trajs(run_dir)
    rows = []
    for path in trajs:
        traj = _load(path)
        stats = traj.get("stats") or {}
        rows.append(
            {
                "instance_id": str(traj.get("instance_id", path.stem)),
                "submitted": bool(traj.get("submitted", traj.get("info", {}).get("submitted", False))),
                "tok_in": int(stats.get("input_tokens", 0)),
            }
        )
    total = len(rows)
    submitted = sum(1 for row in rows if row["submitted"])
    avg_tok_in = (sum(row["tok_in"] for row in rows) / total) if total else 0.0
    return {
        "rows": rows,
        "total": total,
        "submitted": submitted,
        "submission_rate": (submitted / total) if total else 0.0,
        "avg_tok_in": avg_tok_in,
    }


def _verified_summary(run_dir: Path, image: str, submitted_only: bool) -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []
    for traj_path in _find_trajs(run_dir):
        verdicts.append(_evaluate(traj_path, image=image, submitted_only=submitted_only, verbose=False))
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="tree_search run directory")
    parser.add_argument(
        "--custom-cases-roots",
        default="SWE-agent/custom_cases,SWE-agent/custom_cases_2",
        help="Comma-separated case roots (default: SWE-agent/custom_cases,SWE-agent/custom_cases_2)",
    )
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Optional baseline run dir for manual compare")
    parser.add_argument("--image", default=DOCKER_IMAGE, help=f"Docker image for patch verification (default: {DOCKER_IMAGE})")
    parser.add_argument("--submitted-only", action="store_true", help="Only verify trajectories marked submitted")
    parser.add_argument("--min-submission-rate", type=float, default=0.60)
    parser.add_argument("--min-tier3-pass-rate", type=float, default=0.35)
    parser.add_argument("--max-avg-tok-in", type=float, default=50000.0)
    args = parser.parse_args()

    hard_case_ids = _load_hard_case_ids(_parse_case_roots(args.custom_cases_roots))
    traj = _traj_summary(args.run_dir)
    verdicts = _verified_summary(args.run_dir, image=args.image, submitted_only=args.submitted_only)
    verdict_by_id = {str(v.get("iid", "")): v for v in verdicts}

    hard_total = len(hard_case_ids)
    hard_pass = sum(1 for iid in hard_case_ids if verdict_by_id.get(iid, {}).get("verdict") == "pass")
    hard_pass_rate = (hard_pass / hard_total) if hard_total else 0.0

    print("=" * 88)
    print(f"Run: {args.run_dir}")
    print(f"Cases: {traj['total']}  submitted: {traj['submitted']}  submission_rate: {traj['submission_rate']:.2%}")
    print(f"Avg input tokens per case: {traj['avg_tok_in']:.0f}")
    print(f"Tier-3 verified pass rate: {hard_pass}/{hard_total} = {hard_pass_rate:.2%}")
    print("-" * 88)
    print("Acceptance checks")
    ok_submission = traj["submission_rate"] >= args.min_submission_rate
    ok_tier3 = hard_pass_rate >= args.min_tier3_pass_rate
    ok_tokens = traj["avg_tok_in"] <= args.max_avg_tok_in
    print(f"submission_rate >= {args.min_submission_rate:.0%}: {'PASS' if ok_submission else 'FAIL'}")
    print(f"tier3_verified_pass_rate >= {args.min_tier3_pass_rate:.0%}: {'PASS' if ok_tier3 else 'FAIL'}")
    print(f"avg_tok_in <= {args.max_avg_tok_in:.0f}: {'PASS' if ok_tokens else 'FAIL'}")
    print("=" * 88)

    if args.baseline_dir:
        print("Run this for detailed delta view:")
        print(
            "  python3 SWE-agent/scripts/tree_search_custom/compare_runs.py "
            f"{args.baseline_dir} {args.run_dir}"
        )


if __name__ == "__main__":
    main()
