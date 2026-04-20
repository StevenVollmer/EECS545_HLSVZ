#!/usr/bin/env python3
"""Run the combined-agent ablation (variants C–K) on custom_cases_1 and custom_cases_2.

Ablation design:
  A  9b-MCTS baseline              — already run; results in tree_search_runs/
  B  Rafe best (linear)            — already run; results in rafe branch

  ── Remote variants (UMich cluster) ──────────────────────────────────────
  C  Mixed + MCTS                  — 120b planner/reviewer + 30b coder MCTS, heuristic value
  D  Mixed + MCTS + value (30b)    — adds 30b LLM value function
  E  Mixed + MCTS + value + hint   — adds hindsight feedback between dead branches
  F  9b coder + 120b planner/rev   — isolates whether strong planning elevates the 9b coder
  J  30b flat + full swe-search    — 30b for all roles + value fn + hindsight, fair comparison to D/E

  ── Local variants (Ollama, no cluster required) ─────────────────────────
  G  9b + hindsight                — pushes 9b further with cross-branch learning
  H  9b + 9b value function        — tests whether 9b can self-evaluate its own search
  I  9b + full swe-search          — value fn + hindsight on 9b: pre-existing method baseline
  K  9b minimal MCTS + swe-search  — bare UCB1 + swe-search only, no our custom improvements
                                     (no majority vote, no soft gate, no failure surfacing,
                                      no planner/reviewer — the true swe-search-alone baseline)

Key comparisons for the paper:
  K vs A       Our custom techniques (A) vs swe-search alone (K) — same 9b model
  A → I        What swe-search adds on top of our techniques, same 9b model
  J → D/E      What our 120b architecture adds on top of swe-search, same 30b coder
  A → C        What our mixed-size architecture contributes without swe-search
  A → F → C   How much does coder size matter when the planner is 120b?
  H vs D       9b self-eval value fn vs 30b external value fn: does scorer quality matter?

custom_cases_3 is HELD OUT — run it only for final evaluation after ablation is complete.

Usage
-----
  # Dry-run (print all commands):
  python SWE-agent/scripts/combined/run_combined_ablation.py

  # Execute all local variants (G, H) in one terminal:
  python SWE-agent/scripts/combined/run_combined_ablation.py --execute --group local

  # Execute all remote variants (C, D, E, F) in a second terminal:
  python SWE-agent/scripts/combined/run_combined_ablation.py --execute --group remote

  # Only a specific run:
  python SWE-agent/scripts/combined/run_combined_ablation.py --execute --only C_c1

  # Summarize completed runs:
  python SWE-agent/scripts/combined/run_combined_ablation.py --summarize
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model / API constants
# ---------------------------------------------------------------------------

_30B_MODEL    = "openai/Qwen/Qwen3-VL-30B-A3B-Instruct"
_30B_API_BASE = "http://promaxgb10-d668.eecs.umich.edu:8000/v1"
_30B_API_KEY  = "api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy"

_120B_MODEL    = "openai/openai/gpt-oss-120b"
_120B_API_BASE = "http://promaxgb10-d473.eecs.umich.edu:8000/v1"
_120B_API_KEY  = "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"

_9B_MODEL    = "qwen3.5:9b"          # normalized to ollama/qwen3.5:9b in runner
_9B_API_BASE = "http://localhost:11434"
_9B_API_KEY  = "ollama"

# ---------------------------------------------------------------------------
# Test-set paths
# ---------------------------------------------------------------------------

C1 = "SWE-agent/custom_cases"    # Rafe's 20 shared cases
C2 = "SWE-agent/custom_cases_2"  # Steven's 20 cases
C3 = "SWE-agent/custom_cases_3"  # HELD OUT — final evaluation only

# ---------------------------------------------------------------------------
# Shared flag blocks
# ---------------------------------------------------------------------------

_MCTS_COMMON = [
    "--iterations", "18",
    "--expansion-candidates", "1",
    "--edit-vote-samples", "5",
    "--max-node-depth", "18",
    "--agent-architecture", "planner_coder_reviewer",
    "--reviewer-rounds", "2",
    "--reviewer-gate-mode", "soft",
    "--failure-surfacing",
]

# ── Final evaluation baselines ────────────────────────────────────────────

# A: 9b MCTS baseline (all roles = 9b, all our custom techniques, no swe-search)
_A_FLAGS = [
    "--model", _9B_MODEL,
    "--api-base", _9B_API_BASE,
    "--api-key", _9B_API_KEY,
    "--num-ctx", "32768",
    "--max-tokens", "384",
    "--planner-model", _9B_MODEL,
    "--no-hindsight-feedback",
    *_MCTS_COMMON,
]

# B: Rafe linear (120b planner/reviewer + 30b coder, single MCTS pass = no search)
# Runs through run_combined.py with iterations=1 so the evaluator is consistent.
_B_FLAGS = [
    "--model", _30B_MODEL,
    "--api-base", _30B_API_BASE,
    "--api-key", _30B_API_KEY,
    "--planner-model", _120B_MODEL,
    "--planner-api-base", _120B_API_BASE,
    "--planner-api-key", _120B_API_KEY,
    "--reviewer-model", _120B_MODEL,
    "--reviewer-api-base", _120B_API_BASE,
    "--reviewer-api-key", _120B_API_KEY,
    "--max-tokens", "1024",
    "--iterations", "18",
    "--expansion-candidates", "1",
    "--edit-vote-samples", "1",
    "--max-node-depth", "18",
    "--agent-architecture", "planner_coder_reviewer",
    "--reviewer-rounds", "2",
    "--reviewer-gate-mode", "soft",
    "--no-adaptive-branching",
    "--no-failure-surfacing",
    "--no-hindsight-feedback",
]

# ── Remote variants (UMich cluster) ──────────────────────────────────────

# C: 120b planner/reviewer + 30b coder, heuristic value
_C_FLAGS = [
    "--model", _30B_MODEL,
    "--api-base", _30B_API_BASE,
    "--api-key", _30B_API_KEY,
    "--planner-model", _120B_MODEL,
    "--planner-api-base", _120B_API_BASE,
    "--planner-api-key", _120B_API_KEY,
    "--reviewer-model", _120B_MODEL,
    "--reviewer-api-base", _120B_API_BASE,
    "--reviewer-api-key", _120B_API_KEY,
    "--max-tokens", "1024",
    "--no-hindsight-feedback",
    *_MCTS_COMMON,
]

# D: C + 30b LLM value function
_D_FLAGS = [
    *_C_FLAGS,
    "--value-model", _30B_MODEL,
    "--value-api-base", _30B_API_BASE,
    "--value-api-key", _30B_API_KEY,
]

# E: D + hindsight feedback (drop --no-hindsight-feedback inherited from C/D)
_E_FLAGS = [f for f in _D_FLAGS if f != "--no-hindsight-feedback"] + ["--hindsight-feedback"]

# F: 9b coder + 120b planner/reviewer (isolates planner quality vs coder size)
_F_FLAGS = [
    "--model", _9B_MODEL,
    "--api-base", _9B_API_BASE,
    "--api-key", _9B_API_KEY,
    "--num-ctx", "32768",
    "--max-tokens", "512",          # 512 gives planner a bit more room via same flag
    "--planner-model", _120B_MODEL,
    "--planner-api-base", _120B_API_BASE,
    "--planner-api-key", _120B_API_KEY,
    "--reviewer-model", _120B_MODEL,
    "--reviewer-api-base", _120B_API_BASE,
    "--reviewer-api-key", _120B_API_KEY,
    "--no-hindsight-feedback",
    *_MCTS_COMMON,
]

# J: 30b flat + full swe-search (no 120b roles — fair apples-to-apples vs D/E)
# J uses all our custom techniques (majority vote, soft gate, failure surfacing,
# planner-coder-reviewer) but with 30b for every role instead of 120b planner/reviewer.
# J→D isolates exactly what the 120b planner/reviewer contribute on top of swe-search.
_J_FLAGS = [
    "--model", _30B_MODEL,
    "--api-base", _30B_API_BASE,
    "--api-key", _30B_API_KEY,
    "--planner-model", _30B_MODEL,
    "--planner-api-base", _30B_API_BASE,
    "--planner-api-key", _30B_API_KEY,
    "--reviewer-model", _30B_MODEL,
    "--reviewer-api-base", _30B_API_BASE,
    "--reviewer-api-key", _30B_API_KEY,
    "--max-tokens", "1024",
    "--hindsight-feedback",
    "--value-model", _30B_MODEL,
    "--value-api-base", _30B_API_BASE,
    "--value-api-key", _30B_API_KEY,
    *_MCTS_COMMON,
]

# ── Local variants (Ollama only) ─────────────────────────────────────────

# G: 9b + hindsight (fully local)
_G_FLAGS = [
    "--model", _9B_MODEL,
    "--api-base", _9B_API_BASE,
    "--api-key", _9B_API_KEY,
    "--num-ctx", "32768",
    "--max-tokens", "384",
    "--planner-model", _9B_MODEL,
    "--hindsight-feedback",
    *_MCTS_COMMON,
]

# I: 9b + full swe-search (value fn + hindsight, same model as A — swe-search replication)
# This is the pre-existing-method baseline. A→I shows what swe-search alone contributes.
# I→{D,E} shows what our mixed-size architecture adds on top of swe-search.
_I_FLAGS = [
    "--model", _9B_MODEL,
    "--api-base", _9B_API_BASE,
    "--api-key", _9B_API_KEY,
    "--num-ctx", "32768",
    "--max-tokens", "384",
    "--planner-model", _9B_MODEL,
    "--hindsight-feedback",
    "--value-model", _9B_MODEL,
    "--value-api-base", _9B_API_BASE,
    "--value-api-key", _9B_API_KEY,
    *_MCTS_COMMON,
]

# H: 9b + 9b value function (fully local self-evaluation)
_H_FLAGS = [
    "--model", _9B_MODEL,
    "--api-base", _9B_API_BASE,
    "--api-key", _9B_API_KEY,
    "--num-ctx", "32768",
    "--max-tokens", "384",
    "--planner-model", _9B_MODEL,
    "--no-hindsight-feedback",
    "--value-model", _9B_MODEL,
    "--value-api-base", _9B_API_BASE,
    "--value-api-key", _9B_API_KEY,
    *_MCTS_COMMON,
]

# K: minimal MCTS + swe-search (no our custom improvements — true swe-search-alone baseline)
# Strips majority vote (edit_vote_samples=1), adaptive branching, failure surfacing,
# and planner/reviewer entirely. Only bare UCB1 + swe-search value fn + hindsight.
# K vs A is the core comparison: do our techniques beat swe-search alone?
_K_FLAGS = [
    "--model", _9B_MODEL,
    "--api-base", _9B_API_BASE,
    "--api-key", _9B_API_KEY,
    "--num-ctx", "32768",
    "--max-tokens", "384",
    "--agent-architecture", "single",   # no planner or reviewer
    "--edit-vote-samples", "1",         # no majority vote
    "--no-adaptive-branching",
    "--no-failure-surfacing",
    "--hindsight-feedback",
    "--value-model", _9B_MODEL,
    "--value-api-base", _9B_API_BASE,
    "--value-api-key", _9B_API_KEY,
    "--iterations", "18",
    "--expansion-candidates", "1",
    "--max-node-depth", "18",
    "--reviewer-gate-mode", "soft",     # unused with single-agent but required by argparse
]

# ---------------------------------------------------------------------------
# Group membership
# ---------------------------------------------------------------------------

# Local = any run that touches the local GPU (Ollama).
#   F uses local GPU for the 9b coder even though its planner is remote — so F is local.
#   Within the local group, runs that also call a remote LLM are ordered first so they
#   complete before a VPN timeout can interrupt them.
LOCAL_PREFIXES  = {"A", "F", "G", "H", "I", "K"}
REMOTE_PREFIXES = {"B", "C", "D", "E", "J"}        # pure cluster runs, no local GPU


# ---------------------------------------------------------------------------
# Build run list
# ---------------------------------------------------------------------------

def _cmd(run_id: str, extra_flags: list[str], instances_path: str, output_root: Path,
         resume: bool) -> list[str]:
    return [
        "python3",
        "SWE-agent/scripts/combined/run_combined.py",
        "--instances-type", "file",
        "--instances-path", instances_path,
        "--output-dir", str(output_root / run_id),
        "--run-name", run_id,
        *extra_flags,
        *(["--resume"] if resume else []),
    ]


def build_runs(output_root: Path, resume: bool) -> list[tuple[str, list[str], str]]:
    """Return (run_id, command, instances_path) in execution order."""
    runs: list[tuple[str, list[str], str]] = []

    def add(run_id: str, flags: list[str], path: str) -> None:
        runs.append((run_id, _cmd(run_id, flags, path, output_root, resume), path))

    # ── Local group (uses local GPU) ─────────────────────────────────────────
    # Runs that also call a remote LLM come first so they finish before VPN timeout.

    # F: local 9b coder + remote 120b planner/reviewer — remote LLM, so goes first
    add("F_c1_9b_120b_planner",     _F_FLAGS, C1)
    add("F_c2_9b_120b_planner",     _F_FLAGS, C2)

    # Pure-local 9b runs (no remote LLM — safe after VPN drops)
    add("A_c1_9b_mcts",             _A_FLAGS, C1)
    add("A_c2_9b_mcts",             _A_FLAGS, C2)
    add("A_c3_9b_mcts",             _A_FLAGS, C3)
    add("G_c1_9b_hindsight",        _G_FLAGS, C1)
    add("G_c2_9b_hindsight",        _G_FLAGS, C2)
    add("H_c1_9b_selfeval",         _H_FLAGS, C1)
    add("H_c2_9b_selfeval",         _H_FLAGS, C2)
    add("I_c1_sweSearch_9b",        _I_FLAGS, C1)
    add("I_c2_sweSearch_9b",        _I_FLAGS, C2)
    add("K_c1_minimal_sweSearch",   _K_FLAGS, C1)
    add("K_c2_minimal_sweSearch",   _K_FLAGS, C2)

    # ── Remote group (pure cluster, no local GPU) ─────────────────────────
    add("B_c1_rafe_linear",         _B_FLAGS, C1)
    add("B_c2_rafe_linear",         _B_FLAGS, C2)
    add("B_c3_rafe_linear",         _B_FLAGS, C3)
    add("C_c1_mixed_mcts",          _C_FLAGS, C1)
    add("C_c2_mixed_mcts",          _C_FLAGS, C2)
    add("C_c3_mixed_mcts",          _C_FLAGS, C3)
    add("D_c1_mixed_value",         _D_FLAGS, C1)
    add("D_c2_mixed_value",         _D_FLAGS, C2)
    add("E_c1_mixed_hindsight",     _E_FLAGS, C1)
    add("E_c2_mixed_hindsight",     _E_FLAGS, C2)
    add("J_c1_30b_sweSearch",       _J_FLAGS, C1)
    add("J_c2_30b_sweSearch",       _J_FLAGS, C2)

    return runs


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _count_trajs(run_dir: Path) -> int:
    return len(list(run_dir.rglob("*.traj"))) if run_dir.exists() else 0


def _expected_cases(instances_path: str) -> int:
    path = Path(instances_path)
    if not path.exists():
        return 0
    count = 0
    for d in path.iterdir():
        if not d.is_dir():
            continue
        for name in ("case.json", "case.yaml", "case.yml"):
            cf = d / name
            if cf.exists():
                try:
                    import yaml
                    loaded = yaml.safe_load(cf.read_text())
                    count += len(loaded) if isinstance(loaded, list) else 1
                except Exception:
                    count += 1
                break
    return count


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

_RAFE_BENCHMARK_ROOT = Path("SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud")
_RAFE_BEST_VARIANT   = "umich_gptoss_planner_umich_qwen_coder/planner_coder"
_RAFE_CASES_ROOT     = Path("SWE-agent/custom_cases")


def _eval_rafe_patches(variant_dir: Path, cases_root: Path) -> tuple[int, int]:
    """Apply patches from pred files and run success_checks; return (passed, total)."""
    import shutil, tempfile
    passed = 0
    total = 0
    for inst_dir in sorted(variant_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        pred_files = list(inst_dir.rglob("*.pred"))
        if not pred_files:
            total += 1; continue
        pred = json.loads(pred_files[0].read_text())
        patch = pred.get("model_patch", "").strip()
        case_name = inst_dir.name
        case_json = cases_root / case_name / "case.json"
        total += 1
        if not patch or not case_json.exists():
            continue
        case_data = json.loads(case_json.read_text())[0]
        repo_path = cases_root / case_name / case_data["repo_path"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_repo = Path(tmpdir) / "repo"
            shutil.copytree(repo_path, tmp_repo)
            import subprocess as _sp
            _sp.run(["git", "init"], cwd=tmp_repo, capture_output=True)
            patch_file = Path(tmpdir) / "changes.patch"
            patch_file.write_text(patch + "\n")
            r = _sp.run(
                ["git", "apply", "--recount", "--ignore-whitespace", str(patch_file)],
                capture_output=True, text=True, cwd=tmp_repo,
            )
            if r.returncode != 0:
                continue
            ok = True
            for check in case_data["evaluation"]["success_checks"]:
                r = _sp.run(check["command"], shell=True, capture_output=True, text=True, cwd=tmp_repo)
                out = r.stdout + r.stderr
                if r.returncode != check.get("expect_exit_code", 0): ok = False; break
                if any(s not in out for s in check.get("stdout_contains", [])): ok = False; break
                if any(s in out for s in check.get("stdout_not_contains", [])): ok = False; break
            if ok:
                passed += 1
    return passed, total


def _summarize(output_root: Path) -> None:
    runs = sorted(output_root.iterdir()) if output_root.exists() else []
    if not runs:
        print(f"No runs found under {output_root}")
        return

    print(f"\n{'Run':<40}  {'Pass':>4}  {'Total':>5}  {'%':>5}")
    print("-" * 60)

    # Prepend Rafe B baseline rows (evaluated from pred files)
    rafe_variant_dir = _RAFE_BENCHMARK_ROOT / _RAFE_BEST_VARIANT
    if rafe_variant_dir.exists():
        p, t = _eval_rafe_patches(rafe_variant_dir, _RAFE_CASES_ROOT)
        pct = f"{100 * p / t:.0f}%" if t else "—"
        print(f"{'B_c1_rafe_linear (evaluated)':<40}  {p:>4}  {t:>5}  {pct:>5}")
    else:
        print(f"{'B_c1_rafe_linear':<40}  {'—':>4}  {'—':>5}  {'—':>5}")

    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        trajs = list(run_dir.rglob("*.traj"))
        passed = sum(
            1 for t in trajs
            if (lambda d: d.get("submitted") or d.get("info", {}).get("submitted"))(
                json.loads(t.read_text()) if t.exists() else {}
            )
        )
        total = len(trajs)
        pct = f"{100 * passed / total:.0f}%" if total else "—"
        print(f"{run_dir.name:<40}  {passed:>4}  {total:>5}  {pct:>5}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("SWE-agent/tree_search_runs/combined"),
    )
    parser.add_argument("--execute", action="store_true",
                        help="Run commands instead of printing only")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip instances that already have a .traj file (default: on)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated run ID prefixes to run (e.g. C_c1,G_c1)")
    parser.add_argument("--group", choices=["local", "remote"], default=None,
                        help="local = all runs using the local GPU (A,F,G,H,I,K); remote = pure cluster runs (B,C,D,E,J)")
    parser.add_argument("--summarize", action="store_true",
                        help="Print result summary from completed runs and exit")
    args = parser.parse_args()

    if args.summarize:
        _summarize(args.output_root)
        return

    # Determine which run IDs to include.
    # --group and --only can be combined:
    #   - group applies by variant prefix letter (A/B/C...)
    #   - only applies by run_id prefix (e.g. C_c1)
    explicit_only: set[str] | None = None
    if args.only:
        explicit_only = {s.strip() for s in args.only.split(",") if s.strip()}

    group_prefixes: set[str] | None = None
    if args.group == "local":
        group_prefixes = LOCAL_PREFIXES
    elif args.group == "remote":
        group_prefixes = REMOTE_PREFIXES

    runs = build_runs(args.output_root, args.resume)

    print(f"Combined-agent ablation — output root: {args.output_root}")
    if args.group:
        group_map = {"local": LOCAL_PREFIXES, "remote": REMOTE_PREFIXES}
        print(f"Group: {args.group} ({', '.join(sorted(group_map[args.group]))})")
    print(f"{'Run ID':<40}  {'Status':<22}  Command")
    print("-" * 110)

    for run_id, cmd, instances_path in runs:
        if group_prefixes and run_id.split("_")[0] not in group_prefixes:
            print(f"{run_id:<40}  {'[skipped]':<22}")
            continue
        if explicit_only and not any(run_id.startswith(oid) for oid in explicit_only):
            print(f"{run_id:<40}  {'[skipped]':<22}")
            continue

        run_dir = args.output_root / run_id
        done = _count_trajs(run_dir)
        expected = _expected_cases(instances_path)
        status = f"{done}/{expected} done"

        if args.execute and done >= expected > 0:
            print(f"{run_id:<40}  {status + ' — skipping':<22}")
            continue

        printable = " ".join(cmd)
        print(f"{run_id:<40}  {status:<22}  {printable}")

        if args.execute:
            print(f"\n{'='*60}\nStarting: {run_id}  ({done}/{expected} already done)\n{'='*60}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\n[WARNING] {run_id} exited with code {result.returncode} — continuing")

    if args.execute:
        print("\nAll runs complete.")
        _summarize(args.output_root)


if __name__ == "__main__":
    main()
