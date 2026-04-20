#!/usr/bin/env python3
"""Blind reviewer audit — all three reviewer sizes on all trajectories.

Re-runs the reviewer on every completed trajectory WITHOUT the success_check
oracle (no test results, no validation report).  Runs each traj through all
three reviewer sizes (9b / 30b / 120b) so we can compare model-size impact on
reviewer decision quality independently of which agent produced the patch.

Ground truth: traj `submitted` field (True = success_checks passed in runner).

Usage:
  python audit_reviewer.py                          # all trajs × all 3 reviewers
  python audit_reviewer.py --sizes 120b             # one size only
  python audit_reviewer.py --run C_c1               # one run prefix
  python audit_reviewer.py --dry-run                # enumerate, no LLM calls
  python audit_reviewer.py --cache cache.json       # resume interrupted run
  python audit_reviewer.py --csv results.csv

Parallel workers (split by size + shard):
  # Terminal 1 — 9b local (no sharding needed)
  python audit_reviewer.py --sizes 9b

  # Terminals 2-4 — 30b on cluster, 3 workers
  python audit_reviewer.py --sizes 30b --shard 1/3
  python audit_reviewer.py --sizes 30b --shard 2/3
  python audit_reviewer.py --sizes 30b --shard 3/3

  # Terminals 5-7 — 120b on cluster, 3 workers
  python audit_reviewer.py --sizes 120b --shard 1/3
  python audit_reviewer.py --sizes 120b --shard 2/3
  python audit_reviewer.py --sizes 120b --shard 3/3

  # After all workers finish — merge and report
  python audit_reviewer.py --merge "audit_cache_*.json" --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import time
from dataclasses import dataclass, asdict

ROOT          = pathlib.Path(__file__).resolve().parents[3]
COMBINED_RUNS = ROOT / "SWE-agent/tree_search_runs/combined"
LEGACY_A_C1   = ROOT / "SWE-agent/tree_search_runs/all_custom_run_v10"
LEGACY_A_C2   = ROOT / "SWE-agent/tree_search_runs/custom_cases_2_baseline_9b"
CASES_DIRS    = [
    ROOT / "SWE-agent/custom_cases",
    ROOT / "SWE-agent/custom_cases_2",
    ROOT / "SWE-agent/custom_cases_3",
]

# ---------------------------------------------------------------------------
# Named reviewer specs (size label → model, api_base, api_key)
# ---------------------------------------------------------------------------

REVIEWERS: dict[str, tuple[str, str, str]] = {
    "9b":  ("ollama/qwen3.5:9b",
            "http://localhost:11434",
            "ollama"),
    "30b": ("openai/Qwen/Qwen3-VL-30B-A3B-Instruct",
            "http://promaxgb10-d668.eecs.umich.edu:8000/v1",
            "api_RPnuSxgxJQamqW04ma9uJW27vc4TyBdy"),
    "120b":("openai/openai/gpt-oss-120b",
            "http://promaxgb10-d473.eecs.umich.edu:8000/v1",
            "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"),
}
SIZE_ORDER = ["9b", "30b", "120b"]

# ---------------------------------------------------------------------------
# Blind reviewer prompts (no oracle)
# ---------------------------------------------------------------------------

_BLIND_REVIEWER_SYSTEM = """\
You are a code reviewer for a software repair task.

You have NOT been given any test results or acceptance criteria commands.
Evaluate the patch purely from:
  1. The problem description and stated root cause
  2. The quality and correctness of the code change itself

Work through these gates in order. Stop at the first "no":
1. Is the patch non-empty and does it NOT modify test files?
2. Does the change target the code path described in the root cause?
3. Does the implementation look logically correct (no hardcoding, no obvious
   off-by-one, no dropped edge cases)?
4. Is the fix complete (nothing obviously missing)?
All yes → accept.

Output JSON only. No markdown fences. No thinking out loud.
Return exactly these keys:
- decision: "accept" or "revise"
- primary_reason: one sentence
- confidence: "high", "medium", or "low"
- required_changes: array of at most 2 strings (empty if accepting)
"""


def _blind_task_prompt(
    *,
    problem_statement: str,
    root_cause_hypothesis: str,
    files_likely_affected: list[str],
    changed_files: list[str],
    patch: str,
) -> str:
    return json.dumps({
        "problem_statement":              problem_statement[:2000],
        "root_cause_hypothesis":          root_cause_hypothesis[:1000],
        "files_likely_affected_by_planner": files_likely_affected[:10],
        "files_actually_changed":         changed_files[:10],
        "patch":                          patch[:4000],
    }, indent=2)


# ---------------------------------------------------------------------------
# Case lookup
# ---------------------------------------------------------------------------

_case_cache: dict[str, dict | None] = {}

def _find_case(instance_id: str) -> dict | None:
    if instance_id in _case_cache:
        return _case_cache[instance_id]
    base = instance_id.rsplit("_", 1)[0] if instance_id.endswith("_001") else instance_id
    for cases_dir in CASES_DIRS:
        for candidate in [instance_id, base]:
            cj = cases_dir / candidate / "case.json"
            if cj.exists():
                try:
                    rows = json.loads(cj.read_text())
                    result = rows[0] if rows else None
                    _case_cache[instance_id] = result
                    return result
                except Exception:
                    pass
    _case_cache[instance_id] = None
    return None


# ---------------------------------------------------------------------------
# Per-instance audit record (one per traj × reviewer size)
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    run_id:              str
    instance_id:         str
    reviewer_size:       str    # "9b" | "30b" | "120b"
    actually_solved:     bool
    reviewer_decision:   str    # "accept" | "revise" | "error"
    reviewer_confidence: str    # "high" | "medium" | "low" | ""
    reviewer_reason:     str
    has_patch:           bool
    tokens_in:           int = 0
    tokens_out:          int = 0
    cached:              bool = False

    @property
    def tp(self) -> bool: return self.actually_solved     and self.reviewer_decision == "accept"
    @property
    def fp(self) -> bool: return not self.actually_solved and self.reviewer_decision == "accept"
    @property
    def fn(self) -> bool: return self.actually_solved     and self.reviewer_decision == "revise"
    @property
    def tn(self) -> bool: return not self.actually_solved and self.reviewer_decision == "revise"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_blind_reviewer(
    *,
    model: str,
    api_base: str,
    api_key: str,
    task_prompt: str,
    max_tokens: int = 512,
    num_ctx: int | None = 8192,
) -> tuple[dict, int, int]:
    import litellm  # type: ignore
    litellm.suppress_debug_info = True

    is_ollama = model.startswith("ollama/") or "11434" in api_base
    kwargs: dict = {
        "model":      model,
        "api_base":   api_base,
        "api_key":    api_key,
        "messages":   [
            {"role": "system", "content": _BLIND_REVIEWER_SYSTEM},
            {"role": "user",   "content": task_prompt},
        ],
        "temperature": 0.0,
        "max_tokens":  max_tokens,
    }
    if is_ollama:
        kwargs["reasoning_effort"] = "none"
        kwargs["response_format"]  = {"type": "json_object"}
        if num_ctx is not None:
            kwargs["num_ctx"] = num_ctx

    total_in = total_out = 0
    raw = ""
    messages = list(kwargs["messages"])

    for attempt in range(2):
        kwargs["messages"] = messages
        try:
            resp = litellm.completion(**kwargs)
        except Exception as exc:
            return {"decision": "error", "primary_reason": str(exc)[:200],
                    "confidence": "", "required_changes": []}, 0, 0

        usage = getattr(resp, "usage", None)
        total_in  += int(getattr(usage, "prompt_tokens",     0) or 0)
        total_out += int(getattr(usage, "completion_tokens", 0) or 0)
        raw = resp.choices[0].message.content or ""

        stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), total_in, total_out
            except json.JSONDecodeError:
                pass

        if attempt == 0:
            messages = [
                *kwargs["messages"],
                {"role": "assistant", "content": raw},
                {"role": "user",      "content":
                    "Return your answer as one valid JSON object only. No markdown."},
            ]

    return {"decision": "error", "primary_reason": "parse_failed",
            "confidence": "", "required_changes": [], "_raw": raw[:300]}, total_in, total_out


# ---------------------------------------------------------------------------
# Extract reusable context from a traj (done once per traj, shared across sizes)
# ---------------------------------------------------------------------------

@dataclass
class TrajContext:
    run_id:          str
    instance_id:     str
    actually_solved: bool
    patch:           str
    task_prompt:     str   # pre-built blind task prompt


def _load_traj_context(traj_path: pathlib.Path, run_id: str) -> TrajContext:
    d = json.loads(traj_path.read_text())
    instance_id     = d.get("instance_id", traj_path.stem)
    actually_solved = bool(d.get("submitted") or d.get("info", {}).get("submitted"))
    patch           = (d.get("patch") or "").strip()

    ph          = d.get("planner_handoff") or {}
    root_cause  = str(ph.get("root_cause_hypothesis", ""))
    files_plan  = [str(f) for f in (ph.get("files_likely_affected") or [])[:8]]

    changed_files: list[str] = []
    for rnd in (d.get("coder_rounds") or []):
        ls = (rnd.get("loop_state") or {}) if isinstance(rnd, dict) else {}
        changed_files = [str(f) for f in (ls.get("changed_files") or []) if str(f).strip()]

    case         = _find_case(instance_id)
    problem_stmt = str(case.get("problem_statement", "")) if case else ""

    task_prompt = _blind_task_prompt(
        problem_statement=problem_stmt,
        root_cause_hypothesis=root_cause,
        files_likely_affected=files_plan,
        changed_files=changed_files,
        patch=patch,
    )
    return TrajContext(run_id=run_id, instance_id=instance_id,
                       actually_solved=actually_solved, patch=patch,
                       task_prompt=task_prompt)


# ---------------------------------------------------------------------------
# Audit one traj against one reviewer size
# ---------------------------------------------------------------------------

def _audit_one(ctx: TrajContext, size: str,
               cache: dict, dry_run: bool) -> AuditRecord:
    cache_key = f"{ctx.run_id}::{ctx.instance_id}::{size}"
    if cache_key in cache:
        r = cache[cache_key]
        return AuditRecord(**{**r, "cached": True})

    base = {"run_id": ctx.run_id, "instance_id": ctx.instance_id,
            "reviewer_size": size, "actually_solved": ctx.actually_solved,
            "has_patch": bool(ctx.patch)}

    if dry_run:
        return AuditRecord(**base, reviewer_decision="dry_run",
                           reviewer_confidence="", reviewer_reason="", cached=False)

    if not ctx.patch:
        rec = AuditRecord(**base, reviewer_decision="revise",
                          reviewer_confidence="high", reviewer_reason="empty patch",
                          cached=False)
        cache[cache_key] = asdict(rec)
        return rec

    model, api_base, api_key = REVIEWERS[size]
    result, tok_in, tok_out = _call_blind_reviewer(
        model=model, api_base=api_base, api_key=api_key,
        task_prompt=ctx.task_prompt,
    )

    rec = AuditRecord(
        **base,
        reviewer_decision   = str(result.get("decision",     "error")).lower(),
        reviewer_confidence = str(result.get("confidence",   "")),
        reviewer_reason     = str(result.get("primary_reason", "")),
        tokens_in=tok_in, tokens_out=tok_out, cached=False,
    )
    cache[cache_key] = asdict(rec)
    return rec


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def _confusion(records: list[AuditRecord]) -> dict | None:
    valid = [r for r in records if r.reviewer_decision in ("accept", "revise")]
    if not valid:
        return None
    tp = sum(r.tp for r in valid)
    fp = sum(r.fp for r in valid)
    fn = sum(r.fn for r in valid)
    tn = sum(r.tn for r in valid)
    prec = tp / (tp + fp)       if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn)       if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp)       if (tn + fp) > 0 else float("nan")
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else float("nan")
    acc  = (tp + tn) / len(valid)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, specificity=spec, f1=f1, acc=acc,
                n=len(valid))


def _fmt_cm(cm: dict | None) -> str:
    if cm is None:
        return "  (no data)"
    return (f"  TP={cm['tp']:3d} FP={cm['fp']:3d} FN={cm['fn']:3d} TN={cm['tn']:3d}"
            f"  P={cm['precision']:.2f} R={cm['recall']:.2f}"
            f"  F1={cm['f1']:.2f} Acc={cm['acc']:.2f}  n={cm['n']}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(all_records: list[AuditRecord], sizes: list[str]) -> None:
    # ── By reviewer size ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BLIND REVIEWER AUDIT — by reviewer model size")
    print("  TP=correct+accepted  FP=wrong+accepted"
          "  FN=correct+rejected  TN=wrong+rejected")
    print("=" * 80)
    for size in sizes:
        recs = [r for r in all_records if r.reviewer_size == size]
        model_name = REVIEWERS[size][0].split("/")[-1]
        print(f"\n  Reviewer {size} ({model_name})")
        print(_fmt_cm(_confusion(recs)))

    # ── By reviewer size × run_id ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Per-run breakdown")
    print("=" * 80)
    run_ids = sorted({r.run_id for r in all_records})
    # header
    size_hdr = "  ".join(f"{s:>6}" for s in sizes)
    print(f"  {'Run':<40}  {'Size':>4}  TP  FP  FN  TN    P     R    F1   Acc")
    print("-" * 80)
    for run_id in run_ids:
        for size in sizes:
            recs = [r for r in all_records
                    if r.run_id == run_id and r.reviewer_size == size]
            cm = _confusion(recs)
            if cm is None:
                continue
            print(f"  {run_id:<40}  {size:>4}  "
                  f"{cm['tp']:2d}  {cm['fp']:2d}  {cm['fn']:2d}  {cm['tn']:2d}  "
                  f"{cm['precision']:.2f}  {cm['recall']:.2f}  "
                  f"{cm['f1']:.2f}  {cm['acc']:.2f}")
        print()

    # ── Agreement between reviewer sizes ─────────────────────────────────────
    if len(sizes) > 1:
        print("=" * 80)
        print("Inter-reviewer agreement (% of instances where both agree)")
        print("=" * 80)
        # index records by (run_id, instance_id, size)
        idx: dict[tuple, str] = {}
        for r in all_records:
            if r.reviewer_decision in ("accept", "revise"):
                idx[(r.run_id, r.instance_id, r.reviewer_size)] = r.reviewer_decision

        for i, s1 in enumerate(sizes):
            for s2 in sizes[i+1:]:
                agree = total = 0
                for (rid, iid, sz), dec in idx.items():
                    if sz != s1:
                        continue
                    other = idx.get((rid, iid, s2))
                    if other is not None:
                        total += 1
                        agree += int(dec == other)
                pct = agree / total * 100 if total else float("nan")
                print(f"  {s1} vs {s2}: {agree}/{total} agree  ({pct:.1f}%)")
        print()


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------

def _apply_shard(pairs: list, shard_str: str) -> list:
    """Round-robin partition: '2/3' → every item where index % 3 == 1."""
    try:
        i_str, n_str = shard_str.split("/")
        i, n = int(i_str), int(n_str)
        assert 1 <= i <= n
    except Exception:
        raise ValueError(f"--shard must be I/N (e.g. 2/3), got: {shard_str!r}")
    return [p for idx, p in enumerate(pairs) if idx % n == (i - 1)]


def _default_cache_name(sizes: list[str], shard: str | None) -> str:
    size_tag = "_".join(sizes)
    if shard:
        shard_tag = shard.replace("/", "of")
        return f"audit_cache_{size_tag}_{shard_tag}.json"
    return f"audit_cache_{size_tag}.json"


# ---------------------------------------------------------------------------
# Traj discovery
# ---------------------------------------------------------------------------

def _iter_trajs(combined_root: pathlib.Path,
                run_filter: str | None) -> list[tuple[str, pathlib.Path]]:
    pairs: list[tuple[str, pathlib.Path]] = []

    def _add(run_id: str, run_dir: pathlib.Path) -> None:
        for t in sorted(run_dir.rglob("*.traj")):
            pairs.append((run_id, t))

    def _want(key: str) -> bool:
        return run_filter is None or key.upper().startswith(run_filter.upper())

    if _want("A_C1") and LEGACY_A_C1.exists():
        _add("A_c1_9b_mcts", LEGACY_A_C1)
    if _want("A_C2") and LEGACY_A_C2.exists():
        _add("A_c2_9b_mcts", LEGACY_A_C2)

    if combined_root.exists():
        for run_dir in sorted(combined_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_filter and not run_dir.name.upper().startswith(run_filter.upper()):
                continue
            _add(run_dir.name, run_dir)

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _do_merge(pattern: str, csv_path: pathlib.Path | None,
              combined_root: pathlib.Path) -> None:
    """Merge all cache files matching glob pattern, then print full report."""
    import glob as _glob
    files = sorted(_glob.glob(pattern))
    if not files:
        print(f"No files matched: {pattern!r}", file=sys.stderr)
        sys.exit(1)

    merged: dict = {}
    for fpath in files:
        try:
            data = json.loads(pathlib.Path(fpath).read_text())
            merged.update(data)
            print(f"  Loaded {len(data):4d} entries from {fpath}", file=sys.stderr)
        except Exception as exc:
            print(f"  WARNING: could not read {fpath}: {exc}", file=sys.stderr)

    merged_path = pathlib.Path("audit_cache_merged.json")
    merged_path.write_text(json.dumps(merged, indent=2))
    print(f"\nMerged {len(merged)} entries → {merged_path}", file=sys.stderr)

    # Reconstruct AuditRecord objects from merged cache
    all_records: list[AuditRecord] = []
    pairs = _iter_trajs(combined_root, None)
    sizes_seen: set[str] = set()
    for run_id, traj_path in pairs:
        instance_id = traj_path.stem
        # get actually_solved from traj
        try:
            d = json.loads(traj_path.read_text())
            actually_solved = bool(d.get("submitted") or d.get("info", {}).get("submitted"))
            instance_id = d.get("instance_id", instance_id)
        except Exception:
            actually_solved = False

        for size in SIZE_ORDER:
            key = f"{run_id}::{instance_id}::{size}"
            if key in merged:
                r = merged[key]
                all_records.append(AuditRecord(**{**r, "cached": True}))
                sizes_seen.add(size)

    if not all_records:
        print("No records found in merged cache.", file=sys.stderr)
        return

    sizes_present = [s for s in SIZE_ORDER if s in sizes_seen]
    _print_report(all_records, sizes_present)

    if csv_path:
        fields = list(asdict(all_records[0]).keys()) + ["tp", "fp", "fn", "tn"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_records:
                row = asdict(r)
                row.update(tp=int(r.tp), fp=int(r.fp), fn=int(r.fn), tn=int(r.tn))
                w.writerow(row)
        print(f"CSV written: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combined-root", type=pathlib.Path, default=COMBINED_RUNS)
    parser.add_argument("--run",   default=None,
                        help="Filter by run prefix (e.g. C_c1)")
    parser.add_argument("--sizes", default="9b,30b,120b",
                        help="Comma-separated reviewer sizes (default: 9b,30b,120b)")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="Run shard I of N, e.g. --shard 2/3 (round-robin)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache", type=pathlib.Path, default=None,
                        help="Cache file (auto-named from --sizes/--shard if omitted)")
    parser.add_argument("--merge", default=None, metavar="GLOB",
                        help="Merge cache files matching glob and print report, e.g. "
                             "'audit_cache_*.json'")
    parser.add_argument("--csv",   type=pathlib.Path, default=None)
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between LLM calls (default 0.3)")
    args = parser.parse_args()

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        _do_merge(args.merge, args.csv, args.combined_root)
        return

    # ── Normal run mode ───────────────────────────────────────────────────────
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip() in REVIEWERS]
    if not sizes:
        print(f"No valid sizes in --sizes. Choose from: {', '.join(SIZE_ORDER)}",
              file=sys.stderr)
        sys.exit(1)

    cache_path = args.cache or pathlib.Path(
        _default_cache_name(sizes, args.shard)
    )

    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            print(f"Loaded {len(cache)} cached entries from {cache_path}", file=sys.stderr)
        except Exception:
            pass

    pairs = _iter_trajs(args.combined_root, args.run)
    if args.shard:
        pairs = _apply_shard(pairs, args.shard)

    total_calls = len(pairs) * len(sizes)
    shard_info  = f" [shard {args.shard}]" if args.shard else ""
    print(f"{len(pairs)} trajs × {len(sizes)} reviewer(s) = {total_calls} calls"
          f"{shard_info}" + (" (dry run)" if args.dry_run else ""),
          file=sys.stderr)

    all_records: list[AuditRecord] = []
    call_n = 0

    for run_id, traj_path in pairs:
        ctx = _load_traj_context(traj_path, run_id)

        for size in sizes:
            call_n += 1
            rec = _audit_one(ctx, size, cache, args.dry_run)
            all_records.append(rec)

            solved_str = "✓" if rec.actually_solved else "✗"
            cached_str = " (cached)" if rec.cached else ""
            print(f"  [{call_n:4d}/{total_calls}] {run_id:<36} {ctx.instance_id:<26}"
                  f" {size:>4}  actual={solved_str}  "
                  f"{rec.reviewer_decision:<6} {rec.reviewer_confidence or '?':<6}"
                  f"{cached_str}")

            if not rec.cached and not args.dry_run:
                cache_path.write_text(json.dumps(cache, indent=2))
                time.sleep(args.delay)

    if not args.dry_run:
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"\nCache saved: {cache_path}", file=sys.stderr)

    # Single-worker run: print report immediately
    if not args.dry_run and not args.shard:
        _print_report(all_records, sizes)

    if args.csv and all_records and not args.dry_run and not args.shard:
        fields = list(asdict(all_records[0]).keys()) + ["tp", "fp", "fn", "tn"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_records:
                row = asdict(r)
                row.update(tp=int(r.tp), fp=int(r.fp), fn=int(r.fn), tn=int(r.tn))
                w.writerow(row)
        print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()
