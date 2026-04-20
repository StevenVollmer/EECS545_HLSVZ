# Custom Run Scoring

The run grader is implemented in [analyze_custom_runs.py](/Users/rafe/classes/eecs545/project/SWE-agent/scripts/custom/analyze_custom_runs.py).

It scores each run with:

- `functional_correctness` (0-50)
  - based on case-defined `baseline_checks` and `success_checks`
  - success checks dominate the score
- `repair_precision` (about 0-20)
  - rewards small, focused patches
  - rewards overlap with `analysis.likely_fix_paths`
  - penalizes disallowed test edits
- `regression_safety` (about 0-15)
  - rewards passing success checks
  - rewards validation, especially after editing
- `search_grounding` (about 0-10)
  - rewards inspecting and editing the likely fix area
  - otherwise falls back to generic search/view evidence
- `efficiency_control` (0-5)
  - penalizes parse errors, tool errors, long runs, and loop-control triggers

Total score:

```text
total_score =
  functional_correctness +
  repair_precision +
  regression_safety +
  search_grounding +
  efficiency_control
```

Other important outputs:

- `success_passed`: strict case success checks passed
- `observed_success_passed`: effective success ignoring blocked host checks
- `evaluation_blocked`: local grading environment was missing something needed to evaluate
- `relative_compute_to_4o_mini`: normalized compute-burden estimate
- `score_per_compute`: quality divided by normalized compute burden
- `resolved_per_compute`: solved / normalized compute burden

Artifacts used by the grader:

- run `.traj`
- run `.patch`
- run `.info.log`
- run `run_batch.config.yaml`
- case `case.json`

Case metadata that matters:

- `evaluation.baseline_checks`
- `evaluation.success_checks`
- `analysis.likely_fix_paths`
- `policy.allow_test_edits`

Main scripts:

- [run_custom_swebench.py](/Users/rafe/classes/eecs545/project/SWE-agent/scripts/custom/run_custom_swebench.py)
- [judge_custom_case.py](/Users/rafe/classes/eecs545/project/SWE-agent/scripts/custom/judge_custom_case.py)
- [analyze_custom_runs.py](/Users/rafe/classes/eecs545/project/SWE-agent/scripts/custom/analyze_custom_runs.py)
- [render_custom_matrix_report.py](/Users/rafe/classes/eecs545/project/SWE-agent/scripts/custom/render_custom_matrix_report.py)
