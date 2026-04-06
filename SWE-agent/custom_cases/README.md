# Custom Cases

Each custom case lives in its own folder under [custom_cases](/Users/rafe/classes/eecs545/project/SWE-agent/custom_cases).

## Layout

Use this structure:

```text
custom_cases/
  my_case/
    case.json
    repo/
      ...
```

- `repo/` is the directory copied into the Docker runtime
- `case.json` holds the metadata, problem statement, and evaluation rules

Do not make each case its own nested git repo.

## Minimal `case.json`

`case.json` must be a one-item JSON list:

```json
[
  {
    "instance_id": "my_case_001",
    "repo_path": "repo",
    "problem_statement": "Fix the bug described here.",
    "install_commands": ["pip install pytest"],
    "evaluation": {
      "baseline_checks": [
        {
          "name": "bug_reproduces",
          "command": "python -m pytest -q tests/test_example.py",
          "expect_exit_code": 1,
          "stdout_contains": ["AssertionError"]
        }
      ],
      "success_checks": [
        {
          "name": "tests_pass",
          "command": "python -m pytest -q",
          "expect_exit_code": 0,
          "stdout_contains": ["2 passed"]
        }
      ]
    }
  }
]
```

Required fields:

- `instance_id`
- `repo_path`
- `problem_statement`
- `evaluation.baseline_checks`
- `evaluation.success_checks`

Useful optional fields:

- `install_commands`
- `setup_commands`
- `analysis.likely_fix_paths`
- `analysis.showcase`
- `analysis.difficulty`
- `policy.allow_test_edits`

## Recommended metadata

Use `analysis.likely_fix_paths` so the grader can reward finding the right area:

```json
"analysis": {
  "likely_fix_paths": [
    "app/services/example.py",
    "app/utils/formatting.py"
  ],
  "showcase": "planner",
  "difficulty": "medium"
}
```

Use `policy.allow_test_edits` only if the case truly allows tests to be changed:

```json
"policy": {
  "allow_test_edits": false
}
```

## Good case design

Try to make each case:

- small enough to run quickly
- realistic enough to require search and validation
- deterministic to reproduce and grade
- explicit about success through checks

Good progression:

- easy: one-file bug, direct failing test
- medium: multiple likely files, less explicit issue text
- hard: manual repro or behavioral bug not fully covered by tests

## Current Tier List

Current benchmark inventory: 20 cases.

### Tier 1: Easy

- `simple_mean_bug`
- `label_formatter`
- `median_window`
- `status_slug`
- `risk_score`
- `owner_sort`

These are mostly one-file runtime fixes with direct failing tests.

### Tier 2: Medium

- `board_rollup`
- `incident_brief`
- `invoice_footer`
- `milestone_rollup`
- `priority_snapshot`
- `team_digest`
- `workspace_digest`

These usually involve a small app-shaped repo, multiple plausible files, or a service/presenter split.

### Tier 3: Hard

- `budget_snapshot`
- `contact_card`
- `digest_preview`
- `nested_app`
- `owner_recap`
- `renewal_preview`
- `shipment_preview`

These are the most useful for planner/reviewer evaluation. They tend to involve manual repro, behavioral bugs not fully covered by the existing tests, punctuation/casing edge cases, or regression-sensitive fixes.

### Showcase Tags

- planner-favored:
  - `board_rollup`
  - `incident_brief`
  - `invoice_footer`
  - `milestone_rollup`
  - `priority_snapshot`
  - `team_digest`
  - `workspace_digest`
- reviewer-favored:
  - `contact_card`
  - `owner_recap`
  - `renewal_preview`
  - `shipment_preview`
- coder/basic execution:
  - `median_window`
  - `owner_sort`
  - `risk_score`
  - `status_slug`

## Validate a new case

Baseline:

```bash
./env/bin/python SWE-agent/scripts/custom/judge_custom_case.py \
  --case SWE-agent/custom_cases/my_case \
  --mode baseline \
  --run-install
```

The baseline should pass, meaning the bug is reproduced as expected.

## Run the custom runner on a case

```bash
./env/bin/python SWE-agent/scripts/custom/run_custom_swebench.py \
  --preset umich_qwen \
  --instances-type file \
  --instances-path SWE-agent/custom_cases/my_case \
  --filter my_case_001 \
  --output-dir SWE-agent/custom_runs/my_case_umich_qwen
```

## Score a run

```bash
./env/bin/python SWE-agent/scripts/custom/analyze_custom_runs.py \
  SWE-agent/custom_runs/my_case_umich_qwen \
  --json
```

See also:

- [README_custom_runner.md](/Users/rafe/classes/eecs545/project/SWE-agent/scripts/custom/README_custom_runner.md)
- [SCORING.md](/Users/rafe/classes/eecs545/project/SWE-agent/scripts/custom/SCORING.md)
