# Quality Scoring

This document explains how the current matrix scoring works for the local `latest_matrix_easy_results` runs.

The source of truth for the implementation is:
- [`scripts/analyze_traj_quality.py`](/Users/lobs/classes/EECS545_project/SWE-agent/scripts/analyze_traj_quality.py)

The scorer is intentionally not a correctness metric. It is a trajectory-quality metric.

It is meant to answer:
- Did the run make meaningful progress?
- Did it stay grounded in the intended bug?
- Did it waste turns on wrong files or malformed tool calls?

It does not answer:
- Is the patch actually correct?
- Would SWE-bench mark the run as resolved?

## Inputs

The scorer reads a single `.traj` file and inspects the trajectory steps.

For each step it looks at:
- `action`
- `observation`

These are used to infer whether the run:
- stayed on the target file
- discussed the right symbols
- reached handoff
- started coding
- attempted edits
- drifted to the wrong file
- produced syntax/tooling failures

## Target Assumptions

The scorer is currently specialized for the `pydicom__pydicom-1458` experiment.

Hardcoded target file:
- `/testbed/pydicom/pixel_data_handlers/numpy_handler.py`

Hardcoded wrong-file hint:
- `/testbed/pydicom/pixels/pixel_array.py`

Hardcoded target tokens:
- `PixelRepresentation`
- `required_elements`
- `FloatPixelData`
- `DoubleFloatPixelData`

These assumptions are why the scorer is useful for this specific matrix, but not yet generic across arbitrary benchmark tasks.

## Raw Signals

The script computes these raw fields first.

### `planner_handoff`

Set to `true` if any action contains:
- `handoff `

Meaning:
- the planner or coder successfully attempted a structured role transition

Why it matters:
- handoff is a major milestone in multi-role runs

### `coder_started`

Set to `true` if any action contains:
- `cat /testbed/handoff.txt`

Meaning:
- the coder actually began from the planner contract

Why it matters:
- it distinguishes planner-only progress from planner-to-coder execution

### `target_file_reads`

Incremented when the target file path appears in:
- `action`
- or `observation`

Meaning:
- the run is still touching the intended file

Why it matters:
- repeated contact with the correct file is a basic grounding signal

### `target_token_hits`

Incremented when any target token appears in:
- `action`
- or `observation`

Meaning:
- the run is still engaging the right bug concepts

Why it matters:
- a run can mention the right file but still miss the core logic

### `edit_attempts`

Incremented when an action contains:
- `str_replace_editor`

Meaning:
- the run moved from inspection into implementation attempts

Why it matters:
- this is a key progress marker even when the edit fails

### `wrong_file_edits`

Incremented when:
- an action contains `str_replace_editor`
- and the action also contains a known wrong-file hint

Current wrong-file hint:
- `/testbed/pydicom/pixels/pixel_array.py`

Meaning:
- the run attempted to edit a file that is off the intended solution path

Why it matters:
- this is one of the strongest “close but wrong” signals

### `off_target_drift`

Set to `true` when a wrong-file edit is detected.

Meaning:
- the run stopped being grounded in the planner contract and issue

Why it matters:
- drift should be penalized even if the run made earlier progress

### `syntax_errors`

Incremented when the observation contains:
- `syntax error`
- or `usage: str_replace_editor`

Meaning:
- the run issued a malformed command or tool invocation

Why it matters:
- malformed commands are execution failures, not just harmless noise

## Progress Score

`progress_score` is the positive part of the metric.

It rewards getting further through the run pipeline.

Current formula:

- `+2` if `planner_handoff` is true
- `+1` if `coder_started` is true
- `+2` if `target_file_reads > 0`
- `+2` if `target_token_hits > 1`
- `+2` if `edit_attempts > 0`

Maximum current progress score:
- `9`

Interpretation:
- `0-2`: almost no useful forward movement
- `3-5`: partial localization, little execution
- `6-7`: meaningful task engagement
- `8-9`: strong forward progress through the intended workflow

Important limitation:
- progress score does not care whether the edits were correct
- it only measures how far the run got in a plausible repair trajectory

## Penalty Score

`penalty_score` is the negative part of the metric.

It punishes wasted or misleading progress.

Current formula:

- `+min(3, wrong_file_edits)` if `wrong_file_edits > 0`
- `+min(3, syntax_errors)` if `syntax_errors > 0`
- `+1` if `off_target_drift` is true

Interpretation:
- wrong-file edits are expensive because they show diagnosis failure
- syntax errors are expensive because they block execution
- drift is a separate penalty because it captures “lost the plot”

Important detail:
- penalties are capped in places so one repeated failure mode does not completely dominate the score

## Quality Score

`quality_score` is the net trajectory score.

Formula:

- `quality_score = clamp(progress_score - penalty_score, 0, 10)`

Even though the current progress max is `9`, the result is reported on a `/10` scale for readability.

Interpretation:
- `0-2/10`: very weak run
- `3-5/10`: some useful behavior, but major problems
- `6-7/10`: meaningful progress with limited damage
- `8-10/10`: strong trajectory quality, though not necessarily correct

Important limitation:
- `quality_score` is not resolution
- a run can score highly and still fail the benchmark

## Grounding Score

`grounding_score` is a separate score focused only on staying on the intended bug.

Current formula:

- `+2` if `target_file_reads > 0`
- `+2` if `wrong_file_edits == 0`
- `+1` if `off_target_drift` is false

Maximum:
- `5`

Interpretation:
- `5/5`: the run stayed well anchored to the intended file/bug
- `3-4/5`: partially grounded, some instability
- `0-2/5`: strong drift away from the intended task

Why this score exists:
- a run can be low progress but well grounded
- a run can be high progress but poorly grounded
- those are different failure modes and need to be separated

## Why We Use Both `quality_score` and `grounding_score`

These two scores separate two important cases.

### Case 1: High progress, low grounding

Example:
- planner hands off
- coder starts
- edits happen
- but the coder edits the wrong file

This looks active and capable, but it is dangerous.

### Case 2: High grounding, low progress

Example:
- planner keeps inspecting the right file
- the run never reaches handoff or edit

This looks cautious and on-task, but unproductive.

Without separate metrics, both cases can look similar under a single success/failure label.

## Current Matrix Examples

### `v3` style behavior

Observed pattern:
- planner handed off
- coder started
- edit attempts happened
- later drifted to the wrong file

Result:
- high progress
- lower grounding

This is why `v3` looked better than `v4` on progress, but worse on grounding.

### `v4` style behavior

Observed pattern:
- planner stayed on the intended file
- no handoff
- no coder start

Result:
- low progress
- perfect grounding

This is why `v4` looked safer, but less useful.

### `v5` style behavior

Observed pattern:
- more targeted planner reads
- better file-local inspection
- still no handoff before expanding to support files

Result:
- better progress than `v4`
- still good grounding

This is what we mean by “getting closer” even without a final correct patch.

## Why Patch Existence Is Not Enough

A `.patch` file or `submitted` status is not enough to evaluate quality.

A run may:
- autosubmit after format failure
- produce a patch for the wrong reason
- submit after malformed reviewer/coder turns

That is why the scorer looks inside the trajectory rather than trusting only:
- `submitted`
- `.patch` exists
- `preds.json` exists

## Known Weaknesses

The current scorer is intentionally simple and has limitations.

### Task-specific

It is hardcoded to one benchmark issue.

### Token-based

It uses keyword hits rather than semantic understanding.

### Coarse edit detection

Any `str_replace_editor` action counts as an edit attempt, even if it is only a view command.

### No correctness validation

It does not inspect the actual patch content yet.

### No reviewer-specific logic

It does not currently reward or penalize reviewer behavior separately.

## How To Improve It

Natural next improvements:

- distinguish `str_replace_editor view` from real edit commands
- add separate planner, coder, and reviewer sub-scores
- score handoff quality by inspecting handoff JSON fields
- detect whether the final patch modifies the expected lines
- add test-validation scoring:
  - repro attempted
  - repro passed
  - regression checks attempted

## How To Recompute Scores

From the `SWE-agent` repo root:

```bash
python3 scripts/analyze_traj_quality.py latest_matrix_easy_results/all_3_big/pydicom__pydicom-1458/pydicom__pydicom-1458.traj
```

To rebuild the combined latest-matrix summary:

```bash
python3 scripts/summarize_latest_matrix_results.py
```

## Practical Reading Guide

When scanning results, interpret them in this order:

1. `exit_status`
2. `quality_score`
3. `grounding_score`
4. raw trace and patch

Suggested reading:

- If `exit_status` is bad but `quality_score` is high:
  - likely framework or formatting instability
- If `quality_score` is high but `grounding_score` is low:
  - likely active but confused coder behavior
- If `grounding_score` is high but `quality_score` is low:
  - likely planner hesitation or failure to transition roles

This is the intended use of the metric: not to replace benchmark resolution, but to show which kind of failure happened and whether the system is getting closer.
