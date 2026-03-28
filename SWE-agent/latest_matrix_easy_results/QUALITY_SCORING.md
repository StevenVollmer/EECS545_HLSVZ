# Matrix Analysis Scoring

This document describes the current trajectory analysis used by:
- [analyze_traj_quality.py](/Users/lobs/classes/EECS545_project/SWE-agent/scripts/analyze_traj_quality.py)
- [summarize_latest_matrix_results.py](/Users/lobs/classes/EECS545_project/SWE-agent/scripts/summarize_latest_matrix_results.py)

The old analysis mostly measured whether a run could use the SWE-agent interface correctly. That was not enough. Runs were getting credit for reading files, attempting edits, and avoiding malformed commands even when they were not clearly moving toward the correct fix.

The current analysis shifts the emphasis to four questions:
- Did the run work on files that match the issue statement?
- Did it stay focused instead of thrashing across the codebase?
- Did it complete a useful inspect -> edit -> validate workflow?
- Did it execute stably without wasting turns on avoidable errors?

## Inputs

The scorer reads a single `.traj` file and uses:
- the problem statement embedded in the trajectory or replay config
- action commands
- observations
- role/token usage stats from trajectory metadata

It extracts:
- inspected files
- edited files
- validation commands
- issue-referenced file paths
- issue keywords and symbols

## Primary Metrics

### `issue_alignment_score` `/5`

Measures whether the run touched files that match the issue statement.

Signals:
- inspected files that match issue-referenced paths or strong issue keywords
- edited files that match issue-referenced paths or strong issue keywords
- issue symbol mentions that continue to appear during the run

This is the main replacement for the old generic grounding score.

### `solution_focus_score` `/5`

Measures whether the run stayed concentrated on a plausible repair path.

Signals:
- first edit was already aligned with the issue
- edited files were inspected before being edited
- number of edited files stayed small
- number of inspected files stayed reasonably bounded
- edit attempts did not spread across obviously unrelated files

This is intended to reward focused localization, not raw activity.

### `workflow_score` `/5`

Measures whether the run progressed through a useful repair loop.

Signals:
- inspection happened
- at least one successful edit happened
- validation happened after an edit
- the run submitted
- the run finished cleanly or submitted manually

This captures repair progress without pretending it proves correctness.

### `stability_score` `/5`

Measures how much execution quality was lost to avoidable waste.

Penalties:
- malformed commands
- failed edit attempts
- empty steps
- very long or very token-heavy runs

This is a stability metric, not an effectiveness metric.

### `analysis_score` `/20`

This is the sum of:
- `issue_alignment_score`
- `solution_focus_score`
- `workflow_score`
- `stability_score`

It is the primary score used to rank configs in generated summaries.

## Secondary Metrics

The scripts still emit the older fields:
- `quality_score`
- `grounding_score`
- `completion_score`
- `efficiency_score`

These are retained for backwards comparison only. They are no longer the main basis for comparing configurations.

## Useful Raw Fields

The scorer also emits raw fields that are often more informative than any single score:
- `aligned_files_inspected`
- `aligned_files_edited`
- `validation_after_edit`
- `edited_file_alignment`
- `inspected_file_alignment`
- `edited_files`
- `aligned_files`

These are the fields to use when you want to explain why one configuration looks better than another.

## Limitations

This analysis is still not a correctness metric.

It does not directly answer:
- whether the final patch matches the SWE-bench gold patch
- whether the patch resolves the issue
- whether the patch breaks unrelated behavior

What it does provide is a better proxy for whether a run is moving in the right direction on the right part of the repository, which is the gap the earlier analysis failed to cover.
