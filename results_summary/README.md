# Split-Model Findings

This note summarizes the clearest result from the current custom SWE-agent experiments.

Primary sources:

- [benchmark_round_split_compare_cloud/README.md](/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/README.md)
- [benchmark_round_split_compare_cloud/analysis.summary.json](/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/analysis.summary.json)

## Main Finding

The clearest current result is that a strong planner makes the smaller coder much better, and the split setup is now competitive with the all-large baseline.

Current 20-case cloud-only comparison:

- `qwen`
  - avg score: `86.05`
  - strict resolved: `0.700`
  - avg relative compute burden: `3.467`
- `qwen -> qwen`
  - avg score: `79.95`
  - strict resolved: `0.650`
  - avg relative compute burden: `3.163`
- `gpt -> qwen`
  - avg score: `91.15`
  - strict resolved: `0.800`
  - avg relative compute burden: `3.064`
- `gpt -> qwen -> gpt`
  - avg score: `92.25`
  - strict resolved: `0.800`
  - avg relative compute burden: `3.224`
- `gpt`
  - avg score: `92.50`
  - strict resolved: `0.800`
  - avg relative compute burden: `13.873`
- `gpt -> gpt`
  - avg score: `90.65`
  - strict resolved: `0.800`
  - avg relative compute burden: `12.244`

This shows the planner effect much more clearly than the older runs:

- `gpt -> qwen` is better than `qwen`
- `gpt -> qwen` is much better than `qwen -> qwen`
- `gpt -> qwen` matches both `gpt` and `gpt -> gpt` on strict resolved rate
- `gpt -> qwen` does this at about one quarter of the compute burden of the all-large configurations

## What The Planner Is Doing

The planner does not edit code. It gives the coder a structured handoff:

- likely files
- likely symbols
- safe reproduction steps
- validation priorities

The current data suggests:

- a stronger planner can make a smaller coder more competitive
- a weak planner is not enough by itself
- planner quality matters more than just adding a planner role

The most direct planner comparison is:

- `qwen`
  - score `86.05`
  - resolved `0.700`
- `qwen -> qwen`
  - score `79.95`
  - resolved `0.650`
- `gpt -> qwen`
  - score `91.15`
  - resolved `0.800`

So the current 20-case benchmark supports:

- `strong planner + weak coder > weak coder`
- `strong planner + weak coder > weak planner + weak coder`

## What The Reviewer Is Doing

Reviewer is helping, but it is no longer the main story.

Direct comparison:

- `gpt -> qwen`
  - avg score: `91.15`
  - strict resolved: `0.800`
  - compute: `3.064`
- `gpt -> qwen -> gpt`
  - avg score: `92.25`
  - strict resolved: `0.800`
  - compute: `3.224`

So reviewer is adding real value here:

- `+1.10` avg score
- no change in strict resolved rate
- only a small compute increase

That means reviewer is still useful, but the bigger gain in the new 20-case benchmark is the planner improvement, not the reviewer loop by itself.

## Compute Caveat

`Avg relative compute burden to 4o-mini` is a heuristic.

It is useful for internal relative comparison, but it is not a literal API price comparison across providers.
For self-hosted or UMich models, the metric should be interpreted as:

- estimated relative compute burden

not:

- real dollar cost

## Best Current Claim

The strongest current claim is:

> A stronger planner can make a smaller coder competitive with all-large configurations while using far less relative compute burden.

The clearest evidence is:

- `gpt -> qwen`
  - score `91.15`
  - resolved `0.800`
  - estimated relative compute burden `3.064`
- `gpt`
  - score `92.50`
  - resolved `0.800`
  - estimated relative compute burden `13.873`
- `gpt -> gpt`
  - score `90.65`
  - resolved `0.800`
  - estimated relative compute burden `12.244`

So the clearest current split-model result is `gpt -> qwen`, with `gpt -> qwen -> gpt` as a smaller follow-on improvement rather than the main source of the gain.
