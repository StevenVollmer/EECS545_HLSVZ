# Split-Model Findings

This note summarizes the clearest result from the current custom SWE-agent experiments.

Primary sources:

- [benchmark_round_split_compare_cloud/README.md](/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/README.md)
- [benchmark_round_split_compare_cloud/analysis.summary.json](/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/analysis.summary.json)

## Main Finding

The strongest current result is the split setup:

- `gpt planner -> qwen coder -> gpt reviewer`

Current cloud-only comparison:

- `qwen -> qwen`
  - avg score: `81.80`
  - strict resolved: `0.600`
  - avg relative compute burden: `3.150`
- `gpt -> qwen`
  - avg score: `77.50`
  - strict resolved: `0.600`
  - avg relative compute burden: `3.041`
- `gpt -> qwen -> gpt`
  - avg score: `87.30`
  - strict resolved: `0.800`
  - avg relative compute burden: `3.214`
- `gpt -> gpt`
  - avg score: `92.60`
  - strict resolved: `0.800`
  - avg relative compute burden: `12.148`

This is the clearest evidence for the size-split story:

- `gpt -> qwen -> gpt` matches `gpt -> gpt` on strict resolved rate
- `gpt -> qwen -> gpt` is much closer to `gpt -> gpt` in score than in compute burden
- `gpt -> qwen -> gpt` is about one quarter of the compute burden of `gpt -> gpt`

## What The Planner Is Doing

The planner does not edit code. It gives the coder a structured handoff:

- likely files
- likely symbols
- safe reproduction steps
- validation priorities

The current data suggests:

- a stronger planner can make a smaller coder more competitive
- a weak planner is not enough by itself
- the best current result comes from combining a strong planner with a strong reviewer around a smaller coder

## What The Reviewer Is Doing

Reviewer is now materially helping in the split setup we care about.

Direct comparison:

- `gpt -> qwen`
  - avg score: `77.50`
  - strict resolved: `0.600`
  - compute: `3.041`
- `gpt -> qwen -> gpt`
  - avg score: `87.30`
  - strict resolved: `0.800`
  - compute: `3.214`

So reviewer is adding real value here:

- `+9.8` avg score
- `+0.20` strict resolved rate
- only a small compute increase

The biggest reviewer wins in this matrix are on harder semantic cases:

- `digest_preview`
  - `gpt -> qwen`: `42`, failed
  - `gpt -> qwen -> gpt`: `95`, solved
- `nested_app`
  - `gpt -> qwen`: `45`, failed
  - `gpt -> qwen -> gpt`: `98`, solved

So the current evidence supports:

- strong planner + weaker coder + strong reviewer > strong planner + weaker coder

at least for the current cloud-only custom benchmark.

## Compute Caveat

`Avg relative compute burden to 4o-mini` is a heuristic.

It is useful for internal relative comparison, but it is not a literal API price comparison across providers.
For self-hosted or UMich models, the metric should be interpreted as:

- estimated relative compute burden

not:

- real dollar cost

## Best Current Claim

The strongest current claim is:

> A stronger planner and reviewer can make a smaller coder competitive with an all-large setup, while using far less relative compute.

The clearest evidence is:

- `gpt -> qwen -> gpt`
  - score `87.30`
  - resolved `0.800`
  - estimated relative compute burden `3.214`
- `gpt -> gpt`
  - score `92.60`
  - resolved `0.800`
  - estimated relative compute burden `12.148`

So the current data supports the split-model claim most strongly in the `gpt -> qwen -> gpt` configuration.
