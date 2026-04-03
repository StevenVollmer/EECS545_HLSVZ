# Split-Model Findings

This note summarizes the clearest result from the current custom SWE-agent experiments.

Primary source:

- [benchmark_round_split_compare_cloud/README.md](/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/README.md)

## Main Finding

A stronger planner helps a weaker coder.

Current cloud-only comparison:

- `qwen` alone
  - avg score: `78.60`
  - strict resolved: `0.600`
  - avg relative cost: `3.453`
- `qwen -> qwen`
  - avg score: `66.90`
  - strict resolved: `0.400`
  - avg relative cost: `3.148`
- `gpt -> qwen`
  - avg score: `89.20`
  - strict resolved: `0.700`
  - avg relative compute burden: `3.042`
- `gpt` alone
  - avg score: `93.50`
  - strict resolved: `0.800`
  - avg relative compute burden: `13.752`

This supports:

- `strong planner + weak coder > weak coder`
- `strong planner + weak coder > weak planner + weak coder`

In the current data:

- `gpt -> qwen` is better than `qwen`
- `gpt -> qwen` is much better than `qwen -> qwen`
- `gpt -> qwen` is much cheaper than `gpt`
- `gpt -> qwen` is closer to `gpt` in performance than in estimated compute burden

## Interpretation

The planner is not editing code. It gives the coder a structured handoff:

- likely files
- likely symbols
- safe reproduction steps
- validation priorities

The data suggests:

- a strong planner can make a smaller coder more competitive
- a weak planner can hurt a weak coder
- a planner does not automatically beat a strong single model

## Reviewer

Reviewer results are mixed, so they should not be the headline claim.

The clearest result is not "reviewer always helps." The clearer result is that a
stronger planner helps a weaker coder. Reviewer can help in some split setups,
but it is not consistent enough to be the main takeaway.

## Cost Caveat

`Avg relative compute burden to 4o-mini` is currently a heuristic.

It is useful for internal relative comparison, but it is not a literal API price comparison across providers.
For self-hosted or UMich models, the metric is better interpreted as:

- estimated relative compute burden

not:

- real dollar cost

## Best Current Claim

The strongest current claim is:

> A stronger planner can make a weaker coder substantially better, and the
> `gpt -> qwen` split gets much closer to `gpt` than to `qwen` while staying far
> closer to `qwen` on estimated compute burden.

The clearest evidence is the direct comparison above:

- `qwen`
  - score `78.60`
  - resolved `0.600`
  - estimated relative compute burden `3.453`
- `gpt -> qwen`
  - score `89.20`
  - resolved `0.700`
  - estimated relative compute burden `3.042`
- `gpt`
  - score `93.50`
  - resolved `0.800`
  - estimated relative compute burden `13.752`

So the current evidence supports the size-split claim more strongly than any
broader `planner_coder` vs `planner_coder_reviewer` architecture claim.
