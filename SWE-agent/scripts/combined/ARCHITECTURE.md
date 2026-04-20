# Combined Agent: MCTS + Mixed-Size Roles + LLM Value Function

## Motivation

Two prior systems:
- **Steven (MCTS, 9b)**: 75% on custom_cases_1; competitive with 30b linear at ~3× lower cost
- **Rafe (linear, 120b planner + 30b coder)**: 80% on custom_cases_1; better reasoning but no search

Hypothesis: combining MCTS search with Rafe's mixed-size role assignment and a learned value function from swe-search (ICLR 2025) can exceed both approaches.

---

## Three additions over `run_tree_search.py`

### 1. LLM Value Function (`_score_node`)

Replaces heuristic progress flags for UCB1 node scoring. A 30b model evaluates each node on a −100 to 100 scale with state-specific prompts:

| Phase | Node state | Prompt question |
|-------|-----------|-----------------|
| search | no edit yet | Has the agent identified the correct file/root cause? |
| edit | edit made, checks pending | Does the diff correctly fix the bug? |
| terminal | submitted or max depth | Is this a correct, complete fix? |

Score is normalized to 0..100 for UCB1 compatibility and cached on the node to avoid repeat calls (~18 calls per instance).

**Fallback**: when `--value-model` is omitted, heuristic scoring is used unchanged.

### 2. Hindsight Feedback (`_extract_branch_failure`, `_dead_branch_feedback`)

When a branch terminates as a dead end (`stopped_reason` in `{parse_failure, max_depth, stagnation, model_error}`), a one-sentence failure summary is extracted and stored. The next `_expand()` call injects the last 3 failures as a warning into the model's context:

```
Note — these search branches did NOT work (avoid similar approaches):
- Branch failed (parse_failure) — last message: …
- Branch failed (max_depth) — last message: …
```

**Cost**: negligible — string concatenation only.

### 3. Mean Trajectory Reward (`_mean_trajectory_value`)

Current MCTS selects the final result by terminal node value alone. swe-search showed that the **mean** value across all nodes on the root→leaf path is a better predictor of correctness. `_best_terminal()` is updated to rank submitted nodes by `_mean_trajectory_value()` when a value model is configured.

---

## Mixed-size model assignment

| Role | Model | Why |
|------|-------|-----|
| Planner | 120b (gpt-oss) | Reasoning-intensive: hypothesis, file identification |
| Coder | 30b (Qwen3-VL) | Generation-intensive: loops, edits, many calls |
| Reviewer | 120b (gpt-oss) | Reasoning-intensive: patch quality judgment |
| Value fn | 30b (Qwen3-VL) | Cheap scorer: 18 calls per instance |

Separate `--planner-api-base`, `--reviewer-api-base`, `--value-api-base` flags route each role to the correct UMich endpoint.

---

## Ablation variants

| ID | Label | Components |
|----|-------|------------|
| A | 9b-MCTS baseline | Prior results (15/20 c1, 14/20 c2) |
| B | Rafe best linear | Prior results (16/20 c1) |
| C | Mixed + MCTS | 120b planner/reviewer + 30b coder MCTS |
| D | Mixed + MCTS + value | + LLM value function (30b) |
| E | Mixed + MCTS + value + hindsight | + cross-branch failure warnings |

**Isolation logic**:
- B→C: Does MCTS search add value when the planner/reviewer are already 120b?
- C→D: Does LLM value function improve node selection over heuristic?
- D→E: Does hindsight feedback between dead branches help?

**Final evaluation**: run best variant + A + B on `custom_cases_3` (held out throughout ablation).

---

## Files

| File | Purpose |
|------|---------|
| `run_combined.py` | Main runner (fork of `run_tree_search.py` + three additions) |
| `run_combined_ablation.py` | Runs variants C, D, E on c1 + c2 |
| `ARCHITECTURE.md` | This document |

## Running

```bash
# Dry-run: print all commands
python SWE-agent/scripts/combined/run_combined_ablation.py

# Execute all variants (remote UMich cluster required):
python SWE-agent/scripts/combined/run_combined_ablation.py --execute

# Only variant C on custom_cases_1:
python SWE-agent/scripts/combined/run_combined_ablation.py --execute --only C_c1

# Summarize results:
python SWE-agent/scripts/combined/run_combined_ablation.py --summarize
```
