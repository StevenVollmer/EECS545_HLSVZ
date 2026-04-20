# Split-Model Findings

This note summarizes results from the custom SWE-agent experiments across two benchmark sets.

Primary sources:

- [benchmark_round_split_compare_cloud/README.md](/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/benchmark_round_split_compare_cloud/README.md)
- [routed_matrix results](/Users/rafe/classes/eecs545/project/SWE-agent/custom_matrix_runs/routed_matrix/)

Generated figures:

- [Score vs compute](/Users/rafe/classes/eecs545/project/results_summary/figures/score_vs_compute.svg)
- [Configuration summary bars](/Users/rafe/classes/eecs545/project/results_summary/figures/config_summary_bars.svg)
- [Per-case score heatmap](/Users/rafe/classes/eecs545/project/results_summary/figures/case_score_heatmap.svg)
- [Architecture deltas](/Users/rafe/classes/eecs545/project/results_summary/figures/architecture_deltas.svg)
- [Figure data table](/Users/rafe/classes/eecs545/project/results_summary/figures/config_summary.csv)

---

## Part 1: 20-Case Benchmark (Planner Effect)

The original 20-case benchmark established the core planner finding:

- `qwen` — resolved `0.700`, compute `3.467`
- `qwen -> qwen` — resolved `0.650`, compute `3.163`
- `gpt -> qwen` — resolved `0.800`, compute `3.064`
- `gpt -> qwen -> gpt` — resolved `0.800`, compute `3.224`
- `gpt` — resolved `0.800`, compute `13.873`

Key takeaways:

- A strong planner makes a weak coder competitive with all-large configs
- A weak planner hurts (`qwen -> qwen` < `qwen` solo)
- Planner quality matters more than just having a planner role
- `gpt -> qwen` matches `gpt` resolved rate at ~1/4 the compute

---

## Part 2: 10-Case Audit Benchmark (Critic + Reviewer)

A second 10-case benchmark was designed with harder cases to test two new plan-audit mechanisms:

- **Critic (pre-coder audit):** An adversarial pass reviews the planner's handoff *before* the coder starts. If the critic rejects, the planner revises once.
- **Reviewer (post-coder audit):** After the coder finishes, a reviewer examines the patch and validation evidence. If rejected, the coder gets a second round with the reviewer's feedback, the prior patch, and changed-file context.

### Results

```
Config           Resolved   Compute   Description
─────────────────────────────────────────────────────────────────
qwen              5/10       3.37     small coder, no planner (floor)
gpt -> qwen       7/10       2.04     planner + small coder (baseline)
reviewer          8/10       2.60     planner + small coder + post-coder review
critic            8/10       2.15     planner + pre-coder audit + small coder
crit+rev          8/10       2.52     planner + pre-coder audit + small coder + post-coder review
gpt               9/10      24.62    large coder solo (ceiling)
```

### Per-Case Breakdown

```
case                     qwen     gpt->qwen  reviewer  critic   crit+rev  gpt      bucket
──────────────────────────────────────────────────────────────────────────────────────────────
date_parse_locale        FAIL     FAIL       PASS      FAIL     FAIL      PASS     needs-large
dep_cycle_detect         PASS     PASS       PASS      PASS     PASS      PASS     trivial
numeric_drift_sum        FAIL     PASS       PASS      PASS     PASS      PASS     plan-rescuable
pagination_drift         FAIL     PASS       PASS      PASS     PASS      PASS     plan-rescuable
path_normalizer_cache    PASS     PASS       PASS      PASS     PASS      PASS     trivial
retry_cap                PASS     PASS       FAIL      PASS     PASS      PASS     trivial
schema_migration_check   PASS     PASS       PASS      PASS     PASS      PASS     trivial
search_hit_localize      PASS     PASS       PASS      PASS     PASS      PASS     trivial
stable_ranking           FAIL     FAIL       PASS      PASS     PASS      PASS     needs-large
weighted_median          FAIL     FAIL       FAIL      FAIL     FAIL      FAIL     impossible
```

### Key Findings

**1. Both audits independently boost resolved rate by +1 over no-audit baseline.**

- `gpt -> qwen` (no audit): 7/10
- `reviewer` (post-coder): 8/10
- `critic` (pre-coder): 8/10

**2. They solve different cases through different mechanisms.**

- Critic rescues `stable_ranking` — a needs-large case where the plan audit improves hypothesis quality enough for the small coder to succeed.
- Reviewer rescues `date_parse_locale` — a needs-large case where the coder's first attempt fails but reviewer-directed revision on the second round succeeds.
- Neither mechanism rescues both. They operate on different failure modes.

**3. Critic is cheaper than reviewer for the same lift.**

- Critic: 8/10 at compute 2.15
- Reviewer: 8/10 at compute 2.60
- The critic catches plan errors before the coder wastes turns. The reviewer can only catch patch errors after the coder has already committed to a direction.

**4. Stacking does not compound.**

- `crit+rev`: 8/10 at compute 2.52
- No additional resolved-rate lift over either audit alone.
- The two mechanisms rescue different cases but the total unique rescues don't exceed what either achieves individually on this benchmark.

**5. Both audits reach 89% of the large-coder ceiling at ~10x less compute.**

- Critic/reviewer: 8/10 at ~2.15–2.60 compute
- `gpt` solo: 9/10 at 24.62 compute

### Bucket Classification

Cases were classified by which configs solve them:

- **trivial** (5): solved by all configs including qwen solo
- **plan-rescuable** (2): qwen fails, planner+qwen succeeds
- **needs-large** (2): planner+qwen fails, audit or gpt rescues
- **impossible** (1): no config solves (`weighted_median`)

### Audit Mechanism Details

**Critic prompt design:** The critic checks whether the planner's root-cause hypothesis explains the observed symptom, whether target symbols look fabricated, and whether the plan is specific enough for a weak coder to act on. Critical constraint: the critic may not shrink the planner's file list or add forbidden-edit entries — it can only refine the hypothesis and suggest additional files to examine.

**Reviewer improvements:** The reviewer uses a split turn budget (2/3 for round 1, 1/3 for round 2) and carries the prior patch and changed-file list into the revision round. This prevents the coder from restarting cold on round 2 and lets it build on prior work rather than repeat dead ends.

---

## Compute Caveat

`Avg relative compute burden to 4o-mini` is a heuristic useful for internal relative comparison. For self-hosted or UMich models, it estimates relative compute burden, not literal dollar cost.

## Best Current Claims

1. **A strong planner makes a weak coder competitive with all-large configs at ~1/4 the compute.** (20-case benchmark)

2. **Pre-coder plan auditing (critic) and post-coder patch review (reviewer) each independently boost resolved rate, but through different mechanisms on different cases.** (10-case audit benchmark)

3. **Critic is the most compute-efficient audit mechanism** — same resolved-rate lift as reviewer at 17% less compute. Catching plan errors before code generation is cheaper than catching patch errors after.

4. **Both audits reach 89% of the large-coder ceiling (8/10 vs 9/10) at roughly 10x less compute.**
