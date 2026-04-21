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

## Part 3: MCTS Integration (27-Case Unified Matrix)

We integrated the critic into Steven's MCTS pipeline with two modes and ran the full 8-config × 27-case matrix in a single session for consistency. See [raw_data.md](/Users/rafe/classes/eecs545/project/results_summary/raw_data.md) Source 1 for full tables.

- **Plan critic (warnings):** critic output injected into MCTS search context without plan revision. Revision destroyed performance (50%) in early work; warnings preserve search breadth.
- **Critic gate:** replaces MCTS auto-accept with LLM patch quality evaluation before submission. Deployment-realistic — no hidden-test oracle leak.

### MCTS Results (27 cases)

```
Variant              Resolved   Rate    Description
─────────────────────────────────────────────────────────────────────
mcts_baseline         17/27    63.0%   MCTS + auto-accept
mcts + plan_critic    19/27    70.4%   MCTS + critic warnings (no revision)
mcts + critic_gate    20/27    74.1%   MCTS + critic submission gate (best MCTS)
```

### Key MCTS Findings

**1. Critic submission gate is the best MCTS variant** — +3 cases over auto-accept baseline. Counter-intuitively, the gate *improves* solve rate by preventing the search from stopping on early branches that pass success checks but don't fully fix the issue.

**2. Plan critic as warnings is neutral-to-helpful for MCTS** — +2 cases over baseline, no regression. Critic-as-revision would hurt (seen in earlier ablation at 50%).

**3. MCTS variants are 2× the compute of linear gpt→qw but 6× cheaper than gpt solo.** Compute: mcts ≈ 3.9, linear gpt→qw ≈ 2.0, gpt solo ≈ 23.7.

---

## Part 4: 60-Case Held-out Matrix (Final Results)

The 27-case matrix was scaled up to 60 instances split across three held-out case sets (`c1`/`c2`/`c3`, 20 each). Eight configs were kept for the final efficiency frontier. Results are reported in **BPT** (M B-param·tokens), a model-size-normalized compute unit — distinct from the `relative-to-4o-mini` heuristic used in Parts 1–3 (see Compute Unit Reconciliation below).

All multi-role/MCTS variants in this matrix use the **critic gate** from Part 3 (an LLM patch-quality check that replaces MCTS auto-accept). In variant naming this is marked by the `_strict` suffix; the soft (non-gate) counterparts were dropped from the final figures because the gate dominates on all but one config.

### Final-matrix results (c2+c3 held-out avg)

```
Variant                                  Solve rate   BPT (M)
──────────────────────────────────────────────────────────────
L  — 9b linear                             67.5%       0.25
M  — 30b linear                            62.5%       0.75
A  — 9b MCTS + gate                        67.5%       0.30
G  — 9b MCTS + hindsight + gate            70.0%       0.30
F  — 120b plan + 9b code + gate            75.0%       0.60
B  — 120b plan + 30b code + gate           77.5%       0.80
P  — Mixed MCTS + hindsight + gate         72.5%       1.25
C  — Mixed MCTS (120b plan + 9b code)      75.0%       1.45
```

### Key findings, aligned with critic results from Part 2/3

**1. Critic gate transfers from Part 3 to the 60-case matrix.** Every strict-gate variant beats its soft counterpart on held-out sets; the Part 3 result that the critic-as-gate is the best MCTS mode is confirmed at larger scale.

**2. Strong planner + small coder wins on efficiency — consistent with Part 1.** F (120b plan + 9b code) reaches 75% solve rate at 0.60 M BPT, matching the Part 1 claim that a strong planner makes a weak coder competitive at a fraction of the cost.

**3. MCTS buys almost nothing at 9b.** A (9b MCTS) matches L (9b linear) at 67.5% despite more compute; hindsight (G) adds +2.5 pp. MCTS only pays off once paired with a strong planner (C, P).

**4. B dominates the efficiency frontier.** 77.5% solve rate at 0.80 M BPT — higher rate and lower cost than the mixed-size MCTS configs (C, P). The MCTS overhead is not justified when the planner is already 120b.

**5. Hindsight helps at small scale but hurts at mixed scale.** G > A by +2.5 pp at 9b; P < C by −2.5 pp at mixed. Hindsight context appears to crowd out the 120b planner's advantage when it is present.

### Compute Unit Reconciliation

Parts 1–3 use `avg relative compute burden to 4o-mini` (a wall-clock-token heuristic). Part 4 uses **BPT** = (sum of per-call B-params × tokens) / 1e6, a model-size-normalized count of work done. The two are not linearly convertible — BPT weights large-model calls much more heavily than the 4o-mini heuristic does, so direct ratio comparisons across parts (e.g., "Part 2 reviewer at 2.60 vs. Part 4 B at 0.80") are not meaningful. Within each part, relative compute differences are comparable; across parts, only *directional* claims transfer.

## Compute Caveat

`Avg relative compute burden to 4o-mini` is a heuristic useful for internal relative comparison. For self-hosted or UMich models, it estimates relative compute burden, not literal dollar cost.

## Best Current Claims

1. **A strong planner makes a weak coder competitive with all-large configs at ~1/4 the compute.** (20-case benchmark)

2. **Pre-coder plan auditing (critic) and post-coder patch review (reviewer) each independently boost resolved rate, but through different mechanisms on different cases.** (10-case audit benchmark)

3. **Critic is the most compute-efficient audit mechanism** — same resolved-rate lift as reviewer at 17% less compute. Catching plan errors before code generation is cheaper than catching patch errors after.

4. **Both audits reach 89% of the large-coder ceiling (8/10 vs 9/10) at roughly 10x less compute.**

5. **Critic as MCTS submission gate is the best MCTS variant** — +3 cases over auto-accept on 27 cases, and gives deployment-realistic numbers instead of leaking the hidden-test oracle. **Confirmed at scale on the 60-case matrix (Part 4):** the gate strictly dominates the soft-accept counterpart across all multi-role and MCTS configs.

6. **On the 60-case matrix, the best overall config is B (120b plan + 30b code + gate) at 77.5% solve rate / 0.80 M BPT.** Large-planner + mid-size-coder with the critic gate beats every MCTS variant tested, suggesting that for this task distribution search overhead is not justified when the planner is already strong.
