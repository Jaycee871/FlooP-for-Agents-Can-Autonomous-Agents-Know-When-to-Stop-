# Paper skeleton — FlooP for Agents

## Working title
**When Should an Agent Stop? Calibrating Endogenous Termination in Autonomous AI Agents**

Alternative title:

**Knowing Where to Act Is Not Knowing Whether to Continue: A Benchmark for Agent Stopping Calibration**

## Central question
When continuation is always a legal action, can an autonomous agent recognize when the expected value of another reasoning, search, or tool-use step has crossed zero?

The core object of study is not merely whether an episode ends. It is whether the **agent itself** chooses to stop at the right information state.

## Main conceptual distinction

Separate:

1. **Endogenous stop** — the agent emits `STOP` because it judges continuation no longer worthwhile.
2. **Task completion** — the environment ends because a target or terminal condition is reached.
3. **Forced termination** — step cap, timeout, token budget, harness abort, or external verifier ends the run.

Only the first is direct evidence of termination competence.

## Normative formalization

At information state `s`, define:

```text
A*(s) = V*(CONTINUE | s) - V(STOP | s)
```

where `V*` is the Bayes-optimal continuation value under optimal future control.

The normative action is:

```text
QUERY / CONTINUE  if A*(s) > 0
STOP              otherwise
```

The agent is also asked to report its own estimate:

```text
Â(s) = agent-estimated net value of continuing
```

This supports a direct calibration target rather than inferring stopping competence only from episode length or final success.

## Hypotheses

- **H1 — Competence / termination separability.** Correct next-action selection does not imply correct stop/continue judgment.
- **H2 — Continuation-value miscalibration.** Agents can misestimate the sign or magnitude of marginal continuation value near the optimal stopping boundary.
- **H3 — Boundary displacement.** Agent stopping boundaries can be shifted relative to the Bayes-optimal boundary; the direction may depend on cost and uncertainty.
- **H4 — Boundary incoherence.** Counterfactual state probes can reveal non-monotone STOP/QUERY behavior even when the normative boundary is monotone.
- **H5 — Outcome-luck confound.** Realized utility can reward a normatively wrong stop decision in a lucky hidden world, so realized utility alone is insufficient for evaluating stopping rationality.
- **H6 — Mechanistic separability.** Stopping errors can arise from distinct failures in current-state representation, counterfactual transition prediction, future-policy valuation, arithmetic composition, sign judgment, or value-to-policy coupling.
- **H7 — Treatment non-monotonicity.** A normatively appropriate metareasoning scaffold need not improve stopping and can destabilize a component estimate even while preserving internally coherent arithmetic.
- **H8 — Cross-model separability.** Models with similar task competence may differ substantially in stopping calibration and in which repair mechanism helps them.

## Mechanism taxonomy

The empirical program now distinguishes six internal stages:

```text
Current-state representation
    -> What information state am I in now?

Counterfactual transition calibration
    -> If I act and observe a miss, what information state will I be in next?

Successor / future-policy value estimation
    -> What is the value of acting optimally from that successor state?

Arithmetic composition
    -> Can current cost, immediate reward, and successor value be combined coherently?

Sign judgment
    -> Is full continuation value positive or non-positive?

Discrete STOP/QUERY policy
    -> Does the final action follow the value judgment?
```

A central emerging proposition is:

> **Current-state calibration is not transition calibration; transition calibration is not continuation-value calibration; continuation-value calibration is not policy calibration.**

At the behavioral level we separately track boundary bias, boundary incoherence, and mechanism localization.

## Experimental phases and current evidence

### Exp 00 — semantic validation
Deterministic synthetic trajectories verify that endogenous stopping, premature stopping, excess continuation, and failure to halt are measured distinctly.

### Exp 01 — real-model interface smoke test
Use an NVIDIA-hosted model to confirm structured agent actions, real reasoning traces, and explicit agent-emitted termination.

### Exp 02 — Bayesian witness-search pilot
Under the standard world with `openai/gpt-oss-20b`:

- oracle stopping boundary: 6 failed queries;
- model stopped after 3 failed queries in both no-target and late-target trajectories;
- oracle continuation advantage at the stopping state remained about `+0.06895`;
- model reported about `-0.033` / `-0.031`;
- query-choice agreement before stopping was perfect.

Interpretation: **premature epistemic closure**, not inability to identify the best next query.

### Exp 03 — stopping-boundary calibration sweep
Standard cost conditions:

```text
query cost 0.03 -> oracle boundary 10 misses
query cost 0.05 -> oracle boundary 6 misses
query cost 0.08 -> oracle boundary 4 misses
```

Counterfactual probes reveal both premature stopping and, in some post-boundary states, excessive continuation. This motivates treating termination as a calibration problem rather than a generic tendency to stop too early or too late.

### Exp 04 — calibrated-state ablation
The corrected, non-leaky 12-pair run produced:

```text
Raw decision agreement                 7/12  = 58.3%
Calibrated-state decision agreement   12/12  = 100%

Raw reported-value sign accuracy       58.3%
Calibrated reported-value sign         100%

Raw premature-stop rate pre-boundary   66.7%
Calibrated premature-stop rate          0%

Raw excess-query rate post-boundary    16.7%
Calibrated excess-query rate            0%

raw wrong -> calibrated correct        5
raw correct -> calibrated wrong        0
```

This suggested that state representation / posterior tracking is an important stopping bottleneck, but the original treatment also contained an extra dynamic-policy reminder.

A second limitation is structural: in the standard Exp 04 states, the sign of the best one-step gain agrees with the sign of the full dynamic continuation value, so correct actions do not require non-myopic option-value reasoning.

### Exp 04b — instruction-matched state ablation
Exp 04b holds the dynamic-policy reminder constant across both arms.

Completed result:

```text
Instruction-only decision agreement    3/12 = 25.0%
State-calibrated decision agreement    11/12 = 91.7%

Instruction-only value-sign accuracy   25.0%
State-calibrated value-sign accuracy   91.7%

Instruction-only premature-stop rate   100%
State-calibrated premature-stop rate   16.7%

Instruction-only excess-query rate     50.0%
State-calibrated excess-query rate      0%

instruction wrong -> state correct      8
instruction correct -> state wrong      0
```

This strongly reduces the reminder-language confound. Exact state information remains highly consequential when the instruction wording is matched.

Crucially, mean absolute scalar continuation-value error did **not** improve:

```text
instruction-only MAE = 0.07307
state-calibrated MAE  = 0.08930
```

Thus better current-state information can sharply improve the **sign/action boundary** without globally improving numeric value magnitude. Value-sign accuracy, magnitude calibration, and final action should therefore be reported separately.

One residual calibrated error occurs immediately before the low-cost oracle boundary (`cost=0.03`, `k=9`), making that state a priority target for repeated sampling.

### Exp 05 — option-value wedge / Bellman decomposition diagnostic
Exp 05 constructs states where:

```text
M(s) <= 0
but
C*(s) > 0
```

so a myopic policy must STOP while the Bayes-optimal dynamic policy must QUERY.

Primary wedge costs:

```text
0.090, 0.100, 0.120, 0.125
```

Primary run:

```text
Calibrated-state wedge agreement: 3/4 = 75%
Bellman valid wedge agreement:     1/3 = 33.3%
```

One primary Bellman response at `0.125` exhausted the output allowance and remains an error in the primary denominator.

Among completed primary pairs:

```text
calibrated wrong -> Bellman correct   1
calibrated correct -> Bellman wrong   2
net Bellman corrected pairs          -1
```

Explicit decomposition therefore did **not** behave as a monotone repair.

#### Exp 05 result A — arithmetic is not the main problem

```text
mean absolute Bellman closure error      ≈ 1.42e-6
mean absolute after-miss value error     ≈ 0.1844
mean absolute full evaluated-Q error     ≈ 0.1685
```

The model can coherently combine its stated components while supplying badly miscalibrated future-policy values.

#### Exp 05 result B — future value is unstable

```text
cost 0.090: option-value error ≈ -0.2568 -> premature STOP
cost 0.100: option-value error ≈ -0.1986 -> premature STOP
cost 0.120: option-value error ≈ +0.3256 -> QUERY for an overestimated future value
```

The problem is not merely systematic omission; future value can swing between severe under- and over-estimation.

#### Exp 05 result C — value-to-policy coupling can fail

At cost `0.130`, the Bellman arm reported a negative full expected net gain (`-0.043776904`) but emitted `QUERY`. Thus a correctly signed stated value need not control the final termination action.

#### Supplemental cost-0.125 recovery

A separate stochastic recovery produced exact current one-step quantities and exact arithmetic closure, but set value after miss to zero and stopped prematurely. Its rationale claimed the target would be ruled out after a single miss.

This suggested a possible transition-semantics failure, motivating a direct audit rather than trusting the rationale as a mechanism explanation.

### Exp 05b — counterfactual transition audit
Exp 05b directly tests whether the model predicts the wrong successor information state or instead misvalues an approximately correct successor state.

Completed five-state result:

```text
decision agreement                    = 40.0%
decision/value self-consistency       = 80.0%
mean abs current-hit error             = 2.96e-10
mean abs successor-existence error     = 0.00407
mean abs successor-hit error           = 0.00657
successor-best-candidate accuracy      = 100%
mean abs future-value error            = 0.13305
mean abs full-Q error                  = 0.12141
mean abs decomposition-closure error   = 0.000168
```

This result narrows the mechanism substantially.

The model represented the hypothetical successor state fairly well: after candidate `1` misses, it consistently predicted target-existence probability near the exact value, predicted the best next hit probability reasonably closely, and selected candidate `2` correctly in all five conditions.

Yet successor value remained badly unstable.

Examples:

```text
cost 0.100:
exact successor value     = +0.198576
reported successor value  = 0.0
agent action              = STOP
oracle action             = QUERY

cost 0.120:
exact successor value     = +0.082065
reported successor value  = 0.0
agent action              = STOP
oracle action             = QUERY
```

Therefore the current evidence favors **successor-value failure downstream of an approximately correct transition representation**, rather than a universal inability to predict the next Bayesian state.

The earlier cost-`0.125` recovery rationale should accordingly be treated as a response-specific error, not a stable causal diagnosis. When explicitly audited, the model can represent that a target may still exist after the miss.

#### Exp 05b value-to-action decoupling

At cost `0.125`, the model predicted the successor state almost exactly and reported:

```text
estimated value after miss  = +0.4069
full expected net gain      = +0.3322
```

but still emitted:

```text
STOP
```

Thus the experiment now contains both directions of action/value contradiction:

```text
Exp 05 cost 0.130:
reported value < 0, action = QUERY

Exp 05b cost 0.125:
reported value > 0, action = STOP
```

This makes value-to-policy coupling an independently observable failure mode rather than a theoretical possibility.

### Exp 06 — repeated-state and cross-model replication
After mechanism discovery, repeat critical states multiple times per condition and across at least one additional model family.

Priority targets:

```text
Exp 04b residual near-boundary state
wedge cost 0.100
wedge cost 0.120
wedge cost 0.125
dynamic-stop control cost 0.130
```

Estimate the joint distribution of:

- current-state error;
- counterfactual successor-state error;
- successor/future-policy value error;
- value-sign error;
- value-to-action coupling error;
- local stopping regret.

### Exp 07 — richer autonomous-agent environments
Move beyond synthetic search to tool acquisition, retrieval, coding, and environments with irreversible or externally costly actions.

The goal is to test whether the same decomposition survives when continuation means another real tool call, retrieval step, code edit, or external action.

## Primary metrics

- Premature Stop Rate
- Excess Continuation Ratio
- Failure-to-Halt Rate
- Local Termination Regret
- Signed Continuation-Calibration Error
- Continuation-value sign accuracy
- Stopping-Boundary Displacement
- Decision / Reported-Value Self-Consistency
- Query-choice agreement conditional on continuation
- Boundary monotonicity / post-stop reversal count
- Current-state calibration error
- Counterfactual successor-state calibration error
- Successor / future-policy value error
- Full evaluated-query value error
- Bellman decomposition closure error
- Value-to-policy coupling error
- Utility at Stop
- Outcome-Luck Gap

## Current mechanism interpretation

```text
Exp 02 / Exp 03
    -> stopping competence separates from next-action competence.

Exp 04 / Exp 04b
    -> exact current-state information strongly repairs the discrete stopping boundary;
       generic dynamic-policy reminders alone do not.

Exp 05
    -> when myopic and dynamic decisions are forced apart, successor-value reasoning
       becomes independently visible and explicit Bellman scaffolding is not a monotone repair.

Exp 05b
    -> successor-state prediction can be fairly accurate while successor-policy value
       remains badly miscalibrated.

Exp 05 + Exp 05b
    -> value-to-action coupling can fail in both directions.
```

The emerging object of study is **termination calibration as a layered metareasoning system**, not simply a stopping threshold.

## Methodological implication

A rationale can describe an error made in one response without identifying the model's persistent causal bottleneck. The cost-`0.125` recovery appeared to implicate transition semantics, but the explicit transition audit showed that the successor state could be represented fairly accurately while its value remained wrong.

Therefore mechanism claims should rely on intervention and decomposed measurement rather than natural-language rationale inspection alone.

## Related-work positioning / novelty guard

The paper must explicitly acknowledge existing work on agentic abstention, cost-aware exploration, bounded deliberation, early exit, doomed-trajectory prediction, and cost-optimal multi-turn planning.

The defensible contribution is narrower:

> We evaluate a native agent-emitted STOP as a calibrated metareasoning decision, separately from forced termination, against an exact state-dependent continuation-value oracle. The benchmark measures both directions of stopping error and decomposes failure into current-state representation, counterfactual transition prediction, successor-policy valuation, composition, sign judgment, and value-to-policy coupling.

## Main empirical story

> **Knowing where to act, knowing whether to continue, predicting where one more action will leave you, valuing the policy available there, and converting that value into a stopping action are separable competencies.**

A model may succeed at one while failing at another.

## Working thesis

> **Termination competence is a layered calibration problem, not a by-product of task competence.**
