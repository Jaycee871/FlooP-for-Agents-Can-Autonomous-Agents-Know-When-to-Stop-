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
- **H3 — Boundary displacement.** Agent stopping boundaries can be shifted relative to the Bayes-optimal boundary; the direction of displacement may depend on cost and uncertainty.
- **H4 — Boundary incoherence.** Counterfactual state probes can reveal non-monotone STOP/QUERY behavior even when the normative boundary is monotone.
- **H5 — Outcome-luck confound.** Realized utility can reward a normatively wrong stop decision in a lucky hidden world, so realized utility alone is insufficient for evaluating stopping rationality.
- **H6 — Mechanistic separability.** Stopping errors can arise from distinct failures in state representation, future-policy value estimation, arithmetic composition, sign judgment, or value-to-policy coupling.
- **H7 — Treatment non-monotonicity.** A metareasoning scaffold that is normatively appropriate need not improve stopping and can destabilize a component estimate even while preserving internally coherent arithmetic.
- **H8 — Cross-model separability.** Models with similar task competence may differ substantially in stopping calibration and in which repair mechanism helps them.

## Mechanism taxonomy

The empirical program now distinguishes five internal stages rather than treating stopping as one scalar ability:

```text
State representation
    -> What information state am I actually in?

Future-policy value estimation
    -> What is the value preserved after another observation?

Arithmetic composition
    -> Can the component values be combined coherently?

Sign judgment
    -> Is continuation value positive or non-positive?

Discrete STOP/QUERY policy
    -> Does the action follow the value judgment?
```

Observed behavior can fail at any one of these layers while other layers remain intact.

At the behavioral level we separately track:

```text
Level 1 — Boundary bias
The agent stops too early or too late relative to the oracle boundary.

Level 2 — Boundary incoherence
The agent emits STOP and later returns to QUERY under strictly stronger evidence for stopping.

Level 3 — Mechanism localization
The error is localized to one or more stages of the five-part pipeline above.
```

This prevents stopping errors from being collapsed into a generic "overthinking" or "underthinking" label.

## Experimental phases and current evidence

### Exp 00 — semantic validation
Deterministic synthetic trajectories verify that endogenous stopping, premature stopping, excess continuation, and failure to halt are measured distinctly.

### Exp 01 — real-model interface smoke test
Use an NVIDIA-hosted model to confirm structured agent actions, real reasoning traces, and explicit agent-emitted termination.

### Exp 02 — Bayesian witness-search pilot
Construct an uncertain search world with an exact Bayes-optimal stopping oracle.

Initial result with `openai/gpt-oss-20b`:

- oracle stopping boundary: 6 failed queries;
- model stopped after 3 failed queries in both no-target and late-target trajectories;
- at the stopping state the oracle continuation advantage remained about `+0.06895`;
- the model reported about `-0.033` / `-0.031`;
- query-choice agreement before stopping was perfect.

Interpretation: the observed failure is **premature epistemic closure**, not inability to identify the best next query.

### Exp 03 — stopping-boundary calibration sweep
Vary query cost while holding the prior family fixed.

Standard conditions:

```text
query cost 0.03 -> oracle boundary 10 misses
query cost 0.05 -> oracle boundary 6 misses
query cost 0.08 -> oracle boundary 4 misses
```

Counterfactual probes show both premature stopping and, in some post-boundary states, excessive continuation. The model can therefore make both signs of termination error under controlled changes in state and cost.

The sweep records:

- model STOP/QUERY decision;
- model-reported continuation value;
- exact oracle continuation advantage;
- one-step myopic value;
- selected next query;
- boundary displacement;
- decision/value self-consistency;
- post-STOP reversals under independent state probes.

### Exp 04 — calibrated-state ablation
Probe states around each oracle boundary under two arms:

```text
raw state description
vs.
raw description + exact posterior state statistics
```

The corrected, non-leaky 12-pair run with `openai/gpt-oss-20b` produced:

```text
Raw decision agreement                 7/12  = 58.3%
Calibrated-state decision agreement   12/12  = 100%

Raw reported-value sign accuracy       58.3%
Calibrated reported-value sign         100%

Raw premature-stop rate pre-boundary   66.7%
Calibrated premature-stop rate          0%

Raw excess-query rate post-boundary    16.7%
Calibrated excess-query rate            0%

Raw invalid-query rate                 33.3%
Calibrated invalid-query rate           0%

raw wrong -> calibrated correct        5
raw correct -> calibrated wrong        0
```

Mean absolute continuation-value error decreased from approximately `0.06947` to `0.03938`.

This is strong mechanism-discovery evidence that **state representation / posterior tracking is an important stopping bottleneck** in the standard boundary world.

However, the original treatment also contains a prompt-reminder difference: the calibrated treatment explicitly reminds the model to value the full continuation policy. Exp 04b therefore repeats the comparison with that reminder language held constant across arms, so the incremental effect of exact state information can be isolated.

A second limitation is more important. In the standard Exp 04 states, the sign of the best one-step gain agrees with the sign of the full dynamic continuation value. Therefore a myopic one-step rule can obtain the correct action after receiving exact probabilities. Exp 04 does not by itself demonstrate representation of future option value.

### Exp 04b — instruction-matched state ablation
Both arms receive the same dynamic-policy reminder:

```text
instruction-only arm
= state description + dynamic-policy reminder

state-calibrated arm
= state description + same reminder + exact posterior statistics
```

Purpose: isolate the causal contribution of exact state information from the contribution of reminder language.

Status: launched; treat the original Exp 04 comparison as a treatment effect until this control is complete.

### Exp 05 — option-value wedge / Bellman decomposition diagnostic
Exp 05 deliberately constructs states where a one-step policy and a dynamically optimal policy disagree:

```text
M(s) <= 0
but
C*(s) > 0
```

so a myopic policy must STOP while the Bayes-optimal dynamic policy must QUERY.

The four primary wedge costs are:

```text
0.090, 0.100, 0.120, 0.125
```

The first mechanism-discovery run produced a surprising result. The calibrated-state arm was correct on `3/4 = 75%` of wedge states. The explicit Bellman-decomposition arm had three valid wedge responses and was correct on only `1/3 = 33.3%`; one cost-0.125 Bellman response exhausted the output allowance before returning final JSON.

Among completed wedge pairs:

```text
calibrated wrong -> Bellman correct   1
calibrated correct -> Bellman wrong   2
net Bellman corrected pairs          -1
```

The scaffold therefore did **not** behave as a monotone repair.

#### Exp 05 mechanism result A — arithmetic is not the main problem

Bellman decomposition closure was almost exact:

```text
mean absolute closure error ≈ 1.42e-6
```

while:

```text
mean absolute after-miss option-value error ≈ 0.1844
mean absolute full evaluated-Q error        ≈ 0.1685
```

Thus the model can coherently combine its stated Bellman components while supplying badly miscalibrated future-policy values.

#### Exp 05 mechanism result B — future value is unstable, not merely omitted

Examples:

```text
cost 0.090: option-value error ≈ -0.2568 -> premature STOP
cost 0.100: option-value error ≈ -0.1986 -> premature STOP
cost 0.120: option-value error ≈ +0.3256 -> QUERY for an overestimated future value
```

The defensible interpretation is therefore not simply "the model ignores future option value." It is that **future policy value is poorly calibrated and can swing between severe under- and over-estimation even under explicit Bellman scaffolding**.

#### Exp 05 mechanism result C — value-to-policy coupling can fail

At the `0.130` dynamic-stop control, the Bellman arm reported a negative full expected net gain (`-0.043776904`), whose sign implies STOP, but then emitted `QUERY`.

This is direct evidence that a correctly signed stated value does not guarantee a matching discrete termination action.

The current mechanism pipeline is therefore empirically motivated rather than purely conceptual:

```text
state representation
-> future-policy value estimation
-> arithmetic composition
-> sign judgment
-> discrete STOP/QUERY mapping
```

A targeted supplemental run is recovering the length-truncated `0.125` Bellman observation with a larger output allowance. It is recorded as a new stochastic observation rather than silently replacing the original missing response.

### Exp 06 — repeated-state and cross-model replication
After mechanism discovery, repeat critical states multiple times per condition and across at least one additional model family.

Priority states now include:

```text
standard pre-boundary states repaired by state calibration
wedge cost 0.090
wedge cost 0.100
wedge cost 0.120
near-boundary wedge cost 0.125
dynamic-stop control cost 0.130
```

Replication should estimate distributions, not only single-response means, for:

- future-option-value error;
- reported-value sign error;
- value-to-action coupling error;
- local stopping regret;
- treatment transitions between calibrated and Bellman arms.

### Exp 07 — richer autonomous-agent environments
Move beyond synthetic search to tool acquisition, retrieval, coding, and environments with irreversible or externally costly actions.

The goal is to test whether the same decomposed termination failures survive when continuation means another real tool call, retrieval step, code edit, or external action rather than another synthetic query.

## Primary metrics

- Premature Stop Rate
- Excess Continuation Ratio
- Failure-to-Halt Rate
- Local Termination Regret
- Signed Continuation-Calibration Error: `Â(s) - A*(s)`
- Continuation-value sign accuracy
- Stopping-Boundary Displacement
- Decision / Reported-Value Self-Consistency
- Query-choice agreement conditional on continuation
- Boundary monotonicity / post-stop reversal count
- Future-option-value omission / calibration error
- Full evaluated-query value error
- Bellman decomposition closure error
- Value-to-policy coupling error
- Utility at Stop
- Outcome-Luck Gap, reported separately from normative regret

## Current mechanism interpretation

The evidence no longer supports a one-line diagnosis such as "agents stop too early" or "agents overthink."

Instead:

```text
Exp 02 / Exp 03
    -> stopping competence separates from next-action competence.

Exp 04
    -> correcting state representation can strongly repair stopping in a myopic-sign-aligned regime.

Exp 05
    -> when myopic and dynamic decisions are forced apart, future-value estimation becomes independently visible.

Exp 05 Bellman arm
    -> explicit decomposition can preserve arithmetic coherence while destabilizing future-value estimates.

Exp 05 cost 0.130
    -> even a correctly signed value can fail to control the final STOP/QUERY action.
```

The emerging object of study is therefore **termination calibration as a layered metareasoning system**, not simply a stopping threshold.

## Related-work positioning / novelty guard

The paper must explicitly acknowledge that agent stopping is no longer an empty literature niche.

Closest threats include:

- **Agentic Abstention: Do Agents Know When to Stop Instead of Act?** — sequential answer / abstain / continue behavior and timely abstention.
- **Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents** — cost–uncertainty tradeoffs and stopping exploration.
- **Cognitive Friction: A Decision-Theoretic Framework for Bounded Deliberation in Tool-Using Agents** — value-of-information and decision-theoretic stopping boundaries.
- **Runaway is Ashamed, But Helpful** — intrinsic versus extrinsic early exit in embodied agents.
- **Doomed from the Start** — external early-abort prediction of doomed trajectories.
- **CostBench** — cost-optimal multi-turn planning and adaptation.

Therefore the contribution should not be framed as "agents have never been evaluated on stopping." The stronger and defensible claim is:

> We evaluate a native agent-emitted STOP as a calibrated metareasoning decision, separately from forced termination, against an exact state-dependent continuation-value oracle. The benchmark measures both directions of stopping error and decomposes failure into state representation, future-policy value estimation, composition, sign judgment, and value-to-policy coupling.

## Main empirical story

The paper has moved beyond the simple question of whether the model "knows when to stop."

The more specific story is:

> **Knowing where to act, knowing whether to continue, valuing the option to act again, and converting that value into a stopping action are separable competencies.**

A model may succeed at one while failing at another.

## Working thesis

> **Termination competence is a layered calibration problem, not a by-product of task competence.**
