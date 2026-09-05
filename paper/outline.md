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
- **H2 — Continuation-value miscalibration.** Agents systematically misestimate the sign or magnitude of marginal continuation value near the optimal stopping boundary.
- **H3 — Boundary displacement.** Agent stopping boundaries are shifted relative to the Bayes-optimal boundary; the direction of displacement may depend on cost and uncertainty.
- **H4 — Boundary incoherence.** Counterfactual state probes can reveal non-monotone STOP/QUERY behavior even when the normative boundary is monotone.
- **H5 — Outcome-luck confound.** Realized utility can reward a normatively wrong stop decision in a lucky hidden world, so realized utility alone is insufficient for evaluating stopping rationality.
- **H6 — Mechanistic separability.** Stopping errors can arise from at least two distinct sources: miscalibrated state beliefs and misvaluation of future option value after another action.
- **H7 — Prompted metareasoning repair.** Supplying exact state statistics or an explicit Bellman decomposition can selectively repair different failure mechanisms without imposing an external hard stop.
- **H8 — Cross-model separability.** Models with similar task competence may differ substantially in stopping calibration and in which repair mechanism helps them.

## Mechanism taxonomy

The empirical program distinguishes three levels of failure:

```text
Level 1 — Boundary bias
The agent stops too early or too late relative to the oracle boundary.

Level 2 — Boundary incoherence
The agent emits STOP and later returns to QUERY under strictly stronger evidence for stopping.

Level 3 — Value-mechanism failure
The error is localized to belief/state calibration, dynamic future-value estimation,
or the mapping from an estimated value to a discrete action.
```

This taxonomy prevents all stopping errors from being collapsed into a single "overthinking" or "underthinking" label.

## Experimental phases

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

Probe every counterfactual post-miss state through two states beyond the oracle boundary. Record:

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

The calibrated arm reveals exact posterior target-existence probability and exact one-query hit statistics, but does **not** reveal the oracle continuation value, its sign, the stopping boundary, or the oracle STOP/QUERY decision.

Purpose: test whether termination failure is primarily a **belief/state-calibration problem**.

Key paired outcomes:

- raw-wrong / calibrated-correct transitions;
- raw-correct / calibrated-wrong transitions;
- change in continuation-value absolute error;
- change in premature-stop and excess-query rates;
- decision/value self-consistency.

### Exp 05 — Bellman decomposition diagnostic
If exact state calibration is insufficient, compare the calibrated-state arm with an explicit dynamic decomposition:

```text
full continuation value
= - query cost
  + immediate hit value
  + miss probability * optimal value after a miss
```

The model must identify the concrete candidate whose continuation value it is evaluating, including when it ultimately chooses STOP. Exact state-action values are computed only after the response.

Purpose: test whether the remaining failure is specifically a **future-option-value / planning-horizon failure**.

Primary mechanism metrics:

- immediate-hit-value error;
- miss-probability error;
- future-option-value omission error;
- full evaluated-query value error;
- decomposition closure error;
- local stopping regret;
- query-selection regret.

### Exp 06 — cross-model and repeated-state replication
After selecting the mechanism of interest, repeat boundary-neighborhood probes multiple times per state and across at least one additional model family.

Single probes are treated as mechanism-discovery evidence; stable model-level claims require repeated observations.

### Exp 07 — richer autonomous-agent environments
Move beyond synthetic search to tool acquisition, retrieval, coding, and environments with irreversible or externally costly actions.

The goal is to test whether the same calibrated termination failures survive when continuation means another real tool call, retrieval step, code edit, or external action rather than another synthetic query.

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
- Future-option-value omission error
- Decomposition closure error
- Utility at Stop
- Outcome-Luck Gap, reported separately from normative regret

## Mechanism interpretation matrix

```text
Exp 04 improves stopping strongly
    -> state/belief calibration is the main bottleneck.

Exp 04 weak, Exp 05 improves strongly
    -> dynamic decomposition / future option value is the main bottleneck.

Neither improves
    -> failure is deeper than explicit posterior information or Bellman scaffolding;
       investigate policy-level meta-reasoning and action/value coupling.

Values improve but actions do not
    -> value representation and action policy are separable failure modes.
```

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

> We evaluate a native agent-emitted STOP as a calibrated metareasoning action, separately from forced termination, against an exact state-dependent continuation-value oracle. This permits direct measurement of both premature stopping and excessive continuation, calibration of the agent's own claimed marginal value of another action, and mechanism-level decomposition of why that value judgment fails.

## Main empirical story to test

The most interesting outcome is not necessarily runaway overthinking.

A stronger result would be a **two-sided stopping pathology**:

```text
Some states: the agent continues after value < 0.
Other states: the agent stops while value > 0.
```

If the same model can make both errors under controlled changes in cost and uncertainty, the paper becomes about **termination calibration**, not merely verbosity or overthinking.

The stronger mechanistic extension is:

> A model may possess the correct local action ranking yet still fail to assign the correct value to preserving the option to act again after the next observation.

## Working thesis

> **Knowing the best next action is not the same capability as knowing whether another action is worth taking.**
