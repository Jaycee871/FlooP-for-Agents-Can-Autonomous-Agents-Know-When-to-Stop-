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
- **H6 — Prompted metareasoning repair.** Explicit lookahead or value-of-information reasoning can improve stopping calibration without imposing an external hard bound.
- **H7 — Cross-model separability.** Models with similar task competence may differ substantially in stopping calibration.

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

Planned standard conditions:

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

### Exp 04 — metareasoning treatments
Compare baseline prompting with:

- explicit multi-step lookahead;
- self-critique;
- fresh-context verification;
- tool-assisted expected-value calculation;
- privileged verifier assistance.

The critical distinction is whether the treatment improves the agent's own stopping judgment rather than merely moving termination into an external controller.

### Exp 05 — cross-model replication
Repeat the boundary sweep across closed and open-weight model families.

### Exp 06 — richer autonomous-agent environments
Move beyond synthetic search to tool acquisition, retrieval, coding, and environments with irreversible or externally costly actions.

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
- Utility at Stop
- Outcome-Luck Gap, reported separately from normative regret

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

> We evaluate a native agent-emitted STOP as a calibrated metareasoning action, separately from forced termination, against an exact state-dependent continuation-value oracle. This permits direct measurement of both premature stopping and excessive continuation, as well as calibration of the agent's own claimed marginal value of another action.

## Main empirical story to test

The most interesting outcome is not necessarily runaway overthinking.

A stronger result would be a **two-sided stopping pathology**:

```text
Some states: the agent continues after value < 0.
Other states: the agent stops while value > 0.
```

If the same model can make both errors under controlled changes in cost and uncertainty, the paper becomes about **termination calibration**, not merely verbosity or overthinking.

## Working thesis

> **Knowing the best next action is not the same capability as knowing whether another action is worth taking.**
