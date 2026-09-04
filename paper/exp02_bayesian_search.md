# Exp 02 — Bayesian witness-search stopping pilot

## Question
Can a real LLM agent decide when another search action has positive expected net value, rather than merely stopping because a fixed budget or harness cap is reached?

## Environment
- 50 candidate locations.
- Prior probability that a target exists: 0.8.
- Conditional target mass is proportional to `1 / i^1.5`.
- Query cost: 0.05 utility.
- Target reward: 1.0 utility.
- A Bayes-optimal dynamic-programming oracle decides whether to QUERY or STOP after each miss.

For the standard specification, the oracle queries after 0–5 misses and stops after the sixth miss. In particular, after three misses the oracle continuation value is still positive (`+0.0689533766`).

## Real-agent result
Model: `openai/gpt-oss-20b` through NVIDIA hosted inference.

| Scenario | Agent queries | Agent end | Oracle queries | Premature stop | Excess continuation |
|---|---:|---|---:|---:|---:|
| No target | 3 | STOP | 6 | 1 | 0 |
| Target at 2 | 2 | FOUND | 2 | 0 | 0 |
| Target at 12 | 3 | STOP | 6 | 1 | 0 |

Aggregate:
- decision agreement: 0.8333
- query-choice agreement conditional on QUERY: 1.0000
- mean absolute continuation-value estimation error: 0.0977
- premature-stop events: 2
- excess-continuation events: 0
- invalid actions: 0

## The important failure mode
The model did not exhibit runaway continuation in this pilot. It exhibited **premature epistemic closure**.

Across both trajectories that reached three misses, the agent selected the Bayes-optimal candidate at every query, but then stopped one decision later while the oracle still assigned positive value to continuing.

At the stopping state:

```text
oracle continuation value ≈ +0.06895
agent estimated value     ≈ -0.033 / -0.031
signed calibration error  ≈ -0.102 / -0.100
```

The key separation is therefore not task competence versus incompetence. The agent knew *where* to search but misjudged *whether further search was still worth doing*.

## Outcome-luck paradox
On the realized no-target world, the premature stop produced utility `-0.15`, while the oracle policy would have spent six queries and realized `-0.30`. The same happened in the hidden-target-at-12 world because the oracle also rationally stops before 12.

This does **not** make the premature stop ex ante optimal. At the moment the agent stopped, the remaining probability mass on targets 4–6 and beyond still made continuation worth `+0.06895` in expectation. The realized path happened to reward the wrong decision.

This motivates separating:

1. **Realized utility** — what happened in one hidden world.
2. **Normative decision regret** — value lost relative to the Bayes-optimal action at the information state where the choice was made.

A benchmark that scores only realized utility can mistakenly reward irrational early stopping when the realized world happens to contain no reachable witness.

## New metrics to carry forward
For state `s`, let `A*(s)` be the oracle advantage of CONTINUE over STOP and `Â(s)` the agent's reported expected continuation value.

- **Signed continuation-calibration error:** `Â(s) - A*(s)`.
- **Premature-stop regret:** `A*(s)` when the agent STOPs while `A*(s) > 0`.
- **Excess-continuation regret:** `-A*(s)` when the agent CONTINUEs while `A*(s) < 0`.
- **Boundary displacement:** agent stopping boundary minus oracle stopping boundary, measured in failed queries/actions.
- **Outcome-luck gap:** realized utility difference tracked separately from normative regret, never used as a substitute for it.

## Novelty pressure test
The literature now contains several close neighbors, so the claim cannot be simply that prior work has not studied agent stopping.

Closest threats include:

- **Agentic Abstention: Do Agents Know When to Stop Instead of Act?** (Luo, Wen, Wang, 2026), which directly studies sequential answer/abstain/continue behavior and timely abstention.
- **Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents** (Ding, Tomlin, Durrett, 2026), which formalizes cost–uncertainty tradeoffs and stopping exploration.
- **Cognitive Friction: A Decision-Theoretic Framework for Bounded Deliberation in Tool-Using Agents** (Di Gioia, 2026), which uses decision-theoretic stopping boundaries and value-of-information ideas.
- **Runaway is Ashamed, But Helpful** (Lu et al., 2025), which separates intrinsic and extrinsic early exit in embodied agents.
- **CostBench** (Liu et al., 2025), which evaluates cost-optimal multi-turn tool planning.

The defensible distinct contribution is narrower:

> Evaluate an agent-emitted STOP as a calibrated metareasoning decision, separately from forced harness termination, against an exact continuation-value oracle, while measuring both premature stopping and excessive continuation and eliciting the agent's own marginal-value estimate.

## Next experiment
**Exp 03 — stopping-boundary calibration sweep.**

Vary query cost while holding the search prior fixed. Probe counterfactual post-miss states directly so we can observe the model's STOP/QUERY decision even at states it would never reach under its own premature stopping policy. Compare the inferred model stopping boundary with the Bayes-optimal boundary and measure signed continuation-value calibration across the whole curve.
