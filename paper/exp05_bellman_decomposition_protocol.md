# Exp 05 protocol — Bellman decomposition of stopping judgments

## Status

Design-only protocol. Do not launch until the valid Exp 04 calibrated-state ablation is inspected.

## Motivation

Exp 03 showed that the model's native stopping curve can be both premature and non-monotone. Exp 04 tests whether exact posterior state statistics rescue that behavior. If posterior calibration alone is insufficient, the next hypothesis is a **dynamic continuation-value failure**: the model may price only the immediate next query and omit the option value of acting again after a miss.

## Main question

When state uncertainty is already calibrated, can an explicit Bellman-style decomposition improve the model's estimate of the value of continuing without revealing the oracle STOP/CONTINUE decision?

## Arms

Use identical counterfactual states near each oracle boundary.

### A. Calibrated-state arm

Provide only exact state statistics already used in Exp 04:

- posterior probability that a target still exists;
- highest one-query hit probability among remaining candidates;
- candidate with that highest hit probability.

### B. Bellman-decomposition arm

Provide the same state statistics, then require the agent to report the following components before its final action:

```text
immediate_hit_value
query_cost
miss_probability
estimated_value_after_miss
full_expected_net_gain
```

The prompt should remind the model that a miss is not terminal: after a miss, the future policy may either query again or stop.

No oracle continuation value, oracle boundary, or oracle decision is supplied.

## Required JSON schema

```json
{
  "decision": "STOP or QUERY",
  "query": 1,
  "confidence": 0.0,
  "immediate_hit_value": 0.0,
  "query_cost": 0.0,
  "miss_probability": 0.0,
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief rationale"
}
```

If `decision = STOP`, `query = null`.

## Diagnostic quantities

For each state compare the model's decomposition with the exact dynamic-programming oracle:

```text
Q*(s,q) = -c + p_hit(s,q) R + p_miss(s,q) V*(s_q^-)
```

Record separately:

1. immediate-term error;
2. miss-probability error;
3. after-miss option-value error;
4. full continuation-value error;
5. STOP/CONTINUE agreement;
6. query-selection regret;
7. invalid-action rate.

The key mechanistic statistic is the **future-option-value omission error**:

```text
E_option(s) = estimated_value_after_miss - V*(s_q^-)
```

A strongly negative `E_option` near states where continuation is normatively positive would support the planning-horizon hypothesis.

## Interpretation matrix

- Exp 04 rescues behavior, Exp 05 adds little: stopping failure is mainly state/belief calibration.
- Exp 04 does not rescue, Exp 05 does: stopping failure is mainly dynamic decomposition / option-value reasoning.
- Neither rescues: failure is deeper than explicit posterior or Bellman scaffolding and may reflect unstable policy-level meta-reasoning.
- Both improve value estimates but not discrete actions: value representation and action policy are separable.

## Anti-leakage constraints

- Never use `oracle.best_query` in a prompt because the oracle implementation returns `None` at STOP states.
- Derive the highest-prior remaining candidate directly from the posterior masses, independently of the STOP/CONTINUE decision.
- Never expose `oracle_advantage`, oracle boundary, or any sign-equivalent quantity.
- Counterfactual states must be generated independently of the model's prior STOP decisions.

## Replication requirement

After mechanism selection, repeat each boundary-neighborhood state multiple times and across at least one additional model. Single-probe state curves are hypothesis-generating, not sufficient for a stable model-level estimate.
