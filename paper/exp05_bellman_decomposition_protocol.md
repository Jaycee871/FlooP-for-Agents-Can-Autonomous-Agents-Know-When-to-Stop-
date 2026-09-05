# Exp 05 protocol — Bellman decomposition of stopping judgments

## Status

Implementation scaffolded, execution gated. Do not launch until the valid Exp 04 calibrated-state ablation is inspected.

## Motivation

Exp 03 showed that the model's native stopping curve can be both premature and non-monotone. Exp 04 tests whether exact posterior state statistics rescue that behavior. If posterior calibration alone is insufficient, the next hypothesis is a **dynamic continuation-value failure**: the model may price only the immediate next query and omit the option value of acting again after a miss.

## Main question

When state uncertainty is already calibrated, can an explicit Bellman-style decomposition improve the model's estimate of the value of continuing without revealing the oracle STOP/CONTINUE decision?

## Arms

Use identical counterfactual states near each oracle boundary.

### A. Calibrated-state arm

Provide only the exact state statistics already used in Exp 04:

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

The prompt explicitly reminds the model that a miss is not terminal: after a miss, the future policy may either query again or stop.

No oracle continuation value, oracle boundary, or oracle decision is supplied.

## Required JSON schema

```json
{
  "decision": "STOP or QUERY",
  "query": 1,
  "evaluated_query": 1,
  "confidence": 0.0,
  "immediate_hit_value": 0.0,
  "query_cost": 0.0,
  "miss_probability": 0.0,
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief rationale"
}
```

`evaluated_query` is required even when the final decision is STOP. It identifies the concrete continuation candidate whose Bellman decomposition is being priced. If `decision = QUERY`, `query` must equal `evaluated_query`. If `decision = STOP`, `query = null` while `evaluated_query` remains populated. This prevents an otherwise unidentifiable decomposition at STOP states.

## Exact diagnostic decomposition

For an evaluated query q at state s:

```text
Q*(s,q) = -c + p_hit(s,q) R + p_miss(s,q) V*(s_q^-)
```

where `s_q^-` is the posterior state after query q misses.

Record separately:

1. immediate-hit-value error;
2. query-cost error;
3. miss-probability error;
4. after-miss option-value error;
5. full evaluated-query value error;
6. reported decomposition closure error;
7. STOP/CONTINUE agreement;
8. premature-stop regret;
9. excess-continuation regret;
10. query-selection regret;
11. invalid-action rate.

The key mechanistic statistic is the **future-option-value omission error**:

```text
E_option(s,q) = estimated_value_after_miss - V*(s_q^-)
```

A systematically negative `E_option` near states where continuation remains normatively valuable would support the planning-horizon hypothesis.

The **reported decomposition closure error** is:

```text
E_closure = reported_full_expected_net_gain
            - (-reported_query_cost
               + reported_immediate_hit_value
               + reported_miss_probability * reported_value_after_miss)
```

This separates failure to estimate the components from failure to combine them arithmetically.

## Interpretation matrix

- Exp 04 rescues behavior, Exp 05 adds little: stopping failure is mainly state/belief calibration.
- Exp 04 does not rescue, Exp 05 does: stopping failure is mainly dynamic decomposition / option-value reasoning.
- Neither rescues: failure is deeper than explicit posterior or Bellman scaffolding and may reflect unstable policy-level meta-reasoning.
- Both improve value estimates but not discrete actions: value representation and action policy are separable.
- Bellman decomposition closes arithmetically but option-value error stays negative: the failure is not arithmetic; it is specifically future-value estimation.

## Anti-leakage constraints

- Never use `oracle.best_query` in a prompt because the oracle implementation returns `None` at STOP states.
- Derive the highest-prior remaining candidate directly from the remaining target masses, independently of the STOP/CONTINUE decision.
- Never expose `oracle_advantage`, oracle boundary, oracle decision, or any sign-equivalent quantity.
- Counterfactual states are generated independently of the model's prior STOP decisions.
- `evaluated_query` is chosen by the model; its exact Q-value is computed only after the response for diagnostics.

## Failure tolerance

Provider requests use bounded retries for transient timeouts, rate limits, and server errors. Exp 05 also writes each completed probe immediately and records exhausted-request errors rather than discarding the entire sweep. A failed API call therefore cannot erase already collected observations.

## Replication requirement

After mechanism selection, repeat each boundary-neighborhood state multiple times and across at least one additional model. Single-probe state curves are hypothesis-generating, not sufficient for a stable model-level estimate.
