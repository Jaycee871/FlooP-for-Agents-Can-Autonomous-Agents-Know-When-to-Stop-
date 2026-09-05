# Exp 05 protocol — Bellman decomposition in option-value wedge states

## Status

Ready for execution. The corrected Exp 04 result has been inspected.

Exp 04 produced a strong calibrated-state rescue, but its standard boundary states have an important limitation: the sign of the best immediate one-step gain agrees with the sign of the full dynamic continuation value. A model can therefore obtain the correct STOP/QUERY action from supplied probabilities without representing future option value.

Exp 05 removes that shortcut.

## Main question

When the current Bayesian state is already supplied exactly, can an agent recognize that a sequence of future actions is worth initiating even when the **next action looks unprofitable in isolation**?

Equivalently:

> Can the agent value the option to keep acting after the next miss?

## Option-value wedge

Define the best one-step myopic gain:

```text
M(s) = max_q [-c + p_hit(s,q) R]
```

and the exact dynamically controlled continuation value:

```text
C*(s) = max_q [-c + p_hit(s,q) R + p_miss(s,q) V*(s_q^-)].
```

An **option-value wedge state** satisfies:

```text
M(s) <= 0
but
C*(s) > 0.
```

In such a state:

```text
myopic policy  -> STOP
Bayes-optimal dynamic policy -> QUERY
```

The sign difference makes future option value decision-relevant rather than merely an additive refinement.

## Wedge world

Use a deliberately flat target prior:

```text
n_candidates = 10
P(target exists) = 0.8
alpha = 0.05
reward = 1.0
state = initial state, no failed queries yet
```

The flat prior spreads target mass across many candidates. Individual queries can therefore look unattractive while the *sequence* of possible future queries retains substantial value.

The deterministic oracle geometry is:

| Query cost | Best one-step gain | Full continuation value | Oracle action | Role |
|---:|---:|---:|---|---|
| 0.080 | +0.006223 | +0.294141 | QUERY | myopic-positive control |
| 0.090 | -0.003777 | +0.230909 | QUERY | option-value wedge |
| 0.100 | -0.013777 | +0.167677 | QUERY | option-value wedge |
| 0.120 | -0.033777 | +0.041212 | QUERY | option-value wedge |
| 0.125 | -0.038777 | +0.009596 | QUERY | near-boundary option-value wedge |
| 0.130 | -0.043777 | -0.022020 | STOP | dynamic-stop control |
| 0.140 | -0.053777 | -0.053777 | STOP | one-step stop control |

At cost `0.130`, the value after a first miss is still positive, but it is not large enough to compensate for initiating the current query. This is useful for checking that a Bellman-scaffolded model does not simply learn the rule "future value exists, therefore query."

## Arms

Each cost condition is probed under two arms with identical exact state statistics.

### A. Calibrated-state arm

Provide:

- exact posterior probability that a target exists;
- exact highest one-query hit probability;
- the highest-mass remaining candidate.

The prompt reminds the model to optimize the full continuation policy, but does not force an explicit decomposition.

This arm asks whether state calibration alone is sufficient once the one-step and dynamic decisions disagree.

### B. Bellman-decomposition arm

Provide the same exact state statistics, then require the model to report:

```text
immediate_hit_value
query_cost
miss_probability
estimated_value_after_miss
full_expected_net_gain
```

The model must explicitly represent that a miss is not terminal and that the future policy may query again or stop.

No oracle continuation value, oracle action, dynamic boundary, or sign-equivalent quantity is supplied.

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

`evaluated_query` is required even when the final decision is STOP. It identifies the concrete continuation candidate whose Bellman decomposition is being priced. If `decision = QUERY`, `query` must equal `evaluated_query`. If `decision = STOP`, `query = null` while `evaluated_query` remains populated.

## Exact diagnostic decomposition

For evaluated query `q`:

```text
Q*(s,q) = -c + p_hit(s,q) R + p_miss(s,q) V*(s_q^-)
```

The key mechanistic statistic is the **future-option-value omission error**:

```text
E_option(s,q)
= estimated_value_after_miss - V*(s_q^-).
```

A negative value means the model undervalues the future policy preserved after a miss.

Record separately:

1. immediate-hit-value error;
2. query-cost error;
3. miss-probability error;
4. future-option-value omission error;
5. full evaluated-query value error;
6. reported decomposition closure error;
7. STOP/CONTINUE agreement;
8. wedge-only STOP/CONTINUE agreement;
9. premature-stop regret;
10. excess-continuation regret;
11. query-selection regret;
12. invalid-action rate.

## Decomposition closure

Define:

```text
E_closure
= reported_full_expected_net_gain
  - [-reported_query_cost
     + reported_immediate_hit_value
     + reported_miss_probability * reported_value_after_miss].
```

This distinguishes estimating the wrong components from combining stated components incorrectly.

A particularly diagnostic pattern is:

```text
E_closure approximately 0
E_option strongly negative
```

which would mean the model performs the Bellman arithmetic coherently but underprices the future state.

## Primary comparison

The primary treatment comparison is **wedge-only decision agreement**:

```text
Calibrated state vs Bellman decomposition
```

on costs `0.090`, `0.100`, `0.120`, and `0.125`.

Useful paired transitions are:

```text
calibrated STOP -> Bellman QUERY   when oracle = QUERY
calibrated QUERY -> Bellman STOP   when oracle = QUERY
```

The first is a direct treatment rescue of a myopic stopping error.

## Interpretation matrix

- Calibrated arm already succeeds on wedge states: exact state statistics plus natural reasoning are sufficient for dynamic continuation in this world.
- Calibrated arm fails but Bellman arm succeeds: explicit future-option decomposition repairs a planning-horizon failure.
- Both fail on wedge states with negative `E_option`: future policy value remains systematically omitted even under explicit scaffolding.
- Both fail but `E_option` is accurate: the error lies downstream, such as composition or action/value coupling.
- Bellman arm over-continues at cost `0.130`: the scaffold induces a generic continuation bias rather than calibrated dynamic reasoning.

## Anti-leakage constraints

- Never expose `oracle.best_query`; it becomes `None` at oracle STOP states.
- Derive the highest-mass remaining candidate directly from target masses.
- Never expose `oracle_advantage`, oracle decision, a dynamic stopping boundary, or any sign-equivalent quantity.
- The model selects `evaluated_query`; exact `Q*(s,q)` is computed only after the response.
- The condition names are internal records and are never included in the model prompt.

## Failure tolerance

Provider requests use bounded retries for transient timeouts, rate limits, and server errors. Completed probes are written immediately to `probes.jsonl`, and exhausted request errors are recorded instead of aborting the entire sweep.

## Replication requirement

This cost sweep is still a mechanism-discovery experiment. Any apparent wedge effect or Bellman rescue must be repeated multiple times per condition and across at least one additional model family before it is promoted to a stable model-level claim.
