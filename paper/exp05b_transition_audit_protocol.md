# Exp 05b protocol — Counterfactual transition audit

## Motivation

Exp 05 localized a large part of Bellman error to the estimated value after a miss rather than to arithmetic composition. The supplemental cost-`0.125` recovery sharpened the mechanism further: current-state hit probability, miss probability, and arithmetic were essentially exact, but the model set value after a miss to zero and explicitly claimed that a miss would rule out target existence.

That observation leaves two different explanations:

```text
A. transition-model failure
   The model predicts the wrong Bayesian state after a hypothetical miss.

B. successor-value failure
   The model predicts the successor state correctly but assigns the wrong value to acting from it.
```

Exp 05b separates these mechanisms directly.

## Main question

When the current Bayesian state is supplied exactly, can the agent correctly predict the information state that would result from one additional failed query?

The experiment therefore audits the transition:

```text
current state s
    -- query q misses -->
hypothetical successor state s_q^-
```

before asking the model to value continuation from that successor.

## Diagnostic states

Use the same flat-prior option-value world as Exp 05 and audit the most informative conditions:

```text
cost 0.090  option-value wedge
cost 0.100  option-value wedge
cost 0.120  option-value wedge
cost 0.125  near-boundary option-value wedge
cost 0.130  dynamic-stop control
```

The evaluated query is the highest-mass remaining candidate, derived directly from target masses and independent of the oracle STOP/QUERY action.

## Information supplied

The model receives only current-state descriptive quantities:

- prior/search-world specification;
- exact posterior probability that a target currently exists;
- exact hit probability for the highest-mass remaining candidate;
- query cost and reward;
- the identity of the candidate to evaluate.

It does **not** receive any quantity from the hypothetical miss successor state.

In particular, the prompt never reveals:

- posterior target-existence probability after a miss;
- next-best hit probability after a miss;
- value after a miss;
- full dynamic continuation value;
- oracle STOP/QUERY action.

## Required response

For the fixed evaluated query, require:

```json
{
  "decision": "STOP or QUERY",
  "query": 1,
  "evaluated_query": 1,
  "current_hit_probability": 0.0,
  "miss_probability": 0.0,
  "predicted_target_exists_after_miss": 0.0,
  "predicted_best_hit_probability_after_miss": 0.0,
  "predicted_best_candidate_after_miss": 1,
  "estimated_value_after_miss": 0.0,
  "full_expected_net_gain": 0.0,
  "reason": "brief rationale"
}
```

If the final decision is STOP, `query` is null but `evaluated_query` and all counterfactual successor predictions remain required.

## Exact diagnostics

After the response, compute the exact miss-updated state with the benchmark oracle.

Primary transition metrics:

```text
E_exist^- = predicted P(target exists | hypothetical miss)
            - exact P(target exists | hypothetical miss)

E_hit^-   = predicted best next-query hit probability after miss
            - exact best next-query hit probability after miss

SuccessorCandidateCorrect
          = predicted best candidate after miss
            equals exact highest-mass remaining candidate
```

Successor-value error remains:

```text
E_value^- = estimated value after miss - V*(s_q^-)
```

Full evaluated-query error is:

```text
E_Q = reported full expected net gain - Q*(s,q)
```

## Mechanism classification

The central classification is:

```text
Transition wrong + future value wrong
    -> counterfactual state-transition failure is upstream.

Transition approximately right + future value wrong
    -> successor state is represented correctly but valued incorrectly.

Transition and future value right + final action wrong
    -> value-to-policy coupling failure.

All quantities right + action right
    -> locally coherent dynamic termination judgment.
```

This turns the Exp 05 observation into a falsifiable mechanism test rather than inferring the cause from rationale text alone.

## Anti-leakage rule

The candidate under audit is selected from the target-mass ordering, not `oracle.best_query`. Exact successor statistics are computed only after the model response. No successor statistic or sign-equivalent transformation of dynamic continuation value appears in the prompt.

## Replication note

Exp 05b is a targeted mechanism audit. If it confirms systematic transition errors, repeated-state Exp 06 should explicitly measure both current-state and counterfactual-transition calibration across models.
