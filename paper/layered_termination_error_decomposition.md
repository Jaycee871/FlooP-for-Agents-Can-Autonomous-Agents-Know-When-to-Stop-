# Layered termination-error decomposition

## 1. Why a scalar stopping score is insufficient

A STOP/QUERY error can be produced by several different internal failures. The experiments therefore treat termination competence as a composition of estimable mappings rather than a single latent scalar.

For current information state `s`, candidate action `q`, and miss successor `s_q^-`, the exact action value is:

```text
Q*(s,q)
= -c
  + p_hit(s,q) R
  + p_miss(s,q) V*(s_q^-)
```

and the optimal continuation value is:

```text
C*(s) = max_q Q*(s,q).
```

The normative stopping decision is determined only by the sign of `C*(s)` relative to the normalized STOP value zero.

A model, however, must pass through several representational and decision stages before producing a final action.

## 2. Six-stage termination pipeline

Represent the model's decision process diagnostically as:

```text
observed history
    -> current-state representation
    -> hypothetical successor-state representation
    -> successor-policy value estimate
    -> composed continuation-value estimate
    -> sign judgment
    -> STOP/QUERY action
```

The stages are not assumed to correspond to literal neural modules. They are **behaviorally identifiable interfaces** created by controlled elicitation and oracle comparison.

### Stage 1 — current-state representation

True current state:

```text
s
```

elicited model representation:

```text
s_hat
```

Observable components include posterior target-existence probability and current query hit probabilities.

### Stage 2 — counterfactual transition calibration

True miss transition:

```text
s_q^- = T(s, q, miss)
```

model-predicted transition:

```text
s_hat_q^- = T_hat(s_hat, q, miss)
```

This stage asks whether the agent can represent **where one more action would leave it**.

### Stage 3 — successor-policy value estimation

Exact successor value:

```text
V*(s_q^-)
```

model estimate:

```text
V_hat^-(q)
```

This is the value of acting optimally after the hypothetical miss, with STOP still available.

### Stage 4 — arithmetic composition

Given its reported components, the model implicitly or explicitly forms:

```text
Q_tilde(s,q)
= -c_hat
  + p_hat_hit R
  + p_hat_miss V_hat^-(q).
```

The separately elicited full continuation estimate is:

```text
Q_hat(s,q).
```

Define decomposition-closure error:

```text
E_closure
= Q_hat - Q_tilde.
```

Near-zero closure error means the model combines **its own stated inputs** coherently; it does not imply those inputs are correct.

### Stage 5 — value-sign judgment

For a concrete evaluated continuation:

```text
SignCorrect
= 1[sign(Q_hat) = sign(Q*)]
```

and at the policy level:

```text
ContinuationSignCorrect
= 1[sign(C_hat) = sign(C*)].
```

This stage is decision-relevant even when magnitude calibration is poor.

### Stage 6 — value-to-action coupling

With STOP normalized to zero, a self-consistent action obeys:

```text
Q_hat > 0  -> QUERY
Q_hat <= 0 -> STOP
```

Define coupling failure:

```text
E_couple = 1[action contradicts sign(Q_hat)].
```

Keep the two directions separate:

```text
positive value -> STOP
non-positive value -> QUERY
```

because they have different behavioral meanings and both have already appeared in mechanism-discovery runs.

## 3. Additive error decomposition for a fixed query

Ignoring only higher-order notation, the evaluated-query error can be decomposed exactly as:

```text
Q_hat - Q*
=
E_closure
- (c_hat - c)
+ (p_hat_hit - p_hit) R
+ [p_hat_miss V_hat^- - p_miss V*^-].
```

The future term can be split further:

```text
p_hat_miss V_hat^- - p_miss V*^-
=
(p_hat_miss - p_miss) V*^-
+ p_hat_miss (V_hat^- - V*^-).
```

Therefore:

```text
Q_hat - Q*
=
E_closure
- E_cost
+ R E_hit
+ V*^- E_miss
+ p_hat_miss E_successor_value.
```

where:

```text
E_cost            = c_hat - c
E_hit             = p_hat_hit - p_hit
E_miss            = p_hat_miss - p_miss
E_successor_value = V_hat^- - V*^-
```

This equation is useful because it converts an apparently unitary stopping error into separately measurable contributors.

## 4. Transition error versus successor-value error

Successor-value error itself can arise for two different reasons:

```text
V_hat^-(q) - V*(s_q^-)
```

may be large because:

```text
A. the model predicts the wrong successor state;
```

or because:

```text
B. it approximately predicts the correct successor state but values that state badly.
```

Exp 05b operationalizes this distinction by eliciting successor-state statistics before successor value.

A useful two-stage conceptual decomposition is:

```text
V_hat^-(q) - V*(s_q^-)
=
[V_hat(s_hat_q^-) - V*(s_hat_q^-)]
+
[V*(s_hat_q^-) - V*(s_q^-)].
```

The first bracket is **valuation error conditional on the represented successor state**. The second is the value consequence of **transition/state-representation error**.

The benchmark cannot assume access to the model's latent state representation, but explicit successor-statistic elicitation provides behavioral proxies for the second term.

## 5. Boundary error is a downstream observable, not a mechanism

Stopping-boundary displacement is:

```text
Delta_b = b_hat - b*
```

but the same `Delta_b` can be generated by different upstream causes:

```text
wrong current posterior
wrong hypothetical transition
wrong successor value
wrong arithmetic
wrong sign
wrong final action mapping
```

Therefore boundary displacement should be treated as a **behavioral phenotype** whose mechanism is identified only through ablations and component elicitation.

## 6. Decision calibration versus magnitude calibration

The experiments also motivate separating:

```text
magnitude calibration:
|Q_hat - Q*|
```

from:

```text
sign calibration:
1[sign(Q_hat) = sign(Q*)]
```

and from:

```text
action calibration:
1[action = oracle action].
```

A treatment can improve sign/action accuracy while worsening mean absolute magnitude error. Exp 04b provides an initial example of exactly this dissociation.

## 7. Mechanism-identification table

```text
Observed pattern                              Primary diagnosis
--------------------------------------------------------------------------
current state wrong                           current-state representation
current state right, successor state wrong   transition calibration
successor state right, successor value wrong successor-policy valuation
components plausible, closure large           arithmetic composition
reported value has wrong sign                 value/sign calibration
reported sign right, action opposite           value-to-action coupling
```

Multiple failures can coexist in one response. The table identifies the earliest directly observed failure, not a claim about hidden neural causality.

## 8. Empirical predictions

The layered account generates falsifiable predictions.

### P1 — intervention specificity

Supplying exact current-state statistics should primarily reduce Stage-1-derived errors, not necessarily improve successor valuation or scalar magnitude calibration.

### P2 — transition audit

If transition calibration is the dominant bottleneck, explicit successor-state errors should covary strongly with successor-value errors.

### P3 — valuation bottleneck

If successor valuation is independently difficult, successor-state statistics can be accurate while `E_successor_value` remains large.

### P4 — arithmetic dissociation

Near-zero `E_closure` can coexist with large `Q_hat - Q*` when component estimates are wrong.

### P5 — policy-coupling dissociation

A correctly signed reported continuation value need not imply a matching STOP/QUERY action.

### P6 — cross-model mechanism profiles

Two models can have similar aggregate stopping accuracy but different vectors of layer-specific errors. The natural cross-model object is therefore not only a scalar score but a **termination-calibration profile**.

## 9. Termination-calibration profile

For a model `m`, define a descriptive profile over a benchmark distribution `D`:

```text
TCP(m; D) = (
    E_current_state,
    E_transition,
    E_successor_value,
    E_closure,
    E_sign,
    E_coupling,
    R_local
).
```

Each component can be reported with a distribution rather than a single mean.

This creates a stronger benchmark target than ranking models only by episode success or average number of steps: two agents that stop equally often may fail for completely different reasons.

## 10. Core theoretical claim

> **Termination competence is compositionally identifiable. A correct stopping action requires not only task competence, but sufficiently calibrated state representation, counterfactual transition prediction, successor-policy valuation, numerical composition, value-sign judgment, and action coupling. Success at one stage does not logically guarantee success at the next.**

The experiments are designed to test which of these separations are merely conceptual and which are empirically realized across models.
