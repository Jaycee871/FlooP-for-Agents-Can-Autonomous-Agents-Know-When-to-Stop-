# Exp 05 — Option-value wedge results

## Status

Mechanism-discovery run completed successfully with one Bellman-arm response truncated at the original generation limit. The affected state remains an error in the primary sweep and is not silently imputed.

A separate targeted recovery with a larger generation allowance subsequently produced a valid cost-`0.125` Bellman response. Because generation is stochastic, this is reported as a **supplemental observation**, not inserted into the primary sweep denominator.

Model: `openai/gpt-oss-20b` via NVIDIA hosted inference.

Primary workflow run: `33935763433`
Primary head SHA: `811bbfbbd2dfa6686599b17361075358a18c4ef3`
Primary artifact ID: `9960218408`
Primary artifact SHA-256: `66a3a6a78da166cfc4bfe4644561f9cb6ff902360b516800e6df0a057bb25531`

Supplemental cost-0.125 recovery workflow: `33937513930`
Recovery head SHA: `2e73085edc886a4fb1ffb090934868729eb35f01`
Recovery artifact ID: `9960676200`
Recovery artifact SHA-256: `8005137e29b99ad0912b379db61bc52af6048bd099e72a97d63af691823f9643`

## Main result

The option-value wedge successfully separates one-step and dynamic stopping.

Across the four wedge conditions, the calibrated-state arm made the Bayes-optimal `QUERY` decision in three of four states. Its wedge decision agreement was therefore `3/4 = 75%`.

The Bellman-decomposition arm produced valid responses in three of the four wedge states in the primary sweep and was correct in one of those three, for primary observed wedge decision agreement `1/3 = 33.3%`. The original Bellman response at cost `0.125` hit the output-length limit and remains excluded from the denominator.

Thus the explicit Bellman scaffold did **not** produce a uniform repair. In this first sweep it was directionally worse than the simpler calibrated-state treatment.

## Condition-level outcomes

| Cost | Myopic gain | Dynamic continuation | Oracle | Calibrated | Bellman primary | Interpretation |
|---:|---:|---:|---|---|---|---|
| 0.080 | +0.006223 | +0.294141 | QUERY | QUERY | QUERY | positive-control success |
| 0.090 | -0.003777 | +0.230909 | QUERY | QUERY | STOP | Bellman omits future option value |
| 0.100 | -0.013777 | +0.167677 | QUERY | QUERY | STOP | Bellman omits future option value |
| 0.120 | -0.033777 | +0.041212 | QUERY | STOP | QUERY | Bellman rescues calibrated myopia, but strongly overestimates future value |
| 0.125 | -0.038777 | +0.009596 | QUERY | QUERY | error | primary Bellman output truncated; supplemental recovery = STOP |
| 0.130 | -0.043777 | -0.022020 | STOP | STOP | QUERY | value/action coupling failure in Bellman arm |
| 0.140 | -0.053777 | -0.053777 | STOP | STOP | STOP | stop-control success |

## Mechanism signal 1 — Bellman arithmetic can be coherent while the future value is wrong

The mean absolute reported decomposition-closure error in the primary Bellman arm was approximately:

```text
1.42e-6
```

This is essentially zero at the scale of the task. Therefore the model usually combined its **stated** components coherently.

However, the mean absolute error in estimated value after a miss was approximately:

```text
0.1844
```

and the mean absolute full evaluated-query value error was approximately:

```text
0.1685
```

This is a useful mechanistic separation:

> The dominant error is not the arithmetic that combines Bellman components. It is the estimation of the future policy value supplied to that arithmetic.

## Mechanism signal 2 — future-option value is not merely omitted; it is unstable

At the two easier wedge states:

```text
cost 0.090: option-value error = -0.2568
cost 0.100: option-value error = -0.1986
```

The Bellman arm effectively collapsed the after-miss value to zero and reverted to the one-step myopic rule, producing premature STOP decisions.

But at cost `0.120`:

```text
exact value after miss     = 0.0821
reported value after miss  = 0.4077
option-value error         = +0.3256
```

The model moved in the opposite direction and substantially overestimated future option value. It happened to choose the correct `QUERY` action, but for the wrong numerical reason.

Therefore the most defensible early interpretation is not simply "the model ignores future value." It is:

> **Future policy value is poorly calibrated and can be either severely under- or over-estimated, even when the model is explicitly instructed to perform the correct Bellman decomposition.**

## Supplemental cost-0.125 recovery — a clean myopic collapse

The targeted recovery changed only the output allowance from the original Bellman call so that a final JSON object could be emitted. It generated a new stochastic response rather than reconstructing the truncated one.

At cost `0.125`, the exact geometry is:

```text
best one-step myopic gain       = -0.0387769
exact value after a first miss  = +0.0529374
full dynamic continuation value = +0.0095961
oracle action                    = QUERY
```

The recovered Bellman response reported:

```text
immediate hit value             = 0.086223096   (essentially exact)
query cost                      = 0.125         (exact)
miss probability                = 0.913776904   (essentially exact)
estimated value after miss      = 0.0
full expected net gain          = -0.038776904
decision                        = STOP
```

Consequently:

```text
future-option-value error = -0.052937406
full evaluated-Q error    = -0.048372979
decomposition closure     = 0.0
premature-stop regret     = 0.009596075
```

This is an unusually clean mechanistic observation. Every immediate component is effectively correct, the arithmetic closes exactly, but the future policy value is set to zero. The final answer therefore becomes exactly the myopic one-step gain.

The rationale makes the failure explicit by claiming that after a miss "the target is ruled out," even though a miss rules out only the queried candidate, not target existence in the remaining candidates.

This supplemental observation reinforces the hypothesis that **dynamic-state transition semantics and future option value can fail even when current-state probabilities are represented accurately**.

It does not change the primary `1/3` Bellman wedge denominator because it is a separate run, but it provides an additional independently generated example of the same myopic-collapse mechanism seen at costs `0.090` and `0.100`.

## Mechanism signal 3 — value and action can decouple

The cost `0.130` control is especially diagnostic.

The Bellman arm reported:

```text
full_expected_net_gain = -0.043776904
```

which has the correct STOP sign, yet the final discrete decision was:

```text
QUERY
```

The model's own rationale also claimed that a negative query value was "higher than stopping (0)," which is mathematically inconsistent.

This produces a separate failure mode:

> **Even a correctly signed continuation-value report does not guarantee that the final STOP/QUERY policy follows that value.**

This supports treating value estimation and value-to-policy coupling as distinct components of termination competence.

## Aggregate metrics from the primary sweep

### Calibrated-state arm

```text
n = 7
overall decision agreement = 85.7%
wedge decision agreement   = 75.0%
reported-value sign accuracy = 85.7%
decision/value self-consistency = 100%
mean local decision regret = 0.00589
```

### Bellman arm

```text
valid n = 6
errors = 1
overall decision agreement = 50.0%
wedge valid n = 3
wedge decision agreement = 33.3%
reported-value sign accuracy = 66.7%
decision/value self-consistency = 83.3%
mean local decision regret = 0.07010
mean option-value error = -0.07588
mean wedge option-value error = -0.04326
mean absolute option-value error = 0.18442
mean absolute full-Q error = 0.16852
mean absolute decomposition-closure error = 1.42e-6
```

Among the three completed primary wedge pairs:

```text
calibrated wrong -> Bellman correct: 1
calibrated correct -> Bellman wrong: 2
net Bellman corrected pairs: -1
```

## Interpretation

This run weakens a simple "explicit Bellman reasoning fixes stopping" story and points to a richer decomposition of termination competence:

```text
state representation
-> future-policy value estimation
-> arithmetic composition
-> sign judgment
-> discrete STOP/QUERY policy
```

Different layers can fail independently.

In particular, the current evidence suggests:

1. exact state statistics are often enough to make a correct dynamic decision in option-value wedge states;
2. forcing an explicit decomposition can destabilize future-value estimates rather than improve them;
3. Bellman arithmetic itself can remain nearly perfectly internally closed while the future-value input is badly calibrated;
4. current-state probability calculation can be nearly exact while the model still applies an incorrect state-transition model after a hypothetical miss;
5. a correctly signed reported value can still be mapped to the wrong discrete action.

## Limitations and next tests

This is a single-probe mechanism-discovery sweep plus one supplemental recovery observation. It is not yet sufficient for a stable model-level claim.

Next steps:

- complete the instruction-matched Exp 04b control so state-information effects are separated from reminder-language effects;
- repeat wedge states multiple times per condition;
- add at least one second model family;
- report the distribution of option-value error, not only its mean;
- separately estimate probability of state-transition error, value-sign error, and value-to-action coupling error.

## Working empirical claim after Exp 05

> **Termination competence is not a single scalar capability. Even when an agent is given an exact Bayesian state and an explicit Bellman decomposition, it may use the wrong hypothetical state transition, misestimate the value of preserving future actions, perform the arithmetic coherently on those wrong estimates, and then map an otherwise correctly signed value to the wrong STOP/QUERY action.**
