# Exp 05 — Option-value wedge results

## Status

Mechanism-discovery run completed successfully with one Bellman-arm response truncated at the generation limit. The affected state is retained as an infrastructure/model-output error and is not silently imputed.

Model: `openai/gpt-oss-20b` via NVIDIA hosted inference.

Workflow run: `33935763433`
Head SHA: `811bbfbbd2dfa6686599b17361075358a18c4ef3`
Artifact ID: `9960218408`
Artifact SHA-256: `66a3a6a78da166cfc4bfe4644561f9cb6ff902360b516800e6df0a057bb25531`

## Main result

The option-value wedge successfully separates one-step and dynamic stopping.

Across the four wedge conditions, the calibrated-state arm made the Bayes-optimal `QUERY` decision in three of four states. Its wedge decision agreement was therefore `3/4 = 75%`.

The Bellman-decomposition arm produced valid responses in three of the four wedge states and was correct in one of those three, for observed wedge decision agreement `1/3 = 33.3%`. One Bellman response at cost `0.125` hit the output-length limit and is excluded from the denominator.

Thus the explicit Bellman scaffold did **not** produce a uniform repair. In this first sweep it was directionally worse than the simpler calibrated-state treatment.

## Condition-level outcomes

| Cost | Myopic gain | Dynamic continuation | Oracle | Calibrated | Bellman | Interpretation |
|---:|---:|---:|---|---|---|---|
| 0.080 | +0.006223 | +0.294141 | QUERY | QUERY | QUERY | positive-control success |
| 0.090 | -0.003777 | +0.230909 | QUERY | QUERY | STOP | Bellman omits future option value |
| 0.100 | -0.013777 | +0.167677 | QUERY | QUERY | STOP | Bellman omits future option value |
| 0.120 | -0.033777 | +0.041212 | QUERY | STOP | QUERY | Bellman rescues calibrated myopia, but strongly overestimates future value |
| 0.125 | -0.038777 | +0.009596 | QUERY | QUERY | error | Bellman output truncated |
| 0.130 | -0.043777 | -0.022020 | STOP | STOP | QUERY | value/action coupling failure in Bellman arm |
| 0.140 | -0.053777 | -0.053777 | STOP | STOP | STOP | stop-control success |

## Mechanism signal 1 — Bellman arithmetic can be coherent while the future value is wrong

The mean absolute reported decomposition-closure error in the Bellman arm was approximately:

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

## Aggregate metrics

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

Among the three completed wedge pairs:

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
4. a correctly signed reported value can still be mapped to the wrong discrete action.

## Limitations and next tests

This is a single-probe mechanism-discovery sweep. It is not yet sufficient for a stable model-level claim.

Next steps:

- complete the missing cost `0.125` Bellman observation with a larger generation allowance;
- run the instruction-matched Exp 04b control so state-information effects are separated from reminder-language effects;
- repeat wedge states multiple times per condition;
- add at least one second model family;
- report the distribution of option-value error, not only its mean;
- separately estimate probability of value-sign error and value-to-action coupling error.

## Working empirical claim after Exp 05

> **Termination competence is not a single scalar capability. Even when an agent is given an exact Bayesian state and an explicit Bellman decomposition, it may misestimate the value of preserving future actions, perform the arithmetic coherently on those wrong estimates, and then map an otherwise correctly signed value to the wrong STOP/QUERY action.**
