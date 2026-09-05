# Exp 05b — Counterfactual transition audit results

## Question

Exp 05 and the supplemental cost-`0.125` recovery raised a sharper mechanistic possibility: perhaps the model does not merely misvalue future actions, but first predicts the wrong Bayesian state after a hypothetical miss.

Exp 05b tests that hypothesis directly by separating:

```text
transition prediction
from
successor-state valuation
```

## Run provenance

Model: `openai/gpt-oss-20b` via NVIDIA hosted inference.

Workflow run: `33937756661`
Head SHA: `fd99a91e40681228f38182f4f40f4bc064904133`
Artifact ID: `9960751319`
Artifact SHA-256: `84dd9d585857b39e4d28d1a14f930af71d62e231ee32efc3bcc6ee74b5681fed`

All five targeted conditions completed without provider errors.

## Aggregate result

```text
n                                      = 5
decision agreement                     = 40.0%
decision/value self-consistency         = 80.0%
mean abs current-hit error              = 2.96e-10
mean abs successor-existence error      = 0.00407
mean abs successor-hit error            = 0.00657
successor-best-candidate accuracy       = 100%
mean abs future-value error             = 0.13305
mean abs full-Q error                   = 0.12141
mean abs decomposition-closure error    = 0.000168
```

The result is unexpectedly clean: **the model predicts the counterfactual successor state fairly well, yet often assigns the wrong value to acting from that state.**

This argues against a simple universal transition-model-failure explanation.

## Condition-level result

| Cost | Oracle | Agent | Successor existence error | Successor hit error | Future-value error | Interpretation |
|---:|---|---|---:|---:|---:|---|
| 0.090 | QUERY | QUERY | +0.00387 | -0.00786 | -0.00683 | locally coherent dynamic continuation |
| 0.100 | QUERY | STOP | +0.00387 | -0.00786 | -0.19858 | successor represented, value collapsed to zero |
| 0.120 | QUERY | STOP | +0.00437 | -0.00924 | -0.08207 | successor represented, value collapsed to zero |
| 0.125 | QUERY | STOP | +0.00387 | -0.00004 | +0.35396 | successor represented almost exactly, value massively overestimated, then action contradicts positive value |
| 0.130 | STOP | STOP | +0.00437 | -0.00786 | -0.02381 | correct final STOP despite undervalued successor option |

In all five conditions the model correctly identified candidate `2` as the best remaining candidate after candidate `1` hypothetically misses.

## Mechanism result 1 — transition calibration can be good while successor valuation fails

The exact posterior target-existence probability after the hypothetical miss is approximately:

```text
0.781128
```

The model predicted approximately:

```text
0.7850 to 0.7855
```

across all audited costs.

The exact best remaining one-query hit probability is approximately:

```text
0.091145
```

The model predictions were close enough that successor-state identity was plainly preserved, and the best remaining candidate was always correct.

Yet at cost `0.100`, the exact optimal value after a miss is:

```text
+0.198576
```

while the model reports:

```text
0.0
```

At cost `0.120`, exact successor value is:

```text
+0.082065
```

while the model again reports:

```text
0.0
```

So the failure survives even when the model has already represented the next Bayesian state approximately correctly.

This supports a more precise mechanism claim:

> **Knowing the successor state is not the same capability as knowing what that successor state is worth.**

## Mechanism result 2 — the supplemental recovery rationale was not a stable description of the internal bottleneck

The earlier cost-`0.125` recovery said that after a miss "the target is ruled out" and assigned successor value zero. That response suggested an upstream transition-semantics failure.

Exp 05b forces the transition prediction to be explicit. Under this treatment, the same model correctly predicts that target existence remains near `0.785`, correctly identifies the next candidate, and predicts its hit probability almost exactly:

```text
predicted best hit after miss = 0.091100
exact best hit after miss     = 0.091145
```

Therefore the earlier rationale should not be treated as a stable explanation of the mechanism. The model is capable of representing the successor state when explicitly audited.

This is methodologically important:

> **A natural-language rationale can identify an error made in one response without reliably identifying the model's persistent causal bottleneck. Mechanism claims require interventions and decomposed measurements, not rationale inspection alone.**

## Mechanism result 3 — cost 0.125 exposes a second value-to-action decoupling

The cost-`0.125` transition audit is especially striking.

The model predicts the successor state accurately and reports:

```text
estimated value after miss = +0.4069
full expected net gain     = +0.3322
```

Yet it still emits:

```text
STOP
```

The reported continuation value is strongly positive, so the final action directly contradicts the sign of its own value estimate.

Its rationale says the positive continuation value is lower than a supposed "current optimal value" of about `0.368`, but STOP is normalized to zero in this decision problem. This introduces an unsupported comparison baseline and produces a policy error despite a positive continuation estimate.

Together with the Exp 05 cost-`0.130` case, this gives two qualitatively different examples of value-to-action decoupling:

```text
Exp 05 cost 0.130:
reported value < 0, but action = QUERY

Exp 05b cost 0.125:
reported value > 0, but action = STOP
```

Thus action/value coupling can fail in **both directions**.

## Mechanism result 4 — arithmetic closure remains strong

Mean absolute decomposition-closure error is only approximately:

```text
0.000168
```

Most individual states are effectively exact; the aggregate is dominated by the `0.125` response.

The overall pattern remains:

```text
current-state probability: very accurate
successor-state prediction: fairly accurate
arithmetic composition: highly coherent
future-policy value: badly unstable
final value-to-action mapping: sometimes incoherent
```

This substantially narrows the location of the observed failure.

## Revised layered model

The evidence now motivates this decomposition:

```text
1. Current-state representation
2. Counterfactual transition calibration
3. Successor / future-policy value estimation
4. Arithmetic composition
5. Value-sign judgment
6. Value-to-action coupling
```

The strongest current evidence for `openai/gpt-oss-20b` is concentrated at layers 3 and 6, while layers 1, 2, and 4 can be comparatively accurate under explicit elicitation.

That does not imply layers 1 or 2 never fail; Exp 04 showed strong current-state failures without calibrated statistics, and individual free-form rationales can contain transition errors. It means the failure mechanisms are **treatment-dependent and separable**, rather than one permanent defect.

## Refined working claim

> **An autonomous agent can represent its current state accurately, predict the next information state approximately correctly, and combine stated quantities coherently, yet still misvalue the policy available from that successor state or fail to map its own value judgment into the corresponding STOP/QUERY action.**

## Next replication priority

The next repeated-state experiment should sample the same critical conditions multiple times and estimate the joint distribution of:

```text
successor-state error
future-value error
value-sign error
value-to-action coupling error
```

Priority costs are `0.100`, `0.120`, `0.125`, and `0.130` because they distinguish under-valuation, over-valuation, and both directions of action/value inconsistency.

At least one additional model family is required before promoting these mechanisms from model-specific discovery evidence to a broader claim about autonomous agents.
