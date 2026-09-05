# Exp 04b — Instruction-matched state ablation

## Question

Was the strong Exp 04 calibrated-state rescue caused by exact state information itself, or by the fact that the calibrated prompt also contained an explicit reminder to value the full continuation policy?

Exp 04b removes that prompt confound.

## Design

Both arms receive the same dynamic-policy reminder:

```text
Remember that after another miss you may still choose to query again or stop.
Decide the expected NET value of the full continuation policy, not only the
immediate next query.
```

The treatment difference is therefore:

```text
instruction-only
= search state + dynamic-policy reminder

state-calibrated
= same search state + same reminder + exact posterior state statistics
```

The exact state statistics include current posterior target-existence probability, highest one-query hit probability, and the candidate attaining that probability. They do not reveal the oracle continuation value, its sign, the stopping boundary, or the oracle STOP/QUERY action.

## Completed result

Model: `openai/gpt-oss-20b`

Workflow run: `33937428748`
Head SHA: `d4c41ade42155ab72f23c0c9030c50db2a05cfe8`
Artifact ID: `9960699878`
Artifact SHA-256: `07dc3122c4ba02586d91d79d640335717c4f62d8487771c46df8d42a8c9fe6d8`

All 12 paired states completed without provider errors.

| Metric | Instruction-only | State-calibrated |
|---|---:|---:|
| Decision agreement | 3/12 = 25.0% | 11/12 = 91.7% |
| Reported-value sign accuracy | 25.0% | 91.7% |
| Decision/value self-consistency | 100% | 100% |
| Mean absolute continuation-value error | 0.07307 | 0.08930 |
| Premature-stop rate before boundary | 100% | 16.7% |
| Excess-query rate at/after boundary | 50.0% | 0% |
| Invalid-action rate | 8.3% | 0% |

Paired transitions:

```text
instruction wrong -> state correct: 8
instruction correct -> state wrong: 0
net state-corrected pairs: +8
```

## Main interpretation

The strong decision rescue survives after reminder language is held constant.

This substantially weakens the alternative explanation that Exp 04 merely worked because one arm told the model to reason dynamically. In this matched comparison, the dynamic-policy reminder by itself performed poorly: the instruction-only arm stopped prematurely at every pre-boundary probe and over-continued in half of the post-boundary probes.

Adding exact current-state statistics moved decision agreement from `25.0%` to `91.7%`, with eight paired errors corrected and none harmed.

The current evidence therefore supports a narrower mechanism claim:

> **Accurate representation of the current epistemic state is an important causal input to calibrated stopping in the standard boundary regime; a generic instruction to reason about the full continuation policy is not sufficient.**

## Important dissociation: better decisions without better numeric value calibration

The state-calibrated arm made far better STOP/QUERY decisions, yet its mean absolute reported continuation-value error was slightly larger:

```text
instruction-only MAE = 0.07307
state-calibrated MAE  = 0.08930
```

This means the intervention strongly improved the **sign / action boundary** without improving the average numerical magnitude of the model's claimed continuation value.

That dissociation matters because it shows that:

```text
better state information
    -> much better discrete stopping decisions

but not necessarily

better state information
    -> globally better scalar value estimates
```

A stopping benchmark should therefore report value-sign accuracy, magnitude calibration, and final action separately rather than collapsing them into one score.

## The single residual calibrated-state error

At `cost = 0.03`, one state immediately before the oracle boundary (`k = 9`) remained a premature STOP even with exact state statistics. The neighboring calibrated states were otherwise correct.

This residual error is useful for follow-up repeated-state replication because it lies very close to the normative boundary, where the true continuation advantage is small and sign decisions should be especially sensitive to value-estimation noise.

## Relation to Exp 05

Exp 04b establishes that current-state calibration matters even after prompt instructions are matched. It does **not** establish correct dynamic option-value reasoning because the standard boundary regime still allows a myopic one-step rule to recover the same action sign.

Exp 05 removes that shortcut with option-value wedge states. Taken together:

```text
Exp 04b:
Current-state information strongly repairs action-sign calibration.

Exp 05:
When current-state information is already available but myopic and dynamic
signs disagree, errors can move downstream into transition semantics,
future-policy value estimation, or value-to-policy coupling.
```

This supports treating termination competence as a layered mechanism rather than a single threshold.

## Replication requirement

This remains a single-probe mechanism-discovery sweep. The `25% -> 91.7%` difference is large, but stable model-level effect sizes require repeated probes per state and at least one additional model family.
