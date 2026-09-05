# Claim–evidence ledger

This file separates what the current experiments establish from what is still hypothesis-generating. It is intended to keep the manuscript's wording proportional to the evidence as the benchmark expands.

## Claim tier A — benchmark semantics / exact by construction

### A1. Endogenous stopping is distinct from forced termination

Status: **established by benchmark definition and deterministic tests.**

An agent-emitted `STOP` is recorded separately from target completion, step caps, timeouts, verifier aborts, and other harness-imposed termination.

### A2. The Bayesian witness-search arena has an exact state-dependent stopping oracle

Status: **established by dynamic-programming implementation and deterministic tests.**

For every benchmark state, the oracle computes the value of continuing under optimal future control and compares it with the normalized value of stopping.

### A3. Myopic and dynamically optimal stopping can be forced to disagree

Status: **established by deterministic option-value wedge geometry.**

The Exp 05 wedge includes states satisfying:

```text
best one-step gain <= 0
full dynamic continuation value > 0
```

so a one-step policy must STOP while the dynamic oracle must QUERY.

## Claim tier B — replicated within one model / treatment contrast

### B1. Exact current-state information can strongly repair the stopping boundary

Current evidence: **strong within-model mechanism evidence; cross-model replication pending.**

Exp 04b holds the dynamic-policy reminder fixed across arms. With `openai/gpt-oss-20b`:

```text
instruction-only decision agreement = 3/12  = 25.0%
state-calibrated decision agreement = 11/12 = 91.7%
paired wrong -> correct             = 8
paired correct -> wrong             = 0
```

The treatment therefore cannot be reduced to an unmatched reminder sentence.

### B2. Better stopping decisions do not imply better scalar value calibration

Current evidence: **observed clearly in Exp 04b; replication pending.**

Despite the large action/sign improvement:

```text
instruction-only continuation-value MAE = 0.07307
state-calibrated continuation-value MAE = 0.08930
```

This motivates reporting value sign, value magnitude, and discrete action separately.

## Claim tier C — single-model mechanism discovery awaiting repeated-state replication

### C1. Future-policy value can remain badly calibrated after successor-state prediction is approximately correct

Exp 05b with `openai/gpt-oss-20b`:

```text
mean abs current-hit error          = 2.96e-10
mean abs successor-existence error  = 0.00407
mean abs successor-hit error        = 0.00657
successor candidate accuracy        = 100%
mean abs future-policy value error  = 0.13305
```

Status: **mechanism-discovery evidence.** Exp 06 is repeating the critical states and adding a second model family.

### C2. Bellman arithmetic can be internally coherent while its future-value input is wrong

Exp 05 primary Bellman arm:

```text
mean abs decomposition-closure error = 1.42e-6
mean abs after-miss value error       = 0.1844
mean abs full evaluated-Q error       = 0.1685
```

Status: **mechanism-discovery evidence.** Do not phrase as a universal property of LLMs before replication.

### C3. Explicit Bellman decomposition is not necessarily a monotone repair

Exp 05 primary wedge states:

```text
calibrated-state wedge agreement = 3/4 = 75%
Bellman valid wedge agreement     = 1/3 = 33.3%
```

One Bellman wedge response was truncated, so the denominators differ.

Status: **treatment-specific first-sweep result.** The correct claim is not that Bellman prompting is generally harmful, only that an explicit decomposition did not reliably rescue this model in this experiment.

### C4. Value-to-action coupling can fail in both directions

Observed examples:

```text
Exp 05 cost 0.130:
reported continuation value < 0
final action = QUERY

Exp 05b cost 0.125:
reported continuation value > 0
final action = STOP
```

Status: **qualitative mechanism evidence.** Exp 06 estimates whether these contradictions recur under repeated sampling.

### C5. Natural-language rationale is insufficient as a causal mechanism diagnosis

The supplemental cost-0.125 recovery rationale claimed a single miss would rule out target existence. A subsequent explicit transition audit showed that the same model could predict the successor state's target-existence probability, next-hit probability, and next candidate fairly accurately while still misvaluing the successor policy.

Status: **methodological lesson supported by paired interventions**, but keep the wording narrow: rationale text from an individual generation should not be treated as a stable causal explanation without decomposed tests.

## Claim tier D — not yet supported; do not write as a result

The manuscript must not yet claim that:

- all autonomous agents exhibit the same layered failure profile;
- successor-state prediction is generally easy for LLMs;
- future-policy valuation is universally the dominant bottleneck;
- Bellman prompting generally worsens agent performance;
- the observed mechanism transfers unchanged to coding, retrieval, browser, or embodied agents;
- the effect sizes are stable across model families.

These are targets for Exp 06 and Exp 07.

## Current strongest manuscript-safe formulation

> **In a controlled Bayesian search benchmark with an exact continuation-value oracle, stopping behavior can dissociate from next-action competence. For the initial model studied, interventions further separate current-state representation, successor-state prediction, successor-policy valuation, arithmetic composition, value sign, and final STOP/QUERY choice. Repeated-state and cross-model experiments test whether this layered profile generalizes.**

## Promotion rule

A tier-C mechanism may be promoted toward a broader empirical claim only after:

1. repeated observations at the critical states;
2. uncertainty / distribution reporting rather than a single generation;
3. replication in at least one additional model family;
4. preservation of the same exact oracle and treatment semantics;
5. explicit accounting for provider or output-format failures as infrastructure events rather than model decisions.
