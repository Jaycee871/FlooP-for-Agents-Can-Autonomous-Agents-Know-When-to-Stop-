# Exp 06 protocol — Repeated-state and cross-model mechanism replication

## Purpose

Exp 04b, Exp 05, and Exp 05b identify separable candidate failure mechanisms, but each critical state has so far been probed only once per treatment. Exp 06 estimates whether those mechanisms persist across repeated stochastic generations and a second model family.

The goal is not to add more prompt variants. It is to test whether the discovered decomposition is reproducible.

## Primary questions

1. Are successor-state predictions consistently better calibrated than successor-policy values?
2. Are future-value errors systematically biased or highly variable across repeated generations?
3. Does value-to-action coupling fail repeatedly, or were the observed contradictions isolated samples?
4. Do these patterns persist in a second model family under the same task and output schema?

## Critical states

Repeat the most diagnostic Exp 05b conditions:

```text
cost 0.100  wedge: large successor-value underestimation observed
cost 0.120  wedge: successor-value collapse observed
cost 0.125  near-boundary wedge: severe overestimation + positive-value STOP observed
cost 0.130  dynamic-stop control: future value exists but initiating current query is negative
```

The cost-`0.090` state is omitted from the first compact replication because it was locally coherent in Exp 05b and is less diagnostic than the four states above.

## Models

Primary cross-model pair:

```text
openai/gpt-oss-20b
deepseek-ai/deepseek-v4-flash-0731
```

Both are accessed through the same NVIDIA hosted inference interface. A live smoke probe confirmed `deepseek-ai/deepseek-v4-flash-0731` returned a valid chat completion before the replication was launched.

The originally configured `google/gemma-3-27b-it` endpoint returned HTTP 410 because that hosted model had reached end of life. Those provider failures are classified as infrastructure/model-availability events and are **not** counted as model decisions or empirical observations.

A newer `google/gemma-4-31b-it` endpoint was also explored, but a short smoke request timed out. Any separate longer-timeout Gemma 4 run is exploratory until it completes successfully and is frozen with provenance; it is not required for the primary two-model comparison.

The replication uses the same prompt, parsing, oracle, and metrics for both primary models.

To avoid model-specific reasoning-control support becoming a confound, the replication wrapper does not rely on provider-specific `reasoning_effort` behavior. The task instruction itself is held constant.

## Repetitions

The first compact primary replication uses:

```text
3 independent generations per condition per model
4 conditions
2 primary models
= 24 total probes
```

This is a variance-estimation pilot, not the final publication sample size. If the mechanism is stable, increase repetitions before final inference.

## Outcomes

For every response record:

- current hit-probability error;
- successor target-existence error;
- successor best-hit error;
- successor candidate correctness;
- successor/future-policy value error;
- full evaluated-query value error;
- decomposition closure error;
- oracle decision agreement;
- reported-value sign agreement;
- value-to-action self-consistency;
- local stopping regret.

## Primary mechanism comparison

The key cross-layer comparison is:

```text
successor-state calibration error
versus
successor-policy value error
```

If successor-state errors remain small while value errors remain large across repetitions, Exp 05b's mechanism localization is strengthened.

## Value-to-action coupling

For each condition and model estimate:

```text
P(action contradicts reported value sign)
```

and keep the direction of contradiction:

```text
positive reported continuation value -> STOP
negative/non-positive reported continuation value -> QUERY
```

The two directions must not be collapsed because Exp 05 and Exp 05b exhibited both.

## Statistical reporting

For continuous errors report at minimum:

```text
n
mean
median
standard deviation
mean absolute error
```

For binary outcomes report counts and rates. Because this first replication has only three generations per cell, avoid strong significance claims; use the results to select the final repeated-sampling budget.

## Interpretation guard

The experiment does not infer latent reasoning from natural-language rationales. Rationales may be retained for qualitative error inspection, but mechanism claims are based on explicit predicted quantities and intervention-defined comparisons.

Provider/model availability is handled separately from scientific outcomes. A request timeout, retired endpoint, malformed response, or exhausted retry is never silently recoded as STOP, QUERY, or an incorrect model judgment.

## Success criterion for mechanism replication

The strongest replication pattern would be:

```text
small current-state error
small successor-state error
large / unstable successor-value error
small arithmetic-closure error
non-zero value-to-action contradiction rate
```

across more than one model family.

That pattern would support the paper's layered claim that termination competence is not a scalar by-product of task competence.
