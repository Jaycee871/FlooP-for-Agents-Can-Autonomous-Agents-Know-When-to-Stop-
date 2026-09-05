# Exp 04 — Calibrated-state stopping ablation

## Question

Does stopping improve when the agent is given an exact description of its current Bayesian information state, while the actual STOP/CONTINUE decision remains hidden?

Exp 03 identifies stopping-boundary errors but does not by itself tell us whether those errors come from estimating the posterior state incorrectly or from valuing future continuation incorrectly. Exp 04 is a first mechanism-discovery intervention on this distinction.

## Design

For each query-cost condition, locate the exact Bayes-optimal stopping boundary and probe four counterfactual states around it:

```text
boundary - 2
boundary - 1
boundary
boundary + 1
```

Each state is probed independently under two arms.

### Raw arm

The model receives the search world, complete failed-query history, action cost, and reward, and must decide `STOP` or `QUERY` while reporting its estimated expected net gain from continuing.

### Calibrated-state arm

The model receives the same information plus these exact state statistics:

```text
posterior probability that a target still exists
highest one-query hit probability among remaining candidates
candidate attaining that highest hit probability
```

The calibrated prompt also contains an explicit reminder that a miss is not terminal and that the model should value the full continuation policy rather than only the immediate query. This wording difference is treated as a possible intervention confound below and is isolated in Exp 04b.

The statistics themselves are descriptive only. They do not include the optimal continuation value or its sign.

## Anti-leakage rule

A subtle leakage risk occurs if the prompt uses the oracle's `best_query` field directly. The oracle implementation intentionally returns `best_query = None` whenever its optimal action is STOP, so exposing that field would reveal the stopping decision.

The calibrated prompt therefore computes the highest-mass unqueried candidate directly from the remaining target masses, independently of the oracle STOP/CONTINUE decision.

The prompt never exposes:

- oracle STOP/CONTINUE decision;
- oracle continuation advantage;
- oracle stopping boundary;
- any field whose missing/present status depends on the oracle decision;
- any sign-equivalent transformation of the oracle continuation value.

## Corrected run result

Model: `openai/gpt-oss-20b`

The corrected, non-leaky run completed all 12 paired boundary-neighborhood states.

| Metric | Raw | Calibrated treatment |
|---|---:|---:|
| Decision agreement | 7/12 = 58.3% | 12/12 = 100% |
| Reported-value sign accuracy | 7/12 = 58.3% | 12/12 = 100% |
| Decision/value self-consistency | 100% | 100% |
| Mean absolute continuation-value error | 0.06947 | 0.03938 |
| Premature-stop rate before boundary | 66.7% | 0% |
| Excess-query rate at/after boundary | 16.7% | 0% |
| Invalid-query rate among QUERY actions | 33.3% | 0% |

At the paired-state level:

```text
raw wrong -> calibrated correct: 5
raw correct -> calibrated wrong: 0
net corrected pairs: +5
```

The mean absolute continuation-value error decreased by about 43.3%. The median absolute error decreased more sharply, from about `0.03555` to `0.00880`, although one calibrated state still showed a large magnitude error despite choosing the correct action.

## What changed in the failures?

The raw responses reveal concrete state-estimation failures rather than merely noisy thresholding. Examples include underestimating the posterior probability that a target still exists, reverting to the original prior after multiple misses, and in one post-boundary state attempting to query an already-failed candidate.

The calibrated treatment eliminated every observed discrete stopping error in this 12-state sweep. Together with the raw rationales, this strongly motivates the hypothesis that **state representation / posterior tracking is an important bottleneck for endogenous stopping** in this setting.

However, the result should be described as a treatment rescue rather than a pure state-information causal effect until the prompt-matching control is complete.

## Prompt-matching caveat and Exp 04b

The original raw and calibrated prompts are not perfectly instruction-matched. The calibrated arm includes an additional sentence reminding the model that, after another miss, it may still act again and should evaluate the full continuation policy.

Therefore the observed rescue bundles two changes:

```text
exact posterior state statistics
+
explicit dynamic-policy reminder
```

This does **not** create oracle leakage, but it does matter for mechanism attribution.

A dedicated Exp 04b therefore compares:

```text
instruction-only arm
    raw state + the same dynamic-policy reminder

state-calibrated arm
    raw state + exact state statistics + the identical reminder
```

Only the state statistics differ. Exp 04b is the clean test of whether exact state representation adds value beyond wording alone.

There is already circumstantial evidence favoring the state-tracking account: the raw responses contain large posterior errors and even a repeated-query state error. Still, the matched control is required before the paper claims a state-information effect.

## Important limitation: this sweep does not require non-myopic option-value reasoning

The 100% calibrated decision agreement must not be over-interpreted.

In the standard Exp 03 conditions, every probed pre-boundary state also has a positive immediate one-step gain for the highest-probability query:

```text
p_hit * reward - query_cost > 0
```

and every probed stopping state has non-positive one-step gain. Therefore the sign of the myopic one-step rule agrees with the sign of the full dynamic oracle throughout this particular boundary-neighborhood sweep.

Consequently, Exp 04 shows that the calibrated treatment is sufficient to repair the observed decisions **in this regime**, but it does not establish that the model can correctly value the future option to act again after a miss. A model could achieve the observed 12/12 decisions by applying a one-step threshold once the probabilities are supplied.

This limitation directly motivates Exp 05.

## Exp 05 design implication — create an option-value wedge

Exp 05 should not simply repeat the same boundary states with a longer calculation prompt. Instead it constructs states satisfying:

```text
best immediate one-step gain < 0
but
full Bayes-optimal continuation value > 0
```

These **option-value wedge states** make future continuation decision-relevant to the sign of the correct action. A myopic agent must STOP; a dynamically correct agent must QUERY.

The Bellman-decomposition treatment then tests whether explicitly representing the value preserved after a miss repairs this deliberately non-myopic decision.

## Run integrity note

The first implementation of this ablation was discarded for inference because its prompt used an oracle-dependent query field that could leak the STOP state.

The first corrected run was interrupted by a transient hosted-inference read timeout after two completed probes. Because that version only wrote results at the end of the sweep, those partial observations were not promoted to evidence.

The provider adapter was subsequently hardened with bounded retries and a longer timeout before launching the corrected ablation again. The completed corrected run is the result reported above. Infrastructure events are tracked separately from empirical outcomes and are not interpreted as model behavior.

## Replication requirement

This 12-pair sweep is mechanism-discovery evidence, not yet a stable model-level estimate. After the instruction-matched control, the selected treatment effect should be repeated across multiple probes per state and at least one additional model family.
