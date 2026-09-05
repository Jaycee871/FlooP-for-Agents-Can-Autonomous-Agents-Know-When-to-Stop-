# Exp 04 — Calibrated-state stopping ablation

## Question

Does stopping improve when the agent is given an exact description of its current Bayesian information state, while the actual STOP/CONTINUE decision remains hidden?

Exp 03 identifies stopping-boundary errors but does not by itself tell us whether those errors come from estimating the posterior state incorrectly or from valuing future continuation incorrectly. Exp 04 isolates the first possibility.

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

The statistics are descriptive only. They do not include the optimal continuation value or its sign.

## Anti-leakage rule

A subtle leakage risk occurs if the prompt uses the oracle's `best_query` field directly. The oracle implementation intentionally returns `best_query = None` whenever its optimal action is STOP, so exposing that field would reveal the stopping decision.

The calibrated prompt therefore computes the highest-mass unqueried candidate directly from the remaining target masses, independently of the oracle STOP/CONTINUE decision.

The prompt never exposes:

- oracle STOP/CONTINUE decision;
- oracle continuation advantage;
- oracle stopping boundary;
- any field whose missing/present status depends on the oracle decision;
- any sign-equivalent transformation of the oracle continuation value.

## Outcomes

For each arm record:

- STOP/QUERY agreement with the exact oracle;
- sign accuracy of the model-reported continuation value;
- decision/value self-consistency;
- mean absolute continuation-value error;
- premature-stop rate before the oracle boundary;
- excess-query rate at or after the oracle boundary;
- invalid-query rate.

Paired state-level outcomes include:

```text
raw wrong -> calibrated correct
raw correct -> calibrated wrong
net corrected pairs
```

## Interpretation

```text
Strong calibrated-state rescue
    -> belief/state calibration is an important stopping bottleneck.

Little or no rescue
    -> the agent can be told where it is epistemically and still misvalue whether
       preserving another action is worthwhile; test dynamic option-value reasoning.

Value estimates improve but actions do not
    -> value representation and discrete stopping policy are partly separable.
```

The next gated experiment, Exp 05, performs an explicit Bellman decomposition to test the future-option-value hypothesis.

## Run integrity note

The first implementation of this ablation was discarded for inference because its prompt used an oracle-dependent query field that could leak the STOP state.

The first corrected run was interrupted by a transient hosted-inference read timeout after two completed probes. Because that version only wrote results at the end of the sweep, those partial observations were not promoted to evidence.

The provider adapter was subsequently hardened with bounded retries and a longer timeout before launching the corrected ablation again. These infrastructure events are tracked separately from empirical outcomes and must not be interpreted as model behavior.

## Replication requirement

The boundary-neighborhood sweep is a mechanism-discovery experiment. Any model-level claim selected from Exp 04 must subsequently be repeated across multiple probes per state and at least one additional model family.
