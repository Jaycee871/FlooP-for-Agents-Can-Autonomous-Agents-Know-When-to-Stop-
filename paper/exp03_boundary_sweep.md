# Exp 03 — Counterfactual stopping-boundary sweep

## Purpose

Exp 03 asks whether a model has a coherent endogenous stopping boundary when the normative continuation value is known exactly. Each state is probed independently, so the model is evaluated even at counterfactual states that it would never reach after an earlier STOP decision.

Model: `openai/gpt-oss-20b` through NVIDIA hosted inference.

Search world: 50 candidates; target existence prior 0.8; conditional target mass proportional to `1 / i^1.5`; reward 1.0; query costs swept over 0.03, 0.05, and 0.08.

## Primary result

The oracle and model first-stop boundaries diverged strongly:

| Query cost | Oracle boundary | Agent first STOP | Displacement |
|---:|---:|---:|---:|
| 0.03 | 10 | 4 | -6 |
| 0.05 | 6 | 3 | -3 |
| 0.08 | 4 | 1 | -3 |

Mean boundary displacement was **-4 failed queries**, indicating substantial premature stopping in this pilot.

Across all three conditions, mean STOP/QUERY agreement was 0.584 and mean sign accuracy of the model-reported continuation value was also 0.584.

## Stronger finding: the boundary is not coherent

All three cost conditions showed at least one later QUERY after an earlier STOP under a monotonically worsening failed-search history. Thus all three counterfactual stop curves were non-monotone.

The clearest example occurs at query cost 0.03:

```text
failed queries:  3      4      5      6      7      8      9      10
oracle:          QUERY  QUERY  QUERY  QUERY  QUERY  QUERY  QUERY  STOP
agent:           QUERY  STOP   QUERY  QUERY  STOP   QUERY  QUERY  QUERY
```

This pattern cannot be explained only by a single conservative threshold. The model's stopping judgment reverses after additional negative evidence.

## Value/action self-consistency

The model's discrete decision was perfectly consistent with the sign of its own reported continuation value in this pilot (`decision_value_self_consistency = 1.0`). This is useful diagnostically: many wrong STOP/QUERY decisions were not parser accidents or contradictions between a stated value and an action. The model acted consistently with a misestimated value.

## Planning-horizon diagnostic

For costs 0.03 and 0.05, the model's reported continuation value was more often numerically closer to the one-step myopic gain than to the exact dynamic continuation value:

- cost 0.03: 0.538 of probes closer to myopic value;
- cost 0.05: 0.556;
- cost 0.08: 0.286.

This is suggestive, not yet conclusive. It motivates a controlled intervention that separates belief calibration from dynamic option-value reasoning.

## Important secondary failure mode

At several cost-0.03 states the model selected candidates that were explicitly listed as already queried. This is a separate constraint/state-tracking error and should not be conflated with stopping competence. Future metrics should therefore keep:

1. STOP/CONTINUE regret;
2. query-selection regret;
3. invalid-action rate

as separate quantities.

## Interpretation

The strongest pilot claim is not merely that the model stops too early. It is:

> A model can produce internally self-consistent STOP/CONTINUE judgments while those judgments are both normatively miscalibrated and counterfactually non-monotone.

The next experiment should test whether supplying exact posterior state statistics rescues the boundary. If it does, the failure is primarily belief calibration. If it does not, the remaining error points toward dynamic continuation-value reasoning itself.

## Provenance

GitHub Actions run: `33931237475`

Artifact: `exp03-boundary-sweep` (`9958874563`)

Archived summary: `results/exp03_boundary_sweep/summary.json`
