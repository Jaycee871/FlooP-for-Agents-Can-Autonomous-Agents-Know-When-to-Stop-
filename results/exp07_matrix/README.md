# Exp 07 — Oracle substitution audit

This directory freezes the first scientifically complete causal localization experiment following Exp 06.

## Provenance

- GitHub Actions run: `34013784679`
- Source commit: `de40b21f4cd77b0d4804c3b51181bec384872d63`
- Validated summary artifact: `9983305119`
- Artifact digest: `sha256:ed8392d142979480b3519904ac52bc91fd869fe4d946bb2fc7ca35bf31aaccfa`
- Model: `openai/gpt-oss-20b`
- Conditions: 4
- Repeats: 3
- Arms: 2
- Expected probes: 24
- Valid probes: 24
- Error probes: 0
- Scientific validity gate: **PASS**

The complete machine-readable summary is stored in
`gpt_oss_20b_run_34013784679_summary.json`.

## Causal intervention

Exp 06 showed accurate current-state and successor-state estimates but poor stopping
decisions and a negative bias in estimated continuation value. Exp 07 intervenes on
the computation rather than merely observing it.

- `oracle_successor_state`: give the model the exact post-miss state and ask it to
  estimate the optimal future value itself.
- `oracle_successor_value`: give the exact post-miss state **and** the exact optimal
  continuation value, leaving only Bellman composition and STOP/QUERY selection.

## Main result

| Arm | Decision agreement | Value-sign accuracy | Decision/value consistency | Future-value MAE | Full-Q MAE | Mean decision regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle successor state | 0.6667 | 0.6667 | 1.0000 | 0.13709 | 0.12545 | 0.02164 |
| Oracle successor value | **1.0000** | **1.0000** | **1.0000** | **2.41e-10** | **2.92e-06** | **0.0000** |

Supplying the exact successor state alone improves decision accuracy relative to the
Exp 06 observational replication (0.25 -> 0.667), but substantial continuation-value
error remains. Supplying the exact successor value produces a complete rescue: all
12 decisions agree with the oracle, value-sign accuracy is 1.0, regret is zero, and
the reported future value is numerically identical to the supplied oracle value up
to rounding.

Across both arms, decision/value self-consistency remains 1.0. There are no cases
where the model reports a positive continuation value and chooses STOP, or a
non-positive value and chooses QUERY.

## Interpretation

The intervention substantially narrows the failure locus.

1. The model can use exact current and successor-state statistics.
2. The model can compose a supplied continuation value into the current Bellman
   equation with very small numerical error.
3. The final policy action follows the sign of the model's own value estimate.
4. The remaining failure is therefore concentrated in **constructing the future
   continuation value when that value is not supplied**.

This supports a stronger statement than Exp 06:

> The stopping failure is not primarily a failure to perceive the state or to obey
> its own value estimate. It is a failure to construct the option value of future
> search.

The state-only arm is also informative because its residual error is not simply a
stable downward bias. At `cost_0125_wedge`, continuation value is sometimes strongly
overestimated, while at `cost_0100_wedge` and `cost_0120_wedge` it is typically
underestimated. The weakness therefore looks more like unstable recursive valuation
than a single global calibration offset.

## Next experiment: Exp 08 Bellman-depth localization

Exp 08 moves the oracle boundary exactly one step deeper. The model receives the
exact first successor state and the exact value *after a second miss*, but it must
perform one Bellman backup to reconstruct the first successor value before composing
the current action value.

This distinguishes two explanations:

- **Local backup failure:** even one explicit Bellman backup remains unreliable.
- **Recursive-horizon failure:** a single backed-up step works, while unconstrained
  multi-step value construction fails.

Together, Exp 06 -> Exp 07 -> Exp 08 forms a causal localization ladder from state
estimation, to value construction, to recursive Bellman depth.
