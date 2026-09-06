# Exp 06 — Validated GPT-OSS transition replication

This directory freezes the first scientifically complete fault-isolated replication of the Exp 05b transition audit on `openai/gpt-oss-20b`.

## Provenance

- GitHub Actions run: `34004232255`
- Source commit: `79ae8d1b76ac1881f49f6689ee805cc84f87bbb0`
- Validated summary artifact: `9980459759`
- Artifact digest: `sha256:0f4578bfd7cbb3bc0f151eb0fa7da65a1e5fb5c42f8b5c1edb870cfb4078052a`
- Conditions: `cost_0100_wedge`, `cost_0120_wedge`, `cost_0125_wedge`, `cost_0130_dynamic_stop`
- Repeats: 3 per condition
- Expected probes: 12
- Valid probes: 12
- Error probes: 0
- Duplicate probe keys: 0
- Scientific validity gate: **PASS**

The complete machine-readable summary is stored in `gpt_oss_20b_run_34004232255_summary.json`.

## Main result

The replication strongly separates **state/transition understanding** from **continuation-value estimation**.

| Metric | Result |
| --- | ---: |
| Decision agreement with oracle | 0.2500 |
| Reported-value sign accuracy | 0.2500 |
| Decision/value self-consistency | 1.0000 |
| Successor best-candidate accuracy | 1.0000 |
| Mean absolute current-hit error | 2.96e-10 |
| Mean absolute successor-existence error | 0.00733 |
| Mean absolute successor-hit error | 0.01370 |
| Mean future-value error | -0.07191 |
| Mean absolute future-value error | 0.08913 |
| Mean full-Q error | -0.05702 |
| Mean absolute full-Q error | 0.09014 |

The model was essentially exact about the supplied current hit probability, identified the best successor candidate in every probe, and estimated successor-state probabilities with relatively small error. Yet it substantially underestimated the value of the future policy after a miss. Its final decisions were perfectly consistent with the sign of its own reported value estimate: all 12 probes were in the `consistent` coupling category, with no positive-value-to-STOP or nonpositive-value-to-QUERY reversals.

This makes a simple policy-coupling explanation unlikely. The dominant failure is upstream: **the model acts consistently on a mispriced continuation value**.

## Boundary pattern

| Condition | Oracle agreement | Mean future-value error | Mean local decision regret |
| --- | ---: | ---: | ---: |
| `cost_0100_wedge` | 0/3 | -0.19858 | 0.16768 |
| `cost_0120_wedge` | 1/3 | -0.02207 | 0.02747 |
| `cost_0125_wedge` | 0/3 | -0.05294 | 0.00960 |
| `cost_0130_dynamic_stop` | 2/3 | -0.01408 | 0.00734 |

The strongest failure occurs at `cost_0100_wedge`, where all three runs underestimate successor option value by about 0.199 and make the wrong stopping decision. Near the true stopping boundary, smaller valuation errors are sufficient to flip the action.

## Current interpretation

The evidence is most consistent with the following decomposition:

1. **Current-state blindness:** not supported by this replication.
2. **Successor-state identification failure:** weakly supported at most; errors are small and the best successor candidate is always correct.
3. **Dynamic continuation-value miscalibration:** strongly supported.
4. **Value-to-action decoupling:** not supported; decision/value self-consistency is 1.0.

In other words, the agent often knows where it is and what the next state looks like, but still fails to price the option value of continuing search.

## Next experiment

Exp 07 performs an **oracle-substitution audit** with two causal ablations:

- `oracle_successor_state`: provide the exact post-miss state but require the model to estimate the future policy value.
- `oracle_successor_value`: additionally provide the exact optimal value after a miss and require the model to compose the current query value and choose STOP/QUERY.

If state substitution does not rescue decisions but value substitution does, the failure can be localized to dynamic valuation rather than transition prediction or final action selection.
