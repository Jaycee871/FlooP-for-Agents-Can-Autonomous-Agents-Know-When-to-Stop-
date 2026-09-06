# Exp 09 — Recursive Bellman depth ladder

This directory freezes the scientifically complete depth-2/depth-3 recursion audit on `openai/gpt-oss-20b`.

## Provenance

- GitHub Actions run: `34014497459`
- Source commit: `baceae3eefea271ed402148c988c109db80297f0`
- Validated summary artifact: `9983510868`
- Artifact digest: `sha256:41b8f54320457887b19ce25ac5ec072246a32e268dee6cc5c68430c700adbabe`
- Conditions: 4
- Repeats: 3
- Backup depths: 2 and 3
- Expected probes: 24
- Valid probes: 24
- Error probes: 0
- Scientific validity gate: **PASS**

A compact machine-readable freeze is stored in `summary_core.json`; the full aggregate remains traceable to the validated workflow artifact above.

## Main result

| Backup depth | Decision agreement | Value-sign accuracy | First-successor value MAE | Current-Q MAE | All-backup MAE | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | **1.0000** | **1.0000** | 0.000203 | 0.000187 | 0.000186 | **0.0000** |
| 3 | **1.0000** | **1.0000** | 0.010112 | 0.009187 | 0.011147 | **0.0000** |

The key pattern is not an action failure but a **value-quality degradation with recursion depth**. At depth 2, numerical errors remain tiny. At depth 3, value errors increase by roughly two orders of magnitude, yet all 12 decisions still retain the correct sign and incur zero local stopping regret.

The strongest instability appears in `cost_0100_wedge`: depth-3 first-successor value MAE is about 0.0399 and current-Q MAE about 0.0362, while the final action remains correct. In one representative probe the model reports the three backed-up values as 0.3014, 0.2703, and 0.2361 versus oracle values 0.2550, 0.2282, and 0.1986. The error enters at the deepest explicit backup and then propagates upward, but the continuation margin remains large enough that the QUERY decision survives.

## Interpretation

Exp 09 rejects a simple “recursive horizon = immediate decision collapse” story. A more precise account is:

> Explicit recursive structure preserves policy correctness longer than it preserves value calibration.

Together with Exp 06–08, the evidence now separates three things that were previously conflated:

1. **State representation** — the model can use exact supplied states.
2. **Local Bellman execution** — one and two explicit backups are highly accurate; three remain decision-correct but numerically less stable.
3. **Self-instantiation of the value structure** — free-form continuation valuation is still the weakest regime observed so far.

This motivates Exp 10: hold the numerical state trajectory and oracle boundary value fixed, but manipulate only the **representation of the recursion**. The explicit arm receives the Bellman recurrence. The implicit arm receives the same numbers and task semantics in narrative form but no recurrence. If performance separates, the missing capability is not arithmetic recursion itself but spontaneously constructing the correct value representation.
