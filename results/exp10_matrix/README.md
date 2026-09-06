# Exp 10 — Representation ablation

This directory freezes the scientifically complete Exp 10 representation ablation.

## Provenance

- GitHub Actions run: `34038250043`
- Source commit: `4af2261f46073bbff4b83c827bbd205220fee470`
- Summary artifact: `9990907597`
- Artifact digest: `sha256:2d3acce4cbc6be4112283d2090d207a165d657b9fc64c35720971a9a48bac404`
- Model: `openai/gpt-oss-20b`
- Expected probes: 24
- Valid probes: 24
- Errors: 0
- Scientific validity gate: **PASS**

## Main result

The explicit Bellman formula improved numerical calibration but did **not** improve final stopping accuracy.

| Arm | Decision agreement | V(s1) MAE | Current-Q MAE | Closure MAE | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Implicit narrative | 0.7500 | 0.11071 | 0.10694 | 0.02061 | 0.00447 |
| Explicit Bellman | 0.7500 | 0.07551 | 0.06903 | 0.000053 | 0.00447 |

Both arms have identical condition-level decision accuracy: 1.0 at `cost_0100_wedge`, 1.0 at `cost_0120_wedge`, 0.667 at `cost_0125_wedge`, and 0.333 at `cost_0130_dynamic_stop`.

The model remains perfectly action/value self-consistent in both arms: it follows the sign of its own reported value. The explicit recurrence therefore helps arithmetic closure but is not sufficient to recover the correct recursive option value near the stopping boundary.

## Why this changes the next experiment

Exp 09 achieved 100% stopping accuracy at depth 3 while requiring the model to report every intermediate Bellman backup. Exp 10 supplies the same type of exact trajectory information and even gives the Bellman recurrence, but asks only for the final successor value and current action value; accuracy falls to 75%.

That contrast suggests a new causal hypothesis:

> The rescue may come from **externalizing intermediate value states**, not merely from knowing the Bellman formula.

Exp 11 therefore holds the exact facts and Bellman recurrence fixed and changes only whether the model must explicitly output each intermediate backup.
