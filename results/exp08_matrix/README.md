# Exp 08 — One explicit Bellman backup

This directory freezes the scientifically complete Exp 08 depth-2 localization run.

## Provenance

- GitHub Actions run: `34014215051`
- Source commit: `962caf52ec92c41e945bb97068692f5d7a84afd5`
- Model: `openai/gpt-oss-20b`
- Conditions: 4
- Repeats: 3
- Expected probes: 12
- Valid probes: 12
- Error probes: 0
- Summary artifact: `9983413669`
- Artifact digest: `sha256:a83008aa924ae765cacdd6eb77f7912941508bdcffc99aafe33ff107e3de7f71`
- Scientific validity gate: **PASS**

The complete machine-readable summary is stored in
`gpt_oss_20b_run_34014215051_summary.json`.

## Intervention

Exp 07 showed that supplying the exact successor value completely rescued the
stopping decision. Exp 08 moves the oracle boundary one step deeper. The model gets:

1. the exact current state,
2. the exact first post-miss successor state,
3. the exact optimal value only after a *second* miss.

It must reconstruct the first-successor value with one Bellman backup and then
compose the current query value.

## Result

| Metric | Result |
| --- | ---: |
| Decision agreement | **1.0000** |
| Reported-value sign accuracy | **1.0000** |
| Decision/value self-consistency | **1.0000** |
| First-successor candidate accuracy | **1.0000** |
| Mean absolute one-backup value error | **4.11e-06** |
| Mean absolute current-Q error | **1.78e-05** |
| Mean local decision regret | **0.0000** |
| Mean absolute oracle-identity error | **4.51e-17** |

Every one of the 12 probes chose the oracle action. The locally reconstructed
Bellman identity agrees with the exact oracle value to floating-point precision,
and the model's one-step backup is accurate to a few millionths.

## Interpretation

This rules out a simple explanation in which GPT-OSS cannot perform even one local
Bellman backup. It can. Combined with Exp 06 and Exp 07, the evidence now points to a
more specific failure mode:

> The stopping failure emerges when future option value must be constructed through
> an implicit or longer-horizon recursion, not when a single explicit backup is
> required.

The causal ladder is now:

- **Exp 06:** free-form future-value construction -> 25% decision agreement.
- **Exp 07 / exact successor state:** state is fixed, value still self-constructed -> 66.7%.
- **Exp 07 / exact successor value:** value supplied -> 100%.
- **Exp 08 / one explicit backup:** value not supplied, but one explicit recursive step -> 100%.

That pattern motivates Exp 09: increase only the number of explicit Bellman backups
while keeping all state quantities exact. This tests whether there is a measurable
**recursive stopping horizon** at which valuation reliability collapses.
