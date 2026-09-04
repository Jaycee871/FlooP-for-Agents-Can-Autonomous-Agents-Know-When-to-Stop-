# Exp 00 — Synthetic Meta-Termination Pilot

Status: deterministic validation pilot, **not** a model-result claim.

Configuration:

- 10 tasks per family (30 total)
- Families: solvable, unsatisfiable, open-ended
- Horizon / external safety cap: 30 steps
- Per-step cost: 0.01
- Four policies: bounded, unbounded, meta-stop, verifier-stop

A local clean run of `python experiments/exp00_synthetic.py --n-per-family 10 --horizon 30` produced:

| Policy | Mean termination regret | Mean ECR | Premature stop rate | Failure-to-halt rate | Mean stop step |
|---|---:|---:|---:|---:|---:|
| bounded | 0.0549 | 0.4111 | 0.000 | 0.000 | 12.0 |
| unbounded | 0.2437 | 0.7644 | 0.000 | 1.000 | 30.0 |
| meta-stop | 0.0134 | 0.2173 | 0.000 | 0.000 | 8.7 |
| verifier-stop | 0.0000 | 0.0000 | 0.000 | 0.000 | 7.07 |

## Interpretation

The synthetic benchmark behaves as intended:

1. The intentionally unbounded policy never halts endogenously and therefore reaches the external safety cap on every task.
2. Fixed-budget stopping avoids literal non-halting but continues substantially past the oracle stopping point.
3. The simple online meta-stop rule reduces excess continuation and termination regret without premature stopping in this deterministic pilot.
4. The privileged verifier recovers the oracle by construction and therefore serves only as a ceiling/baseline.

These values validate benchmark semantics and metric plumbing. They are **not evidence about real LLM agents**. The next phase replaces synthetic utility trajectories with actual model traces and objective/verifier scoring.
