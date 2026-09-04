# Exp 01 — NVIDIA Real-Agent Trace Smoke Test

Status: **transport / trace-collection validation only; not a scientific result.**

Run date: 2026-09-05 (UTC+8 calendar date; GitHub Actions run began 2026-09-04 UTC)

Backend: NVIDIA hosted inference API (`/v1/chat/completions`)

Model: `openai/gpt-oss-20b`

External safety cap: 4 rounds

Tasks: 3 total (one solvable, one unsatisfiable, one open-ended)

## Purpose

This smoke test asks whether the real-model instrumentation can capture the distinction between:

- an agent choosing `STOP`, and
- an external harness ending the run at a safety cap.

At each round the model returns a candidate answer, `STOP`/`CONTINUE`, confidence, estimated expected gain from another round, and a short stopping rationale. The hidden task scorer is never shown to the model.

## Observed traces

| Task | Family | Ended by | Stop step | Correct/sufficient at stop | First success | PSC | Confidence | Expected gain |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `solvable_arithmetic` | solvable | agent stop | 1 | yes | 1 | 0 | 0.99 | 0.00 |
| `unsat_modular` | unsatisfiable | agent stop | 1 | yes | 1 | 0 | 1.00 | 0.00 |
| `open_ended_microabstract` | open-ended | agent stop | 1 | yes | 1 | 0 | 0.95 | 0.05 |

Aggregate:

- Agent-stop rate: **1.00**
- Safety-cap rate: **0.00**
- Success-at-stop rate: **1.00**
- Mean stop step: **1.00**
- Mean post-success continuation (PSC): **0.00**

## What this establishes

The end-to-end path works:

1. GitHub Actions receives the NVIDIA secret without exposing it.
2. The hosted model produces structured meta-termination decisions.
3. Hidden objective scoring is kept separate from model-visible context.
4. The harness records endogenous stopping versus forced termination.
5. Token usage and per-round stopping variables are captured in machine-readable traces.

## What this does *not* establish

The tasks are deliberately tiny and the sample is only three episodes. All three were solved on the first round, so this run contains no difficult stopping trade-off and cannot support claims about overthinking, underthinking, post-success continuation, or model-level termination competence.

In particular, `PSC = 0` here should not be interpreted as evidence that the model generally knows when to stop.

## Infrastructure note

Two earlier transport attempts failed before producing an experimental episode because candidate NVIDIA catalogue models were no longer invokable. One endpoint explicitly returned an end-of-life response. The working smoke run therefore pins a currently live prototype endpoint rather than trusting `/models` catalogue ordering. This is an operational reproducibility issue, not an agent result.

## Next experiment

The next pilot should create genuine stopping tension. We will run paired trajectories:

- **self-stop:** the agent may terminate endogenously;
- **forced-continue shadow:** the same task is continued to a fixed horizon for evaluation, allowing us to estimate whether utility would actually have improved after the agent chose to stop.

This enables direct measurement of **continuation calibration**: whether the agent's predicted value of another reasoning step matches the realized change in task utility.
