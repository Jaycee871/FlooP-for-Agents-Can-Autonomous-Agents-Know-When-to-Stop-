# FlooP for Agents: Can Autonomous Agents Know When to Stop?

> **Research question:** When continuation is always a legal action, can an autonomous agent recognize when further reasoning, searching, or acting no longer has positive value?

This repository studies **meta-termination**: the ability to decide not merely *how* to continue a task, but *whether continuation remains justified at all*.

The project is inspired by the distinction between bounded and unbounded computation in Hofstadter's discussion of BlooP/FlooP and by the broader formal distinction between working **within** a system and reasoning **about** the system.

## Core experimental design

We compare four stopping regimes:

| Regime | Interpretation | Endogenous stopping? |
|---|---|---:|
| `bounded` | Fixed reasoning budget (BlooP-like) | Yes, by budget |
| `unbounded` | Continue indefinitely unless the harness intervenes (FlooP-like) | No |
| `meta_stop` | Stop when recent marginal reasoning value remains low | Yes |
| `verifier_stop` | Privileged verifier/oracle baseline | Yes |

The benchmark harness always applies an **external safety cap** so an intentionally non-halting policy can be evaluated safely. Reaching that cap is recorded as **failure to halt**, not successful termination.

## Task families

1. **Solvable** — a sufficient answer exists and further work can become redundant or harmful.
2. **Unsatisfiable** — search can continue forever because failure to find a witness does not establish non-existence.
3. **Open-ended** — quality can keep improving, but with diminishing returns.

## Metrics

For task utility `U(s)` after `s` reasoning steps and per-step compute cost `c`, define the oracle stop:

```text
s* = argmax_s [ U(s) - c s ]
```

We then measure:

- **Excess Continuation Ratio (ECR):** fraction of executed steps occurring after `s*`.
- **Premature Stop Rate:** whether an agent stops before `s*`.
- **Failure-to-Halt Rate:** whether the external safety cap, rather than the agent, ends the run.
- **Termination Regret:** net-value gap between the chosen stop and `s*`.
- **Utility at Stop:** task quality when execution ends.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q
python experiments/exp00_synthetic.py
```

The pilot writes:

```text
results/synthetic_pilot.csv
results/synthetic_pilot_summary.json
```

## Repository structure

```text
src/floop_agents/
  tasks.py       synthetic task families
  policies.py    bounded / unbounded / meta / verifier stopping
  simulate.py    safe episode runner
  metrics.py     termination metrics

experiments/
  exp00_synthetic.py

paper/
  outline.md

tests/
```

## Roadmap

**Exp 00 — metric validation.** Deterministic synthetic trajectories establish that the benchmark distinguishes stopping from externally forced termination.

**Exp 01 — model agents.** Replace synthetic trajectories with real reasoning/search traces using an API model backend.

**Exp 02 — open-weight agents.** Run comparable open-weight baselines.

**Exp 03 — skill treatment.** Test whether operational skills improve task performance while leaving termination competence unchanged—or even increasing excess continuation.

**Exp 04 — meta-reasoning ablations.** Compare self-critique, marginal-value estimation, and verifier-assisted stopping.

## Working thesis

> **Competence and termination competence are separable capabilities.**

A more capable agent may have *more* ways to continue, making principled stopping more—not less—important.
