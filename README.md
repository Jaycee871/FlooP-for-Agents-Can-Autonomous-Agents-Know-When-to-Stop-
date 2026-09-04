# FlooP for Agents: Can Autonomous Agents Know When to Stop?

> **Research question:** When continuation is always a legal action, can an autonomous agent recognize when further reasoning, searching, or acting no longer has positive value?

This repository studies **meta-termination**: the ability to decide not merely *how* to continue a task, but *whether continuation remains justified at all*.

The project is inspired by the distinction between bounded and unbounded computation in Hofstadter's discussion of BlooP/FlooP and by the broader formal distinction between working **within** a system and reasoning **about** the system.

## Core distinction

We explicitly separate an agent's own `STOP` decision from termination imposed by the harness.

| Termination mode | Interpretation | Counts as endogenous stopping? |
|---|---|---:|
| Agent emits `STOP` | The agent judges continuation no longer worthwhile | Yes |
| Target / task completion | Environment ends because the task is complete | Not a stopping judgment |
| Fixed budget | Pre-set step/token bound | No |
| Timeout / safety cap | Harness prevents indefinite execution | No |
| External verifier abort | A separate controller halts the agent | No |

The benchmark always applies an **external safety cap** so intentionally non-halting policies can be evaluated safely. Reaching that cap is recorded as **failure to halt**, not successful termination.

## Bayesian witness-search arena

A central benchmark is a search world in which:

- a target may not exist at all;
- failures update the posterior belief that a target exists;
- each query has explicit utility cost;
- a target hit yields reward;
- continuing is always legal, but eventually becomes irrational.

For every information state, an exact dynamic-programming oracle computes the Bayes-optimal `QUERY`/`STOP` action and the **continuation advantage** of one more optimally controlled search step.

This lets us distinguish:

- **premature epistemic closure** — stopping while continuation still has positive value;
- **excess continuation** — continuing after continuation value becomes negative;
- **value miscalibration** — the agent's reported marginal value disagrees with the oracle;
- **forced termination** — the harness, not the agent, ends execution.

## Metrics

For generic task utility `U(s)` after `s` steps and per-step cost `c`, a simple oracle stop is:

```text
s* = argmax_s [ U(s) - c s ]
```

For the Bayesian search arena we use the exact state-dependent continuation value instead of a fixed trajectory optimum.

Primary metrics include:

- **Excess Continuation Ratio (ECR)**
- **Premature Stop Rate**
- **Failure-to-Halt Rate**
- **Termination Regret**
- **Signed Continuation-Calibration Error**: agent-reported continuation value minus oracle continuation advantage
- **Stopping-Boundary Displacement**: agent boundary minus oracle boundary
- **Decision / Value Self-Consistency**
- **Utility at Stop**, tracked separately from normative regret

The separation between realized utility and normative regret matters because a bad stopping decision can look good ex post in a lucky hidden world.

## Current empirical result

**Exp 02 — Bayesian witness-search pilot** used `openai/gpt-oss-20b` through NVIDIA hosted inference.

Under the standard world (`P(target exists)=0.8`, query cost `0.05`, reward `1.0`):

- the Bayes oracle continues through five failed queries and stops after the sixth miss;
- the model correctly chose candidates `1`, `2`, and `3`;
- after three misses, it emitted `STOP` even though oracle continuation value was still about `+0.06895`;
- across the no-target and late-target scenarios this produced two premature-stop events and zero excess-continuation events;
- query-choice agreement conditional on continuing was `1.0`, while overall stopping-decision agreement was `0.8333`.

The pilot therefore exposes a useful separation: **the agent knew where to search, but misjudged whether further search was still worth doing.**

See `paper/exp02_bayesian_search.md` and `results/exp02_bayesian_search/`.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest -q
python experiments/exp00_synthetic.py
```

Real-agent experiments require an NVIDIA API key in the environment.

## Repository structure

```text
src/floop_agents/
  tasks.py           synthetic task families
  policies.py        bounded / unbounded / meta / verifier stopping
  simulate.py        safe episode runner
  metrics.py         termination metrics
  search_arena.py    Bayesian search world + exact stopping oracle
  providers/         real-model adapters

experiments/
  exp00_synthetic.py
  exp01_nvidia_smoke.py
  exp02_bayesian_search.py
  exp03_boundary_sweep.py

paper/
  outline.md
  exp00_pilot.md
  exp01_nvidia_smoke.md
  exp02_bayesian_search.md

results/
```

## Updated roadmap

**Exp 00 — semantic / metric validation.** Deterministic trajectories establish that endogenous stopping differs from external termination.

**Exp 01 — real-model smoke test.** Verify that an API-backed model can emit structured stop/continue actions and that the harness records them correctly.

**Exp 02 — Bayesian witness-search pilot.** Compare real-agent decisions with an exact continuation-value oracle. Initial result: premature stopping despite correct query selection.

**Exp 03 — stopping-boundary calibration sweep.** Vary query cost and probe counterfactual post-miss states to estimate the model's stopping boundary, signed continuation-value calibration, and boundary coherence.

**Exp 04 — metareasoning treatments.** Test whether explicit lookahead, self-critique, verifier assistance, or tool-assisted value calculation repairs stopping without simply imposing another external bound.

**Exp 05 — cross-model / open-weight replication.** Repeat the boundary experiment across model families and inference stacks.

**Exp 06 — richer agent environments.** Extend from synthetic Bayesian search to tool-using and open-ended tasks with irreversible or externally costly actions.

## Novelty guard

Recent work already studies agentic abstention, cost-aware exploration, early exit, and cost-optimal tool planning. The contribution here is therefore **not** the broad claim that no one has studied stopping.

The narrower target is:

> **Evaluate an agent-emitted STOP as a calibrated metareasoning decision, separately from forced harness termination, against an exact normative continuation-value oracle, while measuring both premature stopping and excessive continuation and eliciting the agent's own marginal-value estimate.**

## Working thesis

> **Competence and termination competence are separable capabilities.**

A capable agent can know *what action to take next* while still being wrong about *whether taking another action is worth it at all*.
