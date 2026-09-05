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
- **boundary incoherence** — returning to `QUERY` after a `STOP` under stronger evidence for stopping;
- **value miscalibration** — the agent's reported marginal value disagrees with the oracle;
- **forced termination** — the harness, not the agent, ends execution.

## Exact stopping rule

For candidate query `q` in state `s`:

```text
Q*(s,q) = -c + p_hit(s,q) R + p_miss(s,q) V*(s_q^-)
```

The best continuation value is:

```text
C*(s) = max_q Q*(s,q)
V*(s) = max(0, C*(s))
```

so the normative action is:

```text
QUERY / CONTINUE  iff C*(s) > 0
STOP              otherwise
```

The model separately reports its own estimate of the value of continuing, which permits direct calibration analysis rather than inferring rationality from episode length alone.

## Failure taxonomy

The current experiments distinguish three levels of stopping failure:

```text
Level 1 — Boundary bias
The agent stops too early or too late relative to the oracle boundary.

Level 2 — Boundary incoherence
The agent emits STOP and later returns to QUERY under strictly stronger evidence for stopping.

Level 3 — Value-mechanism failure
The error is localized to state/belief calibration, future-option-value estimation,
or the mapping from a value estimate to a discrete action.
```

This prevents all termination failures from being collapsed into a generic "overthinking" label.

## Metrics

Primary metrics include:

- **Premature Stop Rate**
- **Excess Continuation Ratio**
- **Failure-to-Halt Rate**
- **Local Termination Regret**
- **Signed Continuation-Calibration Error**: agent-reported continuation value minus oracle continuation advantage
- **Continuation-value sign accuracy**
- **Stopping-Boundary Displacement**
- **Decision / Value Self-Consistency**
- **Query-selection regret**
- **Boundary monotonicity / post-STOP reversal count**
- **Future-option-value omission error** in the Bellman diagnostic
- **Decomposition closure error** in the Bellman diagnostic
- **Utility at Stop**, tracked separately from normative regret

The separation between realized utility and normative regret matters because a bad stopping decision can look good ex post in a lucky hidden world.

## Current empirical anchor

**Exp 02 — Bayesian witness-search pilot** used `openai/gpt-oss-20b` through NVIDIA hosted inference.

Under the standard world (`P(target exists)=0.8`, query cost `0.05`, reward `1.0`):

- the Bayes oracle continues through five failed queries and stops after the sixth miss;
- the model correctly chose candidates `1`, `2`, and `3`;
- after three misses, it emitted `STOP` even though oracle continuation value was still about `+0.06895`;
- across the no-target and late-target scenarios this produced two premature-stop events and zero excess-continuation events;
- query-choice agreement conditional on continuing was `1.0`, while overall stopping-decision agreement was `0.8333`.

The pilot therefore exposes a useful separation: **the agent knew where to search, but misjudged whether further search was still worth doing.**

Exp 03 extends this from one trajectory to independently probed stopping-boundary states. Exp 04 and Exp 05 then isolate *why* the continuation judgment fails rather than merely measuring that it fails.

## Mechanism experiments

### Exp 04 — calibrated-state ablation

Compare the raw state prompt against a prompt that additionally provides exact posterior state statistics:

```text
posterior target-existence probability
highest one-query hit probability
highest-mass remaining candidate
```

No oracle continuation value, stopping boundary, STOP/QUERY decision, or sign-equivalent quantity is exposed. The highest-mass candidate is computed independently of the oracle stopping decision to prevent leakage at STOP states.

Purpose: test whether stopping errors are primarily **belief/state-calibration failures**.

### Exp 05 — Bellman decomposition diagnostic

If exact state statistics are insufficient, require the model to decompose the best continuation it can identify:

```text
full continuation value
= - query cost
  + immediate hit value
  + miss probability * estimated optimal value after a miss
```

The key diagnostic is:

```text
future-option-value omission error
= model estimated value after miss - exact V*(state after miss)
```

A systematic negative error would support the hypothesis that the model undervalues the option to keep acting after the next unsuccessful step.

Exp 05 is gated behind inspection of the corrected Exp 04 result rather than being launched automatically.

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
  tasks.py              synthetic task families
  policies.py           bounded / unbounded / meta / verifier stopping
  simulate.py           safe episode runner
  metrics.py            termination metrics
  stopping_metrics.py   exact local regret + calibration decomposition
  search_arena.py       Bayesian search world + exact stopping oracle
  providers/            real-model adapters with bounded retry handling

experiments/
  exp00_synthetic.py
  exp01_nvidia_smoke.py
  exp02_bayesian_search.py
  exp03_boundary_sweep.py
  exp04_calibrated_state_ablation.py
  exp05_bellman_decomposition.py

paper/
  outline.md
  formalization.md
  exp00_pilot.md
  exp01_nvidia_smoke.md
  exp02_bayesian_search.md
  exp03_boundary_sweep.md
  exp04_calibrated_state_ablation.md
  exp05_bellman_decomposition_protocol.md

results/
```

## Updated roadmap

**Exp 00 — semantic / metric validation.** Deterministic trajectories establish that endogenous stopping differs from external termination.

**Exp 01 — real-model smoke test.** Verify that an API-backed model can emit structured stop/continue actions and that the harness records them correctly.

**Exp 02 — Bayesian witness-search pilot.** Compare real-agent decisions with an exact continuation-value oracle. Initial result: premature stopping despite correct query selection.

**Exp 03 — stopping-boundary calibration sweep.** Vary query cost and probe counterfactual post-miss states to estimate the model's stopping boundary, signed continuation-value calibration, and boundary coherence.

**Exp 04 — calibrated-state ablation.** Supply exact posterior state statistics without revealing the oracle decision to isolate belief/state calibration from continuation-value reasoning.

**Exp 05 — Bellman decomposition diagnostic.** If calibrated state information is insufficient, isolate immediate-value, miss-probability, and future-option-value errors.

**Exp 06 — repeated-state and cross-model replication.** Repeat selected boundary states and mechanism treatments across model families.

**Exp 07 — richer agent environments.** Extend from Bayesian search to retrieval, coding, tool acquisition, and externally costly actions.

## Mechanism interpretation

```text
Exp 04 strongly improves stopping
    -> state/belief calibration is the main bottleneck.

Exp 04 weak, Exp 05 strongly improves stopping
    -> dynamic decomposition / future option value is the main bottleneck.

Neither improves
    -> investigate deeper policy-level meta-reasoning and action/value coupling.

Value estimates improve but actions do not
    -> value representation and stopping policy are separable failure modes.
```

## Novelty guard

Recent work already studies agentic abstention, cost-aware exploration, early exit, and cost-optimal tool planning. The contribution here is therefore **not** the broad claim that no one has studied stopping.

The narrower target is:

> **Evaluate an agent-emitted STOP as a calibrated metareasoning decision, separately from forced harness termination, against an exact normative continuation-value oracle, while measuring both premature stopping and excessive continuation and decomposing why the agent's own marginal-value judgment fails.**

## Working thesis

> **Competence and termination competence are separable capabilities.**

A capable agent can know *what action to take next* while still being wrong about *whether taking another action is worth it at all*.
