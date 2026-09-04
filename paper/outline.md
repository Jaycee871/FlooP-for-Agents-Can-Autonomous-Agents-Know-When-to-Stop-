# Paper skeleton — FlooP for Agents

## Working title
**When Success Fails to Halt: Meta-Termination in Autonomous AI Agents**

## Central question
When continuation is always a legal action, can an autonomous agent recognize when further reasoning, searching, or acting no longer has positive epistemic value?

## Hypotheses
- **H1 — Competence/termination separability:** task competence and termination competence are distinct capabilities.
- **H2 — Unbounded continuation cost:** agents without endogenous stopping will accumulate substantial excess continuation after the oracle stopping point.
- **H3 — Meta-stopping advantage:** online marginal-value stopping reduces excess continuation while preserving task utility better than fixed budgets.
- **H4 — Verifier ceiling:** a privileged verifier provides an empirical upper bound on achievable termination quality.
- **H5 — Skill paradox (later phase):** operational skills may improve task competence without proportionally improving termination competence.

## Experimental conditions
1. Bounded policy (BlooP-like fixed bound)
2. Unbounded policy (FlooP-like; only external safety cap)
3. Meta-stop policy (online marginal-value judgment)
4. Verifier-stop policy (privileged oracle baseline)

## Task families
1. Solvable tasks with a recognizable success point
2. Unsatisfiable search tasks where "not found" does not imply "does not exist"
3. Open-ended improvement tasks with diminishing returns

## Primary metrics
- Excess Continuation Ratio (ECR)
- Premature Stop Rate
- Failure-to-Halt Rate
- Termination Regret
- Utility at Stop
- Stop Step / compute budget

## Interpretation
The benchmark deliberately separates an endogenous stop from an external safety cap. Hitting the cap is therefore not counted as successful termination.

## Phases
- Exp 00: deterministic synthetic trajectories to validate semantics and metrics
- Exp 01: LLM reasoning/search tasks with NVIDIA API backend
- Exp 02: open-weight baseline via Hugging Face
- Exp 03: skill/no-skill treatment
- Exp 04: verifier and self-critique ablations
