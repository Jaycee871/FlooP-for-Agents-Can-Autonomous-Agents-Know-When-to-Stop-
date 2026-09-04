# Related-work positioning — endogenous stopping calibration

This note records the closest prior work found in the September 2026 novelty audit and the exact boundary of the claim we should defend.

## 1. Agentic Abstention: Do Agents Know When to Stop Instead of Act? (Luo, Wen, Wang, 2026)

### Strong overlap
- Native terminal action: the agent chooses among `ANSWER`, `ABSTAIN`, and `ACT`.
- Harness exhaustion is distinct from successful abstention.
- Premature abstention and delayed abstention are evaluated separately.
- A task-specific oracle defines the earliest warranted abstention step.

### Important difference
The oracle is an **information-sufficiency / warranted-step oracle**, not a state-dependent expected continuation-value oracle. The paper does not optimize reward in the POMDP formalization and does not elicit a scalar estimate from the agent of the marginal value of another action.

Our distinction therefore cannot be merely "the agent itself stops." Luo et al. already do that.

## 2. Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents (Ding, Tomlin, Durrett, 2026)

### Strongest novelty threat
This is substantially closer than a generic early-exit paper.

- Exploration vs commit is formulated as a sequential decision problem.
- In the Pandora's Box environment, an **exact Bayes-optimal oracle** is used.
- The oracle recursively compares the expected value of verification with the expected value of guessing/committing.
- Model behavior is evaluated against oracle strategy and across exploration-cost regimes.

### Important difference
"Calibration" in CTA means calibrating latent-state or success-probability estimates that are then **provided to the agent as decision context**. It is not primarily a measurement of the agent's spontaneously reported marginal continuation-value estimate.

Our defensible distinction is therefore the combination of:

1. native `STOP` / continuation judgments;
2. exact state-level continuation advantage;
3. **elicitation and calibration of the model's own claimed marginal value of continuing**;
4. explicit two-sided decomposition into premature stopping and excess continuation;
5. realized utility reported separately from ex-ante local decision regret;
6. counterfactual state probes that recover the model's stopping boundary even after the model would have stopped on-policy.

Any paper draft must cite CTA prominently rather than describing the decision-theoretic stopping space as empty.

## 3. Cognitive Friction: A Decision-Theoretic Framework for Bounded Deliberation in Tool-Using Agents (Di Gioia, 2026)

### Strong overlap
- Explicitly prices information acquisition and deliberation.
- Uses an HJB optimal-stopping formulation.
- Defines STOP when the maximum utility of continued information gathering is non-positive.

### Important difference
The implemented stopping boundary is applied by an **external inference-time controller**, not by a native LLM-emitted STOP action. Its practical controller uses an approximate/myopic value-of-information proxy, and the LLM does not report its own marginal continuation value.

This makes it an important conceptual foundation for pricing continuation, but a different empirical object from endogenous termination calibration.

## 4. Runaway is Ashamed, But Helpful (Lu et al., 2025)

### Strong overlap
- Explicitly distinguishes **intrinsic early exit** from **extrinsic early exit**.
- Intrinsic exit is induced by prompting the agent to emit `EXIT`.
- Measures both redundant post-progress steps and damage from premature exit.

### Important difference
There is no normative state-level expected-value stopping oracle and no calibration of the agent's own continuation-value estimate. Redundant Steps and Progress Degradation are trajectory/outcome metrics rather than exact local decision regret.

## Claim we should NOT make

Do not write any variant of:

> Prior work has not studied whether agents know when to stop.

That claim is false as of 2026.

## Narrow contribution we can defend

A stronger formulation is:

> We study **endogenous stopping calibration**: whether an agent's own STOP/CONTINUE action and its stated marginal value of another action agree with an exact state-dependent normative continuation value. Unlike harness-based early exit, task-specific warranted-step labels, or controllers supplied with calibrated priors, our benchmark evaluates the native stopping judgment itself and recovers its counterfactual stopping boundary across cost regimes.

## Empirical differentiators to preserve

The experiment suite should continue to measure all of the following together:

- agent-emitted `STOP` separately from forced termination;
- exact Bayes-optimal continuation advantage at every state;
- agent-reported expected net gain of continuing;
- signed continuation-value calibration error;
- premature-stop regret;
- excess-continuation regret;
- stopping-boundary displacement;
- non-monotone stop/continue reversals under counterfactual probes;
- query/action competence conditional on choosing to continue;
- realized utility separately from normative regret.

The most valuable result would show that action competence can remain high while stopping calibration fails, or that the same model exhibits both premature and excessive continuation under controlled changes in cost.
