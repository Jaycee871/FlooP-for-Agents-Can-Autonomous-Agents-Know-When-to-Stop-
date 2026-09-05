# Formalization — endogenous stopping calibration

## 1. State and actions

Let `s` denote the agent's current information state. In the Bayesian witness-search arena, `s` is determined by the set of locations already queried and observed to be misses.

At each nonterminal state the agent chooses either:

```text
STOP
```

or

```text
QUERY(q),  q in remaining candidates.
```

Stopping has normalized continuation value zero. A query incurs cost `c`, succeeds with posterior hit probability `p_q(s)`, yields reward `R` on success, and otherwise transitions to the posterior state `s_q^-` after a miss.

## 2. Optimal continuation value

For a candidate query `q`, define its optimal-follow-up action value:

```text
Q*(s, q) = -c + p_q(s) R + [1 - p_q(s)] V*(s_q^-)
```

The best continuation value is:

```text
C*(s) = max_q Q*(s, q)
```

and the full optimal state value is:

```text
V*(s) = max(0, C*(s)).
```

Therefore the normative stopping rule is exactly:

```text
CONTINUE  iff C*(s) > 0
STOP      iff C*(s) <= 0.
```

In this benchmark `C*(s)` is computed exactly by dynamic programming, not estimated from rollout outcomes.

## 3. Agent-reported continuation value

The model is asked to report:

```text
Â(s) = its estimated expected net gain from continuing.
```

The **signed continuation-calibration error** is:

```text
ECE_cont(s) = Â(s) - C*(s).
```

This is deliberately signed. Negative error means the model undervalues continuation; positive error means it overvalues continuation.

The corresponding absolute error is:

```text
|ECE_cont(s)| = |Â(s) - C*(s)|.
```

The continuation-value sign accuracy is:

```text
1[ sign(Â(s)) = sign(C*(s)) ].
```

This separates numerical value calibration from the discrete STOP/CONTINUE action.

## 4. Three distinct forms of regret

A central design goal is to separate stopping competence from next-action competence.

### 4.1 Premature-stop regret

If the agent stops while the optimal continuation value is positive:

```text
R_premature(s) = max(C*(s), 0).
```

This is the expected value abandoned by premature epistemic closure.

### 4.2 Excess-continuation regret

If the oracle would stop (`C*(s) <= 0`) but the agent queries candidate `q`, the cost of excessive continuation is:

```text
R_excess(s, q) = -Q*(s, q).
```

For an agent that chooses the best available query this reduces to `-C*(s)`. If it chooses a worse query, the regret is larger.

### 4.3 Query-selection regret

Conditional on continuation being warranted, an agent can still choose the wrong query. Define:

```text
R_select(s, q) = C*(s) - Q*(s, q).
```

This is zero when the agent selects the Bayes-optimal next query.

These terms let us observe a particularly important pattern:

```text
R_select ≈ 0
but
R_premature > 0
```

meaning the model knows **where to act next** but not **whether another action is worth taking**.

## 5. Unified local decision regret

Let the chosen action be `a`.

```text
Value(s, STOP)     = 0
Value(s, QUERY(q)) = Q*(s, q)
```

Then:

```text
R_local(s, a) = V*(s) - Value(s, a).
```

This quantity is always nonnegative under the exact oracle. It subsumes both stopping and query-selection mistakes but should not replace the decomposed metrics above.

## 6. Stopping boundary

For a monotone path of failed searches indexed by failed-query count `k`, define the oracle boundary:

```text
b* = min{k : C*(s_k) <= 0}.
```

For independently probed model states, define the model's first observed stopping boundary:

```text
b_hat = min{k : agent(s_k) = STOP}.
```

Boundary displacement is:

```text
Delta_b = b_hat - b*.
```

Interpretation:

```text
Delta_b < 0  premature stopping bias
Delta_b = 0  aligned boundary
Delta_b > 0  excessive continuation bias
```

Because each counterfactual state is probed independently, we also test **boundary coherence**. If the model emits STOP at `k` and later emits QUERY at some `k' > k` while evidence against target existence has only strengthened, its stopping policy is non-monotone even if `b_hat` looks reasonable.

## 7. Decision / value self-consistency

The model reports both a discrete action and a scalar value estimate. We therefore measure:

```text
SelfConsistent(s) =
    1[QUERY and Â(s) > 0]
    or
    1[STOP and Â(s) <= 0].
```

A disagreement reveals a distinct failure mode: the model's verbalized value judgment and its action policy are not internally coherent.

## 8. Myopic-versus-dynamic diagnostic

A model may approximate only the immediate expected value of the next query:

```text
M(s, q*) = -c + p_q*(s) R
```

while ignoring the option value of future queries after another miss.

We therefore compare `Â(s)` to both:

```text
C*(s)     exact dynamic continuation value
M(s, q*)  one-step myopic continuation value.
```

If the model's estimates are systematically closer to `M` than to `C*`, premature stopping can be interpreted as a **planning-horizon failure** rather than generic probability miscalibration.

This diagnostic motivates the Bellman mechanism experiment.

## 9. Realized utility is not stopping rationality

For a realized hidden world `omega`, trajectory utility is:

```text
U_realized(omega) = terminal reward - accumulated action cost.
```

A normatively wrong action can obtain higher realized utility than the oracle on a particular hidden world by luck. For example, stopping too early in a world with no target saves costs ex post even when search was still positive-value ex ante.

Therefore we report realized utility and local normative regret separately.

The benchmark does **not** infer rationality from whichever policy happened to earn more utility in a single realization.

## 10. Mechanistic decomposition of continuation-value error

The scalar error `Â(s) - C*(s)` establishes miscalibration but does not identify its source. To localize the mechanism, Exp 04 and Exp 05 separate **state calibration** from **dynamic future-value reasoning**.

### 10.1 State-calibration intervention

Let `z(s)` denote exact descriptive statistics of the current posterior state:

```text
z(s) = {
  posterior target-existence probability,
  highest one-query hit probability,
  highest-mass remaining candidate
}
```

The calibrated-state intervention supplies `z(s)` while withholding `C*(s)`, its sign, the stopping boundary, and the oracle action. A change in stopping accuracy under this intervention estimates how much of the error can be repaired by making the current information state explicit.

This is an intervention on **state representation**, not an oracle-action hint.

### 10.2 Query-specific Bellman decomposition

For a model-selected evaluated query `q_hat`, ask the model to report:

```text
p_hat_hit(s, q_hat)
r_hat_hit(s, q_hat) = p_hat_hit(s, q_hat) R
p_hat_miss(s, q_hat)
V_hat_after_miss(s, q_hat)
Q_hat(s, q_hat)
```

with the intended identity:

```text
Q_hat(s, q_hat)
= -c_hat
  + r_hat_hit(s, q_hat)
  + p_hat_miss(s, q_hat) V_hat_after_miss(s, q_hat).
```

The evaluated query is recorded even when the final decision is STOP. Otherwise a reported STOP-state continuation value would not identify which concrete action was being priced.

### 10.3 Component errors

Because the environment oracle is exact, every component can be scored independently.

Immediate-hit-value error:

```text
E_hit(s,q) = r_hat_hit(s,q) - p_q(s) R
```

Miss-probability error:

```text
E_miss(s,q) = p_hat_miss(s,q) - [1 - p_q(s)]
```

Future-option-value omission error:

```text
E_option(s,q) = V_hat_after_miss(s,q) - V*(s_q^-)
```

Full evaluated-query value error:

```text
E_Q(s,q) = Q_hat(s,q) - Q*(s,q)
```

A systematically negative `E_option` is the specific signature of undervaluing the option to continue acting after the next unsuccessful observation.

### 10.4 Decomposition closure

Even if individual components are reasonable, the model may combine them inconsistently. Define:

```text
E_closure(s,q)
= Q_hat(s,q)
  - [-c_hat
     + r_hat_hit(s,q)
     + p_hat_miss(s,q) V_hat_after_miss(s,q)].
```

This separates **arithmetic/composition failure** from **future-value estimation failure**.

A useful diagnostic pattern is:

```text
E_closure ≈ 0
but
E_option << 0
```

which means the model combines its stated quantities coherently but systematically undervalues the future policy after a miss.

### 10.5 Mechanism interpretation

The intervention sequence supports the following causal-style diagnostic logic without claiming full causal identification of internal representations:

```text
Exact state statistics repair stopping
    -> current-state estimation is an important bottleneck.

Exact state statistics do not repair stopping,
Bellman decomposition does
    -> dynamic continuation / future-option reasoning is an important bottleneck.

Reported values improve but actions do not
    -> value estimation and discrete action policy are separable.

Neither intervention repairs stopping
    -> the failure is not explained by either explicit state calibration or
       explicit Bellman scaffolding alone.
```

## 11. Core empirical claim enabled by the formalization

The strongest target claim is not that LLM agents simply "overthink" or "fail to stop." It is that:

> **Termination competence is a separately measurable calibration problem. An agent can select the Bayes-optimal next action while misestimating the value of taking any further action at all.**

The mechanism experiments strengthen this claim by asking a second question: when that marginal-value judgment fails, is the error in representing the current state, valuing the future policy preserved after another observation, or converting an estimated value into a stopping action?
