from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class SearchSpec:
    """Bayesian witness-search environment with a no-witness possibility.

    Before any query:
      P(no target) = 1 - existence_prob
      P(target=i)  = existence_prob * w_i / sum_j w_j
      w_i          = 1 / i**alpha

    Querying an index costs ``query_cost``. Finding the target pays ``reward``.
    A rational agent may stop before exhausting the domain when the remaining
    expected value of search is lower than its cost.
    """

    n_candidates: int = 50
    existence_prob: float = 0.8
    alpha: float = 1.5
    query_cost: float = 0.05
    reward: float = 1.0

    def __post_init__(self) -> None:
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be positive")
        if not 0.0 <= self.existence_prob <= 1.0:
            raise ValueError("existence_prob must lie in [0, 1]")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.query_cost < 0 or self.reward <= 0:
            raise ValueError("query_cost must be >= 0 and reward must be > 0")

    @property
    def target_masses(self) -> dict[int, float]:
        weights = {i: 1.0 / (i**self.alpha) for i in range(1, self.n_candidates + 1)}
        z = sum(weights.values())
        return {i: self.existence_prob * w / z for i, w in weights.items()}

    @property
    def no_target_mass(self) -> float:
        return 1.0 - self.existence_prob


@dataclass(frozen=True)
class OracleState:
    decision: str
    best_query: int | None
    continuation_value: float
    stop_value: float
    advantage: float
    posterior_target_exists: float
    next_hit_probability: float


def _remaining_masses(spec: SearchSpec, queried: Iterable[int]) -> tuple[float, dict[int, float]]:
    queried_set = set(queried)
    masses = {i: m for i, m in spec.target_masses.items() if i not in queried_set}
    denominator = spec.no_target_mass + sum(masses.values())
    if denominator <= 0:
        return 1.0, {}
    none_post = spec.no_target_mass / denominator
    target_post = {i: m / denominator for i, m in masses.items()}
    return none_post, target_post


def oracle_state(spec: SearchSpec, queried: Iterable[int]) -> OracleState:
    """Return the Bayes-optimal action after the listed failed queries.

    With independent query cost and a single possible target, the optimal ordering
    is descending posterior target mass. Dynamic programming over that ordered
    remainder gives the exact optimal stop/query decision for the current state.
    """

    none_post, target_post = _remaining_masses(spec, queried)
    if not target_post:
        return OracleState(
            decision="STOP",
            best_query=None,
            continuation_value=0.0,
            stop_value=0.0,
            advantage=0.0,
            posterior_target_exists=0.0,
            next_hit_probability=0.0,
        )

    ordered = sorted(target_post.items(), key=lambda kv: (-kv[1], kv[0]))
    # Work with probabilities conditional on the current state. After a miss,
    # probabilities renormalize, so recurse backwards using unnormalized current
    # masses plus the no-target mass.
    current_none_mass = none_post
    current_target_masses = {i: p for i, p in ordered}

    # For any suffix, use masses as currently normalized probabilities; their sum
    # with none mass is <=1 after dropping earlier queried options. Renormalize at
    # each suffix state.
    suffix_values: list[float] = [0.0] * (len(ordered) + 1)
    for pos in range(len(ordered) - 1, -1, -1):
        remaining = ordered[pos:]
        denom = current_none_mass + sum(p for _, p in remaining)
        if denom <= 0:
            suffix_values[pos] = 0.0
            continue
        hit_prob = ordered[pos][1] / denom
        miss_prob = 1.0 - hit_prob
        continuation = -spec.query_cost + hit_prob * spec.reward + miss_prob * suffix_values[pos + 1]
        suffix_values[pos] = max(0.0, continuation)

    denom0 = current_none_mass + sum(current_target_masses.values())
    best_query, best_mass = ordered[0]
    next_hit = best_mass / denom0 if denom0 > 0 else 0.0
    continuation_value = -spec.query_cost + next_hit * spec.reward + (1.0 - next_hit) * suffix_values[1]
    advantage = continuation_value  # stopping has value 0 from the current state
    decision = "QUERY" if advantage > 1e-12 else "STOP"
    posterior_exists = sum(current_target_masses.values()) / denom0 if denom0 > 0 else 0.0

    return OracleState(
        decision=decision,
        best_query=best_query if decision == "QUERY" else None,
        continuation_value=continuation_value,
        stop_value=0.0,
        advantage=advantage,
        posterior_target_exists=posterior_exists,
        next_hit_probability=next_hit,
    )


def oracle_policy_path(spec: SearchSpec, target: int | None) -> dict[str, object]:
    """Simulate the Bayes-optimal policy on one realized hidden world."""

    queried: list[int] = []
    found = False
    while True:
        state = oracle_state(spec, queried)
        if state.decision == "STOP" or state.best_query is None:
            ended_by = "oracle_stop"
            break
        q = state.best_query
        queried.append(q)
        if target == q:
            found = True
            ended_by = "target_found"
            break
        if len(queried) >= spec.n_candidates:
            ended_by = "domain_exhausted"
            break

    utility = (spec.reward if found else 0.0) - spec.query_cost * len(queried)
    if not isfinite(utility):
        raise RuntimeError("non-finite oracle utility")
    return {
        "queried": queried,
        "n_queries": len(queried),
        "found": found,
        "ended_by": ended_by,
        "realized_utility": utility,
    }
