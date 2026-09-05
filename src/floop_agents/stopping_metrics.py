from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LocalStoppingMetrics:
    """Normative metrics for one STOP/QUERY decision.

    ``oracle_continuation_value`` is C*(s), the value of the best continuation
    action relative to STOP=0. ``chosen_query_value`` is Q*(s,q) when the agent
    queries a concrete candidate and is None for STOP.
    """

    oracle_decision: str
    agent_decision: str
    oracle_continuation_value: float
    agent_reported_continuation_value: float
    chosen_query_value: float | None
    decision_agreement: bool
    signed_calibration_error: float
    absolute_calibration_error: float
    continuation_sign_accuracy: bool
    decision_value_self_consistency: bool
    premature_stop_regret: float
    excess_continuation_regret: float
    query_selection_regret: float
    local_decision_regret: float


@dataclass(frozen=True)
class StoppingCurveMetrics:
    """Diagnostics for an ordered counterfactual STOP/QUERY decision curve."""

    n_states: int
    first_stop_boundary: int | None
    boundary_displacement: int | None
    decision_agreement_rate: float
    stop_to_query_reversals: int
    post_stop_query_count: int
    nonmonotone: bool
    monotonicity_violation_rate: float


def _oracle_decision(continuation_value: float, tol: float) -> str:
    return "QUERY" if continuation_value > tol else "STOP"


def _reported_decision(reported_value: float, tol: float) -> str:
    return "QUERY" if reported_value > tol else "STOP"


def local_stopping_metrics(
    *,
    agent_decision: str,
    oracle_continuation_value: float,
    agent_reported_continuation_value: float,
    chosen_query_value: float | None = None,
    tol: float = 1e-12,
) -> LocalStoppingMetrics:
    """Compute decomposed stopping and action-selection regret.

    The benchmark normalizes STOP to value zero. The oracle state value is
    ``max(0, C*(s))``. If the agent chooses QUERY, ``chosen_query_value`` must be
    the exact Q*(s,q) of that concrete query under optimal future control.
    """

    agent = agent_decision.strip().upper()
    if agent not in {"STOP", "QUERY"}:
        raise ValueError("agent_decision must be STOP or QUERY")
    if agent == "QUERY" and chosen_query_value is None:
        raise ValueError("chosen_query_value is required for QUERY")
    if agent == "STOP" and chosen_query_value is not None:
        raise ValueError("chosen_query_value must be None for STOP")

    oracle = _oracle_decision(oracle_continuation_value, tol)
    reported = _reported_decision(agent_reported_continuation_value, tol)

    oracle_state_value = max(0.0, oracle_continuation_value)
    chosen_action_value = 0.0 if agent == "STOP" else float(chosen_query_value)
    local_regret = max(0.0, oracle_state_value - chosen_action_value)

    premature = 0.0
    excess = 0.0
    selection = 0.0

    if agent == "STOP" and oracle == "QUERY":
        premature = max(0.0, oracle_continuation_value)
    elif agent == "QUERY":
        q_value = float(chosen_query_value)
        if oracle == "STOP":
            excess = max(0.0, -q_value)
        else:
            selection = max(0.0, oracle_continuation_value - q_value)

    signed_error = agent_reported_continuation_value - oracle_continuation_value

    return LocalStoppingMetrics(
        oracle_decision=oracle,
        agent_decision=agent,
        oracle_continuation_value=oracle_continuation_value,
        agent_reported_continuation_value=agent_reported_continuation_value,
        chosen_query_value=chosen_query_value,
        decision_agreement=agent == oracle,
        signed_calibration_error=signed_error,
        absolute_calibration_error=abs(signed_error),
        continuation_sign_accuracy=reported == oracle,
        decision_value_self_consistency=agent == reported,
        premature_stop_regret=premature,
        excess_continuation_regret=excess,
        query_selection_regret=selection,
        local_decision_regret=local_regret,
    )


def stopping_curve_metrics(
    decisions: Sequence[str],
    oracle_decisions: Sequence[str],
    *,
    oracle_boundary: int,
) -> StoppingCurveMetrics:
    """Summarize an ordered counterfactual stopping curve.

    On the benchmark's failed-search path, increasing the state index means that
    more high-prior candidates have already missed. A coherent threshold-like
    stopping policy therefore should not switch from STOP back to QUERY as the
    failed-query count increases.

    ``monotonicity_violation_rate`` is the number of adjacent STOP->QUERY
    reversals divided by the number of adjacent state pairs. In contrast,
    ``post_stop_query_count`` counts every QUERY occurring after the first STOP.
    """

    if len(decisions) != len(oracle_decisions):
        raise ValueError("decisions and oracle_decisions must have equal length")
    if not decisions:
        raise ValueError("at least one state is required")
    if oracle_boundary < 0:
        raise ValueError("oracle_boundary must be nonnegative")

    normalized = [str(d).strip().upper() for d in decisions]
    oracle = [str(d).strip().upper() for d in oracle_decisions]
    valid = {"STOP", "QUERY"}
    if any(d not in valid for d in normalized + oracle):
        raise ValueError("decisions must contain only STOP or QUERY")

    first_stop = next((i for i, d in enumerate(normalized) if d == "STOP"), None)
    displacement = first_stop - oracle_boundary if first_stop is not None else None
    agreement = sum(d == o for d, o in zip(normalized, oracle)) / len(normalized)

    reversals = sum(
        normalized[i] == "STOP" and normalized[i + 1] == "QUERY"
        for i in range(len(normalized) - 1)
    )
    post_stop_queries = 0
    if first_stop is not None:
        post_stop_queries = sum(d == "QUERY" for d in normalized[first_stop + 1 :])

    pair_count = max(1, len(normalized) - 1)
    return StoppingCurveMetrics(
        n_states=len(normalized),
        first_stop_boundary=first_stop,
        boundary_displacement=displacement,
        decision_agreement_rate=agreement,
        stop_to_query_reversals=reversals,
        post_stop_query_count=post_stop_queries,
        nonmonotone=reversals > 0,
        monotonicity_violation_rate=reversals / pair_count,
    )
