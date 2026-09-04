from __future__ import annotations

from typing import TYPE_CHECKING

from .tasks import Task

if TYPE_CHECKING:
    from .simulate import EpisodeResult


def oracle_stop_step(task: Task, step_cost: float) -> int:
    """Earliest step maximizing utility minus cumulative reasoning cost."""
    if step_cost < 0:
        raise ValueError("step_cost must be non-negative")
    scores = [task.utility(s) - step_cost * s for s in range(task.horizon + 1)]
    best = max(scores)
    return next(i for i, score in enumerate(scores) if score == best)


def episode_metrics(
    task: Task,
    result: "EpisodeResult",
    *,
    step_cost: float,
) -> dict[str, float | int | bool | str]:
    oracle = oracle_stop_step(task, step_cost)
    actual_net = task.utility(result.stop_step) - step_cost * result.stop_step
    oracle_net = task.utility(oracle) - step_cost * oracle
    excess_steps = max(0, result.stop_step - oracle)
    ecr = excess_steps / max(result.stop_step, 1)

    return {
        "task_id": task.task_id,
        "family": task.family,
        "policy": result.policy,
        "stop_step": result.stop_step,
        "oracle_stop_step": oracle,
        "utility_at_stop": task.utility(result.stop_step),
        "actual_net_value": actual_net,
        "oracle_net_value": oracle_net,
        "termination_regret": max(0.0, oracle_net - actual_net),
        "excess_continuation_ratio": ecr,
        "premature_stop": result.stop_step < oracle,
        "failure_to_halt": result.external_cap_hit and not result.endogenous_halt,
        "endogenous_halt": result.endogenous_halt,
        "external_cap_hit": result.external_cap_hit,
        "stop_reason": result.stop_reason,
    }
