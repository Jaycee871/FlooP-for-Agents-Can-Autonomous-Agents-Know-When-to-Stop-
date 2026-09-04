from __future__ import annotations

from dataclasses import dataclass

from .policies import StoppingPolicy
from .tasks import Task


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    policy: str
    stop_step: int
    endogenous_halt: bool
    external_cap_hit: bool
    stop_reason: str
    history: tuple[float, ...]


def run_episode(
    task: Task,
    policy: StoppingPolicy,
    *,
    external_cap: int = 30,
    step_cost: float = 0.01,
) -> EpisodeResult:
    """Run one synthetic reasoning episode.

    The external cap is a benchmark safety mechanism, not an endogenous
    termination rule. This distinction lets us measure failure-to-halt safely.
    """
    if external_cap <= 0:
        raise ValueError("external_cap must be positive")

    cap = min(external_cap, task.horizon)
    history = [task.utility(0)]

    for step in range(1, cap + 1):
        history.append(task.utility(step))
        decision = policy.decide(
            task=task,
            step=step,
            history=history,
            step_cost=step_cost,
        )
        if decision.stop:
            return EpisodeResult(
                task_id=task.task_id,
                policy=policy.name,
                stop_step=step,
                endogenous_halt=True,
                external_cap_hit=False,
                stop_reason=decision.reason,
                history=tuple(history),
            )

    return EpisodeResult(
        task_id=task.task_id,
        policy=policy.name,
        stop_step=cap,
        endogenous_halt=False,
        external_cap_hit=True,
        stop_reason=f"external_cap={cap}",
        history=tuple(history),
    )
