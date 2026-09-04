from floop_agents.policies import BoundedPolicy, MetaStopPolicy
from floop_agents.simulate import run_episode
from floop_agents.tasks import Task


def test_bounded_policy_halts_at_bound():
    task = Task("t", "open_ended", tuple(i / 20 for i in range(21)))
    result = run_episode(task, BoundedPolicy(bound=5), external_cap=20)
    assert result.stop_step == 5
    assert result.endogenous_halt


def test_meta_stop_detects_plateau():
    task = Task("t", "open_ended", (0.0, 0.4, 0.7, 0.705, 0.706, 0.707, 0.708))
    policy = MetaStopPolicy(threshold=0.01, patience=2, min_steps=3)
    result = run_episode(task, policy, external_cap=6, step_cost=0.0)
    assert result.stop_step == 4
    assert result.endogenous_halt
