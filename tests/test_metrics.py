from floop_agents.metrics import episode_metrics, oracle_stop_step
from floop_agents.policies import UnboundedPolicy, VerifierStopPolicy
from floop_agents.simulate import run_episode
from floop_agents.tasks import Task


def test_oracle_prefers_first_plateau_step():
    task = Task("t", "solvable", (0.0, 0.4, 0.8, 1.0, 1.0, 1.0), solved_at=3)
    assert oracle_stop_step(task, step_cost=0.01) == 3


def test_unbounded_hits_external_cap_and_counts_as_failure_to_halt():
    task = Task("t", "open_ended", (0.0, 0.2, 0.3, 0.31, 0.311))
    result = run_episode(task, UnboundedPolicy(), external_cap=4, step_cost=0.01)
    metrics = episode_metrics(task, result, step_cost=0.01)
    assert result.external_cap_hit is True
    assert metrics["failure_to_halt"] is True


def test_verifier_halts_at_oracle():
    task = Task("t", "solvable", (0.0, 0.4, 0.8, 1.0, 0.99), solved_at=3)
    result = run_episode(task, VerifierStopPolicy(), external_cap=4, step_cost=0.01)
    assert result.stop_step == 3
    assert result.endogenous_halt is True
