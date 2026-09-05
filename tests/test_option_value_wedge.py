import pytest

from floop_agents.search_arena import SearchSpec, oracle_state


def wedge_spec(cost: float) -> SearchSpec:
    return SearchSpec(
        n_candidates=10,
        existence_prob=0.8,
        alpha=0.05,
        query_cost=cost,
        reward=1.0,
    )


@pytest.mark.parametrize("cost", [0.090, 0.100, 0.120, 0.125])
def test_option_value_wedge_requires_dynamic_continuation(cost: float):
    spec = wedge_spec(cost)
    state = oracle_state(spec, [])
    myopic_gain = state.next_hit_probability * spec.reward - spec.query_cost

    assert myopic_gain < 0.0
    assert state.advantage > 0.0
    assert state.decision == "QUERY"


def test_myopic_positive_control():
    spec = wedge_spec(0.080)
    state = oracle_state(spec, [])
    myopic_gain = state.next_hit_probability * spec.reward - spec.query_cost

    assert myopic_gain > 0.0
    assert state.advantage > 0.0
    assert state.decision == "QUERY"


def test_dynamic_stop_control_can_have_future_value_after_miss():
    spec = wedge_spec(0.130)
    state = oracle_state(spec, [])
    myopic_gain = state.next_hit_probability * spec.reward - spec.query_cost
    after_miss = oracle_state(spec, [1])

    assert myopic_gain < 0.0
    assert state.advantage < 0.0
    assert state.decision == "STOP"
    assert after_miss.advantage > 0.0


def test_one_step_stop_control_has_no_future_option_value():
    spec = wedge_spec(0.140)
    state = oracle_state(spec, [])
    after_miss = oracle_state(spec, [1])

    assert state.decision == "STOP"
    assert state.advantage < 0.0
    assert after_miss.advantage <= 0.0
