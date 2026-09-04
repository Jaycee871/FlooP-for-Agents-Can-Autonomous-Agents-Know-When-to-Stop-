from floop_agents.search_arena import (
    SearchSpec,
    oracle_policy_path,
    oracle_state,
    query_action_value,
)


def test_oracle_has_interior_stopping_point():
    spec = SearchSpec(n_candidates=50, existence_prob=0.8, alpha=1.5, query_cost=0.05)
    assert oracle_state(spec, range(1, 6)).decision == "QUERY"
    assert oracle_state(spec, range(1, 7)).decision == "STOP"


def test_oracle_queries_highest_prior_first():
    spec = SearchSpec(n_candidates=20, existence_prob=0.8, alpha=1.5, query_cost=0.05)
    state = oracle_state(spec, [])
    assert state.decision == "QUERY"
    assert state.best_query == 1


def test_best_query_action_value_matches_oracle_continuation_value():
    spec = SearchSpec(n_candidates=50, existence_prob=0.8, alpha=1.5, query_cost=0.05)
    queried = [1, 2, 3]
    state = oracle_state(spec, queried)
    assert state.best_query == 4
    assert abs(query_action_value(spec, queried, 4) - state.continuation_value) < 1e-12
    assert query_action_value(spec, queried, 10) <= state.continuation_value


def test_all_queries_are_nonpositive_after_oracle_stop_boundary():
    spec = SearchSpec(n_candidates=50, existence_prob=0.8, alpha=1.5, query_cost=0.05)
    queried = [1, 2, 3, 4, 5, 6]
    state = oracle_state(spec, queried)
    assert state.decision == "STOP"
    remaining_values = [
        query_action_value(spec, queried, q)
        for q in range(1, spec.n_candidates + 1)
        if q not in queried
    ]
    assert max(remaining_values) <= 0.0


def test_oracle_path_finds_early_target():
    spec = SearchSpec(n_candidates=50, existence_prob=0.8, alpha=1.5, query_cost=0.05)
    result = oracle_policy_path(spec, target=2)
    assert result["found"] is True
    assert result["queried"] == [1, 2]
    assert result["realized_utility"] == 0.9


def test_oracle_stops_on_no_target_before_exhaustion():
    spec = SearchSpec(n_candidates=50, existence_prob=0.8, alpha=1.5, query_cost=0.05)
    result = oracle_policy_path(spec, target=None)
    assert result["found"] is False
    assert result["n_queries"] == 6
    assert result["ended_by"] == "oracle_stop"
