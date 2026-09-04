from floop_agents.search_arena import SearchSpec, oracle_policy_path, oracle_state


def test_oracle_has_interior_stopping_point():
    spec = SearchSpec(n_candidates=50, existence_prob=0.8, alpha=1.5, query_cost=0.05)
    assert oracle_state(spec, range(1, 6)).decision == "QUERY"
    assert oracle_state(spec, range(1, 7)).decision == "STOP"


def test_oracle_queries_highest_prior_first():
    spec = SearchSpec(n_candidates=20, existence_prob=0.8, alpha=1.5, query_cost=0.05)
    state = oracle_state(spec, [])
    assert state.decision == "QUERY"
    assert state.best_query == 1


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
