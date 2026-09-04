from floop_agents.search_arena import SearchSpec, oracle_state


def _oracle_boundary(query_cost: float) -> tuple[int, list[int]]:
    spec = SearchSpec(
        n_candidates=50,
        existence_prob=0.8,
        alpha=1.5,
        query_cost=query_cost,
        reward=1.0,
    )
    queried: list[int] = []
    while True:
        state = oracle_state(spec, queried)
        if state.decision == "STOP" or state.best_query is None:
            return len(queried), queried
        queried.append(int(state.best_query))


def test_expected_oracle_boundaries_for_cost_sweep() -> None:
    expected = {
        0.03: 10,
        0.05: 6,
        0.08: 4,
    }
    observed = {}
    for cost, expected_boundary in expected.items():
        boundary, path = _oracle_boundary(cost)
        observed[cost] = boundary
        assert boundary == expected_boundary
        assert len(path) == boundary
        assert path == list(range(1, boundary + 1))

    assert observed == expected
