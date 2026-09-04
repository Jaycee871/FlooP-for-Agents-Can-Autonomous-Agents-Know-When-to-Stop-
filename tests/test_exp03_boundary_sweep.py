from experiments.exp03_boundary_sweep import CONDITIONS, make_spec, oracle_boundary


def test_expected_oracle_boundaries_for_cost_sweep() -> None:
    expected = {
        "cost_003": 10,
        "cost_005": 6,
        "cost_008": 4,
    }
    observed = {}
    for condition in CONDITIONS:
        boundary, path = oracle_boundary(make_spec(condition))
        observed[condition.condition_id] = boundary
        assert len(path) == boundary
        assert path == list(range(1, boundary + 1))

    assert observed == expected
