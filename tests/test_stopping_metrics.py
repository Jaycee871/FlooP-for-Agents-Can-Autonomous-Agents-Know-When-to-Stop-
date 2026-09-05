import pytest

from floop_agents.stopping_metrics import local_stopping_metrics, stopping_curve_metrics


def test_premature_stop_regret_equals_abandoned_continuation_value():
    m = local_stopping_metrics(
        agent_decision="STOP",
        oracle_continuation_value=0.07,
        agent_reported_continuation_value=-0.03,
    )
    assert m.oracle_decision == "QUERY"
    assert m.decision_agreement is False
    assert m.premature_stop_regret == pytest.approx(0.07)
    assert m.excess_continuation_regret == 0.0
    assert m.query_selection_regret == 0.0
    assert m.local_decision_regret == pytest.approx(0.07)
    assert m.signed_calibration_error == pytest.approx(-0.10)
    assert m.continuation_sign_accuracy is False
    assert m.decision_value_self_consistency is True


def test_excess_continuation_uses_value_of_concrete_query():
    m = local_stopping_metrics(
        agent_decision="QUERY",
        oracle_continuation_value=-0.01,
        agent_reported_continuation_value=0.02,
        chosen_query_value=-0.04,
    )
    assert m.oracle_decision == "STOP"
    assert m.excess_continuation_regret == pytest.approx(0.04)
    assert m.local_decision_regret == pytest.approx(0.04)
    assert m.query_selection_regret == 0.0


def test_query_selection_regret_is_separate_from_stopping_regret():
    m = local_stopping_metrics(
        agent_decision="QUERY",
        oracle_continuation_value=0.20,
        agent_reported_continuation_value=0.18,
        chosen_query_value=0.13,
    )
    assert m.oracle_decision == "QUERY"
    assert m.decision_agreement is True
    assert m.premature_stop_regret == 0.0
    assert m.excess_continuation_regret == 0.0
    assert m.query_selection_regret == pytest.approx(0.07)
    assert m.local_decision_regret == pytest.approx(0.07)


def test_decision_value_inconsistency_is_detected():
    m = local_stopping_metrics(
        agent_decision="STOP",
        oracle_continuation_value=-0.02,
        agent_reported_continuation_value=0.10,
    )
    assert m.decision_agreement is True
    assert m.continuation_sign_accuracy is False
    assert m.decision_value_self_consistency is False


def test_query_requires_concrete_query_value():
    with pytest.raises(ValueError):
        local_stopping_metrics(
            agent_decision="QUERY",
            oracle_continuation_value=0.1,
            agent_reported_continuation_value=0.1,
        )


def test_stopping_curve_detects_stop_to_query_reversal():
    decisions = ["QUERY", "QUERY", "STOP", "QUERY", "STOP"]
    oracle = ["QUERY", "QUERY", "QUERY", "STOP", "STOP"]
    m = stopping_curve_metrics(decisions, oracle, oracle_boundary=3)
    assert m.first_stop_boundary == 2
    assert m.boundary_displacement == -1
    assert m.stop_to_query_reversals == 1
    assert m.post_stop_query_count == 1
    assert m.nonmonotone is True
    assert m.monotonicity_violation_rate == pytest.approx(0.25)
    assert m.decision_agreement_rate == pytest.approx(3 / 5)


def test_monotone_aligned_stopping_curve_has_no_violation():
    decisions = ["QUERY", "QUERY", "QUERY", "STOP", "STOP"]
    m = stopping_curve_metrics(decisions, decisions, oracle_boundary=3)
    assert m.first_stop_boundary == 3
    assert m.boundary_displacement == 0
    assert m.stop_to_query_reversals == 0
    assert m.post_stop_query_count == 0
    assert m.nonmonotone is False
    assert m.monotonicity_violation_rate == 0.0
    assert m.decision_agreement_rate == 1.0
