import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp06_repeated_transition_audit import coupling_direction, describe, summarize


def test_describe_reports_distribution_statistics():
    stats = describe([-2.0, 0.0, 2.0])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(0.0)
    assert stats["median"] == pytest.approx(0.0)
    assert stats["stdev"] == pytest.approx(2.0)
    assert stats["mean_abs"] == pytest.approx(4.0 / 3.0)


def test_coupling_direction_distinguishes_both_failure_directions():
    assert coupling_direction({"full_expected_net_gain": 0.1, "decision": "STOP"}) == "positive_value_to_stop"
    assert coupling_direction({"full_expected_net_gain": -0.1, "decision": "QUERY"}) == "nonpositive_value_to_query"
    assert coupling_direction({"full_expected_net_gain": 0.1, "decision": "QUERY"}) == "consistent"
    assert coupling_direction({"full_expected_net_gain": 0.0, "decision": "STOP"}) == "consistent"


def test_summarize_empty_is_well_formed():
    result = summarize([])
    assert result["overall"]["n"] == 0
    assert result["overall"]["n_errors"] == 0
    assert result["overall"]["decision_agreement_rate"] is None
    assert result["by_condition"] == {}
