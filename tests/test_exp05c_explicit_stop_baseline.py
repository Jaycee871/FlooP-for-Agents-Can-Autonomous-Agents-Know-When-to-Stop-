import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp05c_explicit_stop_baseline import EXPLICIT_STOP_RULE, coupling_direction


def test_explicit_stop_rule_defines_zero_baseline_and_sign_mapping():
    assert "STOP has incremental continuation value exactly 0" in EXPLICIT_STOP_RULE
    assert "full_expected_net_gain > 0" in EXPLICIT_STOP_RULE
    assert "full_expected_net_gain <= 0" in EXPLICIT_STOP_RULE
    assert "Do not compare" in EXPLICIT_STOP_RULE


def test_coupling_direction_flags_both_contradictions():
    assert coupling_direction({"full_expected_net_gain": 0.1, "decision": "STOP"}) == "positive_value_to_stop"
    assert coupling_direction({"full_expected_net_gain": -0.1, "decision": "QUERY"}) == "nonpositive_value_to_query"
    assert coupling_direction({"full_expected_net_gain": 0.1, "decision": "QUERY"}) == "consistent"
    assert coupling_direction({"full_expected_net_gain": 0.0, "decision": "STOP"}) == "consistent"
