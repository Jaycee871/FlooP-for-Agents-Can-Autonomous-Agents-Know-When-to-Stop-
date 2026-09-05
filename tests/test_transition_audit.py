import json
import sys
from pathlib import Path

import pytest

# Experiment modules import one another by filename, so expose the experiments directory
# the same way direct experiment execution does.
EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp05_bellman_decomposition import WedgeCondition, make_wedge_spec
from exp05b_transition_audit import parse_transition_response
from floop_agents.search_arena import oracle_state


def test_a_single_miss_does_not_rule_out_target_existence():
    spec = make_wedge_spec(WedgeCondition("audit", 0.125))
    before = oracle_state(spec, [])
    after = oracle_state(spec, [1])

    assert before.posterior_target_exists == pytest.approx(0.8)
    assert 0.0 < after.posterior_target_exists < before.posterior_target_exists
    assert after.next_hit_probability > 0.0
    assert after.advantage > 0.0


def test_cost_0125_successor_retains_positive_option_value():
    spec = make_wedge_spec(WedgeCondition("audit", 0.125))
    after = oracle_state(spec, [1])

    # This is the counterexample to the recovered model rationale that a miss
    # 'rules out the target'. The miss only removes candidate 1.
    assert after.posterior_target_exists > 0.0
    assert after.continuation_value == pytest.approx(0.05293740602796759)


def test_transition_parser_preserves_counterfactual_fields():
    payload = {
        "decision": "QUERY",
        "query": 1,
        "evaluated_query": 1,
        "current_hit_probability": 0.08,
        "miss_probability": 0.92,
        "predicted_target_exists_after_miss": 0.77,
        "predicted_best_hit_probability_after_miss": 0.09,
        "predicted_best_candidate_after_miss": 2,
        "estimated_value_after_miss": 0.05,
        "full_expected_net_gain": 0.01,
        "reason": "test",
    }

    parsed = parse_transition_response(json.dumps(payload))

    assert parsed["decision"] == "QUERY"
    assert parsed["query"] == 1
    assert parsed["evaluated_query"] == 1
    assert parsed["predicted_target_exists_after_miss"] == pytest.approx(0.77)
    assert parsed["predicted_best_hit_probability_after_miss"] == pytest.approx(0.09)
    assert parsed["predicted_best_candidate_after_miss"] == 2
    assert parsed["estimated_value_after_miss"] == pytest.approx(0.05)


def test_transition_parser_requires_no_query_when_stopping():
    payload = {
        "decision": "STOP",
        "query": 7,
        "evaluated_query": 1,
        "current_hit_probability": 0.08,
        "miss_probability": 0.92,
        "predicted_target_exists_after_miss": 0.70,
        "predicted_best_hit_probability_after_miss": 0.09,
        "predicted_best_candidate_after_miss": 2,
        "estimated_value_after_miss": 0.0,
        "full_expected_net_gain": -0.01,
        "reason": "test",
    }

    parsed = parse_transition_response(json.dumps(payload))

    assert parsed["decision"] == "STOP"
    assert parsed["query"] is None
    assert parsed["evaluated_query"] == 1
