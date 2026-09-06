import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp05_bellman_decomposition import make_wedge_spec
from exp07_value_substitution_audit import condition_by_id
from exp11_aggregate_probe_artifacts import duplicate_keys, scientific_validity
from exp11_externalized_backups import EXP11_ARMS, EXP11_CONDITION_IDS, parse_response, prompt_for_arm


def test_exp11_numeric_facts_are_identical_across_arms():
    spec = make_wedge_spec(condition_by_id("cost_0100_wedge"))
    formula_prompt, formula_chain = prompt_for_arm(spec, "formula_only")
    forced_prompt, forced_chain = prompt_for_arm(spec, "forced_backups")

    formula_numeric = formula_prompt.split("\n\nEXTERNALIZATION CONDITION:")[0]
    forced_numeric = forced_prompt.split("\n\nEXTERNALIZATION CONDITION:")[0]
    assert formula_numeric == forced_numeric
    assert formula_chain["boundary_value"] == forced_chain["boundary_value"]
    assert formula_chain["exact_backups"] == forced_chain["exact_backups"]


def test_exp11_forced_parser_requires_all_backups_in_order():
    good = '{"decision":"QUERY","query":1,"evaluated_query":1,"backups":[{"state_depth":3,"value":0.3},{"state_depth":2,"value":0.2},{"state_depth":1,"value":0.1}],"estimated_value_after_miss":0.1,"full_expected_net_gain":0.01,"reason":"ok"}'
    parsed = parse_response(good, "forced_backups")
    assert parsed["reported_backups"] == {3: 0.3, 2: 0.2, 1: 0.1}

    bad = '{"decision":"QUERY","query":1,"evaluated_query":1,"backups":[{"state_depth":1,"value":0.1},{"state_depth":2,"value":0.2},{"state_depth":3,"value":0.3}],"estimated_value_after_miss":0.1,"full_expected_net_gain":0.01,"reason":"wrong order"}'
    try:
        parse_response(bad, "forced_backups")
    except ValueError as exc:
        assert "order" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_exp11_formula_only_parser_does_not_require_backups():
    text = '{"decision":"STOP","query":null,"evaluated_query":1,"estimated_value_after_miss":0.1,"full_expected_net_gain":-0.01,"reason":"stop"}'
    parsed = parse_response(text, "formula_only")
    assert parsed["reported_backups"] is None


def _complete_records(model: str = "model-a", repeats: int = 3):
    records = []
    for arm in EXP11_ARMS:
        for condition_id in EXP11_CONDITION_IDS:
            for repeat in range(repeats):
                records.append({"replication_model": model, "arm": arm, "condition_id": condition_id, "repeat": repeat})
    return records


def test_exp11_validity_requires_full_arm_condition_repeat_grid():
    records = _complete_records()
    result = scientific_validity(records, model="model-a", repeats=3)
    assert result["complete"] is True
    assert result["expected_records"] == 24

    result = scientific_validity(records[:-1], model="model-a", repeats=3)
    assert result["complete"] is False
    assert len(result["missing_probe_keys"]) == 1


def test_exp11_duplicate_key_includes_arm():
    records = _complete_records(repeats=1)
    records.append(dict(records[0]))
    duplicates = duplicate_keys(records)
    assert len(duplicates) == 1
    assert duplicates[0][1] in EXP11_ARMS
