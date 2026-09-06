import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp05_bellman_decomposition import make_wedge_spec
from exp07_value_substitution_audit import condition_by_id
from exp10_aggregate_probe_artifacts import duplicate_keys, scientific_validity
from exp10_representation_ablation import (
    EXP10_ARMS,
    EXP10_CONDITION_IDS,
    common_numeric_prompt,
    parse_exp10_response,
    prompt_for_arm,
)


def test_exp10_common_numeric_prompt_is_shared_across_arms():
    spec = make_wedge_spec(condition_by_id("cost_0100_wedge"))
    common, chain = common_numeric_prompt(spec)
    implicit, _ = prompt_for_arm(spec, "implicit_narrative")
    explicit, _ = prompt_for_arm(spec, "explicit_bellman")
    assert common in implicit
    assert common in explicit
    assert f"{chain['boundary_value']:.9f}" in common


def test_exp10_only_explicit_arm_contains_bellman_recurrence():
    spec = make_wedge_spec(condition_by_id("cost_0120_wedge"))
    implicit, _ = prompt_for_arm(spec, "implicit_narrative")
    explicit, _ = prompt_for_arm(spec, "explicit_bellman")
    assert "V(s_j) = max(0" not in implicit
    assert "V(s_j) = max(0" in explicit


def test_exp10_parser_accepts_minimal_valid_json():
    text = '{"decision":"QUERY","query":1,"evaluated_query":1,"estimated_value_after_miss":0.2,"full_expected_net_gain":0.1,"reason":"ok"}'
    parsed = parse_exp10_response(text)
    assert parsed["decision"] == "QUERY"
    assert parsed["query"] == 1
    assert parsed["estimated_value_after_miss"] == 0.2


def _complete_records(model: str = "model-a", repeats: int = 3):
    records = []
    for arm in EXP10_ARMS:
        for condition_id in EXP10_CONDITION_IDS:
            for repeat in range(repeats):
                records.append(
                    {
                        "replication_model": model,
                        "arm": arm,
                        "condition_id": condition_id,
                        "repeat": repeat,
                    }
                )
    return records


def test_exp10_validity_requires_full_arm_condition_repeat_grid():
    records = _complete_records()
    result = scientific_validity(records, model="model-a", repeats=3)
    assert result["complete"] is True
    assert result["expected_records"] == 24

    result = scientific_validity(records[:-1], model="model-a", repeats=3)
    assert result["complete"] is False
    assert len(result["missing_probe_keys"]) == 1


def test_exp10_duplicate_key_includes_arm():
    records = _complete_records(repeats=1)
    records.append(dict(records[0]))
    duplicates = duplicate_keys(records)
    assert len(duplicates) == 1
    assert duplicates[0][1] in EXP10_ARMS
