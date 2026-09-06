import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp07_aggregate_probe_artifacts import duplicate_keys, scientific_validity
from exp07_value_substitution_audit import (
    EXP07_ARMS,
    EXP07_CONDITION_IDS,
    condition_by_id,
    prompt_for_value_substitution,
)
from exp05_bellman_decomposition import make_wedge_spec


def test_exp07_condition_accepts_replication_condition():
    condition = condition_by_id("cost_0125_wedge")
    assert condition.condition_id == "cost_0125_wedge"


def test_exp07_condition_rejects_non_replication_condition():
    try:
        condition_by_id("cost_0090_wedge")
    except ValueError as exc:
        assert "Unsupported Exp07 condition" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_oracle_successor_value_prompt_contains_exact_value_only_in_value_arm():
    condition = condition_by_id("cost_0100_wedge")
    spec = make_wedge_spec(condition)
    state_prompt = prompt_for_value_substitution(spec, 1, "oracle_successor_state")
    value_prompt = prompt_for_value_substitution(spec, 1, "oracle_successor_value")
    assert "EXPERIMENTAL ARM: ORACLE SUCCESSOR STATE" in state_prompt
    assert "Exact optimal value after the miss:" not in state_prompt
    assert "EXPERIMENTAL ARM: ORACLE SUCCESSOR VALUE" in value_prompt
    assert "Exact optimal value after the miss:" in value_prompt


def _complete_records(model: str = "model-a", repeats: int = 3):
    records = []
    for arm in EXP07_ARMS:
        for condition_id in EXP07_CONDITION_IDS:
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


def test_exp07_scientific_validity_requires_all_arm_condition_repeat_cells():
    records = _complete_records()
    result = scientific_validity(records, model="model-a", repeats=3)
    assert result["complete"] is True
    assert result["expected_records"] == 24

    incomplete = records[:-1]
    result = scientific_validity(incomplete, model="model-a", repeats=3)
    assert result["complete"] is False
    assert len(result["missing_probe_keys"]) == 1


def test_exp07_duplicate_key_includes_arm():
    records = _complete_records(repeats=1)
    duplicate = dict(records[0])
    records.append(duplicate)
    duplicates = duplicate_keys(records)
    assert len(duplicates) == 1
    assert duplicates[0][1] in EXP07_ARMS
