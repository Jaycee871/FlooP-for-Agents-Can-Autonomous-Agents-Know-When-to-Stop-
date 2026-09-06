import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp05_bellman_decomposition import make_wedge_spec
from exp08_aggregate_probe_artifacts import duplicate_keys, scientific_validity
from exp08_bellman_depth_probe import (
    EXP08_CONDITION_IDS,
    condition_by_id,
    depth2_prompt,
)


def test_depth2_prompt_hides_first_successor_value_but_supplies_second_boundary():
    condition = condition_by_id("cost_0100_wedge")
    spec = make_wedge_spec(condition)
    prompt, exact = depth2_prompt(spec, 1)
    assert "EXACT SECOND-SUCCESSOR BOUNDARY VALUE" in prompt
    assert "The exact value of the FIRST successor state is intentionally NOT given." in prompt
    assert f"{exact['second_successor_value']:.9f}" in prompt


def test_one_backup_identity_matches_oracle_first_successor_value():
    for condition_id in sorted(EXP08_CONDITION_IDS):
        condition = condition_by_id(condition_id)
        spec = make_wedge_spec(condition)
        _, exact = depth2_prompt(spec, 1)
        assert abs(exact["one_backup_value"] - exact["successor_value"]) <= 1e-12


def _complete_records(model: str = "model-a", repeats: int = 3):
    records = []
    for condition_id in EXP08_CONDITION_IDS:
        for repeat in range(repeats):
            records.append(
                {
                    "replication_model": model,
                    "condition_id": condition_id,
                    "repeat": repeat,
                    "oracle_identity_error": 0.0,
                }
            )
    return records


def test_exp08_scientific_validity_requires_all_cells():
    records = _complete_records()
    result = scientific_validity(records, model="model-a", repeats=3)
    assert result["complete"] is True
    assert result["expected_records"] == 12

    result = scientific_validity(records[:-1], model="model-a", repeats=3)
    assert result["complete"] is False
    assert len(result["missing_probe_keys"]) == 1


def test_exp08_scientific_validity_rejects_oracle_identity_mismatch():
    records = _complete_records()
    records[0]["oracle_identity_error"] = 1e-4
    result = scientific_validity(records, model="model-a", repeats=3)
    assert result["complete"] is False
    assert len(result["oracle_identity_violations"]) == 1


def test_exp08_duplicate_key_detection():
    records = _complete_records(repeats=1)
    records.append(dict(records[0]))
    duplicates = duplicate_keys(records)
    assert len(duplicates) == 1
