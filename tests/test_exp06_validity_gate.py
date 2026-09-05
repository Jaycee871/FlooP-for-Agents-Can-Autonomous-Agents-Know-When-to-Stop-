import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp06_repeated_transition_audit import scientific_validity


def test_scientific_validity_passes_only_complete_error_free_grid():
    condition_ids = ["a", "b"]
    records = [
        {"condition_id": "a", "repeat": 0},
        {"condition_id": "b", "repeat": 0},
        {"condition_id": "a", "repeat": 1},
        {"condition_id": "b", "repeat": 1},
    ]
    validity = scientific_validity(records, repeats=2, condition_ids=condition_ids)
    assert validity["expected_records"] == 4
    assert validity["valid_records"] == 4
    assert validity["complete"] is True


def test_scientific_validity_rejects_error_record_even_when_count_matches():
    condition_ids = ["a", "b"]
    records = [
        {"condition_id": "a", "repeat": 0},
        {"condition_id": "b", "repeat": 0},
        {"condition_id": "a", "repeat": 1, "error": "truncated"},
        {"condition_id": "b", "repeat": 1},
    ]
    validity = scientific_validity(records, repeats=2, condition_ids=condition_ids)
    assert validity["observed_records"] == 4
    assert validity["valid_records"] == 3
    assert validity["error_records"] == 1
    assert validity["complete"] is False


def test_scientific_validity_rejects_partial_timeout_artifact():
    condition_ids = ["a", "b"]
    records = [
        {"condition_id": "a", "repeat": 0},
        {"condition_id": "b", "repeat": 0},
    ]
    validity = scientific_validity(records, repeats=2, condition_ids=condition_ids)
    assert validity["expected_records"] == 4
    assert validity["observed_records"] == 2
    assert validity["per_condition_complete"] is False
    assert validity["complete"] is False
