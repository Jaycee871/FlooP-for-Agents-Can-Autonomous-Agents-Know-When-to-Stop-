import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp06_aggregate_probe_artifacts import duplicate_keys, load_probe_records
from exp06_single_probe import condition_by_id


def test_condition_by_id_accepts_replication_condition():
    condition = condition_by_id("cost_0125_wedge")
    assert condition.condition_id == "cost_0125_wedge"


def test_condition_by_id_rejects_non_replication_condition():
    try:
        condition_by_id("cost_0090_wedge")
    except ValueError as exc:
        assert "Unsupported Exp06 condition" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_load_probe_records_and_duplicate_detection(tmp_path):
    for index in (1, 2):
        path = tmp_path / str(index) / "probe.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "replication_model": "model-a",
                    "condition_id": "cost_0100_wedge",
                    "repeat": 0,
                }
            ),
            encoding="utf-8",
        )
    records = load_probe_records(tmp_path)
    assert len(records) == 2
    assert duplicate_keys(records) == [("model-a", "cost_0100_wedge", 0)]
