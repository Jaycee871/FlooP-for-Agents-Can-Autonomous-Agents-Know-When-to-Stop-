import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp05_bellman_decomposition import make_wedge_spec
from exp07_value_substitution_audit import condition_by_id
from exp09_aggregate_probe_artifacts import duplicate_keys, scientific_validity
from exp09_recursive_depth_ladder import (
    EXP09_BACKUP_DEPTHS,
    EXP09_CONDITION_IDS,
    build_depth_chain,
    depth_prompt,
    parse_depth_response,
)


def test_exp09_oracle_backup_chain_matches_oracle_values():
    for depth in sorted(EXP09_BACKUP_DEPTHS):
        for condition_id in sorted(EXP09_CONDITION_IDS):
            spec = make_wedge_spec(condition_by_id(condition_id))
            chain = build_depth_chain(spec, depth)
            assert max(abs(v) for v in chain["identity_errors"].values()) <= 1e-12


def test_exp09_prompt_only_supplies_boundary_value():
    spec = make_wedge_spec(condition_by_id("cost_0100_wedge"))
    prompt, chain = depth_prompt(spec, 3)
    assert "BACKUP DEPTH: 3" in prompt
    assert "No intermediate continuation values are supplied." in prompt
    assert "EXACT ORACLE BOUNDARY VALUE AT s_4" in prompt
    assert f"{chain['boundary_value']:.9f}" in prompt


def test_exp09_parser_requires_every_backup_depth():
    good = '{"decision":"QUERY","query":1,"evaluated_query":1,"backup_depth":2,"backups":[{"state_depth":2,"value":0.1},{"state_depth":1,"value":0.2}],"estimated_value_after_miss":0.2,"full_expected_net_gain":0.01,"reason":"ok"}'
    parsed = parse_depth_response(good, 2)
    assert parsed["reported_backups"] == {2: 0.1, 1: 0.2}

    bad = '{"decision":"QUERY","query":1,"evaluated_query":1,"backup_depth":2,"backups":[{"state_depth":1,"value":0.2}],"estimated_value_after_miss":0.2,"full_expected_net_gain":0.01,"reason":"missing"}'
    try:
        parse_depth_response(bad, 2)
    except ValueError as exc:
        assert "Missing backup depths" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def _complete_records(model: str = "model-a", repeats: int = 3):
    records = []
    for depth in EXP09_BACKUP_DEPTHS:
        for condition_id in EXP09_CONDITION_IDS:
            for repeat in range(repeats):
                records.append(
                    {
                        "replication_model": model,
                        "backup_depth": depth,
                        "condition_id": condition_id,
                        "repeat": repeat,
                        "max_abs_oracle_identity_error": 0.0,
                    }
                )
    return records


def test_exp09_validity_requires_full_depth_condition_repeat_grid():
    records = _complete_records()
    result = scientific_validity(records, model="model-a", repeats=3)
    assert result["complete"] is True
    assert result["expected_records"] == 24

    result = scientific_validity(records[:-1], model="model-a", repeats=3)
    assert result["complete"] is False
    assert len(result["missing_probe_keys"]) == 1


def test_exp09_validity_rejects_oracle_identity_mismatch():
    records = _complete_records()
    records[0]["max_abs_oracle_identity_error"] = 1e-4
    result = scientific_validity(records, model="model-a", repeats=3)
    assert result["complete"] is False
    assert len(result["oracle_identity_violations"]) == 1


def test_exp09_duplicate_key_includes_backup_depth():
    records = _complete_records(repeats=1)
    records.append(dict(records[0]))
    assert len(duplicate_keys(records)) == 1
