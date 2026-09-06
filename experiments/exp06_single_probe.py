from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from floop_agents.providers import NvidiaClient

from exp05_bellman_decomposition import WEDGE_CONDITIONS
from exp05b_transition_audit import probe
from exp06_repeated_transition_audit import REPLICATION_CONDITION_IDS


class ProbeClient(NvidiaClient):
    """Exp06 client with explicit, recorded model-specific reasoning controls."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ):
        configured_tokens = int(os.getenv("EXP06_MAX_TOKENS", str(max_tokens)))
        mode = os.getenv("EXP06_REASONING_MODE", "none").strip().lower()
        if mode not in {"none", "preserve"}:
            raise ValueError("EXP06_REASONING_MODE must be 'none' or 'preserve'")
        effective_reasoning = reasoning_effort if mode == "preserve" else None
        return super().chat(
            messages,
            temperature=temperature,
            max_tokens=max(max_tokens, configured_tokens),
            reasoning_effort=effective_reasoning,
        )


def condition_by_id(condition_id: str):
    if condition_id not in REPLICATION_CONDITION_IDS:
        raise ValueError(f"Unsupported Exp06 condition: {condition_id}")
    for condition in WEDGE_CONDITIONS:
        if condition.condition_id == condition_id:
            return condition
    raise ValueError(f"Condition not found in WEDGE_CONDITIONS: {condition_id}")


def run_single_probe(
    *,
    condition_id: str,
    repeat_index: int,
    out_path: Path,
) -> dict[str, Any]:
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")

    client = ProbeClient.from_env()
    model = client.resolve_model()
    condition = condition_by_id(condition_id)
    reasoning_mode = os.getenv("EXP06_REASONING_MODE", "none").strip().lower()

    try:
        record = probe(client, condition)
        record["repeat"] = repeat_index
        record["replication_model"] = model
        record["reasoning_mode"] = reasoning_mode
    except Exception as exc:
        record = {
            "condition_id": condition.condition_id,
            "query_cost": condition.query_cost,
            "repeat": repeat_index,
            "replication_model": model,
            "reasoning_mode": reasoning_mode,
            "error": f"{type(exc).__name__}: {exc}",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(
        "PROBE_VALIDITY: "
        + ("PASS" if "error" not in record else "FAIL")
        + f" model={model} condition={condition_id} repeat={repeat_index}"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--out", type=Path, default=Path("results/exp06_matrix/probe.json"))
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="Exit non-zero after writing the record if this probe contains an error.",
    )
    args = parser.parse_args()

    record = run_single_probe(
        condition_id=args.condition_id,
        repeat_index=args.repeat_index,
        out_path=args.out,
    )
    if args.require_valid and "error" in record:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
