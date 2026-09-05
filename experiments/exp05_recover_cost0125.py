from __future__ import annotations

import argparse
import json
from pathlib import Path

from floop_agents.providers import NvidiaClient

from exp05_bellman_decomposition import WEDGE_CONDITIONS, probe_bellman


TARGET_CONDITION_ID = "cost_0125_wedge"


class LongerFinalAnswerClient(NvidiaClient):
    """NVIDIA client that preserves the Exp 05 prompt but allows a longer final response.

    The original cost-0.125 Bellman observation exhausted the 2048-token generation
    allowance before emitting final JSON. This targeted recovery changes only the
    output allowance; it does not change the state, treatment prompt, temperature,
    or reasoning-effort request used by Exp 05.
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ):
        return super().chat(
            messages,
            temperature=temperature,
            max_tokens=max(4096, max_tokens),
            reasoning_effort=reasoning_effort,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp05_cost0125_recovery"),
    )
    args = parser.parse_args()

    condition = next(c for c in WEDGE_CONDITIONS if c.condition_id == TARGET_CONDITION_ID)
    client = LongerFinalAnswerClient.from_env()
    print(f"Selected NVIDIA model: {client.resolve_model()}")
    print(
        "Recovery semantics: new stochastic observation; "
        "does not replace the original length-truncated record."
    )

    record = probe_bellman(client, condition)
    record["recovery_for"] = TARGET_CONDITION_ID
    record["recovery_reason"] = "original Bellman response hit 2048-token generation limit"
    record["minimum_generation_allowance"] = 4096

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "probe.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
