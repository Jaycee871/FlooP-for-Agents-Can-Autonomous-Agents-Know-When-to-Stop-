from __future__ import annotations

import json
import os
from pathlib import Path

from floop_agents.providers.nvidia import NvidiaClient


CANDIDATES = [
    "google/gemma-4-31b-it",
    "deepseek-ai/deepseek-v4-flash-0731",
    "deepseek-ai/deepseek-v4-pro-0813",
    "nvidia/nemotron-3-super-120b-a12b",
]


def main() -> None:
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVDIA_API_KEY") or ""
    if not api_key:
        raise RuntimeError("NVIDIA API key is required")

    results = []
    for model in CANDIDATES:
        client = NvidiaClient(
            api_key=api_key,
            model=model,
            timeout=60,
            max_retries=0,
        )
        try:
            response = client.chat(
                [{"role": "user", "content": "Reply with exactly OK."}],
                temperature=0.0,
                max_tokens=32,
                reasoning_effort=None,
            )
            results.append(
                {
                    "model": model,
                    "available": True,
                    "returned_model": response.model,
                    "content": response.content[:100],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model": model,
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    out_dir = Path("results/nvidia_candidate_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
