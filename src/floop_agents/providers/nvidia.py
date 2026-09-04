from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class NvidiaResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class NvidiaClient:
    """Minimal OpenAI-compatible client for NVIDIA's hosted inference API.

    The adapter intentionally uses only the Python standard library so the benchmark
    stays lightweight. It never logs or serializes the API key.
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: float = 90.0,
    ) -> None:
        if not api_key:
            raise ValueError("NVIDIA API key is required")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "NvidiaClient":
        # Support the historical misspelling used during the first connectivity test.
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVDIA_API_KEY") or ""
        model = os.getenv("NVIDIA_MODEL") or None
        return cls(api_key=api_key, model=model)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "floop-agents/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"NVIDIA API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"NVIDIA API connection failed: {exc.reason}") from exc

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/models")
        ids: list[str] = []
        for item in payload.get("data", []):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
        return ids

    @staticmethod
    def _chat_candidates(model_ids: Iterable[str]) -> list[str]:
        excluded = ("embed", "rerank", "retrieval", "vision-embed", "nvclip")
        preferred_markers = ("instruct", "chat", "reason", "llama", "qwen", "mistral", "nemotron")
        ids = [m for m in model_ids if not any(term in m.lower() for term in excluded)]
        preferred = [m for m in ids if any(term in m.lower() for term in preferred_markers)]
        return preferred or ids

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        model_ids = self.list_models()
        candidates = self._chat_candidates(model_ids)
        if not candidates:
            raise RuntimeError("No plausible chat-capable NVIDIA model was found")

        # Prefer smaller instruction-tuned models for the smoke experiment to keep
        # latency and cost low. If none match, fall back to the first chat candidate.
        priorities = (
            "8b-instruct",
            "7b-instruct",
            "mini-instruct",
            "small-instruct",
            "llama-3.1-8b",
            "qwen2.5-7b",
        )
        for marker in priorities:
            for model in candidates:
                if marker in model.lower():
                    self.model = model
                    return model
        self.model = candidates[0]
        return self.model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 320,
    ) -> NvidiaResponse:
        model = self.resolve_model()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._request_json("POST", "/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"NVIDIA API returned no choices: {str(data)[:800]}")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"NVIDIA API returned malformed message content: {str(data)[:800]}")
        usage = data.get("usage") or {}
        return NvidiaResponse(
            content=content,
            model=str(data.get("model") or model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
