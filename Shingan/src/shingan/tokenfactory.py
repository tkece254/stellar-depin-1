from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
REQUEST_TIMEOUT_S = 600.0

Role = Literal["builder", "breaker", "triage"]
ROLE_KEYWORDS: dict[Role, str] = {"builder": "ultra", "breaker": "super", "triage": "nano"}

Message = dict[str, str]


@dataclass(frozen=True)
class Completion:
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt += prompt
        self.completion += completion


def tally(usage: dict[str, Usage], completion: Completion) -> None:
    usage.setdefault(completion.model, Usage()).add(completion.prompt_tokens, completion.completion_tokens)


class TokenFactory:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT_S,
            transport=transport,
        )
        self._models: list[str] | None = None

    @classmethod
    def from_env(cls) -> TokenFactory:
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY is not set")
        return cls(api_key, os.environ.get("SHINGAN_BASE_URL", DEFAULT_BASE_URL))

    def models(self) -> list[str]:
        if self._models is None:
            response = self._http.get("/models")
            response.raise_for_status()
            self._models = sorted(m["id"] for m in response.json()["data"])
        return self._models

    def model_for(self, role: Role) -> str:
        override = os.environ.get(f"SHINGAN_{role.upper()}_MODEL")
        if override:
            return override
        keyword = ROLE_KEYWORDS[role]
        matches = [m for m in self.models() if "nemotron" in m.lower() and keyword in m.lower()]
        if not matches:
            raise RuntimeError(
                f"no Nemotron {keyword} model on Token Factory; set SHINGAN_{role.upper()}_MODEL "
                f"(available: {', '.join(self.models())})"
            )
        return matches[0]

    def complete(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 16384,
    ) -> Completion:
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        for attempt in range(MAX_RETRIES + 1):
            response = self._http.post("/chat/completions", json=payload)
            if response.status_code not in RETRY_STATUS or attempt == MAX_RETRIES:
                break
            time.sleep(2**attempt)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return Completion(
            model=model,
            text=data["choices"][0]["message"].get("content") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
