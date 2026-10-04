from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
import time
from typing import Any

from agents.base import ModelResponse


class OpenAICompatibleModel:
    """OpenAI-compatible Chat Completions client with optional SQLite response cache."""

    def __init__(
        self,
        model: str,
        cache_path: str | Path = "results/cache.sqlite3",
        cache_enabled: bool = True,
        timeout: float = 60,
    ):
        self.model = model
        self.cache_path = Path(cache_path)
        self.cache_enabled = cache_enabled
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 512,
    ) -> ModelResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
            "system": system,
            "prompt": prompt,
            "base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL"),
        }
        cache_key = hashlib.sha256(
            json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.cache_enabled:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return ModelResponse(cached["text"], cached["input_tokens"], cached["output_tokens"], True)

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY (or an OpenAI-compatible provider key) before running an LLM evaluation.")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY") and not base_url:
            raise RuntimeError("Set OPENAI_BASE_URL or ANTHROPIC_BASE_URL when using ANTHROPIC_API_KEY.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The 'openai' package is required for live model calls.") from exc

        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=self.timeout,
            max_retries=2,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        result = ModelResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else _estimate_tokens(prompt + system),
            output_tokens=usage.completion_tokens if usage else _estimate_tokens(text),
        )
        if self.cache_enabled:
            self._write_cache(cache_key, result)
        return result

    def _connect(self) -> sqlite3.Connection:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.cache_path, timeout=30)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS model_cache (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        return connection

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM model_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _write_cache(self, cache_key: str, response: ModelResponse) -> None:
        payload = json.dumps(
            {"text": response.text, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO model_cache(cache_key, payload, created_at) VALUES (?, ?, ?)",
                (cache_key, payload, time.time()),
            )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class CostEstimator:
    def __init__(self, prices_per_million: dict[str, tuple[float, float]]):
        self.prices = prices_per_million

    def estimate(self, model: str, tasks: int, agents: int, repeats: int, input_tokens: int, output_tokens: int) -> float:
        input_price, output_price = self.prices.get(model, (0.0, 0.0))
        multiplier = tasks * agents * repeats
        return multiplier * (
            input_tokens * input_price + output_tokens * output_price
        ) / 1_000_000
