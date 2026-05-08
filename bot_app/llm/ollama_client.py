from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from bot_app.config import LLMConfig

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


class OllamaClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout_sec),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
        format_schema: dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> str:
        options: dict[str, Any] = {
            "temperature": self.config.temperature if temperature is None else temperature,
            "num_ctx": self.config.context_window_tokens,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        if extra_options:
            options.update(extra_options)

        payload: dict[str, Any] = {
            "model": self.config.chat_model,
            "messages": messages,
            "stream": False,
            # Для reasoning-моделей лучше явно выключать hidden thinking в router/extractor/json-вызовах,
            # иначе модель может потратить бюджет на рассуждение и вернуть пустой content.
            "think": False,
            "options": options,
        }
        if format_schema is not None:
            payload["format"] = format_schema

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            logger.debug("Ollama returned empty message content. raw=%s", data)
        return content

    async def structured_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        temperature: float,
        num_predict: int = 512,
    ) -> dict[str, Any]:
        raw = await self.chat(
            messages,
            temperature=temperature,
            num_predict=num_predict,
            format_schema=schema,
        )
        return self._parse_json(raw)

    async def embed(self, text: str) -> list[float]:
        """Return embedding vector using Ollama.

        Newer Ollama has /api/embed. Older builds had /api/embeddings.
        We try /api/embed first and fallback to /api/embeddings.
        """
        payload = {"model": self.config.embedding_model, "input": text}
        response = await self._client.post("/api/embed", json=payload)
        if response.status_code < 400:
            data = response.json()
            embeddings = data.get("embeddings") or []
            if embeddings and isinstance(embeddings[0], list):
                return [float(x) for x in embeddings[0]]

        fallback = await self._client.post(
            "/api/embeddings",
            json={"model": self.config.embedding_model, "prompt": text},
        )
        fallback.raise_for_status()
        data = fallback.json()
        emb = data.get("embedding") or []
        return [float(x) for x in emb]

    def clean_visible_answer(self, text: str) -> str:
        text = _THINK_RE.sub("", text)
        # Если модель оборвала тег <think>, грубо отрежем всё до последнего закрывающего маркера.
        if "</think>" in text.lower():
            parts = re.split(r"</think>", text, flags=re.IGNORECASE)
            text = parts[-1]
        return text.strip()

    def _parse_json(self, raw: str) -> dict[str, Any]:
        cleaned = self.clean_visible_answer(raw).strip()
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse strict JSON. Trying fallback extraction. raw=%r", raw[:500])

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
