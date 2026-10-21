from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from jsonschema import validate

from bot_app.config import LLMConfig
from bot_app.llm.model_logging import model_call

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)


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
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
        format_schema: dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
        purpose: str = "chat",
        thinking: bool | None = None,
    ) -> str:
        # Callers specify the answer allowance. Ollama's num_predict is a shared
        # limit for reasoning AND content, so add reasoning room centrally for
        # calls that enable reasoning. A per-request override never mutates config.
        use_thinking = self.config.thinking if thinking is None else thinking
        options: dict[str, Any] = {
            "temperature": self.config.temperature if temperature is None else temperature,
            "num_ctx": self.config.context_window_tokens,
            "num_batch": self.config.num_batch,
        }
        options["num_predict"] = self.config.reserve_response_tokens if num_predict is None else num_predict
        if extra_options:
            options.update(extra_options)
        if use_thinking and options["num_predict"] > 0:
            options["num_predict"] += self.config.reserve_thinking_tokens

        payload: dict[str, Any] = {
            "model": self.config.chat_model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "think": use_thinking,
            "options": options,
        }
        if format_schema is not None:
            if use_thinking:
                # Ollama 0.9.6 constrains reasoning tokens too, suppressing thinking
                # and producing malformed JSON (ollama/ollama#10929). Request the
                # schema in the prompt and validate the final object in structured_json.
                instruction = (
                    "I return my final answer as one JSON object matching this JSON Schema. "
                    "I keep reasoning separate from the final answer. I do not use markdown fences.\n"
                    + json.dumps(format_schema, ensure_ascii=False)
                )
                payload["messages"] = [dict(m) for m in messages]
                if payload["messages"] and payload["messages"][0]["role"] == "system":
                    payload["messages"][0]["content"] += "\n\n" + instruction
                else:
                    payload["messages"].insert(0, {"role": "system", "content": instruction})
            else:
                payload["format"] = format_schema

        async with model_call("CHAT", self.config.chat_model, purpose, "/api/chat",
                              details=f"num_batch={options['num_batch']} keep_alive={self.config.keep_alive} thinking={use_thinking} num_predict={options['num_predict']} images={sum(len(m.get('images', [])) for m in messages)}") as call:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            call.record(data)
            if data.get("done_reason") == "length":
                raise RuntimeError(f"Ollama token limit reached at {purpose}; increase the output/reasoning budget")
            # Reasoning is returned separately by Ollama. Never fall back to it
            # when the model has not produced a final answer.
            content = self.clean_visible_answer(data.get("message", {}).get("content", ""))
            if not content:
                raise RuntimeError(f"Ollama returned no final answer at {purpose}")
            return content

    async def structured_json(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        temperature: float,
        num_predict: int = 512,
        purpose: str = "structured_json",
        thinking: bool | None = None,
    ) -> dict[str, Any]:
        raw = await self.chat(
            messages,
            temperature=temperature,
            num_predict=num_predict,
            format_schema=schema,
            purpose=purpose,
            thinking=thinking,
        )
        data = self._parse_json(raw)
        validate(instance=data, schema=schema)
        return data

    async def embed(self, text: str, *, purpose: str = "embedding") -> list[float]:
        """Return embedding vector using Ollama.

        Newer Ollama has /api/embed. Older builds had /api/embeddings.
        We try /api/embed first and fallback to /api/embeddings.
        """
        async with model_call("EMBEDDING", self.config.embedding_model,
                              purpose, "/api/embed") as call:
            payload = {"model": self.config.embedding_model, "input": text}
            response = await self._client.post("/api/embed", json=payload)
            if response.is_success:
                data = response.json()
                call.record(data)
                embeddings = data.get("embeddings") or []
                if embeddings and isinstance(embeddings[0], list):
                    vector = [float(x) for x in embeddings[0]]
                    call.metrics["dimensions"] = len(vector)
                    return vector
                raise RuntimeError("Ollama /api/embed returned no embedding")
            # A missing model also produces 404. Do not mask it with a legacy retry.
            try:
                error = response.json().get("error", "")
            except (ValueError, AttributeError):
                error = ""
            if error or response.status_code not in (404, 405):
                logger.error("Ollama embedding failed model=%s status=%s error=%s",
                             self.config.embedding_model, response.status_code, error or response.text[:300])
                response.raise_for_status()

            call.log("FALLBACK", "target=/api/embeddings")
            call.endpoint = "/api/embeddings"
            fallback = await self._client.post(
                "/api/embeddings",
                json={"model": self.config.embedding_model, "prompt": text},
            )
            fallback.raise_for_status()
            data = fallback.json()
            call.record(data)
            emb = data.get("embedding") or []
            vector = [float(x) for x in emb]
            call.metrics["dimensions"] = len(vector)
            return vector

    def clean_visible_answer(self, text: str) -> str:
        text = _THINK_RE.sub("", text)
        # Если модель оборвала тег <think>, грубо отрежем всё до последнего закрывающего маркера.
        if "</think>" in text.lower():
            parts = re.split(r"</think>", text, flags=re.IGNORECASE)
            text = parts[-1]
        return text.strip()

    def _parse_json(self, raw: str) -> dict[str, Any]:
        cleaned = self.clean_visible_answer(raw).strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object from Ollama")
        return data
