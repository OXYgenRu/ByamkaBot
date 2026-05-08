from __future__ import annotations

import logging

from bot_app.config import Config
from bot_app.core.context_builder import BuiltPrompt
from bot_app.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class Responder:
    def __init__(self, config: Config, llm: OllamaClient) -> None:
        self.config = config
        self.llm = llm

    async def generate(self, prompt: BuiltPrompt) -> str:
        raw = await self.llm.chat(
            prompt.messages,
            temperature=self.config.llm.temperature,
            num_predict=prompt.num_predict,
        )
        answer = self.llm.clean_visible_answer(raw)
        answer = self._trim(answer)
        return answer

    def _trim(self, text: str) -> str:
        text = text.strip()
        max_chars = self.config.bot.max_visible_answer_chars
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"
