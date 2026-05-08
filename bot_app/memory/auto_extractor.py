from __future__ import annotations

import logging
from dataclasses import dataclass

from bot_app.config import Config
from bot_app.domain.events import MessageEvent
from bot_app.llm.ollama_client import OllamaClient
from bot_app.memory.qdrant_memory import QdrantMemory
from bot_app.personality.interest_field import InterestProfile

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFact:
    text: str
    confidence: float
    kind: str
    scope: str


FACT_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "user_preference",
                            "project_fact",
                            "chat_rule",
                            "running_joke",
                            "decision",
                            "other",
                        ],
                    },
                    "scope": {"type": "string", "enum": ["chat", "global"]},
                },
                "required": ["text", "confidence", "kind", "scope"],
            },
        }
    },
    "required": ["facts"],
}


class AutoMemoryExtractor:
    """Extracts durable facts from ordinary chat messages and stores them in vector memory.

    This module intentionally stores only stable facts. It should reject transient messages,
    secrets, credentials, payment data, and vague emotions.
    """

    def __init__(self, config: Config, llm: OllamaClient, memory: QdrantMemory) -> None:
        self.config = config
        self.llm = llm
        self.memory = memory

    async def maybe_extract_and_store(
        self,
        event: MessageEvent,
        *,
        interest_profile: InterestProfile | None = None,
    ) -> list[str]:
        cfg = self.config.auto_memory
        if not cfg.enabled:
            return []
        if event.is_bot_message:
            return []

        text = event.text.strip()
        if not text:
            return []
        if cfg.ignore_commands and text.startswith("/"):
            return []
        if len(text) < cfg.min_text_chars:
            return []

        facts = await self._extract(event)
        saved_ids: list[str] = []
        for fact in facts[: cfg.max_facts_per_message]:
            fact.text = self._normalize_fact_text(fact.text)
            if not fact.text or len(fact.text) < cfg.min_fact_chars:
                continue
            min_confidence = cfg.min_confidence
            if interest_profile is not None and interest_profile.memory >= self.config.personality_engine.memory_interest_threshold:
                min_confidence = max(0.0, min_confidence - cfg.interest_memory_confidence_discount)
            if fact.confidence < min_confidence:
                continue
            if await self._looks_duplicate(fact.text, event.chat_id):
                logger.info(
                    "auto_memory duplicate skipped rquid=%s chat=%s kind=%s text_preview=%r",
                    event.rquid,
                    event.chat_id,
                    fact.kind,
                    fact.text[:180],
                )
                continue

            chat_id = None if (cfg.allow_global_facts and fact.scope == "global") else event.chat_id
            fact_id = await self.memory.add_fact(
                fact.text,
                chat_id=chat_id,
                user_id=event.user_id,
                source=f"auto_extracted:{fact.kind}",
            )
            saved_ids.append(fact_id)
            logger.info(
                "auto_memory saved rquid=%s fact_id=%s chat=%s user=%s kind=%s confidence=%.2f text_preview=%r",
                event.rquid,
                fact_id,
                chat_id,
                event.user_id,
                fact.kind,
                fact.confidence,
                fact.text[:180],
            )
        return saved_ids

    async def _extract(self, event: MessageEvent) -> list[ExtractedFact]:
        cfg = self.config.auto_memory
        system = (
            "Ты модуль извлечения памяти для Telegram-бота. "
            "Верни строгий JSON по схеме. "
            "Не извлекай временную информацию. "
            "Извлекай только устойчивые факты, которые пригодятся в будущих диалогах. "
            "Не сохраняй одноразовые сообщения, приветствия, эмоции без долгосрочного смысла, "
            "случайные шутки без повторяемости, временные планы без явной важности. "
            "Не сохраняй пароли, токены, ключи API, номера карт, коды подтверждения, адреса проживания, "
            "паспортные данные и другую чувствительную информацию. "
            "Факт должен быть самодостаточным и понятным без исходного сообщения. "
            "Если пользователь просит изменить стиль поведения бота, например не использовать смайлики, "
            "отвечать короче, не шутить или не влезать без повода, сохраняй это как chat_rule "
            "с высокой уверенностью. "
            "Если сохранять нечего, верни {\"facts\": []}."
        )
        user = (
            f"Автор: {event.author_label}\n"
            f"chat_id: {event.chat_id}\n"
            f"Сообщение:\n{event.text}\n\n"
            "Примеры хороших фактов:\n"
            "- В этом чате бот не должен использовать эмодзи и смайлики."
            "- Андрей предпочитает короткие инженерные ответы без воды.\n"
            "- В этом чате локальная шутка: docker-compose называют квест-комнатой.\n"
            "- В проекте бота используется Ollama на localhost:11434.\n\n"
            "Примеры плохих фактов, которые НЕ надо сохранять:\n"
            "- Пользователь интересуется авиакомпанией, которой летает другой пользователь.\n"
            "- Пользователь написал 'ок'.\n"
            "- Пользователь сегодня устал.\n"
            "- Пользователь скинул одноразовый код или токен.\n"
        )
        try:
            data = await self.llm.structured_json(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                FACT_EXTRACTION_SCHEMA,
                temperature=cfg.temperature,
                num_predict=cfg.num_predict,
            )
        except Exception:
            logger.exception("auto_memory extraction failed rquid=%s chat=%s", event.rquid, event.chat_id)
            return []

        raw_facts = data.get("facts") if isinstance(data, dict) else []
        if not isinstance(raw_facts, list):
            return []

        facts: list[ExtractedFact] = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            try:
                facts.append(
                    ExtractedFact(
                        text=str(item.get("text", "")).strip(),
                        confidence=float(item.get("confidence", 0.0)),
                        kind=str(item.get("kind", "other")),
                        scope=str(item.get("scope", "chat")),
                    )
                )
            except (TypeError, ValueError):
                continue
        return facts

    async def _looks_duplicate(self, fact_text: str, chat_id: int) -> bool:
        threshold = self.config.auto_memory.duplicate_score_threshold
        if threshold <= 0:
            return False
        try:
            matches = await self.memory.search(fact_text, chat_id=chat_id, limit=3)
        except Exception:
            logger.exception("auto_memory duplicate check failed")
            return False
        return any(m.score >= threshold for m in matches)

    @staticmethod
    def _normalize_fact_text(text: str) -> str:
        return " ".join(text.strip().split())
