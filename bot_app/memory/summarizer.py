from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from bot_app.config import Config
from bot_app.llm.ollama_client import OllamaClient
from bot_app.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class ChatSummarizer:
    def __init__(self, config: Config, storage: SQLiteStore, llm: OllamaClient) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm
        self._locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def maybe_summarize(self, chat_id: int) -> None:
        if not self.config.summarization.enabled:
            return
        async with self._locks[chat_id]:
            messages = await self.storage.get_unsummarized_old_messages(
                chat_id,
                keep_recent=self.config.summarization.keep_recent_messages,
                limit=self.config.summarization.chunk_size,
            )
            if len(messages) < self.config.summarization.min_unsummarized_messages:
                return

            text = self._format_messages(messages)
            prompt = (
                "Сожми старый фрагмент Telegram-чата в компактное summary.\n"
                "Сохрани: факты, договорённости, локальные шутки, важные предпочтения, нерешённые вопросы.\n"
                "Не пересказывай каждую реплику. Не добавляй того, чего нет.\n\n"
                f"Фрагмент:\n{text}"
            )
            try:
                raw = await self.llm.chat(
                    [
                        {"role": "system", "content": "Ты сжимаешь историю чата для памяти бота."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.llm.summarizer_temperature,
                    num_predict=self.config.summarization.max_summary_tokens,
                )
                summary = self.llm.clean_visible_answer(raw)
                if not summary:
                    return
                message_ids = [int(m["id"]) for m in messages]
                await self.storage.save_summary(
                    chat_id,
                    start_message_id=message_ids[0],
                    end_message_id=message_ids[-1],
                    summary_text=summary,
                )
                await self.storage.mark_messages_summarized(message_ids)
                logger.info(
                    "Summarized chat_id=%s messages=%s..%s",
                    chat_id,
                    message_ids[0],
                    message_ids[-1],
                )
            except Exception:
                logger.exception("Failed to summarize chat_id=%s", chat_id)

    def _format_messages(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            author = msg.get("username") or msg.get("full_name") or f"user:{msg.get('user_id')}"
            lines.append(f"[{msg.get('created_at')}] {author}: {msg.get('text')}")
        return "\n".join(lines)
