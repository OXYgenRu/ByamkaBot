from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from bot_app.config import Config
from bot_app.llm.history_window import HistoryWindow, message_text
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
            window = HistoryWindow(self.config)
            history = await self.storage.get_active_history(chat_id)
            reasoning = self.config.llm.reserve_thinking_tokens if self.config.llm.thinking else 0
            input_budget = (self.config.llm.context_window_tokens - reasoning
                            - self.config.summarization.max_summary_tokens
                            - self.config.context.prompt_safety_tokens - 512)
            messages = window.compaction_prefix(history, max(0, input_budget))
            logger.info("history_window chat=%s tokens=%s capacity=%s trigger=%s keep=%s compact_messages=%s",
                        chat_id, sum(window.cost(m) for m in history), window.capacity,
                        int(window.capacity * self.config.context.history_compact_ratio),
                        int(window.capacity * self.config.context.history_keep_ratio), len(messages))
            if not messages:
                return

            text = self._format_messages(messages)
            prompt = (
                "Сжимаю старый фрагмент Telegram-чата в компактное summary.\n"
                "Сохраняю: факты, договорённости, локальные шутки, важные предпочтения, нерешённые вопросы.\n"
                "Не пересказываю каждую реплику. Не добавляю того, чего нет.\n\n"
                f"Фрагмент:\n{text}"
            )
            try:
                raw = await self.llm.chat(
                    [
                        {"role": "system", "content": self.config.bot.identity_prompt + "Сжимаю историю чата для своей памяти. Сохраняю имена авторов фактов и не приписываю себе чужой опыт."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.llm.summarizer_temperature,
                    num_predict=self.config.summarization.max_summary_tokens,
                    purpose="memory.summarize",
                )
                summary = self.llm.clean_visible_answer(raw)
                if not summary:
                    return
                message_ids = [int(m["id"]) for m in messages]
                await self.storage.commit_compaction(chat_id, message_ids, summary)
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
            lines.append(f"[{msg.get('created_at')}] {author}: {message_text(msg)}")
        return "\n".join(lines)
