"""Shared, conservative token accounting for raw conversation history."""
from __future__ import annotations

import json
from typing import Any

from bot_app.config import Config

from bot_app.llm.token_budget import TokenEstimator


def message_text(row: dict[str, Any]) -> str:
    text = str(row.get('text', ''))
    images = row.get('images') or []
    for image in images:
        description = image.get('description')
        text += ("\n[Описание изображения моделью, возможны ошибки: " + description + "]"
                 if description else "\n[Изображение прикреплено; текстового описания пока нет.]")
    return text


def message_content(row: dict[str, Any]) -> str:
    if row.get('is_bot'):
        return message_text(row)
    return json.dumps({
        'author': row.get('full_name'), 'user_id': row.get('user_id'),
        'username': row.get('username'), 'message_id': row.get('telegram_message_id'),
        'reply_to': row.get('reply_to_message_id'), 'text': message_text(row),
    }, ensure_ascii=False)


class HistoryWindow:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.estimator = TokenEstimator()

    def cost(self, row: dict[str, Any]) -> int:
        return self.estimator.count(message_content(row)) + 12

    @property
    def capacity(self) -> int:
        cfg = self.config
        examples = min(cfg.context.examples_max_tokens, sum(
            self.estimator.count(m['content']) + 12
            for d in cfg.bot.dialogue_examples for m in d))
        output = cfg.llm.reserve_response_tokens
        if cfg.llm.thinking:
            output += cfg.llm.reserve_thinking_tokens
        # Conservative envelopes; ContextBuilder also enforces the actual final budget.
        overhead = (examples + 1024 + max(1024, cfg.context.current_message_extra_tokens)
                    + min(3000, cfg.llm.context_window_tokens // 8)
                    + cfg.context.prompt_safety_tokens)
        return max(256, min(cfg.context.history_max_tokens,
                           cfg.llm.context_window_tokens - output - overhead))

    def newest(self, rows: list[dict[str, Any]], budget: int | None = None) -> list[dict[str, Any]]:
        available = self.capacity if budget is None else max(0, budget)
        selected = []
        for row in reversed(rows):
            cost = self.cost(row)
            if cost > available:
                break
            selected.append(row)
            available -= cost
        return list(reversed(selected))

    def compaction_prefix(self, rows: list[dict[str, Any]], input_budget: int) -> list[dict[str, Any]]:
        """Compact an old contiguous prefix, leaving the newest low-water window."""
        if len(rows) < 2:
            return []
        total = sum(self.cost(row) for row in rows)
        if total < int(self.capacity * self.config.context.history_compact_ratio):
            return []
        keep_budget = int(self.capacity * self.config.context.history_keep_ratio)
        # Always protect the newest message, even if unusually large.
        keep = 1
        used = self.cost(rows[-1])
        for row in reversed(rows[:-1]):
            cost = self.cost(row)
            if used + cost > keep_budget:
                break
            used += cost
            keep += 1
        selected = []
        used = 0
        for row in rows[:-keep]:
            # Include source metadata used by the unchanged summarizer prompt.
            cost = max(self.cost(row), self.estimator.count(str(row.get('text', '')))
                       + self.estimator.count(str(row.get('username') or row.get('full_name') or '')) + 48)
            if used + cost > input_budget:
                break
            selected.append(row)
            used += cost
        return selected
