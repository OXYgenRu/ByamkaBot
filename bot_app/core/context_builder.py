from __future__ import annotations

from dataclasses import dataclass

from bot_app.config import Config
from bot_app.domain.decisions import RouterDecision, ToolCallResult
from bot_app.domain.events import MessageEvent
from bot_app.llm.token_budget import Budget, ContextPacker, TokenEstimator
from bot_app.memory.qdrant_memory import FactSearchResult
from bot_app.personality.interest_field import InterestProfile
from bot_app.storage.sqlite_store import SQLiteStore


@dataclass
class BuiltPrompt:
    messages: list[dict[str, str]]
    num_predict: int
    used_estimated_input_tokens: int


class ContextBuilder:
    def __init__(self, config: Config, storage: SQLiteStore) -> None:
        self.config = config
        self.storage = storage
        self.estimator = TokenEstimator()
        self.packer = ContextPacker(self.estimator)

    async def build(
            self,
            event: MessageEvent,
            decision: RouterDecision,
            facts: list[FactSearchResult],
            tool_results: list[ToolCallResult],
            behavior_facts: list[FactSearchResult] | None = None,
            interest_profile: InterestProfile | None = None,
    ) -> BuiltPrompt:
        behavior_facts = behavior_facts or []
        system_prompt = self._system_prompt(decision, behavior_facts,
                                            interest_profile)
        system_tokens = self.estimator.count(system_prompt)
        budget = Budget(
            context_window_tokens=self.config.llm.context_window_tokens,
            reserve_response_tokens=self.config.llm.reserve_response_tokens,
            reserve_thinking_tokens=self.config.llm.reserve_thinking_tokens,
            system_tokens=system_tokens,
        )

        summaries = await self.storage.get_recent_summaries(
            event.chat_id, self.config.context.summaries_limit
        )
        recent_rows = await self.storage.get_recent_messages(
            event.chat_id, self.config.context.recent_messages_limit
        )

        summary_items = [f"Summary #{i + 1}: {s}" for i, s in
                         enumerate(summaries)]
        fact_items = [f"- {f.text}" for f in
                      facts[: self.config.context.facts_limit]]
        tool_items = [
            f"[{r.tool_name}] {'OK' if r.ok else 'ERROR'}: {r.result or r.error or ''}"
            for r in tool_results[: self.config.context.tool_results_limit]
        ]
        recent_items = [self._format_message_row(row) for row in recent_rows]

        # Распределяем бюджет простым, понятным способом.
        available = budget.input_tokens_available
        current_block = f"Текущее сообщение:\n{event.author_label}: {event.text}"
        current_cost = self.estimator.count(
            current_block) + self.config.context.current_message_extra_tokens
        available = max(512, available - current_cost)

        summaries_budget = min(1200, available // 4)
        facts_budget = min(900, available // 5)
        tools_budget = min(900, available // 5)
        recent_budget = max(512,
                            available - summaries_budget - facts_budget - tools_budget)

        packed_summaries = self.packer.pack_newest_first(summary_items,
                                                         summaries_budget)
        packed_facts = self.packer.pack_newest_first(fact_items, facts_budget)
        packed_tools = self.packer.pack_newest_first(tool_items, tools_budget)
        packed_recent = self.packer.pack_newest_first(recent_items,
                                                      recent_budget)

        context_text = self._build_context_text(
            summaries=packed_summaries,
            facts=packed_facts,
            tool_results=packed_tools,
            recent_messages=packed_recent,
            current_block=current_block,
        )
        used = system_tokens + self.estimator.count(context_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_text},
        ]
        return BuiltPrompt(messages=messages, num_predict=budget.output_tokens,
                           used_estimated_input_tokens=used)

    def _system_prompt(
            self,
            decision: RouterDecision,
            behavior_facts: list[FactSearchResult] | None = None,
            interest_profile: InterestProfile | None = None,
    ) -> str:
        behavior_rules = ""
        if behavior_facts:
            rules = []
            seen = set()

            for fact in behavior_facts:
                text = fact.text.strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                rules.append(f"- {text}")

            if rules:
                behavior_rules = (
                        "\n\nОбязательные правила поведения из памяти. "
                        "Соблюдай их всегда, но не пересказывай пользователю без необходимости:\n"
                        + "\n".join(rules)
                )
        length_rule = {
            "very_short": "Ответь одной короткой фразой, максимум 12 слов.",
            "short": "Ответь коротко, максимум 25 слов. Не задавай больше одного вопроса.",
            "medium": "Ответь по делу, но без лишних вступлений. Максимум 80 слов.",
            "long": "Раскрой тему подробно, но без воды и без ассистентского тона. "
                    "Дай связное объяснение с логикой, примерами и выводом. "
                    "Не растягивай ответ ради длины. "
                    "Ориентир: 150–250 слов, если вопрос действительно сложный."

        }.get(decision.max_reply_length, "Ответь коротко, максимум 25 слов.")
        humor_rule = (
            "Если шутишь, шутка должна быть одной короткой фразой. Не объясняй её и не продолжай вопросами."
            if decision.humor_allowed
            else "Не пытайся шутить, если это не нужно."
        )
        interest_rule = ""
        if interest_profile is not None:
            interest_rule = (
                "\n\nВнутренний интерес Бямки к теме: "
                f"{interest_profile.to_prompt_text()}. "
                "Это внутренний сигнал характера, а не тема ответа. "
                "Не пересказывай эти числа пользователю. "
                "Любимые темы Бямки нельзя вставлять в ответ, если они не связаны напрямую с текущим сообщением. "
                "Если repetition_penalty или novelty_penalty высокие, не возвращайся к похожей теме и отвечай суше. "
                "Если interest высокий, можно быть чуть живее; если низкий — отвечай суше и короче."
            )
        return (
            f"{self.config.bot.personality}\n\n"
            "Стиль ответа:\n"
            "- Не будь разговорчивым.\n"
            "- Не пытайся поддерживать диалог любой ценой.\n"
            "- Не задавай больше одного вопроса.\n"
            "- Не задавай вопрос, если можно просто коротко отреагировать.\n"
            "- Не используй восторженные фразы.\n"
            "- Не используй длинные сравнения и объяснения в обычной болтовне.\n"
            "- Если сообщение пользователя короткое, ответ тоже должен быть коротким.\n"
            "- В обычной реакции допустима одна фраза без пояснений.\n"
            "- Не возвращайся к одной и той же теме или образу несколько раз подряд.\n"
            "- Внутренние интересы — это фон внимания, а не список любимых тем для ответа.\n\n"
            f"Режим ответа: tone={decision.tone}.\n"
            f"{length_rule}\n"
            f"{humor_rule}\n"
            f"{behavior_rules}"
            f"{interest_rule}\n\n"
            "Не показывай внутренние рассуждения. Не используй теги <think>.\n"
            "Если есть результаты инструментов, используй их как факты, но не пересказывай техническую кухню.\n"
            "Пиши сообщение, которое можно сразу отправить в Telegram."
        )

    def _build_context_text(
            self,
            *,
            summaries: list[str],
            facts: list[str],
            tool_results: list[str],
            recent_messages: list[str],
            current_block: str,
    ) -> str:
        parts = []
        if summaries:
            parts.append(
                "Сжатая старая история чата:\n" + "\n".join(summaries))
        if facts:
            parts.append(
                "Дополнительные факты из памяти. "
                "Используй их только если они прямо помогают ответить. "
                "Не упоминай факт просто потому, что он есть в списке:\n"
                + "\n".join(facts)
            )
        if tool_results:
            parts.append(
                "Результаты инструментов:\n" + "\n".join(tool_results))
        if recent_messages:
            parts.append("Недавние сообщения:\n" + "\n".join(recent_messages))
        parts.append(current_block)
        parts.append("Сформируй финальный ответ бота.")
        return "\n\n".join(parts)

    def _format_message_row(self, row: dict) -> str:
        author = row.get("username") or row.get(
            "full_name") or f"user:{row.get('user_id')}"
        if row.get("is_bot"):
            author = "BOT"
        return f"{author}: {row.get('text')}"
