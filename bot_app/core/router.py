from __future__ import annotations

import logging
from datetime import datetime, timezone

from bot_app.config import Config
from bot_app.core.chat_state import ChatState
from bot_app.domain.decisions import RouterDecision
from bot_app.llm.ollama_client import OllamaClient
from bot_app.llm.schemas import ROUTER_SCHEMA
from bot_app.memory.qdrant_memory import FactSearchResult

logger = logging.getLogger(__name__)


class ShouldReplyRouter:
    def __init__(self, config: Config, llm: OllamaClient) -> None:
        self.config = config
        self.llm = llm

    async def decide(self, state: ChatState) -> RouterDecision:
        event = state.event
        if event.is_bot_message:
            return RouterDecision(action="ignore", reason="own_message")

        direct = event.reply_to_bot or event.mentioned_bot
        if self._is_chat_cooling_down(state) and not direct:
            return RouterDecision(action="ignore", reason="chat_cooldown")

        if (
            state.recent_bot_message_count >= self.config.routing.max_bot_replies_in_recent_window
            and not direct
        ):
            return RouterDecision(action="ignore", reason="too_many_recent_bot_replies")

        fallback_action = "reply" if direct else "ignore"
        try:
            llm_decision = await self._llm_decide(state)
        except Exception:
            logger.exception("Router LLM failed rquid=%s", event.rquid)
            return RouterDecision(
                action=fallback_action,
                confidence=1.0 if direct else 0.0,
                reason="router_llm_failed_direct_fallback" if direct else "router_llm_failed",
                tone="helpful" if direct else "neutral",
                needs_memory=direct,
                needs_tools=False,
                humor_allowed=False,
                max_reply_length="short",
            )

        threshold = self._effective_threshold(state, direct)
        if not direct and not self.config.routing.allow_spontaneous_replies:
            return RouterDecision(action="ignore", reason="spontaneous_disabled")

        if llm_decision.action == "ignore" and direct:
            llm_decision.action = "reply"
            llm_decision.confidence = max(llm_decision.confidence, threshold)
            llm_decision.reason = f"direct_override: {llm_decision.reason}"

        if llm_decision.confidence < threshold and not direct:
            return RouterDecision(
                action="ignore",
                confidence=llm_decision.confidence,
                reason=f"below_threshold: {llm_decision.reason}",
            )
        return self._apply_interest_bias(state, llm_decision, direct)

    async def review_after_memory(
        self,
        state: ChatState,
        decision: RouterDecision,
        *,
        behavior_facts: list[FactSearchResult],
        topic_facts: list[FactSearchResult],
    ) -> RouterDecision:
        """Second routing pass after memory retrieval.

        This lets the bot decide that a reply is no longer worth sending after it sees
        that the retrieved facts are irrelevant, repetitive or too thin.
        """
        if not self.config.routing.post_memory_review_enabled:
            return decision
        if decision.action == "ignore":
            return decision

        event = state.event
        direct = event.reply_to_bot or event.mentioned_bot
        if direct and not self.config.routing.allow_post_memory_ignore_direct:
            return decision

        facts_text = self._format_facts(topic_facts, title="Обычные факты")
        behavior_text = self._format_facts(behavior_facts, title="Правила поведения")
        prompt = (
            "Ты второй router Telegram-бота после подтягивания памяти.\n"
            "Первый router уже решил, что бот может ответить. Теперь проверь, действительно ли ответ нужен.\n"
            "Если факты из памяти не помогают, реплика короткая и не требует реакции, или бот будет выглядеть навязчиво — выбери ignore.\n"
            "Если отвечать уместно, оставь reply/tool_reply/ask_clarify и уточни тон.\n"
            "Не заставляй бота отвечать просто потому, что память нашла какие-то факты.\n\n"
            f"Исходное решение: action={decision.action}, confidence={decision.confidence:.2f}, reason={decision.reason}\n"
            f"reply_to_bot={event.reply_to_bot}\nmentioned_bot={event.mentioned_bot}\n"
            f"interest_profile={state.format_interest_profile()}\n\n"
            f"Недавние сообщения:\n{state.format_recent_messages()}\n\n"
            f"{behavior_text}\n\n{facts_text}\n\n"
            f"Текущее сообщение:\n{event.author_label}: {event.text}"
        )
        try:
            data = await self.llm.structured_json(
                [
                    {"role": "system", "content": "Верни строго JSON по схеме. Не пиши объяснений вне JSON."},
                    {"role": "user", "content": prompt},
                ],
                ROUTER_SCHEMA,
                temperature=self.config.llm.router_temperature,
                num_predict=self.config.routing.post_memory_review_num_predict,
            )
            reviewed = RouterDecision.from_dict(data)
        except Exception:
            logger.exception("Post-memory router failed rquid=%s", event.rquid)
            return decision

        if reviewed.action == "ignore" and reviewed.confidence >= self.config.routing.post_memory_ignore_min_confidence:
            reviewed.reason = f"post_memory_ignore: {reviewed.reason}"
            return reviewed
        if reviewed.action != "ignore":
            # Keep tool/memory intent from either pass; the second pass may refine tone/length.
            reviewed.needs_memory = reviewed.needs_memory or decision.needs_memory
            reviewed.needs_tools = reviewed.needs_tools or decision.needs_tools
            reviewed.confidence = max(reviewed.confidence, decision.confidence)
            reviewed.reason = f"post_memory_review: {reviewed.reason}"
            return reviewed
        return decision

    async def _llm_decide(self, state: ChatState) -> RouterDecision:
        event = state.event
        prompt = (
            "Ты router Telegram-бота, который является участником группового чата.\n"
            "Реши, надо ли боту входить в диалог сейчас.\n"
            "Бот не должен отвечать на всё подряд. Лучше молчать, если реплика не нужна.\n"
            "Отвечать стоит, если: бота прямо спросили, упомянули, нужен полезный ответ, есть удачный момент для короткой реплики.\n"
            "У бота есть внутренний interest_profile. Это не приказ отвечать, а слабый bias характера.\n"
            "Если reply/curiosity/final высокие, можно немного повысить шанс короткой реакции.\n"
            "Если repetition_penalty или novelty_penalty высокие, не поощряй повтор той же темы.\n"
            "Если интерес низкий, лучше молчать, если нет прямого обращения.\n"
            "Юмор разрешай только если он контекстный.\n"
            "needs_tools=true только если нужен калькулятор или поиск в памяти.\n\n"
            f"reply_to_bot={event.reply_to_bot}\n"
            f"mentioned_bot={event.mentioned_bot}\n"
            f"recent_bot_message_count={state.recent_bot_message_count}\n"
            f"interest_profile={state.format_interest_profile()}\n\n"
            f"Недавние сообщения:\n{state.format_recent_messages()}\n\n"
            f"Текущее сообщение:\n{event.author_label}: {event.text}"
        )
        data = await self.llm.structured_json(
            [
                {"role": "system", "content": "Верни строго JSON по схеме. Не пиши объяснений вне JSON."},
                {"role": "user", "content": prompt},
            ],
            ROUTER_SCHEMA,
            temperature=self.config.llm.router_temperature,
            num_predict=512,
        )
        return RouterDecision.from_dict(data)

    def _effective_threshold(self, state: ChatState, direct: bool) -> float:
        base = (
            self.config.routing.direct_reply_min_confidence
            if direct
            else self.config.routing.spontaneous_reply_min_confidence
        )
        profile = state.interest_profile
        if direct or profile is None:
            return base
        if profile.repetition_penalty >= self.config.personality_engine.repetition_bias_block_threshold:
            return base
        if profile.reply >= self.config.personality_engine.boost_reply_threshold:
            return max(0.05, base - self.config.personality_engine.router_threshold_discount)
        return base

    def _apply_interest_bias(self, state: ChatState, decision: RouterDecision, direct: bool) -> RouterDecision:
        profile = state.interest_profile
        if direct or profile is None or decision.action == "ignore":
            return decision
        if profile.repetition_penalty >= self.config.personality_engine.repetition_bias_block_threshold:
            decision.reason = f"interest_cooldown: {decision.reason}"
            return decision
        if profile.reply >= self.config.personality_engine.spontaneous_reply_threshold:
            decision.confidence = max(decision.confidence, self.config.routing.spontaneous_reply_min_confidence)
            decision.reason = f"interest_boost: {decision.reason}"
        if profile.humor >= 0.72 and decision.tone == "neutral":
            decision.tone = "light_humor"
            decision.humor_allowed = True
            decision.max_reply_length = "very_short"
        return decision

    def _is_chat_cooling_down(self, state: ChatState) -> bool:
        if not state.last_bot_message_created_at:
            return False
        try:
            last = datetime.fromisoformat(state.last_bot_message_created_at)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            diff = datetime.now(timezone.utc) - last
            return diff.total_seconds() < self.config.routing.chat_cooldown_seconds
        except ValueError:
            return False

    @staticmethod
    def _format_facts(facts: list[FactSearchResult], *, title: str) -> str:
        if not facts:
            return f"{title}: нет"
        lines = [f"{title}:"]
        for fact in facts:
            lines.append(f"- {fact.text} (score={fact.score:.2f}, source={fact.source})")
        return "\n".join(lines)
