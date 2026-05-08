from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from bot_app.config import Config
from bot_app.core.chat_state import ChatStateBuilder
from bot_app.core.context_builder import ContextBuilder
from bot_app.core.responder import Responder
from bot_app.core.router import ShouldReplyRouter
from bot_app.domain.events import MessageEvent, utcnow_iso
from bot_app.memory.auto_extractor import AutoMemoryExtractor
from bot_app.memory.qdrant_memory import QdrantMemory
from bot_app.memory.summarizer import ChatSummarizer
from bot_app.personality.interest_engine import PersonalityInterestEngine
from bot_app.storage.sqlite_store import SQLiteStore
from bot_app.tools.planner import ToolPlanner

logger = logging.getLogger(__name__)


class BotOrchestrator:
    def __init__(
        self,
        config: Config,
        storage: SQLiteStore,
        memory: QdrantMemory,
        summarizer: ChatSummarizer,
        state_builder: ChatStateBuilder,
        router: ShouldReplyRouter,
        tool_planner: ToolPlanner,
        context_builder: ContextBuilder,
        responder: Responder,
        memory_extractor: AutoMemoryExtractor | None = None,
        interest_engine: PersonalityInterestEngine | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.memory = memory
        self.summarizer = summarizer
        self.state_builder = state_builder
        self.router = router
        self.tool_planner = tool_planner
        self.context_builder = context_builder
        self.responder = responder
        self.memory_extractor = memory_extractor
        self.interest_engine = interest_engine
        self._chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def handle_message(self, event: MessageEvent) -> str | None:
        async with self._chat_locks[event.chat_id]:
            message_db_id = await self.storage.save_message(event)

            interest_profile = None
            if self.interest_engine is not None:
                try:
                    interest_profile = await self.interest_engine.evaluate_event(
                        event,
                        message_db_id=message_db_id,
                    )
                except Exception:
                    logger.exception("interest_engine failed rquid=%s chat=%s", event.rquid, event.chat_id)

            if self.memory_extractor is not None:
                try:
                    await self.memory_extractor.maybe_extract_and_store(
                        event,
                        interest_profile=interest_profile,
                    )
                except Exception:
                    logger.exception("auto_memory failed rquid=%s chat=%s", event.rquid, event.chat_id)

            await self.summarizer.maybe_summarize(event.chat_id)

            state = await self.state_builder.build(
                event,
                recent_limit=self.config.context.recent_messages_limit,
                recent_window=self.config.routing.recent_window_messages,
                interest_profile=interest_profile,
            )
            decision = await self.router.decide(state)
            logger.info(
                "router rquid=%s chat=%s action=%s confidence=%.2f reason=%s",
                event.rquid,
                event.chat_id,
                decision.action,
                decision.confidence,
                decision.reason,
            )
            if decision.action == "ignore":
                return None

            behavior_facts, topic_facts = await self._load_memory_for_response(event, decision, interest_profile)

            decision = await self.router.review_after_memory(
                state,
                decision,
                behavior_facts=behavior_facts,
                topic_facts=topic_facts,
            )
            logger.info(
                "post_memory_router rquid=%s chat=%s action=%s confidence=%.2f reason=%s",
                event.rquid,
                event.chat_id,
                decision.action,
                decision.confidence,
                decision.reason,
            )
            if decision.action == "ignore":
                return None

            recent_text = state.format_recent_messages()
            tool_results = await self.tool_planner.maybe_run_tools(event, recent_text, decision)
            prompt = await self.context_builder.build(
                event,
                decision,
                topic_facts,
                tool_results,
                behavior_facts=behavior_facts,
                interest_profile=interest_profile,
            )
            logger.info(
                "prompt rquid=%s estimated_input_tokens=%s num_predict=%s",
                event.rquid,
                prompt.used_estimated_input_tokens,
                prompt.num_predict,
            )
            answer = await self.responder.generate(prompt)
            if not answer:
                return None

            bot_event = MessageEvent(
                rquid=event.rquid,
                chat_id=event.chat_id,
                telegram_message_id=-int(asyncio.get_running_loop().time() * 1000000),
                user_id=None,
                username="BOT",
                full_name=self.config.bot.display_name,
                text=answer,
                created_at=utcnow_iso(),
                is_bot_message=True,
            )
            await self.storage.save_message(bot_event)
            return answer

    async def _load_memory_for_response(self, event: MessageEvent, decision, interest_profile):
        behavior_facts = []
        topic_facts = []
        should_search = decision.needs_memory or decision.action in {"reply", "tool_reply", "ask_clarify"}
        if interest_profile is not None and interest_profile.memory >= self.config.personality_engine.memory_interest_threshold:
            should_search = True
        if not should_search:
            return behavior_facts, topic_facts
        try:
            behavior_facts = await self.memory.search(
                "правила поведения бота, стиль ответа, предпочтения пользователя, запреты, эмодзи, смайлики, emoji",
                chat_id=event.chat_id,
                limit=4,
            )

            topic_facts = await self.memory.search(
                event.text,
                chat_id=event.chat_id,
                limit=self.config.context.facts_limit,
            )
        except Exception:
            logger.exception("Memory search failed rquid=%s", event.rquid)

        behavior_facts = [
            f for f in behavior_facts
            if f.score >= self.config.context.behavior_facts_min_score
        ]
        topic_facts = [
            f for f in topic_facts
            if f.score >= self.config.context.facts_min_score
        ]
        return behavior_facts, topic_facts

    async def render_interest_fingerprint(self):
        if self.interest_engine is None:
            return None
        return await self.interest_engine.render_fingerprint()

    async def format_top_interests(self, chat_id: int, *, limit: int = 8) -> str:
        if self.interest_engine is None:
            return "Движок интересов отключён."
        return await self.interest_engine.top_interests_text(chat_id, limit=limit)
