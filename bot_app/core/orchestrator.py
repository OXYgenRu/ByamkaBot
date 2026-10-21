from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable
from collections import defaultdict
from dataclasses import replace

from bot_app.config import Config
from bot_app.llm.model_logging import model_log_context
from bot_app.llm.history_window import HistoryWindow
from bot_app.core.chat_state import ChatStateBuilder
from bot_app.core.dialogue import is_continuation
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
        image_memory=None,
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
        self.image_memory = image_memory
        self._chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._latest_message: dict[int, int] = {}
        self._last_message_arrival: dict[int, float] = {}
        self._foreground_count = 0
        self._last_activity = 0.0
        self._maintenance_queue: asyncio.Queue = asyncio.Queue()
        self._maintenance_task: asyncio.Task | None = None
        self._maintenance_pending: dict[int, list[MessageEvent]] = {}
        self._maintenance_scheduled: set[int] = set()

    async def handle_message(
        self, event: MessageEvent,
        send_answer: Callable[[str], Awaitable[int]] | None = None,
    ) -> str | None:
        if event.is_bot_message:
            return None
        started = time.monotonic()
        self._foreground_count += 1
        # Register arrival before the first await, including while SQLite is committing.
        self._latest_message[event.chat_id] = max(
            event.telegram_message_id, self._latest_message.get(event.chat_id, 0))
        inserted = False
        try:
            # Ingestion must not wait for the previous LLM response.
            message_db_id, inserted = await self.storage.save_message_with_status(event)
            if not inserted:
                logger.info("duplicate_event rquid=%s message=%s", event.rquid, event.telegram_message_id)
                return None
            self._last_message_arrival[event.chat_id] = time.monotonic()
            logger.info("message_queued rquid=%s chat=%s message=%s locked=%s",
                        event.rquid, event.chat_id, event.telegram_message_id,
                        self._chat_locks[event.chat_id].locked())
            async with self._chat_locks[event.chat_id]:
                await self._wait_for_message_pause(event.chat_id)
                logger.info("message_start rquid=%s message=%s queue_sec=%.3f",
                            event.rquid, event.telegram_message_id, time.monotonic() - started)
                # Coalesce only pending ordinary messages. Once processing starts,
                # newer arrivals do not revoke a reply selected by the router.
                if self._is_pending_superseded(event):
                    return None
                recent = await self.storage.get_messages_for_reply(
                    event.chat_id, event.telegram_message_id,
                    None, active_only=True)
                if self._is_pending_superseded(event):
                    return None
                if self.config.routing.allow_spontaneous_replies and is_continuation(
                    event, recent, self.config.routing.followup_window_seconds
                ):
                    event = replace(event, dialogue_continuation=True)
                with model_log_context(event.rquid, event.chat_id, "foreground"):
                    return await self._respond(event, message_db_id, send_answer, recent)
        finally:
            self._foreground_count -= 1
            self._last_activity = time.monotonic()
            logger.info("message_done rquid=%s elapsed_sec=%.3f", event.rquid, self._last_activity - started)
            if inserted:
                self._maintenance_pending.setdefault(event.chat_id, []).append(event)
                if event.chat_id not in self._maintenance_scheduled:
                    self._maintenance_scheduled.add(event.chat_id)
                    self._maintenance_queue.put_nowait(event.chat_id)
                if self._maintenance_task is None or self._maintenance_task.done():
                    self._maintenance_task = asyncio.create_task(self._run_maintenance())

    async def _wait_for_message_pause(self, chat_id: int) -> None:
        delay = max(0.0, self.config.routing.message_debounce_seconds)
        while delay:
            remaining = self._last_message_arrival.get(chat_id, 0.0) + delay - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    def _is_pending_superseded(self, event: MessageEvent) -> bool:
        stale = (not (event.mentioned_bot or event.reply_to_bot)
                 and self._latest_message.get(event.chat_id, event.telegram_message_id) > event.telegram_message_id)
        if stale:
            logger.info("superseded_message rquid=%s message=%s phase=pending", event.rquid,
                        event.telegram_message_id)
        return stale

    async def _wait_for_idle(self) -> None:
        while self._foreground_count or time.monotonic() - self._last_activity < 5:
            await asyncio.sleep(0.25)

    async def _run_maintenance(self) -> None:
        # One queued/in-flight slot per chat; retain each event for fact extraction.
        while True:
            chat_id = await self._maintenance_queue.get()
            try:
                await self._wait_for_idle()
                events = self._maintenance_pending.pop(chat_id, [])
                if not events:
                    continue
                if self.image_memory is not None:
                    for event in events:
                        if event.image_paths:
                            await self._wait_for_idle()
                            with model_log_context(event.rquid, chat_id, "background"):
                                await self.image_memory.describe_event(event)
                # Compact once per batch, not once for every queued message.
                with model_log_context(events[-1].rquid, chat_id, "background"):
                    try:
                        await self.summarizer.maybe_summarize(chat_id)
                    except Exception:
                        logger.exception("summarizer failed chat=%s", chat_id)
                if self.memory_extractor is not None:
                    for event in events:
                        await self._wait_for_idle()
                        with model_log_context(event.rquid, chat_id, "background"):
                            try:
                                await self.memory_extractor.maybe_extract_and_store(event)
                            except Exception:
                                logger.exception("auto_memory failed rquid=%s", event.rquid)
            finally:
                self._maintenance_scheduled.discard(chat_id)
                if self._maintenance_pending.get(chat_id):
                    self._maintenance_scheduled.add(chat_id)
                    self._maintenance_queue.put_nowait(chat_id)
                self._maintenance_queue.task_done()

    async def close(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)

    async def _respond(
        self, event: MessageEvent, message_db_id: int,
        send_answer: Callable[[str], Awaitable[int]] | None,
        recent_messages: list[dict],
    ) -> str | None:
        interest_profile = None
        if self.interest_engine is not None:
            try:
                interest_profile = await self.interest_engine.evaluate_event(
                    event,
                    message_db_id=message_db_id,
                )
            except Exception:
                logger.exception("interest_engine failed rquid=%s chat=%s", event.rquid, event.chat_id)

        state = await self.state_builder.build(
            event,
            recent_limit=self.config.context.recent_messages_limit,
            recent_window=self.config.routing.recent_window_messages,
            interest_profile=interest_profile,
            recent_messages=HistoryWindow(self.config).newest(
                recent_messages[-self.config.routing.recent_window_messages:],
                self.config.context.router_history_tokens),
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
            recent_messages=recent_messages,
        )
        logger.info(
            "prompt rquid=%s target=%s history_messages=%s estimated_input_tokens=%s num_predict=%s",
            event.rquid,
            prompt.target_message_id,
            len(recent_messages),
            prompt.used_estimated_input_tokens,
            prompt.num_predict,
        )
        answer = await self.responder.generate(prompt)
        if prompt.target_message_id != event.telegram_message_id:
            raise RuntimeError("Response target differs from the context target")
        if not answer:
            return None

        # Send inside the chat lock and persist only a successful delivery.
        if send_answer is None:
            return answer
        telegram_id = await send_answer(answer)
        logger.info("response_sent rquid=%s chat=%s target=%s sent_message=%s",
                    event.rquid, event.chat_id, event.telegram_message_id, telegram_id)
        bot_event = MessageEvent(
            rquid=event.rquid, chat_id=event.chat_id,
            telegram_message_id=telegram_id, user_id=None,
            username="BOT", full_name=self.config.bot.display_name,
            text=answer, created_at=utcnow_iso(), is_bot_message=True,
            reply_to_message_id=event.telegram_message_id,
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
