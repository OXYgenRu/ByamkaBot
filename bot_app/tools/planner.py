from __future__ import annotations

import asyncio
import json
import logging

from bot_app.config import Config
from bot_app.domain.decisions import RouterDecision, ToolCallResult
from bot_app.domain.events import MessageEvent
from bot_app.llm.ollama_client import OllamaClient
from bot_app.llm.schemas import TOOL_PLAN_SCHEMA
from bot_app.storage.sqlite_store import SQLiteStore
from bot_app.tools.base import ToolRunContext
from bot_app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolPlanner:
    def __init__(
        self,
        config: Config,
        llm: OllamaClient,
        registry: ToolRegistry,
        storage: SQLiteStore,
    ) -> None:
        self.config = config
        self.llm = llm
        self.registry = registry
        self.storage = storage

    async def maybe_run_tools(
        self,
        event: MessageEvent,
        recent_messages_text: str,
        decision: RouterDecision,
    ) -> list[ToolCallResult]:
        if not self.config.tools.enabled or not self.config.tools.planner_enabled:
            return []
        if not decision.needs_tools and decision.action != "tool_reply":
            return []

        specs = self.registry.specs_for_prompt(
            allow_side_effects=self.config.tools.allow_side_effect_tools
        )
        if not specs:
            return []

        prompt = (
            "Выбираю инструменты, которые реально нужны для моего ответа.\n"
            "Если инструменты не нужны, возвращаю пустой массив tool_calls.\n"
            "Не вызываю инструменты просто из любопытства.\n\n"
            f"Доступные инструменты:\n{json.dumps(specs, ensure_ascii=False, indent=2)}\n\n"
            f"Недавний контекст:\n{recent_messages_text}\n\n"
            f"Текущее сообщение:\n{event.author_label}: {event.text}"
        )
        try:
            data = await self.llm.structured_json(
                [
                    {"role": "system", "content": self.config.bot.identity_prompt + "Планирую использование безопасных инструментов. Отвечаю строго JSON."},
                    {"role": "user", "content": prompt},
                ],
                TOOL_PLAN_SCHEMA,
                temperature=0.1,
                num_predict=768,
                purpose="tools.plan",
            )
        except Exception:
            logger.exception("Tool planner failed rquid=%s", event.rquid)
            return []

        calls = data.get("tool_calls") or []
        if not isinstance(calls, list):
            return []

        results: list[ToolCallResult] = []
        ctx = ToolRunContext(rquid=event.rquid, chat_id=event.chat_id, user_id=event.user_id)
        for call in calls[: self.config.tools.max_tool_calls]:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name", ""))
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            tool = self.registry.get(tool_name)
            if tool is None:
                continue
            if tool.spec.side_effect and not self.config.tools.allow_side_effect_tools:
                continue
            try:
                result = await asyncio.wait_for(
                    tool.run(args, ctx),
                    timeout=tool.spec.timeout_sec,
                )
                await self.storage.save_tool_event(
                    rquid=event.rquid,
                    chat_id=event.chat_id,
                    tool_name=tool_name,
                    args_json=json.dumps(args, ensure_ascii=False),
                    ok=result.ok,
                    result_text=result.result,
                    error_text=result.error,
                )
                results.append(
                    ToolCallResult(
                        tool_name=tool_name,
                        ok=result.ok,
                        result=result.result,
                        error=result.error,
                    )
                )
            except Exception as exc:
                logger.exception("Tool run failed tool=%s rquid=%s", tool_name, event.rquid)
                results.append(ToolCallResult(tool_name=tool_name, ok=False, result="", error=str(exc)))
        return results
