from __future__ import annotations

from typing import Any

from bot_app.memory.qdrant_memory import QdrantMemory
from bot_app.tools.base import ToolRunContext, ToolRunResult, ToolSpec


class MemorySearchTool:
    def __init__(self, memory: QdrantMemory, default_limit: int = 5) -> None:
        self.memory = memory
        self.default_limit = default_limit

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory_search",
            description="Ищет сохранённые факты и устойчивые знания по текущему чату.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            side_effect=False,
        )

    async def run(self, args: dict[str, Any], ctx: ToolRunContext) -> ToolRunResult:
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", self.default_limit))
        if not query:
            return ToolRunResult(ok=False, result="", error="empty query")
        results = await self.memory.search(query, chat_id=ctx.chat_id, limit=limit)
        if not results:
            return ToolRunResult(ok=True, result="Ничего релевантного в памяти не найдено.")
        lines = [f"- {r.text} (score={r.score:.3f})" for r in results]
        return ToolRunResult(ok=True, result="\n".join(lines))
