from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    side_effect: bool = False
    timeout_sec: int = 15


@dataclass
class ToolRunContext:
    rquid: str
    chat_id: int
    user_id: int | None


@dataclass
class ToolRunResult:
    ok: bool
    result: str
    error: str | None = None


class BaseTool(Protocol):
    @property
    def spec(self) -> ToolSpec:
        ...

    async def run(self, args: dict[str, Any], ctx: ToolRunContext) -> ToolRunResult:
        ...
