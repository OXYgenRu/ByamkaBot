from __future__ import annotations

from dataclasses import dataclass, field

from bot_app.config import Config
from bot_app.memory.qdrant_memory import QdrantMemory
from bot_app.tools.base import BaseTool
from bot_app.tools.calculator import SafeCalculatorTool
from bot_app.tools.memory_search import MemorySearchTool


@dataclass
class ToolRegistry:
    tools: dict[str, BaseTool] = field(default_factory=dict)

    def register(self, tool: BaseTool) -> None:
        self.tools[tool.spec.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self.tools.get(name)

    def specs_for_prompt(self, *, allow_side_effects: bool) -> list[dict]:
        specs = []
        for tool in self.tools.values():
            spec = tool.spec
            if spec.side_effect and not allow_side_effects:
                continue
            specs.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "side_effect": spec.side_effect,
                }
            )
        return specs


def build_default_tool_registry(config: Config, memory: QdrantMemory) -> ToolRegistry:
    registry = ToolRegistry()
    if config.tools.calculator_enabled:
        registry.register(SafeCalculatorTool())
    if config.tools.memory_search_enabled:
        registry.register(MemorySearchTool(memory, default_limit=config.vector_db.search_limit))
    return registry
