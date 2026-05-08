from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RouterDecision:
    action: str = "ignore"  # ignore | reply | tool_reply | ask_clarify
    confidence: float = 0.0
    reason: str = "fallback"
    tone: str = "neutral"
    needs_memory: bool = False
    needs_tools: bool = False
    humor_allowed: bool = False
    max_reply_length: str = "long"  # very_short | short | medium

    @classmethod
    def from_dict(cls, data: dict) -> "RouterDecision":
        allowed_actions = {"ignore", "reply", "tool_reply", "ask_clarify"}
        allowed_tones = {"neutral", "helpful", "light_humor", "serious", "supportive"}
        allowed_lengths = {"very_short", "short", "medium"}

        action = str(data.get("action", "ignore"))
        tone = str(data.get("tone", "neutral"))
        max_reply_length = str(data.get("max_reply_length", "short"))
        return cls(
            action=action if action in allowed_actions else "ignore",
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            reason=str(data.get("reason", "no_reason"))[:500],
            tone=tone if tone in allowed_tones else "neutral",
            needs_memory=bool(data.get("needs_memory", False)),
            needs_tools=bool(data.get("needs_tools", False)),
            humor_allowed=bool(data.get("humor_allowed", False)),
            max_reply_length=max_reply_length if max_reply_length in allowed_lengths else "short",
        )


@dataclass
class ToolCallResult:
    tool_name: str
    ok: bool
    result: str
    error: str | None = None


@dataclass
class ResponseContext:
    summaries: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    recent_messages: list[str] = field(default_factory=list)
    tool_results: list[ToolCallResult] = field(default_factory=list)
