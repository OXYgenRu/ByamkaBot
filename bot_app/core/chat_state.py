from __future__ import annotations

from dataclasses import dataclass

from bot_app.domain.events import MessageEvent
from bot_app.personality.interest_field import InterestProfile
from bot_app.storage.sqlite_store import SQLiteStore


@dataclass
class ChatState:
    event: MessageEvent
    recent_messages: list[dict]
    recent_bot_message_count: int
    last_bot_message_created_at: str | None
    interest_profile: InterestProfile | None = None

    def format_recent_messages(self) -> str:
        lines = []
        for msg in self.recent_messages:
            author = msg.get("username") or msg.get("full_name") or f"user:{msg.get('user_id')}"
            if msg.get("is_bot"):
                author = "BOT"
            lines.append(f"{author}: {msg.get('text')}")
        return "\n".join(lines)

    def format_interest_profile(self) -> str:
        if self.interest_profile is None:
            return "interest_profile=disabled"
        return self.interest_profile.to_prompt_text()


class ChatStateBuilder:
    def __init__(self, storage: SQLiteStore) -> None:
        self.storage = storage

    async def build(
        self,
        event: MessageEvent,
        *,
        recent_limit: int,
        recent_window: int,
        interest_profile: InterestProfile | None = None,
    ) -> ChatState:
        recent_messages = await self.storage.get_recent_messages(event.chat_id, recent_limit)
        recent_bot_count = await self.storage.get_recent_bot_message_count(event.chat_id, recent_window)
        last_bot_at = await self.storage.get_last_bot_message_created_at(event.chat_id)
        return ChatState(
            event=event,
            recent_messages=recent_messages,
            recent_bot_message_count=recent_bot_count,
            last_bot_message_created_at=last_bot_at,
            interest_profile=interest_profile,
        )
