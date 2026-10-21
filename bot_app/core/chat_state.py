from __future__ import annotations

from dataclasses import dataclass

from bot_app.domain.events import MessageEvent
from bot_app.llm.history_window import message_text
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
            lines.append(f"[id={msg.get('telegram_message_id')} at={msg.get('created_at')} reply_to={msg.get('reply_to_message_id')}] {author}: {message_text(msg)}")
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
        recent_messages: list[dict] | None = None,
    ) -> ChatState:
        if recent_messages is None:
            recent_messages = await self.storage.get_messages_for_reply(
                event.chat_id, event.telegram_message_id, max(recent_limit, recent_window))
        recent_bot_count = sum(bool(row.get("is_bot")) for row in recent_messages[-recent_window:])
        last_bot_at = next((row["created_at"] for row in reversed(recent_messages) if row.get("is_bot")), None)
        return ChatState(
            event=event,
            recent_messages=recent_messages,
            recent_bot_message_count=recent_bot_count,
            last_bot_message_created_at=last_bot_at,
            interest_profile=interest_profile,
        )
