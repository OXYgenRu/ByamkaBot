from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MessageEvent:
    rquid: str
    chat_id: int
    telegram_message_id: int
    user_id: int | None
    username: str | None
    full_name: str | None
    text: str
    created_at: str
    reply_to_message_id: int | None = None
    reply_to_bot: bool = False
    mentioned_bot: bool = False
    is_bot_message: bool = False
    dialogue_continuation: bool = False
    image_paths: tuple[str, ...] = ()

    @property
    def author_label(self) -> str:
        if self.username:
            return f"@{self.username}"
        if self.full_name:
            return self.full_name
        if self.user_id is not None:
            return f"user:{self.user_id}"
        return "unknown"
