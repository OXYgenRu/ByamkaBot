from __future__ import annotations

from datetime import datetime, timezone

from bot_app.domain.events import MessageEvent


def is_continuation(event: MessageEvent, rows: list[dict], window_sec: int = 180) -> bool:
    if event.user_id is None or event.is_bot_message:
        return False
    # Replies explicitly addressed to somebody else do not belong to this dialogue.
    if event.reply_to_message_id and not event.reply_to_bot:
        target = next((m for m in rows if m.get("telegram_message_id") == event.reply_to_message_id), None)
        if not target or not target.get("is_bot"):
            return False
    # A reply delivered while this event was queued also establishes dialogue.
    prior = [m for m in rows if m.get("telegram_message_id") != event.telegram_message_id
             and (m.get("is_bot") or m.get("created_at", "") <= event.created_at)]
    index = next((i for i in range(len(prior) - 1, -1, -1) if prior[i].get("is_bot")), None)
    if index is None or len(prior) - index - 1 > 4:
        return False
    bot = prior[index]
    # Stop automatic continuation if another participant has taken over.
    if any(m.get("user_id") != event.user_id for m in prior[index + 1:] if not m.get("is_bot")):
        return False
    target_id = bot.get("reply_to_message_id")
    addressed = next((m for m in reversed(prior[:index]) if not m.get("is_bot")
                      and (target_id is None or m.get("telegram_message_id") == target_id)), None)
    if not addressed or addressed.get("user_id") != event.user_id:
        return False
    if addressed.get("created_at", "") > event.created_at:
        return False
    try:
        then = datetime.fromisoformat(bot["created_at"])
        now = datetime.fromisoformat(event.created_at)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return abs((now - then).total_seconds()) <= window_sec
    except (KeyError, ValueError):
        return False
