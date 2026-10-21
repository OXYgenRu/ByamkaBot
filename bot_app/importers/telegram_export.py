from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from bot_app.config import Config
from bot_app.domain.events import MessageEvent, utcnow_iso
from bot_app.llm.ollama_client import OllamaClient
from bot_app.personality.interest_engine import PersonalityInterestEngine
from bot_app.storage.sqlite_store import SQLiteStore

if TYPE_CHECKING:
    from bot_app.memory.qdrant_memory import QdrantMemory

logger = logging.getLogger(__name__)


IMPORT_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "chat_rule",
                            "user_preference",
                            "participant_preference",
                            "project_fact",
                            "decision",
                            "running_joke",
                            "local_topic",
                            "social_pattern",
                            "other",
                        ],
                    },
                },
                "required": ["text", "confidence", "kind"],
            },
        }
    },
    "required": ["facts"],
}


@dataclass
class ImportedMessage:
    event: MessageEvent
    source_message_id: int
    message_db_id: int | None = None
    inserted: bool = False


@dataclass
class ImportStats:
    source_path: str
    target_chat_id: int
    source_chat_name: str | None = None
    source_chat_id: int | None = None
    total_json_messages: int = 0
    parsed_text_messages: int = 0
    skipped_service: int = 0
    skipped_empty: int = 0
    imported_messages: int = 0
    duplicate_messages: int = 0
    summaries_created: int = 0
    empty_summaries: int = 0
    facts_saved: int = 0
    duplicate_facts: int = 0
    facts_by_kind: Counter[str] = field(default_factory=Counter)
    interest_messages_processed: int = 0
    interest_messages_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    before_counts: dict[str, int] = field(default_factory=dict)
    after_counts: dict[str, int] = field(default_factory=dict)
    top_interests: list[dict[str, Any]] = field(default_factory=list)
    recent_facts: list[dict[str, Any]] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "Telegram import statistics",
            f"source_path: {self.source_path}",
            f"source_chat: {self.source_chat_name or '-'} ({self.source_chat_id or '-'})",
            f"target_chat_id: {self.target_chat_id}",
            "",
            "Messages:",
            f"- total_json_messages: {self.total_json_messages}",
            f"- parsed_text_messages: {self.parsed_text_messages}",
            f"- imported_messages: {self.imported_messages}",
            f"- duplicate_messages: {self.duplicate_messages}",
            f"- skipped_service: {self.skipped_service}",
            f"- skipped_empty: {self.skipped_empty}",
            "",
            "Memory:",
            f"- summaries_created: {self.summaries_created}",
            f"- empty_summaries: {self.empty_summaries}",
            f"- facts_saved: {self.facts_saved}",
            f"- duplicate_facts: {self.duplicate_facts}",
        ]
        if self.facts_by_kind:
            lines.append("- facts_by_kind:")
            for kind, count in self.facts_by_kind.most_common():
                lines.append(f"  - {kind}: {count}")
        lines.extend([
            "",
            "Interests:",
            f"- interest_messages_processed: {self.interest_messages_processed}",
            f"- interest_messages_skipped: {self.interest_messages_skipped}",
        ])
        if self.before_counts or self.after_counts:
            lines.append("")
            lines.append("DB counters:")
            keys = sorted(set(self.before_counts) | set(self.after_counts))
            for key in keys:
                before = self.before_counts.get(key, 0)
                after = self.after_counts.get(key, 0)
                lines.append(f"- {key}: {before} -> {after} (delta {after - before:+d})")
        if self.top_interests:
            lines.append("")
            lines.append("Top learned interests:")
            for row in self.top_interests[:10]:
                lines.append(
                    "- "
                    f"{row.get('topic_text')} "
                    f"(interest={float(row.get('learned_interest', 0.0)):.2f}, "
                    f"seen={int(row.get('seen_count', 0))})"
                )
        if self.recent_facts:
            lines.append("")
            lines.append("Recent saved facts:")
            for row in self.recent_facts[:10]:
                lines.append(f"- [{row.get('source')}] {row.get('fact_text')}")
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(f"- {err}" for err in self.errors)
        return "\n".join(lines)


class TelegramExportImporter:
    """Offline importer for Telegram Desktop JSON exports.

    The importer turns a JSON history into bot memory:
    messages, summaries, facts/rules/jokes, learned interests and local topics.
    """

    def __init__(
        self,
        *,
        config: Config,
        storage: SQLiteStore,
        llm: OllamaClient,
        memory: QdrantMemory | None = None,
        interest_engine: PersonalityInterestEngine | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm
        self.memory = memory
        self.interest_engine = interest_engine
        self.dry_run = dry_run

    async def import_file(
        self,
        *,
        path: str | Path,
        target_chat_id: int,
        source_chat_id: int | None = None,
        source_chat_name: str | None = None,
        import_messages: bool | None = None,
        build_summaries: bool | None = None,
        extract_facts: bool | None = None,
        update_interests: bool | None = None,
    ) -> ImportStats:
        cfg = self.config.telegram_import
        path = Path(path)
        root = json.loads(path.read_text(encoding="utf-8"))
        chat = self._select_chat(root, source_chat_id=source_chat_id, source_chat_name=source_chat_name)
        raw_messages = chat.get("messages") or []
        if not isinstance(raw_messages, list):
            raise ValueError("Selected Telegram export chat has no messages list")

        stats = ImportStats(
            source_path=str(path),
            target_chat_id=target_chat_id,
            source_chat_name=str(chat.get("name") or chat.get("title") or source_chat_name or ""),
            source_chat_id=_safe_int(chat.get("id")),
            total_json_messages=len(raw_messages),
        )
        stats.before_counts = await self._db_counts(target_chat_id)

        do_messages = cfg.import_messages if import_messages is None else import_messages
        do_summaries = cfg.build_summaries if build_summaries is None else build_summaries
        do_facts = cfg.extract_facts if extract_facts is None else extract_facts
        do_interests = cfg.update_interests if update_interests is None else update_interests

        imported = await self._parse_and_import_messages(
            raw_messages,
            target_chat_id=target_chat_id,
            do_messages=do_messages,
            stats=stats,
        )
        if do_summaries:
            await self._build_summaries(imported, stats)
        if do_facts:
            await self._extract_and_store_facts(imported, stats)
        if do_interests:
            await self._replay_interests(imported, stats)

        stats.after_counts = await self._db_counts(target_chat_id)
        stats.top_interests = await self.storage.get_interest_topics(target_chat_id, limit=10)
        stats.recent_facts = await self.storage.get_recent_facts(target_chat_id, limit=10)
        return stats

    async def _parse_and_import_messages(
        self,
        raw_messages: list[Any],
        *,
        target_chat_id: int,
        do_messages: bool,
        stats: ImportStats,
    ) -> list[ImportedMessage]:
        cfg = self.config.telegram_import
        parsed: list[ImportedMessage] = []
        total = len(raw_messages)
        for idx, item in enumerate(raw_messages, start=1):
            if not isinstance(item, dict):
                stats.skipped_service += 1
                continue
            if cfg.skip_service_messages and item.get("type") not in (None, "message"):
                stats.skipped_service += 1
                self._progress("messages", idx, total, stats=stats)
                continue

            text = normalize_telegram_text(item.get("text"))
            if cfg.skip_empty_text and not text.strip():
                stats.skipped_empty += 1
                self._progress("messages", idx, total, stats=stats)
                continue

            source_id = _safe_int(item.get("id")) or idx
            telegram_message_id = self._map_message_id(source_id)
            reply_source_id = _safe_int(item.get("reply_to_message_id"))
            event = MessageEvent(
                rquid=f"import-{uuid.uuid4()}",
                chat_id=target_chat_id,
                telegram_message_id=telegram_message_id,
                user_id=_parse_user_id(item.get("from_id") or item.get("actor_id")),
                username=None,
                full_name=str(item.get("from") or item.get("actor") or "unknown"),
                text=text.strip(),
                created_at=normalize_telegram_date(item),
                reply_to_message_id=self._map_message_id(reply_source_id) if reply_source_id is not None else None,
                reply_to_bot=False,
                mentioned_bot=False,
                is_bot_message=False,
            )
            imported = ImportedMessage(event=event, source_message_id=source_id)
            stats.parsed_text_messages += 1
            if do_messages and not self.dry_run:
                message_db_id, inserted = await self.storage.save_message_with_status(event)
                imported.message_db_id = message_db_id
                imported.inserted = inserted
                if inserted:
                    stats.imported_messages += 1
                else:
                    stats.duplicate_messages += 1
            parsed.append(imported)
            self._progress("messages", idx, total, stats=stats)
        self._progress("messages", total, total, stats=stats, force=True)
        return parsed

    async def _build_summaries(self, imported: list[ImportedMessage], stats: ImportStats) -> None:
        if self.dry_run:
            self._print("summaries: dry-run, skipped LLM writes")
            return
        rows = [m for m in imported if m.message_db_id is not None]
        if not rows:
            self._print("summaries: no imported DB messages, skipped")
            return
        cfg = self.config.telegram_import
        chunks = list(_chunks(rows, cfg.summary_chunk_size))
        for idx, chunk in enumerate(chunks, start=1):
            text = self._format_messages([m.event for m in chunk], max_text_chars=1200)
            prompt = (
                "Сжимаю фрагмент импортированной истории Telegram-чата для своей памяти.\n"
                "Нужно сохранить не хронологию, а полезный слой памяти:\n"
                "- кто с кем общается и какой стиль у чата;\n"
                "- важные факты, решения и договорённости;\n"
                "- локальные темы, шутки и повторяющиеся паттерны;\n"
                "- правила поведения, которые мне нужно учитывать в будущих разговорах.\n"
                "Не добавляю того, чего нет в сообщениях. Пишу компактно.\n\n"
                f"Фрагмент:\n{text}"
            )
            try:
                raw = await self.llm.chat(
                    [
                        {"role": "system", "content": self.config.bot.identity_prompt + "Делаю offline-summary истории чата для своей памяти. Сохраняю имена авторов фактов и не приписываю себе чужой опыт."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=cfg.summary_temperature,
                    num_predict=cfg.summary_max_tokens,
                    purpose="import.summarize",
                )
                summary = self.llm.clean_visible_answer(raw)
                if not summary:
                    stats.empty_summaries += 1
                else:
                    ids = [int(m.message_db_id) for m in chunk if m.message_db_id is not None]
                    await self.storage.save_summary(
                        stats.target_chat_id,
                        start_message_id=ids[0],
                        end_message_id=ids[-1],
                        summary_text=summary,
                    )
                    if cfg.mark_imported_messages_summarized:
                        await self.storage.mark_messages_summarized(ids)
                    stats.summaries_created += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("telegram import summary failed chunk=%s", idx)
                stats.errors.append(f"summary chunk {idx}: {exc}")
            self._progress("summaries", idx, len(chunks), stats=stats, chunks=True)
        self._progress("summaries", len(chunks), len(chunks), stats=stats, force=True, chunks=True)

    async def _extract_and_store_facts(self, imported: list[ImportedMessage], stats: ImportStats) -> None:
        if self.memory is None:
            stats.errors.append("facts requested, but QdrantMemory is not initialized")
            return
        if self.dry_run:
            self._print("facts: dry-run, skipped LLM/vector writes")
            return
        cfg = self.config.telegram_import
        candidates = [m for m in imported if len(m.event.text.strip()) >= cfg.min_text_chars_for_facts]
        chunks = list(_chunks(candidates, cfg.fact_chunk_size))
        for idx, chunk in enumerate(chunks, start=1):
            extracted = await self._extract_facts_from_chunk([m.event for m in chunk])
            for fact in extracted:
                text = " ".join(str(fact.get("text") or "").split())
                if not text:
                    continue
                try:
                    confidence = float(fact.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                if confidence < cfg.fact_min_confidence:
                    continue
                kind = str(fact.get("kind") or "other")
                if await self._fact_is_duplicate(text, stats.target_chat_id):
                    stats.duplicate_facts += 1
                    continue
                await self.memory.add_fact(
                    text,
                    chat_id=stats.target_chat_id,
                    user_id=None,
                    source=f"telegram_import:{kind}",
                )
                stats.facts_saved += 1
                stats.facts_by_kind[kind] += 1
            self._progress("facts", idx, len(chunks), stats=stats, chunks=True)
        self._progress("facts", len(chunks), len(chunks), stats=stats, force=True, chunks=True)

    async def _extract_facts_from_chunk(self, events: list[MessageEvent]) -> list[dict[str, Any]]:
        cfg = self.config.telegram_import
        text = self._format_messages(events, max_text_chars=900)
        system = (
            self.config.bot.identity_prompt + "Извлекаю память из импортированной истории Telegram-чата. Сохраняю имена участников, к которым относятся факты. "
            "Возвращаю строгий JSON по схеме. Извлекаю только устойчивые знания, которые должны повлиять "
            "на мои будущие ответы в этом чате. Особенно важны chat_rule, running_joke, local_topic, "
            "participant_preference, project_fact, decision и social_pattern. "
            "Не сохраняю одноразовые реплики, случайные эмоции, секреты, токены, пароли, номера карт, "
            "адреса проживания и чувствительные данные. "
            "Факты должны быть самодостаточными и без ссылок на 'это сообщение'."
        )
        user = (
            f"Максимум фактов из чанка: {cfg.max_facts_per_chunk}.\n"
            "Хорошие примеры:\n"
            "- В этом чате не любят длинные ассистентские ответы.\n"
            "- В чате принято обсуждать новые версии проекта по пятницам.\n"
            "- Участник предпочитает примеры кода на Python.\n"
            "- В проекте бота используется Ollama и локальная LLM.\n\n"
            f"Сообщения:\n{text}"
        )
        try:
            data = await self.llm.structured_json(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                IMPORT_FACT_SCHEMA,
                temperature=cfg.fact_temperature,
                num_predict=cfg.fact_num_predict,
                purpose="import.extract_facts",
            )
        except Exception:
            logger.exception("telegram import fact extraction failed")
            return []
        facts = data.get("facts") if isinstance(data, dict) else []
        if not isinstance(facts, list):
            return []
        return [x for x in facts[: cfg.max_facts_per_chunk] if isinstance(x, dict)]

    async def _fact_is_duplicate(self, text: str, chat_id: int) -> bool:
        if self.memory is None:
            return False
        threshold = self.config.telegram_import.fact_duplicate_score_threshold
        if threshold <= 0:
            return False
        try:
            matches = await self.memory.search(text, chat_id=chat_id, limit=3)
        except Exception:
            logger.exception("telegram import duplicate fact check failed")
            return False
        return any(item.score >= threshold for item in matches)

    async def _replay_interests(self, imported: list[ImportedMessage], stats: ImportStats) -> None:
        if self.interest_engine is None:
            stats.errors.append("interests requested, but personality_engine is disabled")
            return
        if self.dry_run:
            self._print("interests: dry-run, skipped embedding writes")
            return
        cfg = self.config.telegram_import
        candidates = [m for m in imported if len(m.event.text.strip()) >= cfg.min_text_chars_for_interests]
        if cfg.interest_max_messages and cfg.interest_max_messages > 0:
            candidates = candidates[-int(cfg.interest_max_messages):]
        total = len(candidates)
        for idx, item in enumerate(candidates, start=1):
            try:
                await self.interest_engine.evaluate_event(item.event, message_db_id=item.message_db_id)
                stats.interest_messages_processed += 1
            except Exception as exc:  # noqa: BLE001
                stats.interest_messages_skipped += 1
                logger.exception("telegram import interest replay failed idx=%s", idx)
                stats.errors.append(f"interest message {idx}: {exc}")
            self._progress("interests", idx, total, stats=stats)
        self._progress("interests", total, total, stats=stats, force=True)

    def _select_chat(
        self,
        root: Any,
        *,
        source_chat_id: int | None,
        source_chat_name: str | None,
    ) -> dict[str, Any]:
        if not isinstance(root, dict):
            raise ValueError("Telegram export root must be a JSON object")
        if isinstance(root.get("messages"), list):
            return root

        chats = root.get("chats")
        chat_list = chats.get("list") if isinstance(chats, dict) else None
        if not isinstance(chat_list, list):
            raise ValueError("Unsupported Telegram export: expected top-level messages or chats.list")

        candidates = [c for c in chat_list if isinstance(c, dict) and isinstance(c.get("messages"), list)]
        if source_chat_id is not None:
            for chat in candidates:
                if _safe_int(chat.get("id")) == source_chat_id:
                    return chat
            raise ValueError(f"No chat with id={source_chat_id} in export")

        if source_chat_name:
            needle = source_chat_name.strip().lower()
            matches = [
                chat for chat in candidates
                if needle in str(chat.get("name") or chat.get("title") or "").lower()
            ]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError(f"No chat matching name={source_chat_name!r} in export")
            names = ", ".join(str(m.get("name") or m.get("title") or m.get("id")) for m in matches[:10])
            raise ValueError(f"Multiple chats match {source_chat_name!r}: {names}. Use --source-chat-id.")

        if len(candidates) == 1:
            return candidates[0]
        preview = "; ".join(
            f"id={c.get('id')} name={c.get('name') or c.get('title')} messages={len(c.get('messages') or [])}"
            for c in candidates[:20]
        )
        raise ValueError(
            "Full Telegram export contains multiple chats. Pass --source-chat-id or --source-chat-name. "
            f"Candidates: {preview}"
        )

    def _map_message_id(self, source_id: int | None) -> int:
        if source_id is None:
            return -1
        mode = self.config.telegram_import.message_id_mode
        if mode == "preserve":
            return int(source_id)
        if mode == "negative_export_id":
            return -abs(int(source_id))
        raise ValueError(f"Unsupported telegram_import.message_id_mode={mode!r}")

    def _format_messages(self, events: list[MessageEvent], *, max_text_chars: int) -> str:
        lines: list[str] = []
        for event in events:
            text = event.text.replace("\n", " ").strip()
            if len(text) > max_text_chars:
                text = text[:max_text_chars].rstrip() + "..."
            lines.append(f"[{event.created_at}] {event.author_label}: {text}")
        return "\n".join(lines)

    async def _db_counts(self, chat_id: int) -> dict[str, int]:
        return {
            "messages": await self.storage.count_messages(chat_id),
            "summaries": await self.storage.count_summaries(chat_id),
            "facts": await self.storage.count_facts(chat_id),
            "interest_topics": len(await self.storage.get_interest_topics(chat_id, limit=100000)),
        }

    def _progress(
        self,
        label: str,
        current: int,
        total: int,
        *,
        stats: ImportStats,
        force: bool = False,
        chunks: bool = False,
    ) -> None:
        cfg = self.config.telegram_import
        every = cfg.progress_every_chunks if chunks else cfg.progress_every_messages
        if not force and every > 0 and current % every != 0:
            return
        pct = 100.0 if total <= 0 else (current / max(1, total)) * 100.0
        self._print(
            f"{label}: {current}/{total} ({pct:.1f}%) | "
            f"messages imported={stats.imported_messages}, duplicates={stats.duplicate_messages}, "
            f"summaries={stats.summaries_created}, facts={stats.facts_saved}, "
            f"interests={stats.interest_messages_processed}"
        )

    @staticmethod
    def _print(text: str) -> None:
        print(text, file=sys.stderr, flush=True)


def normalize_telegram_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts).strip()
    return ""


def normalize_telegram_date(item: dict[str, Any]) -> str:
    raw = item.get("date")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    unix_raw = item.get("date_unixtime")
    try:
        return datetime.fromtimestamp(int(str(unix_raw)), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return utcnow_iso()


def _parse_user_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"-?\d+", str(value))
    return int(match.group(0)) if match else None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunks(items: list[ImportedMessage], size: int) -> list[list[ImportedMessage]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]
