from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import aiosqlite

from bot_app.domain.events import MessageEvent, utcnow_iso

logger = logging.getLogger(__name__)


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.commit()
        await self.init_schema()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteStore is not connected")
        return self._db

    async def init_schema(self) -> None:
        await self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                telegram_message_id INTEGER NOT NULL,
                user_id INTEGER,
                username TEXT,
                full_name TEXT,
                text TEXT NOT NULL,
                reply_to_message_id INTEGER,
                created_at TEXT NOT NULL,
                is_bot INTEGER NOT NULL DEFAULT 0,
                is_summarized INTEGER NOT NULL DEFAULT 0,
                UNIQUE(chat_id, telegram_message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id
            ON messages(chat_id, id);

            CREATE INDEX IF NOT EXISTS idx_messages_chat_unsummarized
            ON messages(chat_id, is_summarized, id);

            CREATE TABLE IF NOT EXISTS message_images (
                chat_id INTEGER NOT NULL,
                telegram_message_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                description TEXT,
                PRIMARY KEY(chat_id, telegram_message_id, path)
            );
            CREATE INDEX IF NOT EXISTS idx_message_images_path ON message_images(path);

            CREATE TABLE IF NOT EXISTS chat_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                start_message_id INTEGER NOT NULL,
                end_message_id INTEGER NOT NULL,
                summary_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_summaries_chat_id_id
            ON chat_summaries(chat_id, id);

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                chat_id INTEGER,
                user_id INTEGER,
                fact_text TEXT NOT NULL,
                source TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rquid TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                args_json TEXT NOT NULL,
                ok INTEGER NOT NULL,
                result_text TEXT,
                error_text TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS interest_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rquid TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                user_id INTEGER,
                message_db_id INTEGER,
                text_preview TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_interest_observations_chat_id_id
            ON interest_observations(chat_id, id);

            CREATE TABLE IF NOT EXISTS interest_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                topic_text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                learned_interest REAL NOT NULL,
                curiosity REAL NOT NULL,
                humor REAL NOT NULL,
                memory REAL NOT NULL,
                reply REAL NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_interest_topics_chat_score
            ON interest_topics(chat_id, learned_interest DESC, last_seen_at DESC);
            """
        )
        await self.db.commit()

    async def save_message(self, event: MessageEvent) -> int:
        message_id, _inserted = await self.save_message_with_status(event)
        return message_id

    async def save_message_with_status(self, event: MessageEvent) -> tuple[int, bool]:
        cursor = await self.db.execute(
            """
            INSERT OR IGNORE INTO messages(
                chat_id, telegram_message_id, user_id, username, full_name,
                text, reply_to_message_id, created_at, is_bot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.chat_id,
                event.telegram_message_id,
                event.user_id,
                event.username,
                event.full_name,
                event.text,
                event.reply_to_message_id,
                event.created_at,
                1 if event.is_bot_message else 0,
            ),
        )
        inserted = cursor.rowcount > 0
        if inserted:
            for path in event.image_paths:
                await self.db.execute(
                    "INSERT OR IGNORE INTO message_images(chat_id,telegram_message_id,path,description) "
                    "VALUES (?,?,?,(SELECT description FROM message_images WHERE path=? AND description IS NOT NULL LIMIT 1))",
                    (event.chat_id, event.telegram_message_id, path, path))
        await self.db.commit()
        row = await self.db.execute_fetchall(
            "SELECT id FROM messages WHERE chat_id=? AND telegram_message_id=?",
            (event.chat_id, event.telegram_message_id),
        )
        return int(row[0]["id"]), inserted

    async def get_message_images(self, chat_id: int, message_id: int) -> list[dict[str, Any]]:
        rows = await self.db.execute_fetchall(
            "SELECT path,description FROM message_images WHERE chat_id=? AND telegram_message_id=? ORDER BY rowid",
            (chat_id, message_id))
        return [dict(r) for r in rows]

    async def get_image_description(self, path: str) -> str | None:
        rows = await self.db.execute_fetchall(
            "SELECT description FROM message_images WHERE path=? AND description IS NOT NULL LIMIT 1", (path,))
        return str(rows[0]["description"]) if rows else None

    async def save_image_description(self, path: str, description: str) -> None:
        await self.db.execute("UPDATE message_images SET description=? WHERE path=?", (description, path))
        await self.db.commit()

    async def _with_images(self, rows) -> list[dict[str, Any]]:
        result = [dict(r) for r in rows]
        if not result:
            return result
        # Fetch metadata in one query rather than a query for every history message.
        images = await self.db.execute_fetchall(
            "SELECT telegram_message_id,path,description FROM message_images WHERE chat_id=? ORDER BY rowid",
            (result[0]["chat_id"],))
        by_message = {}
        for image in images:
            by_message.setdefault(image["telegram_message_id"], []).append(dict(image))
        for row in result:
            row["images"] = by_message.get(row["telegram_message_id"], [])
        return result

    async def get_recent_messages(self, chat_id: int, limit: int) -> list[dict[str, Any]]:
        rows = await self.db.execute_fetchall(
            """
            SELECT * FROM messages
            WHERE chat_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        return await self._with_images(reversed(rows))

    async def get_messages_for_reply(self, chat_id: int, target_id: int, limit: int | None, *, active_only: bool = False) -> list[dict[str, Any]]:
        # A delayed request must not see newer human questions. Answers delivered
        # while it waited are useful only when they belong to earlier requests.
        rows = await self.db.execute_fetchall(
            """
            SELECT * FROM messages
            WHERE chat_id=? AND (?=0 OR is_summarized=0) AND (
                (is_bot=0 AND telegram_message_id <= ?) OR
                (is_bot=1 AND (
                    reply_to_message_id < ? OR
                    (reply_to_message_id IS NULL AND telegram_message_id < ?)
                ))
            )
            ORDER BY id DESC LIMIT ?
            """,
            (chat_id, int(active_only), target_id, target_id, target_id, -1 if limit is None else limit),
        )
        return await self._with_images(reversed(rows))

    async def get_recent_bot_message_count(self, chat_id: int, window: int) -> int:
        rows = await self.db.execute_fetchall(
            """
            SELECT COUNT(*) AS cnt FROM (
                SELECT is_bot FROM messages
                WHERE chat_id=?
                ORDER BY id DESC
                LIMIT ?
            ) WHERE is_bot=1
            """,
            (chat_id, window),
        )
        return int(rows[0]["cnt"])

    async def get_last_bot_message_created_at(self, chat_id: int) -> str | None:
        rows = await self.db.execute_fetchall(
            """
            SELECT created_at FROM messages
            WHERE chat_id=? AND is_bot=1
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id,),
        )
        return str(rows[0]["created_at"]) if rows else None

    async def get_active_history(self, chat_id: int) -> list[dict[str, Any]]:
        rows = await self.db.execute_fetchall(
            "SELECT * FROM messages WHERE chat_id=? AND is_summarized=0 ORDER BY id", (chat_id,))
        return await self._with_images(rows)

    async def commit_compaction(self, chat_id: int, message_ids: list[int], summary: str) -> None:
        if not message_ids:
            return
        # A dedicated short transaction cannot be committed midway by another task
        # using the shared ingestion connection. Never hold it while calling the LLM.
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                for mid in message_ids:
                    cursor = await db.execute(
                        "UPDATE messages SET is_summarized=1 WHERE id=? AND chat_id=? AND is_summarized=0",
                        (mid, chat_id))
                    if cursor.rowcount != 1:
                        raise RuntimeError("Compaction source changed; refusing duplicate summary")
                await db.execute(
                    "INSERT INTO chat_summaries(chat_id,start_message_id,end_message_id,summary_text,created_at) VALUES (?,?,?,?,?)",
                    (chat_id, message_ids[0], message_ids[-1], summary, utcnow_iso()))
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    async def get_unsummarized_old_messages(
        self,
        chat_id: int,
        *,
        keep_recent: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        cutoff_rows = await self.db.execute_fetchall(
            """
            SELECT MIN(id) AS cutoff_id FROM (
                SELECT id FROM messages
                WHERE chat_id=?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (chat_id, keep_recent),
        )
        cutoff_id = cutoff_rows[0]["cutoff_id"]
        if cutoff_id is None:
            return []

        rows = await self.db.execute_fetchall(
            """
            SELECT * FROM messages
            WHERE chat_id=? AND id < ? AND is_summarized=0
            ORDER BY id ASC
            LIMIT ?
            """,
            (chat_id, int(cutoff_id), limit),
        )
        return [dict(r) for r in rows]

    async def mark_messages_summarized(self, message_ids: list[int]) -> None:
        if not message_ids:
            return
        await self.db.executemany(
            "UPDATE messages SET is_summarized=1 WHERE id=?",
            [(mid,) for mid in message_ids],
        )
        await self.db.commit()

    async def save_summary(
        self,
        chat_id: int,
        start_message_id: int,
        end_message_id: int,
        summary_text: str,
    ) -> int:
        cursor = await self.db.execute(
            """
            INSERT INTO chat_summaries(chat_id, start_message_id, end_message_id, summary_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_id, start_message_id, end_message_id, summary_text, utcnow_iso()),
        )
        await self.db.commit()
        return int(cursor.lastrowid)

    async def get_recent_summaries(
        self, chat_id: int, limit: int, *, before_telegram_id: int | None = None,
    ) -> list[str]:
        rows = await self.db.execute_fetchall(
            """
            SELECT summary_text FROM chat_summaries
            WHERE chat_id=? AND (? IS NULL OR end_message_id < (
                SELECT id FROM messages WHERE chat_id=? AND telegram_message_id=?
            ))
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, before_telegram_id, chat_id, before_telegram_id, limit),
        )
        return [str(r["summary_text"]) for r in reversed(rows)]

    async def save_fact(
        self,
        fact_id: str,
        fact_text: str,
        *,
        chat_id: int | None,
        user_id: int | None,
        source: str | None,
    ) -> None:
        await self.db.execute(
            """
            INSERT OR REPLACE INTO facts(id, chat_id, user_id, fact_text, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (fact_id, chat_id, user_id, fact_text, source, utcnow_iso()),
        )
        await self.db.commit()

    async def save_tool_event(
        self,
        *,
        rquid: str,
        chat_id: int,
        tool_name: str,
        args_json: str,
        ok: bool,
        result_text: str | None,
        error_text: str | None,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO tool_events(rquid, chat_id, tool_name, args_json, ok, result_text, error_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (rquid, chat_id, tool_name, args_json, 1 if ok else 0, result_text, error_text, utcnow_iso()),
        )
        await self.db.commit()

    async def save_interest_observation(
        self,
        *,
        rquid: str,
        chat_id: int,
        user_id: int | None,
        message_db_id: int | None,
        text_preview: str,
        embedding: list[float],
        profile: dict[str, float],
    ) -> int:
        cursor = await self.db.execute(
            """
            INSERT INTO interest_observations(
                rquid, chat_id, user_id, message_db_id, text_preview,
                embedding_json, profile_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rquid,
                chat_id,
                user_id,
                message_db_id,
                text_preview,
                json.dumps([float(x) for x in embedding], ensure_ascii=False),
                json.dumps(profile, ensure_ascii=False),
                utcnow_iso(),
            ),
        )
        await self.db.commit()
        return int(cursor.lastrowid)

    async def create_interest_topic(
        self,
        *,
        chat_id: int,
        topic_text: str,
        embedding: list[float],
        learned_interest: float,
        curiosity: float,
        humor: float,
        memory: float,
        reply: float,
    ) -> int:
        now = utcnow_iso()
        cursor = await self.db.execute(
            """
            INSERT INTO interest_topics(
                chat_id, topic_text, embedding_json, learned_interest,
                curiosity, humor, memory, reply, seen_count, created_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                chat_id,
                topic_text,
                json.dumps([float(x) for x in embedding], ensure_ascii=False),
                float(learned_interest),
                float(curiosity),
                float(humor),
                float(memory),
                float(reply),
                now,
                now,
            ),
        )
        await self.db.commit()
        return int(cursor.lastrowid)

    async def update_interest_topic(
        self,
        *,
        topic_id: int,
        embedding: list[float],
        learned_interest: float,
        curiosity: float,
        humor: float,
        memory: float,
        reply: float,
        seen_count: int,
        topic_text: str,
    ) -> None:
        rows = await self.db.execute_fetchall(
            "SELECT embedding_json FROM interest_topics WHERE id=?",
            (topic_id,),
        )
        centroid = [float(x) for x in embedding]
        if rows:
            old = _decode_vector(str(rows[0]["embedding_json"]))
            old_weight = max(1, int(seen_count) - 1)
            if len(old) == len(centroid):
                centroid = [
                    (old[i] * old_weight + centroid[i]) / (old_weight + 1)
                    for i in range(len(centroid))
                ]
        await self.db.execute(
            """
            UPDATE interest_topics
            SET topic_text=?, embedding_json=?, learned_interest=?, curiosity=?, humor=?, memory=?, reply=?,
                seen_count=?, last_seen_at=?
            WHERE id=?
            """,
            (
                topic_text,
                json.dumps(centroid, ensure_ascii=False),
                float(learned_interest),
                float(curiosity),
                float(humor),
                float(memory),
                float(reply),
                int(seen_count),
                utcnow_iso(),
                int(topic_id),
            ),
        )
        await self.db.commit()

    async def find_similar_interest_topics(
        self,
        *,
        chat_id: int,
        embedding: list[float],
        min_similarity: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows = await self.db.execute_fetchall(
            """
            SELECT * FROM interest_topics
            WHERE chat_id=?
            ORDER BY learned_interest DESC, last_seen_at DESC
            LIMIT 200
            """,
            (chat_id,),
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            vector = _decode_vector(str(item.get("embedding_json") or "[]"))
            similarity = _cosine(embedding, vector)
            if similarity < min_similarity:
                continue
            item["similarity"] = similarity
            results.append(item)
        results.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
        return results[:limit]

    async def get_interest_topics(self, chat_id: int, *, limit: int) -> list[dict[str, Any]]:
        rows = await self.db.execute_fetchall(
            """
            SELECT id, chat_id, topic_text, learned_interest, curiosity, humor, memory, reply, seen_count, created_at, last_seen_at
            FROM interest_topics
            WHERE chat_id=?
            ORDER BY learned_interest DESC, seen_count DESC, last_seen_at DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        return [dict(r) for r in rows]

    async def count_messages(self, chat_id: int) -> int:
        rows = await self.db.execute_fetchall(
            "SELECT COUNT(*) AS cnt FROM messages WHERE chat_id=?",
            (chat_id,),
        )
        return int(rows[0]["cnt"])

    async def count_summaries(self, chat_id: int) -> int:
        rows = await self.db.execute_fetchall(
            "SELECT COUNT(*) AS cnt FROM chat_summaries WHERE chat_id=?",
            (chat_id,),
        )
        return int(rows[0]["cnt"])

    async def count_facts(self, chat_id: int) -> int:
        rows = await self.db.execute_fetchall(
            "SELECT COUNT(*) AS cnt FROM facts WHERE chat_id=?",
            (chat_id,),
        )
        return int(rows[0]["cnt"])

    async def get_recent_facts(self, chat_id: int, *, limit: int) -> list[dict[str, Any]]:
        rows = await self.db.execute_fetchall(
            """
            SELECT id, fact_text, source, created_at
            FROM facts
            WHERE chat_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        return [dict(r) for r in rows]


def _decode_vector(raw: str) -> list[float]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [float(x) for x in data]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return dot / (na * nb)
