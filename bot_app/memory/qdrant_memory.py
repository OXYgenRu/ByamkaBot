from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from bot_app.config import VectorDBConfig
from bot_app.llm.ollama_client import OllamaClient
from bot_app.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class FactSearchResult:
    fact_id: str
    text: str
    score: float
    chat_id: int | None
    user_id: int | None
    source: str | None


class QdrantMemory:
    def __init__(
        self,
        config: VectorDBConfig,
        llm: OllamaClient,
        storage: SQLiteStore,
    ) -> None:
        self.config = config
        self.llm = llm
        self.storage = storage
        Path(config.qdrant_path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=config.qdrant_path)
        self._vector_size: int | None = None

    async def ensure_collection(self) -> None:
        if self._collection_exists():
            return
        probe = await self.llm.embed("probe")
        self._vector_size = len(probe)
        if self._vector_size <= 0:
            raise RuntimeError("Embedding vector is empty; check Ollama embedding model")

        await asyncio.to_thread(
            self.client.create_collection,
            collection_name=self.config.collection,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection %s size=%s", self.config.collection, self._vector_size)

    def _collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == self.config.collection for c in collections)

    async def add_fact(
        self,
        fact_text: str,
        *,
        chat_id: int | None,
        user_id: int | None,
        source: str | None = None,
    ) -> str:
        await self.ensure_collection()
        fact_id = str(uuid.uuid4())
        vector = await self.llm.embed(fact_text)
        payload = {
            "fact_id": fact_id,
            "text": fact_text,
            "chat_id": chat_id,
            "user_id": user_id,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        point = models.PointStruct(id=fact_id, vector=vector, payload=payload)
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.config.collection,
            points=[point],
        )
        await self.storage.save_fact(
            fact_id,
            fact_text,
            chat_id=chat_id,
            user_id=user_id,
            source=source,
        )
        return fact_id

    async def search(
        self,
        query: str,
        *,
        chat_id: int | None,
        limit: int | None = None,
    ) -> list[FactSearchResult]:
        if not query.strip():
            return []
        await self.ensure_collection()
        vector = await self.llm.embed(query)
        limit = limit or self.config.search_limit

        def _query() -> Any:
            if hasattr(self.client, "query_points"):
                return self.client.query_points(
                    collection_name=self.config.collection,
                    query=vector,
                    limit=max(limit * 3, limit),
                    with_payload=True,
                )
            return self.client.search(
                collection_name=self.config.collection,
                query_vector=vector,
                limit=max(limit * 3, limit),
                with_payload=True,
            )

        raw = await asyncio.to_thread(_query)
        points = getattr(raw, "points", raw)
        results: list[FactSearchResult] = []
        for point in points:
            payload = point.payload or {}
            point_chat_id = payload.get("chat_id")
            # Показываем глобальные факты и факты текущего чата.
            if point_chat_id is not None and chat_id is not None and int(point_chat_id) != int(chat_id):
                continue
            results.append(
                FactSearchResult(
                    fact_id=str(payload.get("fact_id") or point.id),
                    text=str(payload.get("text") or ""),
                    score=float(getattr(point, "score", 0.0)),
                    chat_id=int(point_chat_id) if point_chat_id is not None else None,
                    user_id=int(payload["user_id"]) if payload.get("user_id") is not None else None,
                    source=payload.get("source"),
                )
            )
            if len(results) >= limit:
                break
        return results
