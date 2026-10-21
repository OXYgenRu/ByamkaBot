from __future__ import annotations

import argparse
import asyncio

from bot_app.config import load_config
from bot_app.llm.ollama_client import OllamaClient
from bot_app.memory.qdrant_memory import QdrantMemory
from bot_app.storage.sqlite_store import SQLiteStore


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--chat-id", type=int, default=None)
    parser.add_argument("fact")
    args = parser.parse_args()

    config = load_config(args.config)
    llm = OllamaClient(config.llm)
    storage = SQLiteStore(config.storage.sqlite_path)
    await storage.connect()
    memory = QdrantMemory(config.vector_db, llm, storage)
    fact_id = await memory.add_fact(args.fact, chat_id=args.chat_id, user_id=None, source="seed_script")
    print(f"saved fact {fact_id}")
    await storage.close()
    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
