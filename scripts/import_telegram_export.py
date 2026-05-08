from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot_app.config import load_config
from bot_app.importers.telegram_export import TelegramExportImporter
from bot_app.llm.ollama_client import OllamaClient
from bot_app.logging_setup import setup_logging
from bot_app.personality.interest_engine import PersonalityInterestEngine
from bot_app.storage.sqlite_store import SQLiteStore

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.example.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Telegram Desktop JSON export into Byamka bot memory."
    )
    parser.add_argument("--json", required=True, help="Path to Telegram Desktop result.json or single chat JSON")
    parser.add_argument("--target-chat-id", required=True, type=int, help="Chat id whose memory should be affected")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--source-chat-id", type=int, default=None, help="Source chat id inside full Telegram export")
    parser.add_argument("--source-chat-name", default=None, help="Source chat name substring inside full Telegram export")

    parser.add_argument("--dry-run", action="store_true", help="Parse JSON and show stats without writing DB/Qdrant")
    parser.add_argument("--no-messages", action="store_true", help="Do not import raw messages into SQLite")
    parser.add_argument("--no-summaries", action="store_true", help="Do not build summaries")
    parser.add_argument("--no-facts", action="store_true", help="Do not extract facts/rules/jokes")
    parser.add_argument("--no-interests", action="store_true", help="Do not replay learned interests")
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.app.log_level)

    storage = SQLiteStore(config.storage.sqlite_path)
    await storage.connect()
    llm = OllamaClient(config.llm)
    memory: QdrantMemory | None = None
    interest_engine: PersonalityInterestEngine | None = None

    try:
        needs_memory = not args.no_facts and not args.dry_run
        needs_interests = not args.no_interests and not args.dry_run and config.personality_engine.enabled

        if needs_memory:
            from bot_app.memory.qdrant_memory import QdrantMemory

            memory = QdrantMemory(config.vector_db, llm, storage)
            await memory.ensure_collection()
        elif not args.no_facts:
            # Dry-run: no Qdrant writes/checks are needed.
            memory = None

        if needs_interests:
            interest_engine = PersonalityInterestEngine(config, llm, storage)
            await interest_engine.ensure_ready()

        importer = TelegramExportImporter(
            config=config,
            storage=storage,
            llm=llm,
            memory=memory,
            interest_engine=interest_engine,
            dry_run=args.dry_run,
        )
        stats = await importer.import_file(
            path=args.json,
            target_chat_id=args.target_chat_id,
            source_chat_id=args.source_chat_id,
            source_chat_name=args.source_chat_name,
            import_messages=not args.no_messages,
            build_summaries=not args.no_summaries,
            extract_facts=not args.no_facts,
            update_interests=not args.no_interests,
        )
        print("\n" + stats.to_text())
    finally:
        await llm.close()
        await storage.close()


if __name__ == "__main__":
    asyncio.run(amain())
