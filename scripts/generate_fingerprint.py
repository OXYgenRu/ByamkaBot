from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from bot_app.config import load_config
from bot_app.llm.ollama_client import OllamaClient
from bot_app.personality.interest_engine import PersonalityInterestEngine
from bot_app.storage.sqlite_store import SQLiteStore


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="data/interest_fingerprint.svg")
    args = parser.parse_args()

    config = load_config(args.config)
    llm = OllamaClient(config.llm)
    storage = SQLiteStore(config.storage.sqlite_path)
    await storage.connect()
    engine = PersonalityInterestEngine(config, llm, storage)
    path = await engine.render_fingerprint(Path(args.output))
    print(path)
    await storage.close()
    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
