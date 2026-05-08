from __future__ import annotations

import argparse
import asyncio

from bot_app.config import load_config
from bot_app.llm.ollama_client import OllamaClient


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.example.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    llm = OllamaClient(config.llm)
    answer = await llm.chat(
        [
            {"role": "system", "content": "Отвечай коротко."},
            {"role": "user", "content": "Скажи, что Ollama работает."},
        ],
        num_predict=80,
    )
    print(llm.clean_visible_answer(answer))
    emb = await llm.embed("test embedding")
    print(f"embedding size={len(emb)}")
    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
