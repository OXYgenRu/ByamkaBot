from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message

from bot_app.config import load_config
from bot_app.core.chat_state import ChatStateBuilder
from bot_app.core.context_builder import ContextBuilder
from bot_app.core.orchestrator import BotOrchestrator
from bot_app.core.responder import Responder
from bot_app.core.router import ShouldReplyRouter
from bot_app.domain.events import MessageEvent, utcnow_iso
from bot_app.llm.ollama_client import OllamaClient
from bot_app.logging_setup import setup_logging
from bot_app.memory.auto_extractor import AutoMemoryExtractor
from bot_app.memory.qdrant_memory import QdrantMemory
from bot_app.memory.summarizer import ChatSummarizer
from bot_app.personality.interest_engine import PersonalityInterestEngine
from bot_app.storage.sqlite_store import SQLiteStore
from bot_app.tools.planner import ToolPlanner
from bot_app.tools.registry import build_default_tool_registry

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.example.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.getenv("BOT_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)),
        help="Path to YAML config",
    )
    return parser.parse_args()


async def build_orchestrator(config_path: str) -> tuple[BotOrchestrator, Bot, OllamaClient, SQLiteStore]:
    config = load_config(config_path)
    setup_logging(config.app.log_level)

    llm = OllamaClient(config.llm)
    storage = SQLiteStore(config.storage.sqlite_path)
    await storage.connect()

    memory = QdrantMemory(config.vector_db, llm, storage)
    interest_engine = PersonalityInterestEngine(config, llm, storage) if config.personality_engine.enabled else None
    try:
        await memory.ensure_collection()
    except Exception:
        logger.exception("Could not initialize vector memory. Check Ollama embedding model/Qdrant path.")
        raise

    if interest_engine is not None:
        await interest_engine.ensure_ready()

    registry = build_default_tool_registry(config, memory)
    summarizer = ChatSummarizer(config, storage, llm)
    state_builder = ChatStateBuilder(storage)
    router = ShouldReplyRouter(config, llm)
    tool_planner = ToolPlanner(config, llm, registry, storage)
    context_builder = ContextBuilder(config, storage)
    responder = Responder(config, llm)
    memory_extractor = AutoMemoryExtractor(config, llm, memory)
    orchestrator = BotOrchestrator(
        config=config,
        storage=storage,
        memory=memory,
        summarizer=summarizer,
        state_builder=state_builder,
        router=router,
        tool_planner=tool_planner,
        context_builder=context_builder,
        responder=responder,
        memory_extractor=memory_extractor,
        interest_engine=interest_engine,
    )
    if config.telegram.parse_mode:
        bot = Bot(
            token=config.telegram_token,
            default=DefaultBotProperties(parse_mode=config.telegram.parse_mode),
        )
    else:
        bot = Bot(token=config.telegram_token)
    me = await bot.get_me()
    config.bot.username = me.username
    logger.info("Bot started as @%s", me.username)
    return orchestrator, bot, llm, storage


def normalize_message(message: Message, bot_username: str | None) -> MessageEvent | None:
    text = message.text or message.caption or ""
    if not text.strip():
        return None
    user = message.from_user
    username = user.username if user else None
    full_name = user.full_name if user else None
    user_id = user.id if user else None
    lower_text = text.lower()
    is_private_chat = message.chat.type == "private"
    mentioned_bot = is_private_chat or bool(bot_username and (f"@{bot_username.lower()}" in lower_text))
    reply_to_bot = bool(
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.is_bot
    )
    return MessageEvent(
        rquid=str(uuid.uuid4()),
        chat_id=message.chat.id,
        telegram_message_id=message.message_id,
        user_id=user_id,
        username=username,
        full_name=full_name,
        text=text.strip(),
        created_at=utcnow_iso(),
        reply_to_message_id=message.reply_to_message.message_id if message.reply_to_message else None,
        reply_to_bot=reply_to_bot,
        mentioned_bot=mentioned_bot,
        is_bot_message=bool(user and user.is_bot),
    )


async def main() -> None:
    args = parse_args()
    orchestrator, bot, llm, storage = await build_orchestrator(args.config)
    dp = Dispatcher()

    @dp.message(Command("ping"))
    async def ping(message: Message) -> None:
        await message.answer("pong")

    @dp.message(Command("remember"))
    async def remember(message: Message) -> None:
        event = normalize_message(message, orchestrator.config.bot.username)
        if event:
            await orchestrator.storage.save_message(event)
        fact = (message.text or "").split(maxsplit=1)
        if len(fact) < 2 or not fact[1].strip():
            await message.answer("Напиши так: /remember факт, который нужно запомнить")
            return
        fact_text = fact[1].strip()
        fact_id = await orchestrator.memory.add_fact(
            fact_text,
            chat_id=message.chat.id,
            user_id=message.from_user.id if message.from_user else None,
            source="telegram_command",
        )
        await message.answer(f"Запомнил факт: {fact_id[:8]}")


    @dp.message(Command("fingerprint"))
    async def fingerprint(message: Message) -> None:
        path = await orchestrator.render_interest_fingerprint()
        if path is None:
            await message.answer("Движок интересов отключён.")
            return
        await message.answer_document(
            FSInputFile(path),
            caption="fingerprint характера Бямки по seed интересов",
        )

    @dp.message(Command("interests"))
    async def interests(message: Message) -> None:
        text = await orchestrator.format_top_interests(message.chat.id, limit=8)
        await message.answer(text[:3500])

    @dp.message(Command("memory"))
    async def memory_search(message: Message) -> None:
        query = (message.text or "").split(maxsplit=1)
        if len(query) < 2 or not query[1].strip():
            await message.answer("Напиши так: /memory что искать")
            return
        results = await orchestrator.memory.search(
            query[1].strip(),
            chat_id=message.chat.id,
            limit=orchestrator.config.vector_db.search_limit,
        )
        if not results:
            await message.answer("В памяти ничего похожего не нашёл.")
            return
        text = "\n".join(f"- {r.text} ({r.score:.3f})" for r in results)
        await message.answer(text[:3500])

    @dp.message(F.text | F.caption)
    async def handle_text(message: Message) -> None:
        event = normalize_message(message, orchestrator.config.bot.username)
        if event is None:
            return
        try:
            answer = await orchestrator.handle_message(event)
            if answer:
                await message.answer(answer)
        except Exception:
            logger.exception("Failed to handle message")
            # В группе лучше не шуметь технической ошибкой. Но для прямых упоминаний можно коротко сказать.
            if event.reply_to_bot or event.mentioned_bot:
                await message.answer("Я споткнулся на обработке сообщения. Проверь логи бота.")

    try:
        await dp.start_polling(bot)
    finally:
        await llm.close()
        await storage.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
