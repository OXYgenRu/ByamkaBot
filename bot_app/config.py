from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class AppConfig:
    name: str = "local-chat-participant"
    log_level: str = "INFO"
    data_dir: str = "./data"


@dataclass
class TelegramConfig:
    token_env: str = "TELEGRAM_BOT_TOKEN"
    parse_mode: str | None = None


@dataclass
class BotConfig:
    display_name: str = "Локальный бот"
    username: str | None = None
    identity_aliases: list[str] = field(default_factory=list)
    personality: str = "Я участник Telegram-чата. Отвечаю естественно и коротко."
    dialogue_examples: list[list[dict[str, str]]] = field(default_factory=list)
    dialogue_examples_file: str | None = None
    max_visible_answer_chars: int = 1800
    default_language: str = "ru"

    @property
    def identity_prompt(self) -> str:
        username = f" Мой username — @{self.username}." if self.username else ""
        aliases = " Меня также называют: " + ", ".join(self.identity_aliases) + "." if self.identity_aliases else ""
        return (
            f"Я — {self.display_name}, участник этого Telegram-чата.{username}{aliases} "
            "В истории текущего чата BOT и сообщения assistant обозначают мои ответы. "
            "Упоминания меня в третьем лице связываю с собой по контексту, "
            "но не считаю любое «он» обращением ко мне. "
        )

    def __post_init__(self) -> None:
        if not isinstance(self.dialogue_examples, list):
            raise ValueError("bot.dialogue_examples must be a list of conversations")
        for dialogue in self.dialogue_examples:
            if not isinstance(dialogue, list) or not dialogue or len(dialogue) % 2:
                raise ValueError("Each dialogue example must contain complete user/assistant pairs")
            for index, message in enumerate(dialogue):
                role = "user" if index % 2 == 0 else "assistant"
                if (not isinstance(message, dict) or message.get("role") != role
                        or not isinstance(message.get("content"), str) or not message["content"].strip()):
                    raise ValueError("Dialogue examples require alternating user/assistant roles and nonempty content")


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "deepseek-r1:8b"
    embedding_model: str = "nomic-embed-text"
    thinking: bool = True
    num_batch: int = 1024
    keep_alive: str = "3h"
    context_window_tokens: int = 8192
    reserve_response_tokens: int = 700
    reserve_thinking_tokens: int = 1200
    temperature: float = 0.65
    router_temperature: float = 0.1
    summarizer_temperature: float = 0.2
    request_timeout_sec: int = 180


@dataclass
class VisionConfig:
    enabled: bool = True
    max_download_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 25_000_000
    max_side: int = 1024
    max_images_per_request: int = 1
    reserve_image_tokens: int = 4096
    description_tokens: int = 512

    def __post_init__(self) -> None:
        if min(self.max_download_bytes, self.max_pixels, self.max_side,
               self.max_images_per_request, self.reserve_image_tokens, self.description_tokens) <= 0:
            raise ValueError("Vision limits must be positive")


@dataclass
class RoutingConfig:
    message_debounce_seconds: float = 1.0
    answer_on_direct_reply: bool = True
    answer_on_mention: bool = True
    allow_spontaneous_replies: bool = True
    spontaneous_reply_min_confidence: float = 0.72
    direct_reply_min_confidence: float = 0.2
    chat_cooldown_seconds: int = 12
    max_bot_replies_in_recent_window: int = 3
    recent_window_messages: int = 12
    followup_window_seconds: int = 180
    dialogue_reply_min_confidence: float = 0.5

    # Second routing pass after memory/facts are retrieved.
    post_memory_review_enabled: bool = True
    allow_post_memory_ignore_direct: bool = False
    post_memory_ignore_min_confidence: float = 0.65
    post_memory_review_num_predict: int = 512


@dataclass
class ContextConfig:
    history_max_tokens: int = 14000
    history_compact_ratio: float = 0.90
    history_keep_ratio: float = 0.50
    router_history_tokens: int = 2500
    prompt_safety_tokens: int = 1024
    examples_max_tokens: int = 1200
    recent_messages_limit: int = 40
    recent_messages_soft_limit_tokens: int = 2500
    summaries_limit: int = 6
    facts_limit: int = 5
    tool_results_limit: int = 4
    current_message_extra_tokens: int = 400
    facts_min_score: float = 0.55
    behavior_facts_min_score: float = 0.55

    def __post_init__(self) -> None:
        if not 0 < self.history_keep_ratio < self.history_compact_ratio < 1:
            raise ValueError("History thresholds must satisfy 0 < keep < compact < 1")
        if min(self.history_max_tokens, self.router_history_tokens, self.prompt_safety_tokens) <= 0:
            raise ValueError("History and safety token budgets must be positive")


@dataclass
class SummarizationConfig:
    enabled: bool = True
    keep_recent_messages: int = 35
    min_unsummarized_messages: int = 25
    chunk_size: int = 30
    max_summary_tokens: int = 600


@dataclass
class StorageConfig:
    sqlite_path: str = "./data/bot.sqlite3"


@dataclass
class VectorDBConfig:
    provider: str = "qdrant_local"
    qdrant_path: str = "./data/qdrant"
    collection: str = "bot_facts"
    search_limit: int = 5


@dataclass
class PersonalityEngineConfig:
    enabled: bool = True
    seed: int = 73421
    latent_dim: int = 6
    noise_scale: float = 2.7
    octaves: int = 4
    persistence: float = 0.55

    # Final interest formula. Innate interest is capped so seeded noise is a
    # background preference, not an obsession.
    base_interest: float = 0.35
    innate_weight: float = 0.20
    learned_weight: float = 0.35
    novelty_weight: float = 0.28
    max_innate_contribution: float = 0.12

    # Learned interest evolves per chat when similar topics reappear.
    learned_similarity_threshold: float = 0.72
    learned_topics_limit: int = 5
    topic_merge_similarity: float = 0.86
    learn_rate: float = 0.08
    learn_decay: float = 0.015
    learn_negative_rate: float = 0.04
    learn_repetition_decay: float = 0.10
    min_learned_interest: float = 0.02

    # Anti-loop mechanics. Similar topics become less novel, and recently seen
    # topics get cooldown penalty.
    topic_cooldown_minutes: int = 180
    repeated_topic_penalty: float = 0.35
    novelty_seen_count_strength: float = 0.08
    max_novelty_penalty: float = 0.45
    repetition_bias_block_threshold: float = 0.18

    # Behavioral thresholds. Interest is a bias, not an order.
    boost_reply_threshold: float = 0.66
    spontaneous_reply_threshold: float = 0.84
    memory_interest_threshold: float = 0.76
    save_interest_threshold: float = 0.72
    router_threshold_discount: float = 0.12

    # 2D SVG fingerprint rendering.
    fingerprint_size: int = 96
    fingerprint_cell_px: int = 5


@dataclass
class TelegramImportConfig:
    enabled: bool = True
    import_messages: bool = True
    build_summaries: bool = True
    extract_facts: bool = True
    update_interests: bool = True

    # Telegram Desktop export can be a single-chat JSON with top-level messages
    # or a full account export with chats.list. The script supports both.
    message_id_mode: str = "negative_export_id"
    skip_service_messages: bool = True
    skip_empty_text: bool = True
    min_text_chars_for_facts: int = 10
    min_text_chars_for_interests: int = 4

    summary_chunk_size: int = 80
    summary_max_tokens: int = 900
    summary_temperature: float = 0.15
    mark_imported_messages_summarized: bool = True

    fact_chunk_size: int = 60
    max_facts_per_chunk: int = 12
    fact_min_confidence: float = 0.76
    fact_duplicate_score_threshold: float = 0.91
    fact_temperature: float = 0.0
    fact_num_predict: int = 1100

    interest_max_messages: int = 0
    progress_every_messages: int = 100
    progress_every_chunks: int = 5


@dataclass
class ToolsConfig:
    enabled: bool = True
    planner_enabled: bool = True
    allow_side_effect_tools: bool = False
    max_tool_calls: int = 3
    calculator_enabled: bool = True
    memory_search_enabled: bool = True


@dataclass
class AutoMemoryConfig:
    enabled: bool = True
    ignore_commands: bool = True
    min_text_chars: int = 12
    min_fact_chars: int = 20
    max_facts_per_message: int = 3
    min_confidence: float = 0.72
    duplicate_score_threshold: float = 0.92
    temperature: float = 0.0
    num_predict: int = 700
    allow_global_facts: bool = False
    interest_memory_confidence_discount: float = 0.08


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    personality_engine: PersonalityEngineConfig = field(default_factory=PersonalityEngineConfig)
    telegram_import: TelegramImportConfig = field(default_factory=TelegramImportConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    auto_memory: AutoMemoryConfig = field(default_factory=AutoMemoryConfig)


    @property
    def telegram_token(self) -> str:
        token = os.getenv(self.telegram.token_env, "").strip()
        if not token:
            raise RuntimeError(
                f"Telegram token is empty. Set env variable {self.telegram.token_env}."
            )
        return token


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value



def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_config(path: str | Path) -> Config:
    load_dotenv()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bot_options = dict(_section(raw, "bot"))
    if bot_options.get("dialogue_examples_file"):
        if bot_options.get("dialogue_examples"):
            raise ValueError("Use either bot.dialogue_examples or bot.dialogue_examples_file")
        examples_path = path.resolve().parent / bot_options["dialogue_examples_file"]
        examples_data = yaml.safe_load(examples_path.read_text(encoding="utf-8"))
        if not isinstance(examples_data, dict) or not isinstance(examples_data.get("dialogues"), list):
            raise ValueError("Dialogue examples file must contain a dialogues list")
        if examples_data.get("speaker") != bot_options.get("display_name", BotConfig.display_name):
            raise ValueError("Dialogue examples speaker must match bot.display_name")
        bot_options["dialogue_examples"] = examples_data["dialogues"]
    cfg = Config(
        app=AppConfig(**_section(raw, "app")),
        telegram=TelegramConfig(**_section(raw, "telegram")),
        bot=BotConfig(**bot_options),
        llm=LLMConfig(**_section(raw, "llm")),
        vision=VisionConfig(**_section(raw, "vision")),
        routing=RoutingConfig(**_section(raw, "routing")),
        context=ContextConfig(**_section(raw, "context")),
        summarization=SummarizationConfig(**_section(raw, "summarization")),
        storage=StorageConfig(**_section(raw, "storage")),
        vector_db=VectorDBConfig(**_section(raw, "vector_db")),
        personality_engine=PersonalityEngineConfig(**_section(raw, "personality_engine")),
        telegram_import=TelegramImportConfig(**_section(raw, "telegram_import")),
        tools=ToolsConfig(**_section(raw, "tools")),
        auto_memory=AutoMemoryConfig(**_section(raw, "auto_memory")),
    )

    # Resolve relative storage paths against the project root inferred from the config file.
    # This avoids creating SQLite/Qdrant in a surprising working directory.
    base_dir = path.resolve().parent.parent if path.resolve().parent.name == "config" else path.resolve().parent

    cfg.app.data_dir = _resolve_path(base_dir, cfg.app.data_dir)
    cfg.storage.sqlite_path = _resolve_path(base_dir, cfg.storage.sqlite_path)
    cfg.vector_db.qdrant_path = _resolve_path(base_dir, cfg.vector_db.qdrant_path)

    Path(cfg.app.data_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.storage.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.vector_db.qdrant_path).mkdir(parents=True, exist_ok=True)
    return cfg
