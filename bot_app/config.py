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
    personality: str = "Ты участник Telegram-чата. Отвечай естественно и коротко."
    max_visible_answer_chars: int = 1800
    default_language: str = "ru"


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "deepseek-r1:8b"
    embedding_model: str = "nomic-embed-text"
    context_window_tokens: int = 8192
    reserve_response_tokens: int = 700
    reserve_thinking_tokens: int = 1200
    temperature: float = 0.65
    router_temperature: float = 0.1
    summarizer_temperature: float = 0.2
    request_timeout_sec: int = 180


@dataclass
class RoutingConfig:
    answer_on_direct_reply: bool = True
    answer_on_mention: bool = True
    allow_spontaneous_replies: bool = True
    spontaneous_reply_min_confidence: float = 0.72
    direct_reply_min_confidence: float = 0.2
    chat_cooldown_seconds: int = 12
    max_bot_replies_in_recent_window: int = 3
    recent_window_messages: int = 12

    # Second routing pass after memory/facts are retrieved.
    post_memory_review_enabled: bool = True
    allow_post_memory_ignore_direct: bool = False
    post_memory_ignore_min_confidence: float = 0.65
    post_memory_review_num_predict: int = 512


@dataclass
class ContextConfig:
    recent_messages_limit: int = 40
    recent_messages_soft_limit_tokens: int = 2500
    summaries_limit: int = 6
    facts_limit: int = 5
    tool_results_limit: int = 4
    current_message_extra_tokens: int = 400
    facts_min_score: float = 0.55
    behavior_facts_min_score: float = 0.55

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
    cfg = Config(
        app=AppConfig(**_section(raw, "app")),
        telegram=TelegramConfig(**_section(raw, "telegram")),
        bot=BotConfig(**_section(raw, "bot")),
        llm=LLMConfig(**_section(raw, "llm")),
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
