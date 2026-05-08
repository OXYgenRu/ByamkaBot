from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from bot_app.config import Config
from bot_app.domain.events import MessageEvent
from bot_app.llm.ollama_client import OllamaClient
from bot_app.personality.fingerprint import InterestFingerprintRenderer
from bot_app.personality.interest_field import InterestProfile, PerlinInterestField
from bot_app.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class PersonalityInterestEngine:
    """Evaluates and evolves Byamka's interests.

    Innate interest is deterministic from the seed. Learned interest is stored per chat.
    Anti-loop mechanics keep a high-interest semantic area from becoming an obsession:
    repeated topics get lower novelty, recent topics get cooldown penalty, and learned
    interest can both grow and decay.
    """

    def __init__(self, config: Config, llm: OllamaClient, storage: SQLiteStore) -> None:
        self.config = config
        self.llm = llm
        self.storage = storage
        self._field: PerlinInterestField | None = None

    async def ensure_ready(self) -> None:
        if self._field is not None:
            return
        probe = await self.llm.embed("interest field probe")
        if not probe:
            raise RuntimeError("Embedding vector is empty; cannot initialize interest engine")
        cfg = self.config.personality_engine
        self._field = PerlinInterestField(
            embedding_dim=len(probe),
            latent_dim=cfg.latent_dim,
            seed=cfg.seed,
            noise_scale=cfg.noise_scale,
            octaves=cfg.octaves,
            persistence=cfg.persistence,
        )
        logger.info(
            "interest engine initialized seed=%s embedding_dim=%s latent_dim=%s",
            cfg.seed,
            len(probe),
            cfg.latent_dim,
        )

    @property
    def field(self) -> PerlinInterestField:
        if self._field is None:
            raise RuntimeError("Interest engine is not initialized")
        return self._field

    async def evaluate_event(self, event: MessageEvent, *, message_db_id: int | None = None) -> InterestProfile | None:
        cfg = self.config.personality_engine
        if not cfg.enabled:
            return None
        if event.is_bot_message or not event.text.strip():
            return None

        await self.ensure_ready()
        embedding = await self.llm.embed(event.text)
        similar_topics = await self.storage.find_similar_interest_topics(
            chat_id=event.chat_id,
            embedding=embedding,
            min_similarity=cfg.learned_similarity_threshold,
            limit=cfg.learned_topics_limit,
        )

        learned, max_similarity, best_topic = self._weighted_learned_interest(similar_topics)
        novelty, novelty_penalty = self._compute_novelty(max_similarity=max_similarity, best_topic=best_topic)
        repetition_penalty = self._compute_repetition_penalty(best_topic)

        profile, _ = self.field.evaluate_embedding(
            embedding,
            learned_interest=learned,
            novelty=novelty,
            novelty_penalty=novelty_penalty,
            repetition_penalty=repetition_penalty,
            base_interest=cfg.base_interest,
            innate_weight=cfg.innate_weight,
            learned_weight=cfg.learned_weight,
            novelty_weight=cfg.novelty_weight,
            max_innate_contribution=cfg.max_innate_contribution,
        )

        await self.storage.save_interest_observation(
            rquid=event.rquid,
            chat_id=event.chat_id,
            user_id=event.user_id,
            message_db_id=message_db_id,
            text_preview=event.text[:500],
            embedding=embedding,
            profile=profile.to_dict(),
        )

        await self._maybe_evolve_topic(event, embedding, profile, similar_topics)
        logger.info(
            "interest rquid=%s chat=%s %s",
            event.rquid,
            event.chat_id,
            profile.to_prompt_text(),
        )
        return profile

    def _weighted_learned_interest(self, similar_topics: list[dict]) -> tuple[float, float, dict | None]:
        learned = 0.0
        max_similarity = 0.0
        best_topic = similar_topics[0] if similar_topics else None
        if not similar_topics:
            return learned, max_similarity, best_topic

        weighted_score = 0.0
        total_weight = 0.0
        for topic in similar_topics:
            similarity = float(topic.get("similarity", 0.0))
            score = float(topic.get("learned_interest", 0.0))
            weight = max(0.0, similarity)
            weighted_score += score * weight
            total_weight += weight
            max_similarity = max(max_similarity, similarity)
        if total_weight > 0:
            learned = weighted_score / total_weight
        return learned, max_similarity, best_topic

    def _compute_novelty(self, *, max_similarity: float, best_topic: dict | None) -> tuple[float, float]:
        cfg = self.config.personality_engine
        raw_novelty = max(0.0, min(1.0, 1.0 - max_similarity))
        if best_topic is None:
            return 1.0, 0.0

        seen_count = max(1, int(best_topic.get("seen_count", 1)))
        novelty_penalty = min(
            float(cfg.max_novelty_penalty),
            math.log1p(seen_count) * float(cfg.novelty_seen_count_strength),
        )
        novelty = max(0.0, raw_novelty - novelty_penalty)
        return novelty, novelty_penalty

    def _compute_repetition_penalty(self, best_topic: dict | None) -> float:
        cfg = self.config.personality_engine
        if best_topic is None or cfg.topic_cooldown_minutes <= 0:
            return 0.0

        last_seen_raw = str(best_topic.get("last_seen_at") or "")
        try:
            last_seen = datetime.fromisoformat(last_seen_raw)
        except ValueError:
            return 0.0
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)

        elapsed_minutes = (datetime.now(timezone.utc) - last_seen).total_seconds() / 60.0
        if elapsed_minutes < 0 or elapsed_minutes >= cfg.topic_cooldown_minutes:
            return 0.0

        # Strongest right after the topic was seen; fades linearly to zero by cooldown end.
        similarity = float(best_topic.get("similarity", 1.0))
        cooldown_factor = 1.0 - (elapsed_minutes / max(1.0, float(cfg.topic_cooldown_minutes)))
        return max(0.0, min(1.0, cfg.repeated_topic_penalty * cooldown_factor * max(0.0, similarity)))

    async def _maybe_evolve_topic(
        self,
        event: MessageEvent,
        embedding: list[float],
        profile: InterestProfile,
        similar_topics: list[dict],
    ) -> None:
        cfg = self.config.personality_engine
        should_consider_topic = (
            profile.final_interest >= cfg.save_interest_threshold
            or profile.memory >= cfg.memory_interest_threshold
            or bool(similar_topics)
        )
        if not should_consider_topic:
            return

        best_topic = similar_topics[0] if similar_topics else None
        if best_topic and float(best_topic.get("similarity", 0.0)) >= cfg.topic_merge_similarity:
            old_score = float(best_topic.get("learned_interest", 0.0))
            seen_count = int(best_topic.get("seen_count", 0))
            new_score = self._evolve_learned_score(old_score=old_score, profile=profile)
            await self.storage.update_interest_topic(
                topic_id=int(best_topic["id"]),
                embedding=embedding,
                learned_interest=new_score,
                curiosity=profile.curiosity,
                humor=profile.humor,
                memory=profile.memory,
                reply=profile.reply,
                seen_count=seen_count + 1,
                topic_text=event.text[:240],
            )
            logger.info(
                "interest_topic updated chat=%s topic_id=%s old=%.2f new=%.2f seen=%s repetition=%.2f",
                event.chat_id,
                best_topic["id"],
                old_score,
                new_score,
                seen_count + 1,
                profile.repetition_penalty,
            )
            return

        # New topics should not be created just because innate noise is high. Require
        # enough novelty or memory signal so random attractors do not flood storage.
        if profile.final_interest < cfg.save_interest_threshold and profile.memory < cfg.memory_interest_threshold:
            return

        initial_learned = max(
            cfg.min_learned_interest,
            min(1.0, 0.45 * profile.final_interest + 0.20 * profile.memory),
        )
        await self.storage.create_interest_topic(
            chat_id=event.chat_id,
            topic_text=event.text[:240],
            embedding=embedding,
            learned_interest=initial_learned,
            curiosity=profile.curiosity,
            humor=profile.humor,
            memory=profile.memory,
            reply=profile.reply,
        )
        logger.info(
            "interest_topic created chat=%s learned=%.2f text_preview=%r",
            event.chat_id,
            initial_learned,
            event.text[:80],
        )

    def _evolve_learned_score(self, *, old_score: float, profile: InterestProfile) -> float:
        cfg = self.config.personality_engine
        old_score = max(0.0, min(1.0, old_score))

        positive_signal = max(0.0, profile.final_interest - cfg.save_interest_threshold)
        negative_signal = max(0.0, cfg.save_interest_threshold - profile.final_interest)

        positive_delta = cfg.learn_rate * positive_signal
        negative_delta = cfg.learn_negative_rate * negative_signal
        repetition_delta = cfg.learn_repetition_decay * profile.repetition_penalty

        new_score = (
            old_score * (1.0 - cfg.learn_decay)
            + positive_delta
            - negative_delta
            - repetition_delta
        )
        return max(float(cfg.min_learned_interest), min(1.0, new_score))

    async def render_fingerprint(self, output_path: str | Path | None = None) -> Path:
        await self.ensure_ready()
        cfg = self.config.personality_engine
        if output_path is None:
            output_path = Path(self.config.app.data_dir) / "interest_fingerprint.svg"
        renderer = InterestFingerprintRenderer(self.field)
        return renderer.render_svg(output_path, size=cfg.fingerprint_size, cell=cfg.fingerprint_cell_px)

    async def top_interests_text(self, chat_id: int, *, limit: int = 8) -> str:
        rows = await self.storage.get_interest_topics(chat_id, limit=limit)
        if not rows:
            return "У Бямки пока нет выраженных интересов в этом чате."
        lines = []
        for row in rows:
            lines.append(
                "- "
                f"{row['topic_text']} "
                f"(interest={float(row['learned_interest']):.2f}, seen={int(row['seen_count'])})"
            )
        return "\n".join(lines)
