from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterestProfile:
    """Snapshot of Byamka's inner reaction to a message/topic.

    Values are normalized to 0..1. They are not commands; the router uses them as bias.
    `repetition_penalty` and `novelty_penalty` are anti-loop signals: they reduce
    final_interest when a similar topic was seen too often or too recently.
    """

    innate_interest: float
    learned_interest: float
    final_interest: float
    curiosity: float
    humor: float
    memory: float
    reply: float
    novelty: float
    novelty_penalty: float = 0.0
    repetition_penalty: float = 0.0

    def to_prompt_text(self) -> str:
        return (
            f"innate={self.innate_interest:.2f}, learned={self.learned_interest:.2f}, "
            f"final={self.final_interest:.2f}, curiosity={self.curiosity:.2f}, "
            f"humor={self.humor:.2f}, memory={self.memory:.2f}, reply={self.reply:.2f}, "
            f"novelty={self.novelty:.2f}, novelty_penalty={self.novelty_penalty:.2f}, "
            f"repetition_penalty={self.repetition_penalty:.2f}"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "innate_interest": self.innate_interest,
            "learned_interest": self.learned_interest,
            "final_interest": self.final_interest,
            "curiosity": self.curiosity,
            "humor": self.humor,
            "memory": self.memory,
            "reply": self.reply,
            "novelty": self.novelty,
            "novelty_penalty": self.novelty_penalty,
            "repetition_penalty": self.repetition_penalty,
        }


class PerlinInterestField:
    """Seeded Perlin-like field over a low-dimensional semantic projection.

    Embeddings are projected to a small latent space and then evaluated by several
    independent fractal Perlin fields. This gives the bot stable, smooth "innate"
    interests: close semantic points usually produce close scores.
    """

    AXIS_SEED_OFFSETS = {
        "innate_interest": 11,
        "curiosity": 23,
        "humor": 37,
        "memory": 41,
        "reply": 53,
        "fingerprint_r": 67,
        "fingerprint_g": 71,
        "fingerprint_b": 79,
    }

    def __init__(
        self,
        *,
        embedding_dim: int,
        latent_dim: int,
        seed: int,
        noise_scale: float = 2.7,
        octaves: int = 4,
        persistence: float = 0.55,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.latent_dim = max(2, int(latent_dim))
        self.seed = int(seed)
        self.noise_scale = float(noise_scale)
        self.octaves = max(1, int(octaves))
        self.persistence = float(persistence)

        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(
            0.0,
            1.0 / math.sqrt(max(1, self.latent_dim)),
            size=(self.embedding_dim, self.latent_dim),
        ).astype(np.float32)

    def project_embedding(self, embedding: list[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(embedding, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.size != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {x.size}")
        norm = float(np.linalg.norm(x))
        if norm > 0:
            x = x / norm
        z = x @ self.projection
        return np.tanh(z).astype(np.float32)

    def evaluate_latent(
        self,
        z: np.ndarray,
        *,
        learned_interest: float = 0.0,
        novelty: float = 0.5,
        novelty_penalty: float = 0.0,
        repetition_penalty: float = 0.0,
        base_interest: float = 0.35,
        innate_weight: float = 0.20,
        learned_weight: float = 0.35,
        novelty_weight: float = 0.28,
        max_innate_contribution: float = 0.12,
    ) -> InterestProfile:
        innate = self.axis_value(z, "innate_interest")
        curiosity = self.axis_value(z, "curiosity")
        humor = self.axis_value(z, "humor")
        memory = self.axis_value(z, "memory")
        reply = self.axis_value(z, "reply")

        learned = _clamp01(learned_interest)
        novelty = _clamp01(novelty)
        novelty_penalty = _clamp01(novelty_penalty)
        repetition_penalty = _clamp01(repetition_penalty)

        # Innate interest is capped deliberately: a seed-generated preference must
        # never become an obsession. Novelty is used as a modifier around neutral 0.5,
        # not as a full topic score by itself.
        innate_contribution = min(float(max_innate_contribution), float(innate_weight) * innate)
        novelty_contribution = float(novelty_weight) * (novelty - 0.5)
        final = _clamp01(
            float(base_interest)
            + innate_contribution
            + float(learned_weight) * learned
            + novelty_contribution
            - repetition_penalty
        )

        return InterestProfile(
            innate_interest=innate,
            learned_interest=learned,
            final_interest=final,
            curiosity=curiosity,
            humor=humor,
            memory=memory,
            reply=reply,
            novelty=novelty,
            novelty_penalty=novelty_penalty,
            repetition_penalty=repetition_penalty,
        )

    def evaluate_embedding(
        self,
        embedding: list[float] | np.ndarray,
        *,
        learned_interest: float = 0.0,
        novelty: float = 0.5,
        novelty_penalty: float = 0.0,
        repetition_penalty: float = 0.0,
        base_interest: float = 0.35,
        innate_weight: float = 0.20,
        learned_weight: float = 0.35,
        novelty_weight: float = 0.28,
        max_innate_contribution: float = 0.12,
    ) -> tuple[InterestProfile, np.ndarray]:
        z = self.project_embedding(embedding)
        return self.evaluate_latent(
            z,
            learned_interest=learned_interest,
            novelty=novelty,
            novelty_penalty=novelty_penalty,
            repetition_penalty=repetition_penalty,
            base_interest=base_interest,
            innate_weight=innate_weight,
            learned_weight=learned_weight,
            novelty_weight=novelty_weight,
            max_innate_contribution=max_innate_contribution,
        ), z

    def axis_value(self, z: np.ndarray, axis: str) -> float:
        seed = self.seed + self.AXIS_SEED_OFFSETS.get(axis, 101)
        coords = np.asarray(z, dtype=np.float32) * self.noise_scale
        return self._fbm(coords, seed)

    def fingerprint_value(self, x: float, y: float, axis: str) -> float:
        """2D field used only for visual fingerprint rendering."""
        seed = self.seed + self.AXIS_SEED_OFFSETS.get(axis, 101)
        coords = np.asarray([x * self.noise_scale, y * self.noise_scale], dtype=np.float32)
        return self._fbm(coords, seed)

    def _fbm(self, coords: np.ndarray, seed: int) -> float:
        value = 0.0
        amplitude = 1.0
        total = 0.0
        frequency = 1.0
        for octave in range(self.octaves):
            value += amplitude * self._perlin_nd(coords * frequency, seed + octave * 1009)
            total += amplitude
            amplitude *= self.persistence
            frequency *= 2.0
        if total <= 0:
            return 0.5
        # Perlin is roughly in [-1, 1], normalize and clamp for safety.
        return _clamp01((value / total + 1.0) * 0.5)

    def _perlin_nd(self, coords: np.ndarray, seed: int) -> float:
        dims = int(coords.size)
        floors = np.floor(coords).astype(np.int64)
        frac = coords - floors
        fade = frac * frac * frac * (frac * (frac * 6 - 15) + 10)

        result = 0.0
        # 2^8 corners is fine for chat-bot traffic. Keep latent_dim modest in config.
        for bits in itertools.product((0, 1), repeat=dims):
            corner = np.asarray(bits, dtype=np.int64)
            grid = floors + corner
            distance = frac - corner
            grad = self._gradient(grid, seed, dims)
            dot = float(np.dot(grad, distance))
            weight = 1.0
            for i, bit in enumerate(bits):
                weight *= fade[i] if bit else 1.0 - fade[i]
            result += weight * dot
        # Scale softly to avoid tiny ranges in high dimensions.
        return float(max(-1.0, min(1.0, result * 1.8)))

    def _gradient(self, grid: np.ndarray, seed: int, dims: int) -> np.ndarray:
        h = hashlib.blake2b(digest_size=16)
        h.update(str(seed).encode("ascii"))
        h.update(b"|")
        h.update(",".join(str(int(v)) for v in grid).encode("ascii"))
        raw_seed = int.from_bytes(h.digest()[:8], "little", signed=False)
        rng = np.random.default_rng(raw_seed)
        grad = rng.normal(size=dims).astype(np.float32)
        norm = float(np.linalg.norm(grad))
        if norm <= 1e-8:
            grad[0] = 1.0
            return grad
        return grad / norm


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
