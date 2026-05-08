from __future__ import annotations

from dataclasses import dataclass
import math


class TokenEstimator:
    """Грубая оценка токенов без tokenizer-зависимости.

    Для русского текста символов на токен часто меньше, чем для английского.
    Берём консервативную оценку: примерно 3.2 символа на токен.
    """

    def __init__(self, chars_per_token: float = 3.2) -> None:
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.chars_per_token))


@dataclass(frozen=True)
class Budget:
    context_window_tokens: int
    reserve_response_tokens: int
    reserve_thinking_tokens: int
    system_tokens: int = 0

    @property
    def output_tokens(self) -> int:
        return self.reserve_response_tokens + self.reserve_thinking_tokens

    @property
    def input_tokens_available(self) -> int:
        value = (
            self.context_window_tokens
            - self.reserve_response_tokens
            - self.reserve_thinking_tokens
            - self.system_tokens
        )
        return max(512, value)


class ContextPacker:
    def __init__(self, estimator: TokenEstimator) -> None:
        self.estimator = estimator

    def pack_newest_first(
        self,
        items: list[str],
        budget_tokens: int,
        separator: str = "\n",
    ) -> list[str]:
        used = 0
        selected_reversed: list[str] = []
        for item in reversed(items):
            cost = self.estimator.count(item) + self.estimator.count(separator)
            if used + cost > budget_tokens:
                break
            selected_reversed.append(item)
            used += cost
        return list(reversed(selected_reversed))
