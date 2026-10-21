from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count
from typing import Any, AsyncIterator, Iterator

logger = logging.getLogger(__name__)
WAIT_INTERVAL_SEC = 5.0
_ids = count(1)
_context: ContextVar[str] = ContextVar(
    "model_log_context", default="rquid=- chat=- mode=standalone"
)


@contextmanager
def model_log_context(rquid: str, chat_id: int, mode: str) -> Iterator[None]:
    token = _context.set(f"rquid={rquid} chat={chat_id} mode={mode}")
    try:
        yield
    finally:
        _context.reset(token)


@dataclass
class ModelCall:
    kind: str
    model: str
    stage: str
    endpoint: str
    call_id: int = field(default_factory=lambda: next(_ids))
    started: float = field(default_factory=time.monotonic)
    context: str = field(default_factory=_context.get)
    metrics: dict[str, Any] = field(default_factory=dict)

    def log(self, state: str, details: str = "", level: int = logging.INFO) -> None:
        logger.log(
            level,
            "MODEL %s call=%s kind=%s model=%s stage=%s endpoint=%s "
            "%s elapsed_sec=%.3f%s",
            state, self.call_id, self.kind, self.model, self.stage, self.endpoint,
            self.context, time.monotonic() - self.started,
            " " + details if details else "",
        )

    def record(self, data: dict[str, Any]) -> None:
        message = data.get("message") or {}
        if "thinking" in message:
            self.metrics["thinking_chars"] = len(message["thinking"] or "")
        if "done_reason" in data:
            self.metrics["done_reason"] = data["done_reason"]
        # Ollama durations are nanoseconds; report seconds, omit unavailable fields.
        for source, label in (
            ("total_duration", "server_sec"),
            ("load_duration", "load_sec"),
            ("prompt_eval_duration", "prompt_sec"),
            ("eval_duration", "generation_sec"),
        ):
            value = data.get(source)
            if isinstance(value, (int, float)):
                self.metrics[label] = f"{value / 1_000_000_000:.3f}"
        for source, label in (
            ("prompt_eval_count", "input_tokens"),
            ("eval_count", "output_tokens"),
        ):
            if source in data:
                self.metrics[label] = data[source]

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(WAIT_INTERVAL_SEC)
            self.log("WAIT", "waiting_for=ollama_response")


@asynccontextmanager
async def model_call(
    kind: str, model: str, stage: str, endpoint: str, *, details: str = ""
) -> AsyncIterator[ModelCall]:
    call = ModelCall(kind, model, stage, endpoint)
    call.log("START", details)
    heartbeat = asyncio.create_task(call.heartbeat())
    try:
        yield call
    except asyncio.CancelledError:
        call.log("CANCEL")
        raise
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", "-")
        metrics = " ".join(f"{key}={value}" for key, value in call.metrics.items())
        call.log("ERROR", f"error={type(exc).__name__} status={status} {metrics}".rstrip(), logging.ERROR)
        raise
    else:
        call.log("END", " ".join(f"{key}={value}" for key, value in call.metrics.items()))
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
