
import hashlib
import json
import logging
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from shiar.config import Settings

logger = logging.getLogger("shiar.tracing")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_NUMBER = re.compile(r"\b\d[\d -]{7,}\d\b")  # card, phone, IBAN-like numbers


def redact(text: str) -> str:
    """Mask obvious personal data even when text capture is enabled."""
    return _LONG_NUMBER.sub("[NUMBER]", _EMAIL.sub("[EMAIL]", text))


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Stage:
    name: str
    started_at: float  # epoch seconds
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestTrace:
    trace_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )  # 32 hex = Langfuse/OTel format
    started_at: float = field(default_factory=time.time)
    name: str = "generate"
    status: str = "ok"
    status_message: str = ""
    provider: str = ""
    model: str = ""
    cache_hit: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    price_known: bool = True
    latency_ms: float = 0.0
    quality_score: float | None = None
    quality_reason: str = ""
    prompt: str = ""  # masked or fingerprinted before export
    answer: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    stages: list[Stage] = field(default_factory=list)

    def stage(self, name: str) -> "_StageTimer":
        return _StageTimer(self, name)


class _StageTimer:
    """``with trace.stage("cache") as s: ...`` records one timed stage."""

    def __init__(self, trace: RequestTrace, name: str):
        self.stage = Stage(name=name, started_at=time.time())
        self._trace = trace
        self._t0 = 0.0

    def __enter__(self) -> Stage:
        self._t0 = time.perf_counter()
        return self.stage

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stage.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        if exc_type is not None:
            self.stage.metadata["error"] = exc_type.__name__
        self._trace.stages.append(self.stage)


class TraceSink(Protocol):
    def export(self, trace: RequestTrace) -> None: ...
    def flush(self) -> None: ...


def _public_view(trace: RequestTrace, settings: Settings) -> dict[str, Any]:
    """Serialisable copy of the trace with text masked according to settings."""
    data = asdict(trace)
    for key in ("prompt", "answer"):
        text = data[key]
        if settings.trace_capture_text:
            data[key] = redact(text)
        else:
            data[key] = (
                {"sha256": fingerprint(text), "chars": len(text)} if text else None
            )
    data["environment"] = settings.environment
    data["release"] = settings.release
    return data


class NoopTraceSink:
    def export(self, trace: RequestTrace) -> None:
        return None

    def flush(self) -> None:
        return None


class JsonlTraceSink:
    """Append one JSON object per request to a local file (offline evidence)."""

    def __init__(self, path: str | Path, settings: Settings):
        self.path = Path(path)
        self.settings = settings

    def export(self, trace: RequestTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_public_view(trace, self.settings)) + "\n")

    def flush(self) -> None:
        return None


class LangfuseTraceSink:
    """Export a finished request as a Langfuse observation tree.

    request (root span)
      |- cache            (span: hit / miss)
      |- prepare          (span: prompt + policy version)   [miss only]
      |- generation       (generation: model, usage, cost)  [miss only]
      |- quality          (span)                            [miss only]
    + score "answer_quality" on the trace

    The tree is exported AFTER the request (the lesson's recorder pattern).
    The SDK has no public way to back-date a start time, so each observation
    starts at export time and we end it at ``start + measured duration``.
    Result: durations (and latency widgets) are correct; absolute timestamps
    are shifted by a few milliseconds. The measured values are also kept in
    metadata (``duration_ms``, ``latency_ms``).
    """

    def __init__(self, settings: Settings):
        from langfuse import Langfuse

        self.settings = settings
        self.client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
            environment=settings.environment,
            release=settings.release,
            sample_rate=1.0,  # sampling is decided once, in should_sample()
        )

    def export(self, trace: RequestTrace) -> None:
        view = _public_view(trace, self.settings)
        root = self.client.start_observation(
            trace_context={"trace_id": trace.trace_id},
            name=f"shiar-{trace.name}",
            as_type="span",
            input={"prompt": view["prompt"], "params": trace.params},
            output={"answer": view["answer"], "cache_hit": trace.cache_hit},
            level="ERROR" if trace.status != "ok" else "DEFAULT",
            status_message=trace.status_message or None,
            metadata={
                "status": trace.status,
                "provider": trace.provider,
                "model": trace.model,
                "latency_ms": trace.latency_ms,
                "cache_hit": trace.cache_hit,
                "price_known": trace.price_known,
                "started_at": datetime.fromtimestamp(
                    trace.started_at, tz=UTC
                ).isoformat(),
            },
        )
        root_started_ns = time.time_ns()  # right after the root was created
        for stage in trace.stages:
            if stage.name == "generation":
                child = root.start_observation(
                    name="generation",
                    as_type="generation",
                    model=trace.model,
                    model_parameters=trace.params,
                    usage_details={
                        "input": trace.input_tokens,
                        "output": trace.output_tokens,
                    },
                    cost_details={"total": trace.cost_usd},
                    metadata={**stage.metadata, "duration_ms": stage.duration_ms},
                    level="ERROR" if "error" in stage.metadata else "DEFAULT",
                )
            else:
                child = root.start_observation(
                    name=stage.name,
                    as_type="span",
                    metadata={**stage.metadata, "duration_ms": stage.duration_ms},
                )
            _end_after(child, stage.duration_ms)
        if trace.quality_score is not None:
            root.score_trace(
                name="answer_quality",
                value=trace.quality_score,
                comment=trace.quality_reason,
            )
        _end_after(root, trace.latency_ms, root_started_ns)

    def flush(self) -> None:
        self.client.flush()


def _end_after(
    observation: Any, duration_ms: float, started_ns: int | None = None
) -> None:
    """End an observation ``duration_ms`` after it started (default: now)."""
    start = time.time_ns() if started_ns is None else started_ns
    observation.end(end_time=start + int(duration_ms * 1_000_000))


def should_sample(trace: RequestTrace, settings: Settings) -> bool:
    """Always keep errors; sample successful requests at TRACE_SAMPLE_RATE."""
    if trace.status != "ok":
        return True
    return random.random() < settings.trace_sample_rate


def build_trace_sink(settings: Settings) -> TraceSink:
    if settings.trace_sink == "langfuse" and settings.langfuse_public_key:
        try:
            return LangfuseTraceSink(settings)
        except Exception:  # telemetry must never take the API down
            logger.exception("Langfuse init failed, falling back to JSONL traces")
    if settings.trace_sink in {"jsonl", "langfuse"}:
        return JsonlTraceSink(settings.trace_path, settings)
    return NoopTraceSink()
