
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_PROMPT_CHARS = 2000
# Zero-width space, zero-width (non-)joiner and byte-order mark: invisible text.
INVISIBLE_CHARS = {chr(0x200B), chr(0x200C), chr(0x200D), chr(0xFEFF)}


class GenerateRequest(BaseModel):

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(
        min_length=1,
        max_length=MAX_PROMPT_CHARS,
        description="Customer-support question in plain text.",
        examples=["What evidence is required for a duplicate-payment refund?"],
    )
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=300, ge=16, le=1024)

    @field_validator("prompt")
    @classmethod
    def prompt_must_have_text(cls, value: str) -> str:
        # str_strip_whitespace already removed outer spaces; this also rejects
        # prompts made only of invisible characters such as zero-width spaces.
        visible = "".join(ch for ch in value if ch not in INVISIBLE_CHARS)
        if not visible.strip():
            raise ValueError("prompt must contain visible text")
        return value


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class GenerateResponse(BaseModel):
    answer: str
    cache_hit: bool
    trace_id: str
    provider: str
    model: str
    usage: Usage
    cost_usd: float
    latency_ms: float
    quality_score: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    provider: str
    model: str
    environment: str
    release: str
    problems: list[str] = []


class ErrorResponse(BaseModel):
    error: str
    detail: str
    trace_id: str | None = None
