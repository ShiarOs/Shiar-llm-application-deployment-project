import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "dry",
    "groq",
    "mistral",
    "openrouter",
    "google",
    "openai",
)

PROVIDER_ENV: dict[str, tuple[str, str]] = {
    "groq": ("GROQ_API_KEY", "GROQ_MODEL"),
    "mistral": ("MISTRAL_API_KEY", "MISTRAL_MODEL"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"),
    "google": ("GOOGLE_API_KEY", "GOOGLE_MODEL"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL"),
}


class SettingsError(ValueError):
    """Raised when the environment holds an unusable configuration."""


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    value = int(raw) if raw else default
    if value < minimum:
        raise SettingsError(f"{name} must be >= {minimum}, got {value}")
    return value


def _float(name: str, default: float, low: float, high: float) -> float:
    raw = os.getenv(name, "").strip()
    value = float(raw) if raw else default
    if not low <= value <= high:
        raise SettingsError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable service configuration."""

    provider: str = "dry"
    model: str = "dry-policy-v1"
    api_key: str = field(default="", repr=False)  # repr=False keeps keys out of logs
    openrouter_fallback_model: str = ""

    cache_ttl_seconds: int = 300
    cache_max_entries: int = 512
    max_concurrent_requests: int = 4
    queue_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 30.0

    trace_sink: str = "jsonl"
    trace_path: str = "reports/generated/traces.jsonl"
    trace_sample_rate: float = 1.0
    trace_capture_text: bool = False
    langfuse_public_key: str = field(default="", repr=False)
    langfuse_secret_key: str = field(default="", repr=False)
    langfuse_base_url: str = "https://cloud.langfuse.com"

    environment: str = "development"
    release: str = "local"

    @property
    def is_dry(self) -> bool:
        return self.provider == "dry"

    def readiness_problems(self) -> list[str]:
       
        problems: list[str] = []
        if not self.is_dry and not self.api_key:
            key_env, _ = PROVIDER_ENV[self.provider]
            problems.append(f"{key_env} is missing for provider '{self.provider}'")
        if self.trace_sink == "langfuse" and not (
            self.langfuse_public_key and self.langfuse_secret_key
        ):
            problems.append("TRACE_SINK=langfuse but Langfuse keys are missing")
        return problems


def load_settings() -> Settings:
    
    if os.getenv("PYTHON_DOTENV_DISABLED", "").strip() not in {"1", "true"}:
        load_dotenv(override=False)  # real env vars (Render) win over .env

    provider = os.getenv("MODEL_PROVIDER", "dry").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise SettingsError(
            f"MODEL_PROVIDER must be one of {SUPPORTED_PROVIDERS}, got '{provider}'"
        )

    if provider == "dry":
        model, api_key = "dry-policy-v1", ""
    else:
        key_env, model_env = PROVIDER_ENV[provider]
        api_key = os.getenv(key_env, "").strip()
        model = os.getenv(model_env, "").strip()
        if not model:
            raise SettingsError(f"{model_env} must be set for provider '{provider}'")

    trace_sink = os.getenv("TRACE_SINK", "jsonl").strip().lower()
    if trace_sink not in {"none", "jsonl", "langfuse"}:
        raise SettingsError(
            f"TRACE_SINK must be none, jsonl or langfuse, got '{trace_sink}'"
        )

    return Settings(
        provider=provider,
        model=model,
        api_key=api_key,
        openrouter_fallback_model=os.getenv("OPENROUTER_FALLBACK_MODEL", "").strip(),
        cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 300),
        cache_max_entries=_int("CACHE_MAX_ENTRIES", 512),
        max_concurrent_requests=_int("MAX_CONCURRENT_REQUESTS", 4),
        queue_timeout_seconds=_float("QUEUE_TIMEOUT_SECONDS", 10.0, 0.1, 300),
        request_timeout_seconds=_float("REQUEST_TIMEOUT_SECONDS", 30.0, 0.1, 300),
        trace_sink=trace_sink,
        trace_path=os.getenv("TRACE_PATH", "reports/generated/traces.jsonl").strip(),
        trace_sample_rate=_float("TRACE_SAMPLE_RATE", 1.0, 0.0, 1.0),
        trace_capture_text=_bool("TRACE_CAPTURE_TEXT", False),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
        langfuse_base_url=os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        ).strip(),
        environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development").strip(),
        release=os.getenv("LANGFUSE_RELEASE", "").strip()
        or os.getenv("RENDER_GIT_COMMIT", "local")[:12],
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached after the first call)."""
    return load_settings()