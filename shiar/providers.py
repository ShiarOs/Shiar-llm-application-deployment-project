
import asyncio
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from shiar.config import Settings
from shiar.policy import DECLINE_MESSAGE, match_policy


@dataclass(frozen=True)
class GenerationParams:
    """Every parameter that can change the model output (all go in the cache key)."""

    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class ProviderResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class ProviderError(RuntimeError):
    """The upstream LLM failed (network error, 5xx, rate limit, bad payload...)."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def generate(
        self, system_prompt: str, prompt: str, params: GenerationParams
    ) -> ProviderResult: ...


def _estimate_tokens(text: str) -> int:
    # Rough rule of thumb: ~4 characters per token for English text.
    return max(1, len(text) // 4)


class DryProvider:
    """Deterministic offline provider: no network, no key, no cost.

    It answers from the approved policies with the same routing the real
    model is instructed to follow, so functional cases can run in CI.
    """

    name = "dry"

    def __init__(self, model: str = "dry-policy-v1", delay_seconds: float = 0.02):
        self.model = model
        self.delay_seconds = delay_seconds  # simulates a small network round trip

    async def generate(
        self, system_prompt: str, prompt: str, params: GenerationParams
    ) -> ProviderResult:
        await asyncio.sleep(self.delay_seconds)
        policy = match_policy(prompt)
        text = policy.rule if policy else DECLINE_MESSAGE
        return ProviderResult(
            text=text,
            model=self.model,
            input_tokens=_estimate_tokens(system_prompt + prompt),
            output_tokens=_estimate_tokens(text),
        )


class LangChainProvider:
    """Adapter that wraps any LangChain chat model behind ``LLMProvider``."""

    def __init__(self, name: str, model: str, chat_factory: Any):
        self.name = name
        self.model = model
        # The factory builds a chat model for given params. LangChain chat
        # models are cheap to construct, and building one per parameter set
        # keeps temperature/max_tokens explicit instead of mutating shared state.
        self._chat_factory = chat_factory
        self._clients: dict[GenerationParams, Any] = {}

    def _client(self, params: GenerationParams) -> Any:
        if params not in self._clients:
            self._clients[params] = self._chat_factory(params)
        return self._clients[params]

    async def generate(
        self, system_prompt: str, prompt: str, params: GenerationParams
    ) -> ProviderResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        try:
            message = await self._client(params).ainvoke(messages)
        except asyncio.CancelledError:
            raise  # let our own timeout cancel the call cleanly
        except Exception as exc:  # vendor SDKs raise many different types
            raise ProviderError(
                f"{self.name} call failed: {type(exc).__name__}"
            ) from exc

        text = message.text if isinstance(message.text, str) else str(message.content)
        text = text.strip()
        if not text:
            raise ProviderError(f"{self.name} returned an empty answer")
        usage = getattr(message, "usage_metadata", None) or {}
        return ProviderResult(
            text=text,
            model=self.model,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )


def build_provider(settings: Settings) -> LLMProvider:
    """Create the provider selected by ``MODEL_PROVIDER``.

    Imports are lazy so the dry provider never loads vendor SDKs.
    Retries are disabled in the SDK (max_retries=0): our service owns the
    timeout budget, and hidden SDK retries would silently multiply latency.
    """
    if settings.is_dry:
        return DryProvider()

    common: dict[str, Any] = {
        "model": settings.model,
        "api_key": settings.api_key,
        "max_retries": 0,
        "timeout": settings.request_timeout_seconds,
    }

    def factory(params: GenerationParams) -> Any:
        kwargs: dict[str, Any] = {
            **common,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        match settings.provider:
            case "groq":
                from langchain_groq import ChatGroq

                return ChatGroq(**kwargs)
            case "mistral":
                from langchain_mistralai import ChatMistralAI

                return ChatMistralAI(**kwargs)
            case "openrouter":
                from langchain_openrouter import ChatOpenRouter

                if settings.openrouter_fallback_model:
                    kwargs["model_kwargs"] = {
                        "models": [settings.openrouter_fallback_model]
                    }
                return ChatOpenRouter(**kwargs)
            case "google":
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(**kwargs)
            case "openai":
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(**kwargs, reasoning_effort="none")
        raise ValueError(f"unsupported provider {settings.provider}")

    return LangChainProvider(settings.provider, settings.model, factory)
