

import asyncio
import logging
import time
from dataclasses import dataclass

from shiar.cache import TTLCache, make_cache_key
from shiar.config import Settings
from shiar.policy import POLICY_VERSION, SYSTEM_PROMPT
from shiar.pricing import estimate_cost
from shiar.providers import (
    GenerationParams,
    LLMProvider,
    ProviderError,
    ProviderResult,
)
from shiar.quality import score_answer
from shiar.schemas import GenerateRequest, GenerateResponse, Usage
from shiar.tracing import RequestTrace, TraceSink, should_sample

logger = logging.getLogger("shiar.service")


class ServiceError(Exception):
    """A failure that the API turns into an explicit HTTP error response."""

    status_code = 500
    error = "internal_error"

    def __init__(self, detail: str, trace_id: str):
        super().__init__(detail)
        self.detail = detail
        self.trace_id = trace_id


class UpstreamTimeout(ServiceError):
    status_code = 504
    error = "upstream_timeout"


class UpstreamFailure(ServiceError):
    status_code = 502
    error = "upstream_failure"


class Overloaded(ServiceError):
    status_code = 503
    error = "overloaded"


@dataclass
class CachedAnswer:
    """What we store in the cache: the provider result plus its quality score."""

    result: ProviderResult
    quality_score: float
    quality_reason: str


class GenerationService:
    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        cache: TTLCache,
        trace_sink: TraceSink,
    ):
        self.settings = settings
        self.provider = provider
        self.cache = cache
        self.trace_sink = trace_sink
        # Bounds concurrent PROVIDER calls (cache hits never wait for it).
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self.provider_calls = 0  # simple counter, handy when debugging

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        trace = RequestTrace(provider=self.provider.name, model=self.provider.model)
        params = GenerationParams(request.temperature, request.max_tokens)
        trace.prompt = request.prompt
        trace.params = {
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        t0 = time.perf_counter()

        try:
            with trace.stage("cache") as cache_stage:
                key = make_cache_key(
                    provider=self.provider.name,
                    model=self.provider.model,
                    prompt=request.prompt,
                    params=params,
                )
                cache_stage.metadata["key_prefix"] = key[:12]

            async def compute() -> CachedAnswer:
                return await self._call_provider(request.prompt, params, trace)

            cached, hit = await self.cache.get_or_compute(key, compute)
            cache_stage.metadata["hit"] = hit
            trace.cache_hit = hit
            trace.answer = cached.result.text
            if not hit:
                trace.quality_score = cached.quality_score
                trace.quality_reason = cached.quality_reason
            return self._response(trace, cached, hit, t0)
        except ServiceError as exc:
            trace.status, trace.status_message = exc.error, exc.detail
            raise
        except Exception as exc:  # unexpected bug: still traced, still explicit
            trace.status, trace.status_message = "internal_error", type(exc).__name__
            logger.exception("unexpected error in generate")
            raise ServiceError("unexpected internal error", trace.trace_id) from exc
        finally:
            trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            self._export(trace)

    async def _call_provider(
        self, prompt: str, params: GenerationParams, trace: RequestTrace
    ) -> CachedAnswer:
        with trace.stage("prepare") as prep:
            prep.metadata.update(
                {"policy_version": POLICY_VERSION, "prompt_chars": len(prompt)}
            )

        # 1) wait for a free provider slot, but not forever (back-pressure)
        try:
            await asyncio.wait_for(
                self.semaphore.acquire(), timeout=self.settings.queue_timeout_seconds
            )
        except TimeoutError as exc:
            raise Overloaded(
                "too many requests in flight, retry later", trace.trace_id
            ) from exc

        # 2) call the provider with a hard deadline
        try:
            with trace.stage("generation") as gen:
                self.provider_calls += 1
                try:
                    result = await asyncio.wait_for(
                        self.provider.generate(SYSTEM_PROMPT, prompt, params),
                        timeout=self.settings.request_timeout_seconds,
                    )
                except TimeoutError as exc:
                    raise UpstreamTimeout(
                        f"provider did not answer within "
                        f"{self.settings.request_timeout_seconds}s",
                        trace.trace_id,
                    ) from exc
                except ProviderError as exc:
                    raise UpstreamFailure(str(exc), trace.trace_id) from exc
                gen.metadata.update(
                    {
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    }
                )
        finally:
            self.semaphore.release()

        trace.input_tokens, trace.output_tokens = (
            result.input_tokens,
            result.output_tokens,
        )
        trace.cost_usd, trace.price_known = estimate_cost(
            result.model, result.input_tokens, result.output_tokens
        )
        with trace.stage("quality") as q:
            score, reason = score_answer(prompt, result.text)
            q.metadata.update({"score": score, "reason": reason})
        return CachedAnswer(result, score, reason)

    def _response(
        self, trace: RequestTrace, cached: CachedAnswer, hit: bool, t0: float
    ) -> GenerateResponse:
        result = cached.result
        # A cache hit costs nothing and uses no new tokens: report zeros.
        usage = (
            Usage()
            if hit
            else Usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.input_tokens + result.output_tokens,
            )
        )
        return GenerateResponse(
            answer=result.text,
            cache_hit=hit,
            trace_id=trace.trace_id,
            provider=self.provider.name,
            model=result.model,
            usage=usage,
            cost_usd=0.0 if hit else trace.cost_usd,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            quality_score=cached.quality_score,
        )

    def _export(self, trace: RequestTrace) -> None:
        if not should_sample(trace, self.settings):
            return
        try:
            self.trace_sink.export(trace)
        except Exception:  # telemetry failures must never break the response
            logger.exception("trace export failed for %s", trace.trace_id)
