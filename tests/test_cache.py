
import asyncio

import pytest

from shiar.cache import TTLCache, make_cache_key
from shiar.providers import GenerationParams
from shiar.schemas import GenerateRequest
from shiar.service import GenerationService, Overloaded
from tests.conftest import make_settings
from tests.fakes import CountingProvider

P = GenerationParams(temperature=0.0, max_tokens=300)


def key(provider="dry", model="m", prompt="hello", params=P):
    return make_cache_key(provider=provider, model=model, prompt=prompt, params=params)


def test_key_is_stable_and_parameter_aware():
    assert key() == key()
    assert key() == key(prompt="  hello  ")  # whitespace-only differences collapse
    assert key() != key(prompt="Hello")  # case can change meaning: keep distinct
    assert key() != key(model="other")
    assert key() != key(provider="groq")
    assert key() != key(params=GenerationParams(0.5, 300))
    assert key() != key(params=GenerationParams(0.0, 100))


def test_ttl_expiry_with_fake_clock():
    now = [1000.0]
    cache = TTLCache(ttl_seconds=10, clock=lambda: now[0])
    cache.set("k", "v")
    now[0] += 9.9
    assert cache.get("k") == "v"
    now[0] += 0.2  # 10.1 s after insert
    assert cache.get("k") is None


def test_lru_eviction():
    cache = TTLCache(ttl_seconds=60, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # "a" is now most recently used
    cache.set("c", 3)  # evicts "b"
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3


@pytest.mark.anyio
async def test_single_flight_deduplicates_concurrent_misses():
    cache = TTLCache(ttl_seconds=60)
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "answer"

    results = await asyncio.gather(
        *(cache.get_or_compute("k", compute) for _ in range(5))
    )
    assert calls == 1
    assert [hit for _, hit in results].count(False) == 1  # exactly one real miss


class PeakTrackingProvider(CountingProvider):
    """Records the highest number of provider calls running at the same time."""

    def __init__(self):
        super().__init__(delay_seconds=0.05)
        self.active = self.peak = 0

    async def generate(self, system_prompt, prompt, params):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            return await super().generate(system_prompt, prompt, params)
        finally:
            self.active -= 1


@pytest.mark.anyio
async def test_semaphore_bounds_provider_concurrency():
    provider = PeakTrackingProvider()
    settings = make_settings(max_concurrent_requests=2, request_timeout_seconds=5)
    service = GenerationService(settings, provider, TTLCache(60), _NullSink())
    prompts = [f"How should support handle locked account {i}?" for i in range(6)]
    await asyncio.gather(
        *(service.generate(GenerateRequest(prompt=p)) for p in prompts)
    )
    assert provider.peak == 2
    assert provider.calls == 6


@pytest.mark.anyio
async def test_overload_returns_503_when_queue_wait_expires():
    provider = CountingProvider(delay_seconds=0.3)
    settings = make_settings(
        max_concurrent_requests=1, queue_timeout_seconds=0.05, request_timeout_seconds=5
    )
    service = GenerationService(settings, provider, TTLCache(60), _NullSink())
    results = await asyncio.gather(
        service.generate(GenerateRequest(prompt="refund question A")),
        service.generate(GenerateRequest(prompt="refund question B")),
        return_exceptions=True,
    )
    assert any(isinstance(r, Overloaded) for r in results)


class _NullSink:
    def export(self, trace):
        pass

    def flush(self):
        pass


@pytest.fixture
def anyio_backend():
    return "asyncio"
