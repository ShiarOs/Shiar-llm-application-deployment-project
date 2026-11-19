
import asyncio

from shiar.providers import (
    DryProvider,
    GenerationParams,
    ProviderError,
    ProviderResult,
)


class CountingProvider(DryProvider):
    """Dry provider that counts calls, so tests can prove cache reuse."""

    def __init__(self, delay_seconds: float = 0.0):
        super().__init__(delay_seconds=delay_seconds)
        self.calls = 0

    async def generate(
        self, system_prompt, prompt, params: GenerationParams
    ) -> ProviderResult:
        self.calls += 1
        return await super().generate(system_prompt, prompt, params)


class SlowProvider(CountingProvider):
    """Never answers in time -> triggers our timeout."""

    async def generate(self, system_prompt, prompt, params):
        self.calls += 1
        await asyncio.sleep(10)
        raise AssertionError("unreachable")


class FailingProvider(CountingProvider):
    """Simulates the vendor returning an error (e.g. HTTP 500 / rate limit)."""

    async def generate(self, system_prompt, prompt, params):
        self.calls += 1
        raise ProviderError("dry upstream returned 500")
