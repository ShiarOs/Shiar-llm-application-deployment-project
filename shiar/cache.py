
import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from shiar.policy import POLICY_VERSION
from shiar.providers import GenerationParams


def make_cache_key(
    *, provider: str, model: str, prompt: str, params: GenerationParams
) -> str:
    """Hash EVERYTHING that can change the answer.

    If any of these differ, the answer may differ, so the key must differ:
    provider, model, policy version (system prompt), prompt text,
    temperature and max_tokens.
    """
    payload = {
        "provider": provider,
        "model": model,
        "policy_version": POLICY_VERSION,
        "prompt": " ".join(prompt.split()),  # collapse repeated whitespace only
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class _Entry:
    value: Any
    expires_at: float


class TTLCache:
    """Async-safe in-memory cache: entries expire after ``ttl_seconds`` and the
    least-recently-used entry is evicted when ``max_entries`` is reached."""

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int = 512,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock  # injectable so tests can "travel in time"
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at <= self._clock():
            del self._data[key]  # expired: behave exactly like a miss
            self.misses += 1
            return None
        self._data.move_to_end(key)  # mark as recently used
        self.hits += 1
        return entry.value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = _Entry(value, self._clock() + self.ttl_seconds)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)  # evict least recently used

    def clear(self) -> None:
        self._data.clear()
        self.hits = self.misses = 0

    async def get_or_compute(
        self, key: str, compute: Callable[[], Awaitable[Any]]
    ) -> tuple[Any, bool]:
        """Return ``(value, cache_hit)``.

        Single-flight: if 5 identical requests arrive together, only the first
        calls the provider; the others await the same result instead of
        paying for 4 duplicate provider calls. Failures are never cached.
        """
        cached = self.get(key)
        if cached is not None:
            return cached, True

        if key in self._inflight:  # someone is already computing this key
            return await asyncio.shield(self._inflight[key]), True

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            value = await compute()
        except asyncio.CancelledError:
            future.cancel()  # waiting duplicates are cancelled too
            raise
        except Exception as exc:
            future.set_exception(exc)  # waiting duplicates receive the same error
            future.exception()  # mark as retrieved to avoid "never retrieved" warnings
            raise
        else:
            self.set(key, value)
            future.set_result(value)
            return value, False
        finally:
            self._inflight.pop(key, None)

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "entries": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
