
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from shiar.cache import TTLCache
from shiar.config import Settings
from shiar.main import create_app
from shiar.service import GenerationService
from shiar.tracing import JsonlTraceSink
from tests.fakes import CountingProvider


def make_settings(**overrides) -> Settings:
    base: dict[str, Any] = {
        "provider": "dry",
        "model": "dry-policy-v1",
        "cache_ttl_seconds": 300,
        "max_concurrent_requests": 4,
        "queue_timeout_seconds": 1.0,
        "request_timeout_seconds": 0.2,
        "trace_sink": "jsonl",
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def trace_file(tmp_path):
    return tmp_path / "traces.jsonl"


@pytest.fixture
def build_client(trace_file) -> Iterator:
    """Factory: ``build_client(fake=..., **settings)`` -> (client, service)."""
    clients: list[TestClient] = []

    def _build(fake=None, **setting_overrides):
        settings = make_settings(**setting_overrides)
        service = GenerationService(
            settings=settings,
            provider=fake or CountingProvider(),
            cache=TTLCache(settings.cache_ttl_seconds),
            trace_sink=JsonlTraceSink(trace_file, settings),
        )
        client = TestClient(create_app(service))
        client.__enter__()  # runs the lifespan (startup)
        clients.append(client)
        return client, service

    yield _build
    for client in clients:
        client.__exit__(None, None, None)
