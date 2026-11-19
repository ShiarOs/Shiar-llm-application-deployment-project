
import json

import pytest

from tests.fakes import CountingProvider, FailingProvider, SlowProvider

REFUND = "What evidence is required for a duplicate-payment refund?"


def test_health_is_200(build_client):
    client, _ = build_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_with_dry_provider(build_client):
    client, service = build_client()
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert service.provider.calls == 0  # readiness never calls the provider


def test_ready_fails_when_live_key_missing(build_client):
    client, _ = build_client(provider="groq", model="openai/gpt-oss-20b", api_key="")
    # provider="groq" here is only a setting; the injected provider stays fake
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert "GROQ_API_KEY" in body["problems"][0]


def test_generate_valid_request(build_client):
    client, _ = build_client()
    response = client.post("/generate", json={"prompt": REFUND})
    assert response.status_code == 200
    body = response.json()
    assert "receipt" in body["answer"] and "transaction" in body["answer"]
    assert body["cache_hit"] is False
    assert len(body["trace_id"]) == 32
    assert response.headers["X-Trace-Id"] == body["trace_id"]
    assert body["usage"]["total_tokens"] > 0
    assert body["quality_score"] == 1.0


def test_second_identical_request_hits_cache(build_client):
    client, service = build_client()
    first = client.post("/generate", json={"prompt": REFUND}).json()
    second = client.post("/generate", json={"prompt": REFUND}).json()
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["answer"] == first["answer"]
    assert second["usage"]["total_tokens"] == 0  # no new provider tokens
    assert second["cost_usd"] == 0.0
    assert service.provider.calls == 1  # the proof: only one provider call


def test_different_parameters_do_not_share_cache(build_client):
    client, service = build_client()
    client.post("/generate", json={"prompt": REFUND, "temperature": 0.0})
    other = client.post("/generate", json={"prompt": REFUND, "temperature": 0.7}).json()
    assert other["cache_hit"] is False
    assert service.provider.calls == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "   "},
        {"prompt": chr(0x200B) * 3},
        {"prompt": "x" * 2001},
        {},
        {"prompt": 123},
        {"prompt": "ok", "temperature": 3},
        {"prompt": "ok", "max_tokens": 0},
        {"prompt": "ok", "unexpected": "field"},
    ],
    ids=[
        "empty",
        "blank",
        "invisible",
        "too_long",
        "missing",
        "not_string",
        "bad_temperature",
        "bad_max_tokens",
        "extra_field",
    ],
)
def test_invalid_input_is_rejected_before_provider(build_client, payload):
    client, service = build_client()
    response = client.post("/generate", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert service.provider.calls == 0  # rejected BEFORE the provider


def test_malformed_json_is_rejected(build_client):
    client, service = build_client()
    response = client.post(
        "/generate", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert service.provider.calls == 0


def test_provider_timeout_returns_504(build_client, trace_file):
    client, service = build_client(fake=SlowProvider())
    response = client.post("/generate", json={"prompt": REFUND})
    assert response.status_code == 504
    body = response.json()
    assert body["error"] == "upstream_timeout"
    assert body["trace_id"]
    assert service.provider.calls == 1  # this failure DID reach the provider
    trace = json.loads(trace_file.read_text().splitlines()[-1])
    assert trace["status"] == "upstream_timeout"
    assert trace["quality_score"] is None  # failures are not quality samples


def test_upstream_failure_returns_502_and_is_not_cached(build_client):
    client, service = build_client(fake=FailingProvider())
    first = client.post("/generate", json={"prompt": REFUND})
    second = client.post("/generate", json={"prompt": REFUND})
    assert first.status_code == second.status_code == 502
    assert first.json()["error"] == "upstream_failure"
    assert service.provider.calls == 2  # errors are never cached


def test_unsupported_policy_is_declined(build_client):
    client, _ = build_client()
    body = client.post(
        "/generate", json={"prompt": "Can we promise a lifetime subscription discount?"}
    ).json()
    assert "cannot confirm" in body["answer"].lower()
    assert "approved policy" in body["answer"].lower()


def test_trace_masks_prompt_by_default(build_client, trace_file):
    client, _ = build_client()
    client.post("/generate", json={"prompt": "Locked account for jane@example.com"})
    trace = json.loads(trace_file.read_text().splitlines()[-1])
    assert "jane@example.com" not in trace_file.read_text()
    assert set(trace["prompt"]) == {"sha256", "chars"}
    assert [s["name"] for s in trace["stages"]] == [
        "cache",
        "prepare",
        "generation",
        "quality",
    ]


def test_cache_hit_trace_has_no_generation_stage(build_client, trace_file):
    client, _ = build_client()
    client.post("/generate", json={"prompt": REFUND})
    client.post("/generate", json={"prompt": REFUND})
    hit = json.loads(trace_file.read_text().splitlines()[-1])
    assert hit["cache_hit"] is True
    assert [s["name"] for s in hit["stages"]] == ["cache"]
    assert hit["input_tokens"] == hit["output_tokens"] == 0


def test_counting_provider_is_a_real_provider():
    from shiar.providers import LLMProvider

    assert isinstance(CountingProvider(), LLMProvider)
