
import json
from pathlib import Path

import pytest

from shiar.config import SettingsError, load_settings
from shiar.policy import DECLINE_MESSAGE, SYSTEM_PROMPT, match_policy
from shiar.pricing import estimate_cost
from shiar.quality import score_answer
from shiar.tracing import redact

CASES = json.loads(Path("data/evaluation/cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_dry_answers_contain_expected_terms(case):
    """Uses the shared labels as-is (never edited)."""
    policy = match_policy(case["prompt"])
    answer = (policy.rule if policy else DECLINE_MESSAGE).lower()
    for term in case["expected_terms"]:
        assert term in answer


def test_system_prompt_mentions_every_policy_and_decline_rule():
    for term in ("receipt", "identity", "privacy", "status page", "cannot confirm"):
        assert term in SYSTEM_PROMPT.lower()


def test_quality_rubric():
    refund_q = "What evidence is required for a duplicate-payment refund?"
    assert score_answer(refund_q, "Ask for the receipt and transaction ID.")[0] == 1.0
    assert score_answer(refund_q, "Ask for the receipt.")[0] == pytest.approx(0.6)
    assert score_answer("Lifetime discount?", DECLINE_MESSAGE)[0] == 1.0
    assert score_answer("Lifetime discount?", "Sure, 50% forever!")[0] == 0.0


def test_cost_estimation():
    cost, known = estimate_cost("openai/gpt-oss-20b", 1_000_000, 1_000_000)
    assert known and cost == pytest.approx(0.375)
    assert estimate_cost("unknown-model", 10, 10) == (0.0, False)


def test_redact_masks_email_and_long_numbers():
    text = redact("mail jane.doe@example.com card 4111 1111 1111 1111")
    assert "example.com" not in text and "4111" not in text


def test_settings_reject_unknown_provider(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("MODEL_PROVIDER", "skynet")
    with pytest.raises(SettingsError):
        load_settings()


def test_settings_reject_bad_numbers(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("MODEL_PROVIDER", "dry")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "0")
    with pytest.raises(SettingsError):
        load_settings()


def test_api_key_never_appears_in_repr(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_super_secret")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    assert "gsk_super_secret" not in repr(load_settings())
