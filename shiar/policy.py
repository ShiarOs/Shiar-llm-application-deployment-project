

from dataclasses import dataclass

POLICY_VERSION = "2026-11-v1"

DECLINE_MESSAGE = (
    "I cannot confirm that. It is not covered by an approved policy, "
    "so please escalate it to a team lead before promising anything to the customer."
)


@dataclass(frozen=True)
class Policy:
    topic: str
    keywords: tuple[str, ...]  # used by the dry provider and the quality rubric
    rule: str
    anchors: tuple[str, ...]  # terms a correct answer is expected to contain


POLICIES: tuple[Policy, ...] = (
    Policy(
        topic="duplicate_payment_refund",
        keywords=("refund", "duplicate", "double charge", "charged twice"),
        rule=(
            "Duplicate-payment refunds require the customer's receipt and the "
            "transaction ID of both charges. Verify that both transactions "
            "exist before approving the refund."
        ),
        anchors=("receipt", "transaction"),
    ),
    Policy(
        topic="account_access",
        keywords=("locked", "lock", "login", "log in", "password", "account access"),
        rule=(
            "For a locked account, verify the customer's identity first, then "
            "send a password reset link. Never share or change credentials over chat."
        ),
        anchors=("identity", "reset"),
    ),
    Policy(
        topic="privacy_request",
        keywords=("delete", "personal data", "gdpr", "privacy", "erase"),
        rule=(
            "Data deletion and other personal-data requests go to the privacy "
            "team for human review. Support must not delete data directly."
        ),
        anchors=("privacy", "human review"),
    ),
    Policy(
        topic="incident_communication",
        keywords=("incident", "outage", "downtime", "status"),
        rule=(
            "During an active incident, point customers to the status page and "
            "share only confirmed information. Do not speculate about causes or ETAs."
        ),
        anchors=("status page", "confirmed"),
    ),
)


def match_policy(prompt: str) -> Policy | None:
    """Return the first policy whose keywords appear in the prompt, if any."""
    text = prompt.lower()
    for policy in POLICIES:
        if any(keyword in text for keyword in policy.keywords):
            return policy
    return None


def build_system_prompt() -> str:
    """System prompt that lists every approved policy and the decline rule."""
    rules = "\n".join(f"- {p.rule}" for p in POLICIES)
    return (
        "You are the Shiar-Northstar customer-support assistant for support agents.\n"
        "Answer ONLY from the approved policies below. Be concise (max 4 sentences).\n"
        "Use the policy's own key terms in your answer.\n\n"
        f"Approved policies (version {POLICY_VERSION}):\n{rules}\n\n"
        "If the question is not covered by these policies, reply exactly:\n"
        f'"{DECLINE_MESSAGE}"\n'
        "Never invent discounts, promises, prices or exceptions."
    )


SYSTEM_PROMPT = build_system_prompt()
