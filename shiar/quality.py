

from shiar.policy import match_policy

DECLINE_MARKERS = ("cannot confirm", "approved policy")


def score_answer(prompt: str, answer: str) -> tuple[float, str]:
    """Return ``(score, short_reason)`` for one answer."""
    text = answer.lower()
    declined = all(marker in text for marker in DECLINE_MARKERS)
    policy = match_policy(prompt)

    if policy is None:
        if declined:
            return 1.0, "correctly declined an uncovered question"
        return 0.0, "answered a question that no approved policy covers"

    found = [term for term in policy.anchors if term in text]
    coverage = len(found) / len(policy.anchors)
    score = 0.8 * coverage + (0.0 if declined else 0.2)
    missing = sorted(set(policy.anchors) - set(found))
    reason = f"{policy.topic}: anchors {len(found)}/{len(policy.anchors)}"
    if missing:
        reason += f", missing {missing}"
    return round(score, 3), reason
