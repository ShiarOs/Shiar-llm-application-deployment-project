

PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "dry-policy-v1": (0.0, 0.0),
    "openai/gpt-oss-20b": (0.075, 0.30),  # Groq, check console.groq.com/pricing
    "ministral-14b-latest": (0.20, 0.20),  # Mistral, check mistral.ai/pricing
    "google/gemma-4-26b-a4b-it:free": (0.0, 0.0),  # OpenRouter free tier
    "openrouter/free": (0.0, 0.0),
    "gemma-4-26b-a4b-it": (0.0, 0.0),  # Google AI Studio free tier
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> tuple[float, bool]:
    """Return ``(cost_usd, price_known)``."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return 0.0, False
    input_price, output_price = prices
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return round(cost, 8), True
