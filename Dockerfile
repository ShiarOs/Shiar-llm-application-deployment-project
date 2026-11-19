# syntax=docker/dockerfile:1
# shiar support assistant: same image locally and on Render.

# ---------- stage 1: build the locked virtual environment ----------
FROM python:3.13-slim AS builder

# uv binary copied from its official image (pinned version, no curl | sh)
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Copy only dependency files first: this layer is cached until they change.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# ---------- stage 2: small runtime image ----------
FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_DOTENV_DISABLED=1 \
    PORT=8000

# Run as an unprivileged user, never as root.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
# The app writes local JSONL traces under reports/: give the user that folder.
RUN mkdir -p /app/reports/generated && chown -R appuser:appuser /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser shiar ./shiar
COPY --chown=appuser:appuser data ./data

USER appuser
EXPOSE 8000

# Lightweight liveness probe using only the standard library.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=2)"

# Render injects $PORT. One worker = one process-local cache (see report).
CMD ["sh", "-c", "exec uvicorn shiar.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
