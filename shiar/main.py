

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from shiar import __version__
from shiar.cache import TTLCache
from shiar.config import Settings, get_settings
from shiar.providers import build_provider
from shiar.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ReadyResponse,
)
from shiar.service import GenerationService, ServiceError
from shiar.tracing import build_trace_sink

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


def build_service(settings: Settings) -> GenerationService:
    return GenerationService(
        settings=settings,
        provider=build_provider(settings),
        cache=TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries),
        trace_sink=build_trace_sink(settings),
    )


router = APIRouter()


def get_service(request: Request) -> GenerationService:
    return request.app.state.service


async def service_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ServiceError)  # registered only for ServiceError
    body = ErrorResponse(error=exc.error, detail=exc.detail, trace_id=exc.trace_id)
    headers = {"X-Trace-Id": exc.trace_id}
    if exc.status_code == 503:
        headers["Retry-After"] = "5"
    return JSONResponse(body.model_dump(), status_code=exc.status_code, headers=headers)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Invalid input is rejected here, BEFORE the service or provider runs.
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}"
        for err in exc.errors()
    )
    body = ErrorResponse(error="invalid_request", detail=problems)
    return JSONResponse(body.model_dump(), status_code=422)


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness: the process is up and serving HTTP. No dependencies checked."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
async def ready(request: Request) -> JSONResponse:
    """Readiness: configuration is usable. Never calls the provider."""
    service = get_service(request)
    settings = service.settings
    problems = settings.readiness_problems()
    body = ReadyResponse(
        status="not_ready" if problems else "ready",
        provider=settings.provider,
        model=settings.model,
        environment=settings.environment,
        release=settings.release,
        problems=problems,
    )
    return JSONResponse(body.model_dump(), status_code=503 if problems else 200)


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid input"},
        502: {"model": ErrorResponse, "description": "Provider failed"},
        503: {"model": ErrorResponse, "description": "Service overloaded"},
        504: {"model": ErrorResponse, "description": "Provider timed out"},
    },
    tags=["inference"],
)
async def generate(payload: GenerateRequest, request: Request) -> JSONResponse:
    result = await get_service(request).generate(payload)
    return JSONResponse(
        result.model_dump(),
        headers={
            "X-Trace-Id": result.trace_id,
            "X-Cache": "HIT" if result.cache_hit else "MISS",
        },
    )


@router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "shiar-support-assistant", "docs": "/docs"}


def create_app(service: GenerationService | None = None) -> FastAPI:
    """App factory. Tests pass a service with fake parts; production passes None."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.service = service or build_service(get_settings())
        yield
        app.state.service.trace_sink.flush()  # deliver buffered telemetry on shutdown

    app = FastAPI(
        title="shiar Support Assistant",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(router)
    return app


app = create_app()
