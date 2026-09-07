"""Точка входа FastAPI-приложения Brain3D AI (заменяет прежний api/main.py).

Запуск: uvicorn api.fastapi_app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from api import routes
from api.dependencies import limiter
from core.config import get_settings
from core.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201, ARG001 — сигнатура задана FastAPI
    """Инициализация при старте (логирование, хранилище, кэш моделей)."""
    settings = get_settings()
    setup_logging(level=settings.log_level)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    yield


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Единообразный JSON-ответ 429 при превышении лимита запросов."""
    return JSONResponse(status_code=429, content={"detail": f"Слишком много запросов: {exc.detail}"})


def create_app() -> FastAPI:
    """Фабрика FastAPI-приложения (удобна для тестов и uvicorn --factory)."""
    settings = get_settings()

    app = FastAPI(
        title="Brain3D AI API",
        description="3D-визуализация КТ/МРТ головного мозга с сегментацией на готовых предобученных моделях",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes.router)

    return app


app = create_app()
