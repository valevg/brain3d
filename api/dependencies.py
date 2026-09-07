"""FastAPI-зависимости: настройки, конфигурация, реестр моделей и rate limiting через DI."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import PipelineConfig, Settings, get_pipeline_config, get_settings
from models.pretrained_models import PretrainedModelRegistry
from pipelines.brain_pipeline import ModelCacheManager

SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineConfigDep = Annotated[PipelineConfig, Depends(get_pipeline_config)]


@lru_cache
def get_model_registry() -> PretrainedModelRegistry:
    """Возвращает закэшированный (в рамках процесса) реестр предобученных моделей.

    Использует settings.model_cache_dir как директорию кэша — единая точка правды
    для всех эндпоинтов управления моделями (api/routes.py).
    """
    settings = get_settings()
    return PretrainedModelRegistry(cache_dir=settings.model_cache_dir)


@lru_cache
def get_cache_manager() -> ModelCacheManager:
    """Возвращает закэшированный ModelCacheManager поверх того же реестра/кэша."""
    return ModelCacheManager(get_settings().model_cache_dir, get_model_registry())


ModelRegistryDep = Annotated[PretrainedModelRegistry, Depends(get_model_registry)]
CacheManagerDep = Annotated[ModelCacheManager, Depends(get_cache_manager)]

#: Общий rate limiter процесса (in-memory — для многопроцессного деплоя за балансировщиком
#: нужен storage_uri на Redis, см. https://slowapi.readthedocs.io). Ключ — IP клиента.
limiter = Limiter(key_func=get_remote_address)
