"""Схемы данных для эндпоинтов управления моделями, ресурсами и пайплайном на готовых моделях."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Метаданные одной модели реестра для GET /api/models/available."""

    name: str
    status: str = Field(description="loaded | not_downloaded | downloading | error | api_available")
    size_gb: float = Field(
        description="Фактический размер в кэше, если загружена; иначе оценка (0 для API-моделей)"
    )
    source: str = Field(description="totalsegmentator | monai_bundle | vista3d | med3d")
    description: str
    last_used: datetime | None = Field(
        default=None, description="Когда модель последний раз реально использовалась"
    )
    use_count: int = Field(default=0, description="Сколько раз модель была использована в этом процессе")


class ModelListResponse(BaseModel):
    """Ответ GET /api/models/available."""

    models: list[ModelInfo]


class ModelDownloadRequest(BaseModel):
    """Тело запроса POST /api/models/download."""

    model_name: str = Field(description="Имя модели из реестра, см. GET /api/models/available")


class TaskAcceptedResponse(BaseModel):
    """Ответ на постановку фоновой задачи в очередь (скачивание модели, запуск пайплайна)."""

    task_id: str
    status: str = Field(default="accepted")
    websocket_url: str = Field(description="Путь WebSocket для отслеживания прогресса этой задачи")


class ProcessingRequest(BaseModel):
    """Тело запроса POST /api/process/{study_id}."""

    modality: str = "auto"
    tumor_model: str = "auto"
    vessel_model: str = "auto"
    use_fast_mode: bool = False
    offline_mode: bool = False
    export_format: str = "glb"


class ResourceInfo(BaseModel):
    """Ответ POST /api/detect-resources."""

    gpu_available: bool
    gpu_memory_gb: float
    gpu_model: str
    ram_gb: float
    disk_free_gb: float
    internet_available: bool
    recommended_models: list[str] = Field(
        default_factory=list, description="Имена стратегий, рекомендованных под текущие ресурсы"
    )


class PipelineStatusResponse(BaseModel):
    """Ответ GET /api/pipeline/status/{task_id}."""

    task_id: str
    status: str = Field(description="running | done | failed | unknown")
    current_stage: str
    progress_percent: int
    message: str | None = None
    selected_models: dict[str, str] = Field(default_factory=dict)
    result: dict | None = Field(
        default=None, description="Полный результат process_with_progress(), когда готов"
    )


class CachedModelInfo(BaseModel):
    """Одна запись кэша для GET /api/cache/info."""

    name: str
    source: str
    task: str
    size_gb: float
    path: str


class CacheInfoResponse(BaseModel):
    """Ответ GET /api/cache/info."""

    total_size_gb: float
    cache_dir: str
    models: list[CachedModelInfo]


class CacheClearResponse(BaseModel):
    """Ответ POST /api/cache/clear."""

    freed_gb: float
    cache_dir: str
