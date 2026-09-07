"""Все HTTP/WebSocket-маршруты Brain3D AI API.

Объединяет: загрузку исследований, управление предобученными моделями (список/
скачивание/кэш), определение ресурсов, запуск пайплайна на готовых моделях
(pipelines.brain_pipeline.BrainModelPipeline) с отслеживанием прогресса по
WebSocket, и отдачу результата.

In-memory хранилища (_PIPELINE_TASKS, _DOWNLOAD_STATE, progress_tracker, счётчики
использования моделей) живут только в памяти одного процесса — как и в исходном
api/routers/segmentation.py, для многопроцессного/распределённого деплоя это нужно
заменить на Redis/БД с pub/sub для WebSocket.
"""

# Намеренно БЕЗ "from __future__ import annotations" в этом файле: PEP 563 (отложенное
# вычисление аннотаций) в паре с декоратором @limiter.limit() (slowapi) ломает резолвинг
# типов у параметров БЕЗ значения по умолчанию (BackgroundTasks, Pydantic-модель тела
# запроса) — FastAPI перестаёт видеть их природу и трактует как обязательные
# query-параметры (проверено эмпирически). Python 3.10+ и так поддерживает `X | Y` и
# `dict[str, X]` как настоящие объекты в рантайме, future-импорт здесь не нужен.

import asyncio
import io
import shutil
import time
import uuid
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse

from api.dependencies import get_cache_manager, get_model_registry, limiter
from api.schemas.models import (
    CachedModelInfo,
    CacheClearResponse,
    CacheInfoResponse,
    ModelDownloadRequest,
    ModelInfo,
    ModelListResponse,
    PipelineStatusResponse,
    ProcessingRequest,
    ResourceInfo,
    TaskAcceptedResponse,
)
from api.schemas.study import StudyUploadResponse
from core.config import Settings, get_settings
from core.exceptions import Brain3DError, ModelNotFoundError, StudyLoadError
from core.io import DicomLoader, NiftiLoader
from core.io.base_loader import BaseLoader
from core.logging_config import get_logger
from models.pretrained_models import ModelSpec, PretrainedModelRegistry
from pipelines.brain_pipeline import (
    STRATEGY_TO_REGISTRY_MODEL,
    TUMOR_STRATEGY_MAP,
    VESSEL_STRATEGY_MAP,
    BrainModelPipeline,
    BrainPipelineConfig,
    ModelCacheManager,
    ResourceDetector,
)

logger = get_logger(__name__)
router = APIRouter()

#: Загрузчики для определения модальности при upload. Реализована локально (а не через
#: pipelines.pipeline_factory.resolve_loader), т.к. pipeline_factory тянет models.registry
#: (и transitively torch) на уровне импорта модуля — тяжёлая зависимость, не нужная
#: для простого определения формата/модальности при загрузке файла.
_UPLOAD_LOADERS: tuple[BaseLoader, ...] = (DicomLoader(), NiftiLoader())


def _resolve_upload_loader(path: Path) -> BaseLoader:
    """Находит подходящий загрузчик (DICOM/NIfTI) для пути."""
    for loader in _UPLOAD_LOADERS:
        if loader.can_load(path):
            return loader
    raise StudyLoadError(f"Формат исследования не распознан: {path}")


# --------------------------------------------------------------------------- #
# Прогресс-трекер (общий для скачивания моделей и запуска пайплайна)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ProgressEvent:
    """Одно событие прогресса фоновой задачи."""

    percent: int | None
    message: str
    timestamp: datetime


class ProgressTracker:
    """In-memory трекер прогресса фоновых задач для WebSocket /ws/progress/{task_id}.

    WebSocket-обработчик ОПРАШИВАЕТ список событий с коротким интервалом, а не
    использует настоящий push через очередь — проще, не требует моста между
    синхронным progress_callback (вызывается из фонового потока BackgroundTasks)
    и asyncio-циклом, и вполне достаточно для одного backend-процесса.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[ProgressEvent]] = defaultdict(list)
        self._final_status: dict[str, str] = {}

    def report(self, task_id: str, percent: int | None, message: str) -> None:
        self._events[task_id].append(
            ProgressEvent(percent=percent, message=message, timestamp=datetime.now(timezone.utc))
        )

    def set_final_status(self, task_id: str, status_value: str) -> None:
        self._final_status[task_id] = status_value

    def get_status(self, task_id: str) -> str:
        if task_id in self._final_status:
            return self._final_status[task_id]
        return "running" if task_id in self._events else "unknown"

    def get_events_since(self, task_id: str, index: int) -> tuple[list[ProgressEvent], int]:
        events = self._events.get(task_id, [])
        return events[index:], len(events)


progress_tracker = ProgressTracker()

#: Стадии пайплайна по контрольным точкам прогресса (см. BrainModelPipeline.process_with_progress).
_STAGE_LABELS: tuple[tuple[int, str], ...] = (
    (0, "Загрузка DICOM"),
    (10, "Выбор и загрузка моделей"),
    (20, "Препроцессинг"),
    (50, "Сегментация"),
    (80, "Построение 3D-мешей"),
    (100, "Экспорт"),
)


def _stage_label_for_percent(percent: int | None) -> str:
    """Переводит процент прогресса в человекочитаемое название текущей стадии."""
    if percent is None:
        return "Завершено"
    label = _STAGE_LABELS[0][1]
    for threshold, name in _STAGE_LABELS:
        if percent >= threshold:
            label = name
    return label


# --------------------------------------------------------------------------- #
# Метрики использования моделей (сколько раз использована, когда в последний раз)
# --------------------------------------------------------------------------- #

_MODEL_USAGE_COUNT: dict[str, int] = defaultdict(int)
_MODEL_LAST_USED: dict[str, datetime] = {}

#: Переходные статусы скачивания, не выводимые из наличия файлов в кэше напрямую
#: ("downloading" во время фоновой задачи, "error" после неудачной попытки).
_DOWNLOAD_STATE: dict[str, str] = {}


def _record_model_usage(strategy_or_model_name: str, registry: PretrainedModelRegistry) -> None:
    """Учитывает использование модели/стратегии в счётчиках для GET /api/models/available.

    Классические методы без записи в реестре (frangi, threshold) тоже считаются —
    под собственным именем стратегии, просто не отображаются в списке моделей
    реестра (у них там нет записи).
    """
    catalog_names = {spec.name for spec in registry.list_available_models()}
    key = (
        strategy_or_model_name
        if strategy_or_model_name in catalog_names
        else STRATEGY_TO_REGISTRY_MODEL.get(strategy_or_model_name, strategy_or_model_name)
    )
    _MODEL_USAGE_COUNT[key] += 1
    _MODEL_LAST_USED[key] = datetime.now(timezone.utc)


def _retry(
    func, attempts: int = 3, base_delay_s: float = 1.0
):  # noqa: ANN001, ANN201 — небольшой внутренний хелпер
    """Повторяет func() при Brain3DError с экспоненциальной задержкой.

    Используется для операций, обращающихся к внешним сервисам (скачивание весов
    моделей) — единичный сетевой сбой не должен сразу считаться окончательной ошибкой.

    Args:
        func: Вызываемое без аргументов; при неудаче должно бросать Brain3DError.
        attempts: Максимальное число попыток.
        base_delay_s: База экспоненциальной задержки между попытками (в секундах).

    Returns:
        Результат успешного вызова func().

    Raises:
        Brain3DError: Последняя ошибка, если все попытки исчерпаны.
    """
    last_error: Brain3DError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Brain3DError as exc:
            last_error = exc
            logger.warning("Попытка %d/%d не удалась: %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(base_delay_s * (2 ** (attempt - 1)))
    assert last_error is not None  # noqa: S101 — достижимо только если attempts >= 1
    raise last_error


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


@router.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Проверка готовности сервиса (Docker HEALTHCHECK, CI)."""
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Исследования: загрузка
# --------------------------------------------------------------------------- #


@router.post(
    "/studies/upload",
    response_model=StudyUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["studies"],
)
async def upload_study(file: UploadFile, settings: Settings = Depends(get_settings)) -> StudyUploadResponse:
    """Принимает исследование (.zip с DICOM-серией или .nii/.nii.gz) и сохраняет его.

    Args:
        file: Загружаемый файл (multipart/form-data).
        settings: Настройки приложения (путь к хранилищу, лимит размера).

    Returns:
        StudyUploadResponse с идентификатором исследования и определённой модальностью.

    Raises:
        HTTPException 400: Формат файла не распознан или файл повреждён.
        HTTPException 413: Файл превышает upload_max_size_mb.
    """
    study_id = str(uuid.uuid4())
    study_dir = settings.storage_dir / study_id
    study_dir.mkdir(parents=True, exist_ok=True)

    raw_path = study_dir / (file.filename or "upload.bin")
    size_bytes = 0
    max_bytes = settings.upload_max_size_mb * 1024 * 1024

    with raw_path.open("wb") as destination:
        while chunk := await file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > max_bytes:
                shutil.rmtree(study_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Файл превышает лимит {settings.upload_max_size_mb} МБ",
                )
            destination.write(chunk)

    study_path = _prepare_study_path(raw_path, study_dir)

    try:
        loader = _resolve_upload_loader(study_path)
        volume = loader.load(study_path)
    except StudyLoadError as exc:
        shutil.rmtree(study_dir, ignore_errors=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info("Исследование %s загружено, модальность=%s", study_id, volume.modality.value)
    return StudyUploadResponse(study_id=study_id, modality=volume.modality, num_slices=volume.voxels.shape[0])


def _prepare_study_path(raw_path: Path, study_dir: Path) -> Path:
    """Распаковывает zip с DICOM-серией либо возвращает путь как есть для NIfTI."""
    if raw_path.suffix.lower() == ".zip":
        extract_dir = study_dir / "dicom"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(raw_path) as archive:
            archive.extractall(extract_dir)
        raw_path.unlink(missing_ok=True)
        return extract_dir
    return raw_path


def _resolve_study_input_path(study_dir: Path) -> Path:
    """Находит исходный файл/директорию исследования внутри study_dir (минуя processed/)."""
    candidates = [p for p in study_dir.glob("*") if p.name != "processed"]
    if not candidates:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="В исследовании не найдено файлов для обработки"
        )
    return candidates[0]


# --------------------------------------------------------------------------- #
# Модели: список / скачивание
# --------------------------------------------------------------------------- #


def _model_status(spec: ModelSpec, registry: PretrainedModelRegistry) -> str:
    """Определяет статус модели: api_available (vista3d) | downloading | error | loaded | not_downloaded."""
    if spec.source == "vista3d":
        return "api_available"
    transient = _DOWNLOAD_STATE.get(spec.name)
    if transient in ("downloading", "error"):
        return transient
    return "loaded" if registry.is_cached(spec.name) else "not_downloaded"


@router.get("/api/models/available", response_model=ModelListResponse, tags=["models"])
def list_available_models(
    registry: PretrainedModelRegistry = Depends(get_model_registry),
    cache_manager: ModelCacheManager = Depends(get_cache_manager),
) -> ModelListResponse:
    """Возвращает список всех моделей реестра с их текущим статусом, размером и метриками использования."""
    cached_sizes_gb = {
        entry["name"]: entry["size_bytes"] / 1024**3 for entry in cache_manager.list_cached_models()
    }

    models = [
        ModelInfo(
            name=spec.name,
            status=_model_status(spec, registry),
            size_gb=round(cached_sizes_gb.get(spec.name, spec.expected_size_gb or 0.0), 3),
            source=spec.source,
            description=spec.description,
            last_used=_MODEL_LAST_USED.get(spec.name),
            use_count=_MODEL_USAGE_COUNT.get(spec.name, 0),
        )
        for spec in registry.list_available_models()
    ]
    return ModelListResponse(models=models)


@router.post(
    "/api/models/download",
    response_model=TaskAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["models"],
)
@limiter.limit("10/minute")
def download_model_endpoint(
    request: Request,  # noqa: ARG001 — требуется slowapi для определения клиента по IP
    body: ModelDownloadRequest,
    background_tasks: BackgroundTasks,
    registry: PretrainedModelRegistry = Depends(get_model_registry),
) -> TaskAcceptedResponse:
    """Запускает фоновое скачивание модели в кэш.

    Args:
        body: Имя модели из реестра (см. GET /api/models/available).
        background_tasks: Механизм FastAPI для фоновой обработки без блокировки ответа.
        registry: Реестр предобученных моделей.

    Returns:
        task_id и адрес WebSocket для отслеживания прогресса.

    Raises:
        HTTPException 404: Модель с таким именем не найдена в реестре.
    """
    try:
        registry.get_model(body.model_name)
    except ModelNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    task_id = str(uuid.uuid4())
    _DOWNLOAD_STATE[body.model_name] = "downloading"
    progress_tracker.report(task_id, 0, f"Начало загрузки модели {body.model_name}")
    background_tasks.add_task(_download_model_task, task_id, body.model_name, registry)

    return TaskAcceptedResponse(task_id=task_id, websocket_url=f"/ws/progress/{task_id}")


def _download_model_task(task_id: str, model_name: str, registry: PretrainedModelRegistry) -> None:
    """Фоновая задача скачивания модели с retry-логикой и обновлением прогресса/статуса."""
    try:
        _retry(lambda: registry.download_model(model_name))
        _DOWNLOAD_STATE[model_name] = "loaded"
        progress_tracker.report(task_id, 100, f"Модель {model_name} загружена")
        progress_tracker.set_final_status(task_id, "done")
        logger.info("Модель %s успешно загружена (task_id=%s)", model_name, task_id)
    except Brain3DError as exc:
        _DOWNLOAD_STATE[model_name] = "error"
        progress_tracker.report(task_id, None, f"Ошибка загрузки {model_name}: {exc}")
        progress_tracker.set_final_status(task_id, "failed")
        logger.exception("Не удалось загрузить модель %s (task_id=%s)", model_name, task_id)


# --------------------------------------------------------------------------- #
# Ресурсы
# --------------------------------------------------------------------------- #


@router.post("/api/detect-resources", response_model=ResourceInfo, tags=["resources"])
def detect_resources(registry: PretrainedModelRegistry = Depends(get_model_registry)) -> ResourceInfo:
    """Определяет доступные ресурсы (GPU/RAM/диск/интернет) и рекомендует модели.

    Рекомендация моделей переиспользует ту же логику выбора, что и реальный запуск
    пайплайна (BrainModelPipeline.select_models), а не отдельную копию правил —
    чтобы рекомендация не могла разойтись с тем, что пайплайн выберет на практике.
    """
    detector = ResourceDetector()
    optimal = detector.get_optimal_config()
    gpu = optimal["gpu"]

    pipeline = BrainModelPipeline(BrainPipelineConfig(), model_cache_dir=registry.cache_dir)
    resources = pipeline_resources_from_optimal(optimal)
    selected = pipeline.select_models("MRI", resources)

    return ResourceInfo(
        gpu_available=gpu["available"],
        gpu_memory_gb=gpu["memory_gb"] or 0.0,
        gpu_model=gpu["name"] or "",
        ram_gb=round(optimal["ram_gb"], 1),
        disk_free_gb=round(optimal["disk_free_gb"], 1),
        internet_available=optimal["has_internet"],
        recommended_models=sorted(set(selected.values())),
    )


def pipeline_resources_from_optimal(
    optimal: dict[str, Any],
):  # noqa: ANN201 — избегаем цикл. импорта ComputeResources
    """Строит ComputeResources из ResourceDetector.get_optimal_config() для переиспользования селектора."""
    from models.pretrained_models import ComputeResources

    gpu = optimal["gpu"]
    return ComputeResources(
        has_gpu=gpu["available"],
        gpu_memory_gb=gpu["memory_gb"],
        has_internet=optimal["has_internet"],
        cpu_count=1,
    )


# --------------------------------------------------------------------------- #
# Пайплайн: запуск / статус / скачивание результата
# --------------------------------------------------------------------------- #

#: task_id -> {"study_id", "pipeline": BrainModelPipeline, "result": dict | None}
_PIPELINE_TASKS: dict[str, dict[str, Any]] = {}


@router.post(
    "/api/process/{study_id}",
    response_model=TaskAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["pipeline"],
)
@limiter.limit("5/minute")
def start_processing(
    request: Request,  # noqa: ARG001 — требуется slowapi
    study_id: str,
    body: ProcessingRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    registry: PretrainedModelRegistry = Depends(get_model_registry),
) -> TaskAcceptedResponse:
    """Запускает полный пайплайн (BrainModelPipeline) для ранее загруженного исследования.

    Args:
        study_id: Идентификатор исследования, возвращённый /studies/upload.
        body: Параметры обработки (модальность/модели/режимы — см. ProcessingRequest).

    Returns:
        task_id и адрес WebSocket для отслеживания прогресса.

    Raises:
        HTTPException 404: Исследование не найдено.
        HTTPException 400: В директории исследования нет файлов для обработки.
    """
    study_dir = settings.storage_dir / study_id
    if not study_dir.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Исследование не найдено")

    dicom_path = _resolve_study_input_path(study_dir)
    task_id = str(uuid.uuid4())

    config = BrainPipelineConfig(
        modality_override=None if body.modality == "auto" else body.modality.upper(),
        tumor_model_strategy=TUMOR_STRATEGY_MAP.get(body.tumor_model, body.tumor_model),
        vessel_model_strategy=VESSEL_STRATEGY_MAP.get(body.vessel_model, body.vessel_model),
        fast_mode=body.use_fast_mode,
        offline=body.offline_mode,
        export_format=body.export_format,
        output_dir=study_dir / "processed" / task_id,
    )
    pipeline = BrainModelPipeline(config, model_cache_dir=settings.model_cache_dir)
    _PIPELINE_TASKS[task_id] = {"study_id": study_id, "pipeline": pipeline, "result": None}

    background_tasks.add_task(_run_pipeline_task, task_id, dicom_path, registry)
    logger.info("Пайплайн запущен: study_id=%s, task_id=%s, конфиг=%s", study_id, task_id, config)

    return TaskAcceptedResponse(task_id=task_id, websocket_url=f"/ws/progress/{task_id}")


def _run_pipeline_task(task_id: str, dicom_path: Path, registry: PretrainedModelRegistry) -> None:
    """Фоновая задача полного пайплайна с обогащением прогресса именами реально выбранных моделей.

    Точные пер-модельные проценты (например, "Swin UNETR: 60%") недоступны из
    BrainModelPipeline.process_with_progress — она даёт только 6 контрольных точек
    (0/10/20/50/80/100%). Вместо того чтобы выдумывать несуществующую гранулярность,
    сообщения обогащаются РЕАЛЬНО выбранными моделями (доступны с 20%, после стадии
    выбора моделей) — что и требовалось по сути примера из ТЗ.
    """
    entry = _PIPELINE_TASKS[task_id]
    pipeline: BrainModelPipeline = entry["pipeline"]

    def _callback(percent: int, message: str) -> None:
        selected = pipeline.selected_models
        if selected and percent >= 20:
            message = (
                f"{message} (мозг={selected.get('brain')}, "
                f"опухоль={selected.get('tumor')}, сосуды={selected.get('vessels')})"
            )
        progress_tracker.report(task_id, percent, message)

    try:
        result = pipeline.process_with_progress(dicom_path, progress_callback=_callback)
        entry["result"] = result
        progress_tracker.set_final_status(task_id, "done" if result.get("status") == "ok" else "failed")

        for _task_name, info in result.get("model_info", {}).items():
            _record_model_usage(info["strategy"], registry)
    except Exception as exc:  # noqa: BLE001 — фоновая задача не должна падать молча
        logger.exception("Пайплайн task_id=%s упал неожиданно", task_id)
        entry["result"] = {"status": "failed", "error": str(exc)}
        progress_tracker.set_final_status(task_id, "failed")


@router.get("/api/pipeline/status/{task_id}", response_model=PipelineStatusResponse, tags=["pipeline"])
def get_pipeline_status(task_id: str) -> PipelineStatusResponse:
    """Возвращает текущий статус выполнения пайплайна: стадию, прогресс, выбранные модели.

    Args:
        task_id: Идентификатор задачи, полученный от POST /api/process/{study_id}.

    Raises:
        HTTPException 404: Задача с таким task_id не найдена.
    """
    entry = _PIPELINE_TASKS.get(task_id)
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Задача пайплайна не найдена")

    events, _ = progress_tracker.get_events_since(task_id, 0)
    latest = events[-1] if events else None
    pipeline: BrainModelPipeline = entry["pipeline"]

    return PipelineStatusResponse(
        task_id=task_id,
        status=progress_tracker.get_status(task_id),
        current_stage=_stage_label_for_percent(latest.percent if latest else 0),
        progress_percent=(
            latest.percent if latest and latest.percent is not None else (100 if entry["result"] else 0)
        ),
        message=latest.message if latest else None,
        selected_models=pipeline.selected_models or {},
        result=entry["result"],
    )


@router.get("/api/process/{task_id}/download", tags=["pipeline"], response_model=None)
def download_result(task_id: str) -> FileResponse | StreamingResponse:
    """Отдаёт результат экспорта: файл сцены (glb/obj) либо zip с файлами структур (stl/ply).

    Args:
        task_id: Идентификатор задачи.

    Raises:
        HTTPException 404: Задача не найдена либо результат ещё не готов/не успешен.
    """
    entry = _PIPELINE_TASKS.get(task_id)
    if entry is None or entry["result"] is None or entry["result"].get("status") != "ok":
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Результат не найден или ещё не готов")

    export_path = entry["result"]["export"]["path"]
    if isinstance(export_path, str):
        file_path = Path(export_path)
        return FileResponse(file_path, filename=file_path.name)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path_str in export_path.values():
            archive.write(path_str, arcname=Path(path_str).name)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{task_id}.zip"'},
    )


# --------------------------------------------------------------------------- #
# WebSocket прогресса
# --------------------------------------------------------------------------- #


@router.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str) -> None:
    """Отправляет клиенту новые события прогресса задачи (скачивание модели или пайплайн).

    Закрывается автоматически, как только задача переходит в статус done/failed и
    все накопленные события отправлены.
    """
    await websocket.accept()
    last_index = 0
    try:
        while True:
            new_events, last_index = progress_tracker.get_events_since(task_id, last_index)
            for event in new_events:
                await websocket.send_json({"percent": event.percent, "message": event.message})

            current_status = progress_tracker.get_status(task_id)
            if current_status in ("done", "failed") and not new_events:
                await websocket.send_json(
                    {
                        "percent": None,
                        "message": f"Задача завершена: {current_status}",
                        "status": current_status,
                    }
                )
                break

            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        logger.info("WebSocket отключён клиентом (task_id=%s)", task_id)


# --------------------------------------------------------------------------- #
# Кэш моделей
# --------------------------------------------------------------------------- #


@router.get("/api/cache/info", response_model=CacheInfoResponse, tags=["cache"])
def get_cache_info(cache_manager: ModelCacheManager = Depends(get_cache_manager)) -> CacheInfoResponse:
    """Возвращает сведения о локальном кэше весов моделей: размер и список моделей."""
    entries = cache_manager.list_cached_models()
    return CacheInfoResponse(
        total_size_gb=round(cache_manager.get_cache_size() / 1024**3, 3),
        cache_dir=str(cache_manager.cache_dir),
        models=[
            CachedModelInfo(
                name=e["name"],
                source=e["source"],
                task=e["task"],
                size_gb=round(e["size_bytes"] / 1024**3, 3),
                path=e["path"],
            )
            for e in entries
        ],
    )


@router.post("/api/cache/clear", response_model=CacheClearResponse, tags=["cache"])
@limiter.limit("3/minute")
def clear_cache_endpoint(  # noqa: ARG001
    request: Request, cache_manager: ModelCacheManager = Depends(get_cache_manager)
) -> CacheClearResponse:
    """Полностью очищает кэш весов моделей и освобождает место на диске."""
    freed_bytes = cache_manager.get_cache_size()
    cache_manager.clear_cache()
    _DOWNLOAD_STATE.clear()
    logger.info("Кэш моделей очищен: освобождено %.2f ГБ", freed_bytes / 1024**3)
    return CacheClearResponse(
        freed_gb=round(freed_bytes / 1024**3, 3), cache_dir=str(cache_manager.cache_dir)
    )
