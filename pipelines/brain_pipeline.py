"""Оркестрация полного пайплайна Brain3D AI на готовых предобученных моделях.

Пайплайн больше не обучает модели — вместо этого он:
    1. Загружает готовые модели при первом использовании (models.pretrained_models).
    2. Кэширует их для повторного использования (ModelCacheManager).
    3. Выбирает оптимальную модель под доступные ресурсы (ResourceDetector +
       models.pretrained_models.SmartModelSelector).
    4. Откатывается на альтернативу, если выбранная модель недоступна (FallbackHandler).

Собирает воедино модули, написанные ранее в этой сессии:
    - core.dicom_loader.DICOMLoader — загрузка DICOM + определение модальности/взвешивания;
    - core.preprocessing.UnifiedPreprocessor — препроцессинг КТ/МРТ;
    - core.vessel_segmentation.VesselSegmenter — адаптивная сегментация сосудов;
    - core.mesh_generator.MeshGenerator — сборка сцены и экспорт;
    - models.pretrained_models — реестр моделей, обёртки TotalSegmentator/VISTA-3D/MONAI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from core.dicom_loader import DICOMLoader
from core.exceptions import Brain3DError, ChecksumMismatchError, VesselSegmentationError
from core.logging_config import get_logger
from core.mesh_generator import MeshGenerator
from core.preprocessing import UnifiedPreprocessor
from core.vessel_segmentation import VesselSegmenter
from models.pretrained_models import (
    ComputeResources,
    PretrainedModelRegistry,
    SmartModelSelector,
    TotalSegmentatorWrapper,
    VISTA3DWrapper,
    detect_compute_resources,
    is_online,
    load_tumor_model,
)

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "brain3d"

#: Соответствие имени стратегии (как их возвращает SmartModelSelector/load_tumor_model)
#: имени модели в PretrainedModelRegistry — используется download_required_models() для
#: определения, ЧТО именно нужно скачать. Стратегии без записи (None) не требуют
#: отдельного скачивания: "frangi"/"threshold" — классическая обработка без весов,
#: "vista3d" — облачный API/Docker-образ, которым не управляет этот реестр.
STRATEGY_TO_REGISTRY_MODEL: dict[str, str | None] = {
    "totalsegmentator": "totalsegmentator_total",
    "brats_pretrained": "monai_brain_tumor_swinunetr",
    "monai_vessel_model": "monai_brain_vessel",
    "vista3d": None,
    "frangi": None,
    "threshold": None,
    "threshold+frangi": None,
}

#: Порядок стадий process_with_progress() — используется для перезапуска с конкретной
#: стадии (resume_from).
_STAGE_ORDER: tuple[str, ...] = ("load", "select_and_download", "preprocess", "segment", "mesh", "export")


class BrainPipelineConfig(BaseModel):
    """Конфигурация BrainModelPipeline."""

    modality_override: str | None = Field(
        default=None, description="'CT'/'MRI' — принудительно, вместо автоопределения"
    )
    tumor_model_strategy: str = Field(
        default="auto", description="'auto'|'brats_pretrained'|'vista3d'|'totalsegmentator'"
    )
    vessel_model_strategy: str = Field(
        default="auto", description="'auto'|'frangi'|'totalsegmentator'|'vista3d'"
    )
    fast_mode: bool = Field(
        default=False, description="Быстрые (менее точные) варианты моделей, где применимо"
    )
    offline: bool = Field(default=False, description="Не обращаться к сети (API, скачивание моделей)")
    force_cpu: bool = Field(
        default=False,
        description="Не использовать GPU, даже если он физически доступен (аналогично offline "
        "для сети — подменяет has_gpu=False перед выбором моделей в select_models)",
    )
    export_format: str = Field(default="glb", description="'glb'|'stl'|'obj'|'ply'")
    output_dir: Path = Field(default=Path("./output"))
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @field_validator("output_dir", mode="after")
    @classmethod
    def _expand_output_dir(cls, value: Path) -> Path:
        return value.expanduser()


# --------------------------------------------------------------------------- #
# ResourceDetector
# --------------------------------------------------------------------------- #


class ResourceDetector:
    """Определяет доступные ресурсы окружения и рекомендует конфигурацию пайплайна."""

    def detect_gpu(self) -> dict[str, Any]:
        """Возвращает {"available": bool, "memory_gb": float | None, "name": str | None}."""
        try:
            import torch
        except ImportError:
            return {"available": False, "memory_gb": None, "name": None}

        if not torch.cuda.is_available():
            return {"available": False, "memory_gb": None, "name": None}

        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {"available": True, "memory_gb": memory_gb, "name": torch.cuda.get_device_name(0)}

    def detect_ram(self) -> int:
        """Возвращает объём ДОСТУПНОЙ оперативной памяти в байтах (0, если psutil не установлен)."""
        try:
            import psutil
        except ImportError:
            logger.debug("psutil не установлен — объём RAM неизвестен")
            return 0
        return int(psutil.virtual_memory().available)

    def detect_disk_space(self, path: str | Path = ".") -> int:
        """Возвращает свободное место на диске (в байтах) для раздела, содержащего path."""
        return shutil.disk_usage(Path(path)).free

    def detect_internet(self) -> bool:
        """Проверяет наличие интернет-соединения (см. models.pretrained_models.is_online)."""
        return is_online()

    def get_optimal_config(self) -> dict[str, Any]:
        """Собирает снимок ресурсов и рекомендует базовые флаги конфигурации пайплайна."""
        gpu_info = self.detect_gpu()
        has_internet = self.detect_internet()

        return {
            "gpu": gpu_info,
            "ram_gb": self.detect_ram() / 1024**3,
            "disk_free_gb": self.detect_disk_space() / 1024**3,
            "has_internet": has_internet,
            "recommended_fast_mode": not gpu_info["available"],
            "recommended_offline_mode": not has_internet,
        }


# --------------------------------------------------------------------------- #
# ModelCacheManager
# --------------------------------------------------------------------------- #


class ModelCacheManager:
    """Управляет локальным дисковым кэшем весов моделей (~/.cache/brain3d по умолчанию)."""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        registry: PretrainedModelRegistry | None = None,
    ) -> None:
        """Инициализирует менеджер кэша.

        Args:
            cache_dir: Директория кэша.
            registry: Реестр моделей; по умолчанию создаётся новый на том же cache_dir.
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.registry = registry or PretrainedModelRegistry(cache_dir=self.cache_dir)

    def get_cache_size(self) -> int:
        """Возвращает суммарный размер кэша в байтах."""
        if not self.cache_dir.exists():
            return 0
        return sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())

    def list_cached_models(self) -> list[dict[str, Any]]:
        """Возвращает список закэшированных моделей с их размером и путём на диске."""
        entries: list[dict[str, Any]] = []
        for spec in self.registry.list_cached_models():
            path = self.cache_dir / spec.source / spec.name
            size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0
            entries.append(
                {
                    "name": spec.name,
                    "source": spec.source,
                    "task": spec.task,
                    "path": str(path),
                    "size_bytes": size_bytes,
                }
            )
        return entries

    def manage_cache(self, max_size_gb: float = 10.0) -> list[str]:
        """Освобождает место в кэше, если его размер превышает max_size_gb.

        Наивная эвристика (нет данных о времени последнего использования на уровне
        реестра): удаляются модели от самой большой к самой маленькой, пока
        суммарный размер не уложится в лимит — гарантированно освобождает нужный
        объём за минимум удалений.

        Args:
            max_size_gb: Максимально допустимый размер кэша, ГБ.

        Returns:
            Имена удалённых моделей.
        """
        max_size_bytes = int(max_size_gb * 1024**3)
        entries = sorted(self.list_cached_models(), key=lambda e: e["size_bytes"], reverse=True)
        total_size = sum(e["size_bytes"] for e in entries)

        removed: list[str] = []
        for entry in entries:
            if total_size <= max_size_bytes:
                break
            shutil.rmtree(entry["path"], ignore_errors=True)
            total_size -= entry["size_bytes"]
            removed.append(entry["name"])
            logger.info(
                "ModelCacheManager: удалена модель %r (%.1f МБ) — лимит кэша %.1f ГБ",
                entry["name"],
                entry["size_bytes"] / 1024**2,
                max_size_gb,
            )
        return removed

    def clear_cache(self) -> None:
        """Полностью очищает кэш моделей."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Кэш моделей очищен: %s", self.cache_dir)

    def prefetch_models(self, model_names: list[str]) -> dict[str, bool]:
        """Заранее скачивает список моделей реестра (например, при установке приложения).

        Args:
            model_names: Имена моделей ИЗ РЕЕСТРА (см. PretrainedModelRegistry), а не
                имена стратегий — для перевода стратегии в имя модели см.
                STRATEGY_TO_REGISTRY_MODEL / BrainModelPipeline.download_required_models.

        Returns:
            {имя_модели: успех} — False, если скачивание не удалось.
        """
        results: dict[str, bool] = {}
        for name in tqdm(model_names, desc="Предзагрузка моделей", unit="модель"):
            try:
                self.registry.download_model(name)
                results[name] = True
            except Brain3DError as exc:
                logger.warning("prefetch_models: не удалось загрузить %r: %s", name, exc)
                results[name] = False
        return results


# --------------------------------------------------------------------------- #
# FallbackHandler
# --------------------------------------------------------------------------- #


class FallbackHandler:
    """Централизованная логика отката на альтернативную модель/стратегию при сбое."""

    #: Детерминированные альтернативы по умолчанию для известных стратегий.
    _KNOWN_FALLBACKS: dict[str, str] = {
        "vista3d": "totalsegmentator",
        "totalsegmentator": "frangi",
        "brats_pretrained": "totalsegmentator",
        "monai_vessel_model": "frangi",
    }

    def __init__(self, registry: PretrainedModelRegistry | None = None) -> None:
        self.registry = registry or PretrainedModelRegistry()

    def handle_model_failure(self, model_name: str, error: Exception) -> str:
        """Определяет альтернативную стратегию/модель после сбоя основной.

        Логика:
            - ChecksumMismatchError (повреждённые веса) -> веса перекачиваются
              (force=True) и возвращается ТО ЖЕ имя — для повторной попытки с тем
              же методом, но уже целыми весами.
            - Иначе (VISTA-3D недоступен, TotalSegmentator не установлен/упал,
              MONAI bundle недоступен и т.п.) -> известная альтернатива по
              _KNOWN_FALLBACKS, либо "frangi" как крайний вариант.

        Args:
            model_name: Имя модели/стратегии, которая упала.
            error: Возникшее исключение.

        Returns:
            Имя альтернативной стратегии для повторной попытки. Если оно совпадает
            с исходным model_name, вызывающий код должен считать, что альтернативы
            нет (например, после неудачной перекачки повреждённых весов).
        """
        logger.warning("Сбой модели %r: %s — подбирается альтернатива", model_name, error)

        if isinstance(error, ChecksumMismatchError):
            logger.info("Повреждённые веса %r — перекачивание", model_name)
            try:
                self.registry.download_model(model_name, force=True)
            except Brain3DError as redownload_error:
                logger.warning("Повторная загрузка %r не удалась: %s", model_name, redownload_error)
            return model_name

        fallback = self._KNOWN_FALLBACKS.get(model_name, "frangi")
        logger.info("Модель %r заменена на %r", model_name, fallback)
        return fallback


# --------------------------------------------------------------------------- #
# BrainModelPipeline
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _PipelineState:
    """Внутреннее состояние выполнения пайплайна между стадиями (для resume_from)."""

    loader: DICOMLoader | None = None
    volume: np.ndarray | None = None
    modality: str | None = None
    mr_weighting: str | None = None
    spacing: tuple[float, float, float] | None = None
    selected_models: dict[str, str] | None = None
    processed_volume: np.ndarray | None = None
    brain_mask: np.ndarray | None = None
    tumor_mask: np.ndarray | None = None
    vessel_mask: np.ndarray | None = None
    meshes: dict[str, Any] | None = None


class BrainModelPipeline:
    """Оркестрирует полный пайплайн: DICOM -> модальность -> модели -> сегментация -> сцена -> экспорт."""

    def __init__(
        self,
        config: BrainPipelineConfig | None = None,
        model_cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        """Инициализирует пайплайн.

        Args:
            config: Конфигурация пайплайна. По умолчанию — BrainPipelineConfig().
            model_cache_dir: Директория кэша весов моделей.
        """
        self.config = config or BrainPipelineConfig()
        self.model_cache_dir = Path(model_cache_dir).expanduser()

        self.registry = PretrainedModelRegistry(cache_dir=self.model_cache_dir)
        self.selector = SmartModelSelector(self.registry)
        self.cache_manager = ModelCacheManager(self.model_cache_dir, self.registry)
        self.fallback_handler = FallbackHandler(self.registry)
        self.resource_detector = ResourceDetector()
        self.vessel_segmenter = VesselSegmenter(registry=self.registry)
        self.mesh_generator = MeshGenerator()

        self._state = _PipelineState()
        self._last_result: dict[str, Any] = {}

        cached_names = [spec.name for spec in self.registry.list_cached_models()]
        logger.info(
            "BrainModelPipeline: в кэше уже загружено %d моделей: %s", len(cached_names), cached_names
        )

    # --- Определение модальности ----------------------------------------------------- #

    def detect_modality(self, dicom_path: str | Path) -> str:
        """Определяет модальность DICOM-серии (и, для МРТ, тип взвешивания).

        Самостоятельная утилита (загружает DICOM заново) — внутри process()/
        process_with_progress() модальность определяется как часть стадии "load"
        без повторной загрузки серии.

        Args:
            dicom_path: Путь к DICOM-серии.

        Returns:
            "CT" или "MRI".
        """
        loader = DICOMLoader()
        loader.load_dicom_series(dicom_path)
        modality = loader.detect_modality()

        if loader.mr_weighting is not None:
            logger.info("Определена модальность: %s, взвешивание: %s", modality, loader.mr_weighting.value)
        else:
            logger.info("Определена модальность: %s", modality)

        return modality

    # --- Выбор моделей ------------------------------------------------------------------ #

    def select_models(self, modality: str, resources: ComputeResources | None = None) -> dict[str, str]:
        """Выбирает модели/стратегии сегментации под модальность и доступные ресурсы.

        Args:
            modality: "CT" или "MRI".
            resources: Доступные ресурсы; по умолчанию определяются автоматически
                (с учётом self.config.offline).

        Returns:
            {"brain": "...", "tumor": "...", "vessels": "..."} — например
            {"brain": "totalsegmentator", "tumor": "brats_pretrained", "vessels": "frangi"}.
        """
        resources = resources or detect_compute_resources(check_internet=not self.config.offline)
        if self.config.offline and resources.has_internet:
            resources = ComputeResources(
                has_gpu=resources.has_gpu,
                gpu_memory_gb=resources.gpu_memory_gb,
                has_internet=False,
                cpu_count=resources.cpu_count,
            )

        tumor_model = (
            self.config.tumor_model_strategy
            if self.config.tumor_model_strategy != "auto"
            else self.selector.choose_best_model(
                modality=modality, task="tumor", available_resources=resources
            )
        )
        vessel_model = (
            self.config.vessel_model_strategy
            if self.config.vessel_model_strategy != "auto"
            else self.selector.choose_best_model(
                modality=modality, task="vessel", available_resources=resources
            )
        )
        brain_model = (
            "vista3d"
            if resources.has_gpu and (resources.gpu_memory_gb or 0.0) >= 48.0 and not self.config.offline
            else "totalsegmentator"
        )

        selected = {"brain": brain_model, "tumor": tumor_model, "vessels": vessel_model}
        logger.info("select_models(modality=%s): %s", modality, selected)
        return selected

    def download_required_models(self, model_names: list[str]) -> dict[str, bool]:
        """Скачивает в кэш веса, недостающие для перечисленных стратегий.

        "frangi"/"threshold"/"threshold+frangi" (не требуют весов) и "vista3d"
        (облачный API/Docker-образ) не скачиваются этим методом и считаются
        успешными автоматически. Это best-effort предзагрузка: TotalSegmentator
        всё равно скачивает per-task веса самостоятельно при первом сегментировании
        конкретной задачи (см. TotalSegmentatorWrapper.segment) — так что для задач,
        отличных от "total", реальная загрузка может произойти позже, при вызове.

        Args:
            model_names: Имена стратегий (значения из select_models()).

        Returns:
            {имя_стратегии: успех}.
        """
        results: dict[str, bool] = {}
        registry_keys = {name: STRATEGY_TO_REGISTRY_MODEL.get(name) for name in set(model_names)}
        downloadable = [(name, key) for name, key in registry_keys.items() if key is not None]

        for name, key in tqdm(downloadable, desc="Загрузка моделей", unit="модель"):
            try:
                self.registry.download_model(key)
                results[name] = True
            except ChecksumMismatchError as exc:
                fallback = self.fallback_handler.handle_model_failure(name, exc)
                results[name] = fallback == name
            except Brain3DError as exc:
                logger.warning("Не удалось загрузить модель %r (%s): %s", name, key, exc)
                results[name] = False

        for name, key in registry_keys.items():
            if key is None:
                results.setdefault(name, True)

        return results

    # --- Полный пайплайн ----------------------------------------------------------------- #

    def process(self, dicom_path: str | Path) -> dict[str, Any]:
        """Выполняет полный пайплайн без отслеживания прогресса.

        Args:
            dicom_path: Путь к DICOM-серии.

        Returns:
            Словарь с результатами каждой стадии (см. process_with_progress()).
        """
        return self.process_with_progress(dicom_path, progress_callback=None)

    def process_with_progress(
        self,
        dicom_path: str | Path,
        progress_callback: Callable[[int, str], None] | None = None,
        resume_from: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Выполняет полный пайплайн с отчётом о прогрессе (10/20/50/80/100%).

        Args:
            dicom_path: Путь к DICOM-серии.
            progress_callback: Вызывается как callback(percent, message) на каждой
                контрольной точке. None — прогресс только логируется.
            resume_from: Имя стадии из _STAGE_ORDER, с которой продолжить выполнение
                после предыдущего сбоя (более ранние стадии не повторяются, их
                результат берётся из состояния предыдущего вызова).
            should_cancel: Опциональный callback без аргументов, вызываемый перед
                КАЖДОЙ стадией; если возвращает True, выполнение прерывается с
                status="failed" (см. desktop.workers — используется для кнопки
                "Отмена" в UI). Отмена кооперативная — уже начатая стадия (например,
                сам инференс модели) до конца не прерывается, проверка происходит
                только на границах стадий, что безопаснее принудительного убийства
                потока/процесса на середине вычислений.

        Returns:
            Словарь: {"loading", "model_selection", "preprocessing", "brain_segmentation",
            "tumor_segmentation", "vessel_segmentation", "mesh_generation", "export",
            "model_info", "status"} — с результатами соответствующих стадий; при сбое
            (включая отмену) дополнительно {"status": "failed", "error": "..."}.
        """

        def _report(percent: int, message: str) -> None:
            logger.info("[%3d%%] %s", percent, message)
            if progress_callback is not None:
                progress_callback(percent, message)

        def _check_cancelled() -> None:
            if should_cancel is not None and should_cancel():
                raise Brain3DError("Обработка отменена пользователем")

        stages_to_run = self._stages_from(resume_from)
        result: dict[str, Any] = dict(self._last_result) if resume_from else {"dicom_path": str(dicom_path)}
        state = self._state

        try:
            if "load" in stages_to_run:
                _check_cancelled()
                stage_started_at = time.perf_counter()
                _report(0, "Загрузка DICOM...")
                loader = DICOMLoader()
                volume = loader.load_dicom_series(dicom_path)
                detected_modality = loader.detect_modality()
                modality = self.config.modality_override or detected_modality
                mr_weighting = loader.mr_weighting.value if loader.mr_weighting else None
                spacing = loader.get_volume_info()["spacing"]

                state.loader, state.volume = loader, volume
                state.modality, state.mr_weighting, state.spacing = modality, mr_weighting, spacing

                result["loading"] = {
                    "status": "ok",
                    "shape": tuple(volume.shape),
                    "modality": modality,
                    "mr_weighting": mr_weighting,
                    "elapsed_seconds": round(time.perf_counter() - stage_started_at, 3),
                }
                _report(
                    10,
                    f"DICOM загружен: модальность={modality}"
                    + (f" ({mr_weighting})" if mr_weighting else ""),
                )

            if "select_and_download" in stages_to_run:
                _check_cancelled()
                stage_started_at = time.perf_counter()
                resources = detect_compute_resources(check_internet=not self.config.offline)
                selected_models = self.select_models(state.modality, resources)
                download_status = self.download_required_models(list(selected_models.values()))
                state.selected_models = selected_models
                result["model_selection"] = {
                    "selected": selected_models,
                    "download_status": download_status,
                    "elapsed_seconds": round(time.perf_counter() - stage_started_at, 3),
                }

            if "preprocess" in stages_to_run:
                _check_cancelled()
                stage_started_at = time.perf_counter()
                preprocessor = UnifiedPreprocessor(state.modality)
                extra_kwargs = (
                    {"modality": state.mr_weighting} if state.modality == "MRI" and state.mr_weighting else {}
                )
                processed_volume = preprocessor.preprocess(state.volume, state.spacing, **extra_kwargs)
                state.processed_volume = processed_volume
                result["preprocessing"] = {
                    "status": "ok",
                    "shape": tuple(processed_volume.shape),
                    "elapsed_seconds": round(time.perf_counter() - stage_started_at, 3),
                }
                _report(20, "Препроцессинг завершён")

            if "segment" in stages_to_run:
                _check_cancelled()
                resources = detect_compute_resources(check_internet=not self.config.offline)
                selected_models = state.selected_models

                brain_started_at = time.perf_counter()
                brain_mask = self._segment_brain(
                    state.processed_volume, state.spacing, selected_models["brain"]
                )
                brain_elapsed = round(time.perf_counter() - brain_started_at, 3)

                tumor_started_at = time.perf_counter()
                tumor_mask = self._segment_tumor(
                    state.processed_volume, state.spacing, selected_models["tumor"]
                )
                tumor_elapsed = round(time.perf_counter() - tumor_started_at, 3)

                vessel_started_at = time.perf_counter()
                vessel_mask, vessel_method, vessel_confidence = self._segment_vessels(
                    state.processed_volume, state.spacing, state.modality, resources
                )
                vessel_elapsed = round(time.perf_counter() - vessel_started_at, 3)

                state.brain_mask, state.tumor_mask, state.vessel_mask = brain_mask, tumor_mask, vessel_mask

                result["brain_segmentation"] = {**_mask_summary(brain_mask), "elapsed_seconds": brain_elapsed}
                result["tumor_segmentation"] = {**_mask_summary(tumor_mask), "elapsed_seconds": tumor_elapsed}
                result["vessel_segmentation"] = {
                    **_mask_summary(vessel_mask),
                    "method": vessel_method,
                    "confidence": vessel_confidence,
                    "elapsed_seconds": vessel_elapsed,
                }
                _report(50, "Сегментация завершена")

            if "mesh" in stages_to_run:
                _check_cancelled()
                stage_started_at = time.perf_counter()
                meshes = self.mesh_generator.create_full_brain_scene(
                    state.brain_mask,
                    tumor_mask=state.tumor_mask,
                    vessel_mask=state.vessel_mask,
                    spacing=state.spacing,
                )
                state.meshes = meshes
                result["mesh_generation"] = {
                    "structures": sorted(meshes),
                    "face_counts": {name: len(mesh.faces) for name, mesh in meshes.items()},
                    "elapsed_seconds": round(time.perf_counter() - stage_started_at, 3),
                }
                _report(80, f"Построено {len(meshes)} мешей")

            if "export" in stages_to_run:
                _check_cancelled()
                stage_started_at = time.perf_counter()
                if not state.meshes:
                    raise Brain3DError("Экспорт невозможен: не построено ни одного меша")
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                export_path = self.config.output_dir / f"scene.{self.config.export_format}"
                exported = self.mesh_generator.export_scene(
                    state.meshes, export_path, format=self.config.export_format
                )
                result["export"] = {
                    "path": _paths_to_str(exported),
                    "elapsed_seconds": round(time.perf_counter() - stage_started_at, 3),
                }
                _report(100, "Экспорт завершён")

            result["model_info"] = self.get_model_info()
            result["status"] = "ok"
        except Brain3DError as exc:
            logger.exception("Пайплайн упал при обработке %s", dicom_path)
            result["status"] = "failed"
            result["error"] = str(exc)

        self._last_result = result
        return result

    def _stages_from(self, resume_from: str | None) -> tuple[str, ...]:
        """Возвращает список стадий для выполнения, начиная с resume_from (или все)."""
        if resume_from is None:
            return _STAGE_ORDER
        if resume_from not in _STAGE_ORDER:
            raise Brain3DError(
                f"Неизвестная стадия для перезапуска: {resume_from!r}. Доступны: {_STAGE_ORDER}"
            )
        if resume_from != "load" and self._state.volume is None:
            raise Brain3DError("Нет сохранённого состояния предыдущего запуска — resume_from недоступен")
        return _STAGE_ORDER[_STAGE_ORDER.index(resume_from) :]

    # --- Диспетчеры сегментации по конкретной стратегии ------------------------------- #

    def _segment_brain(
        self, volume: np.ndarray, spacing: tuple[float, float, float], strategy: str
    ) -> np.ndarray | None:
        """Сегментирует мозг выбранной стратегией с одним уровнем автоматического fallback."""
        try:
            if strategy == "vista3d":
                wrapper = VISTA3DWrapper(mode="local")
                if not wrapper.is_available():
                    raise Brain3DError("VISTA-3D недоступен")
                return wrapper.segment(volume, spacing, classes=["brain"]).get("brain")

            wrapper = TotalSegmentatorWrapper(fast=self.config.fast_mode, task="total")
            wrapper.segment(volume, spacing)
            return wrapper.get_brain_mask()
        except Brain3DError as exc:
            fallback = self.fallback_handler.handle_model_failure(strategy, exc)
            if fallback == strategy or fallback not in ("vista3d", "totalsegmentator"):
                logger.error("Сегментация мозга (%s) не удалась без доступной альтернативы", strategy)
                return None
            return self._segment_brain(volume, spacing, fallback)

    def _segment_tumor(
        self, volume: np.ndarray, spacing: tuple[float, float, float], strategy: str
    ) -> np.ndarray | None:
        """Сегментирует опухоль выбранной стратегией с одним уровнем автоматического fallback."""
        try:
            model = load_tumor_model(strategy=strategy, device="cpu", registry=self.registry)

            if isinstance(model, TotalSegmentatorWrapper):
                # TotalSegmentator не выделяет опухоли как отдельную анатомическую структуру.
                logger.warning("TotalSegmentator не сегментирует опухоли — маска опухоли недоступна")
                return None
            if isinstance(model, VISTA3DWrapper):
                if not model.is_available():
                    raise Brain3DError("VISTA-3D недоступен")
                return model.segment(volume, spacing, classes=["tumor"]).get("tumor")

            return self._run_monai_segmentation_model(model, volume)
        except Brain3DError as exc:
            fallback = self.fallback_handler.handle_model_failure(strategy, exc)
            if fallback == strategy:
                logger.error("Сегментация опухоли (%s) не удалась без доступной альтернативы", strategy)
                return None
            return self._segment_tumor(volume, spacing, fallback)

    def _segment_vessels(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        modality: str,
        resources: ComputeResources,
    ) -> tuple[np.ndarray | None, str | None, float]:
        """Сегментирует сосуды через VesselSegmenter.segment_adaptive (уже содержит фолбэк-каскад)."""
        try:
            mask, method, confidence = self.vessel_segmenter.segment_adaptive(
                volume, spacing, modality=modality, resources=resources
            )
            return mask, method, confidence
        except VesselSegmentationError as exc:
            logger.error("Сегментация сосудов полностью не удалась: %s", exc)
            return None, None, 0.0

    @staticmethod
    def _run_monai_segmentation_model(model: Any, volume: np.ndarray) -> np.ndarray:
        """Прогоняет объём через загруженную MONAI bundle-модель (torch.nn.Module)."""
        import torch

        model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(np.asarray(volume, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
            logits = model(input_tensor)
            probabilities = (
                torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)[:, 1:2]
            )
            return (probabilities > 0.5).squeeze().to("cpu").numpy().astype(np.uint8)

    # --- Отчётность -------------------------------------------------------------------- #

    @property
    def selected_models(self) -> dict[str, str] | None:
        """Модели, выбранные на стадии "select_and_download" последнего запуска.

        None, если процесс ещё не дошёл до этой стадии (или ещё не запускался).
        Публичный доступ к внутреннему состоянию — используется api.routes для
        отчёта о статусе выполнения (GET /api/pipeline/status/{task_id}) без
        необходимости дожидаться полного результата process_with_progress().
        """
        return self._state.selected_models

    def get_model_info(self) -> dict[str, Any]:
        """Возвращает информацию об использованных в последнем запуске моделях.

        Используется для отчётов и встраивания в метаданные сцены (см.
        core.mesh_generator.SceneMetadata.segmentation_models).

        Returns:
            {задача: {strategy, registry_name, source, description, cached}}.
        """
        selected_models = self._state.selected_models or {}
        cached_specs = {spec.name: spec for spec in self.registry.list_cached_models()}

        info: dict[str, Any] = {}
        for task, strategy in selected_models.items():
            registry_key = STRATEGY_TO_REGISTRY_MODEL.get(strategy)
            spec = cached_specs.get(registry_key) if registry_key else None
            info[task] = {
                "strategy": strategy,
                "registry_name": registry_key,
                "source": spec.source if spec else None,
                "description": spec.description if spec else None,
                "cached": bool(registry_key and self.registry.is_cached(registry_key)),
            }
        return info


def _mask_summary(mask: np.ndarray | None) -> dict[str, Any]:
    """Строит компактную JSON-совместимую сводку по маске структуры для результата пайплайна."""
    if mask is None:
        return {"found": False, "voxels": 0}
    return {"found": bool(np.any(mask)), "voxels": int(mask.sum())}


def _paths_to_str(exported: Path | dict[str, Path]) -> str | dict[str, str]:
    """Приводит результат MeshGenerator.export_scene() к JSON-совместимому виду."""
    if isinstance(exported, dict):
        return {name: str(path) for name, path in exported.items()}
    return str(exported)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

TUMOR_STRATEGY_MAP: dict[str, str] = {
    "auto": "auto",
    "monai": "brats_pretrained",
    "vista3d": "vista3d",
    "totalsegmentator": "totalsegmentator",
}
VESSEL_STRATEGY_MAP: dict[str, str] = {
    "auto": "auto",
    "frangi": "frangi",
    "totalseg": "totalsegmentator",
    "vista3d": "vista3d",
}


def _build_cli_parser() -> argparse.ArgumentParser:
    """Строит парсер аргументов командной строки для run_pipeline_cli()."""
    parser = argparse.ArgumentParser(
        prog="brain_pipeline", description="Brain3D AI — пайплайн сегментации на готовых моделях"
    )
    parser.add_argument("dicom_path", help="Путь к DICOM-серии")
    parser.add_argument("output_dir", help="Директория для экспорта результата")
    parser.add_argument("--modality", choices=["auto", "ct", "mri"], default="auto")
    parser.add_argument("--tumor-model", choices=sorted(TUMOR_STRATEGY_MAP), default="auto")
    parser.add_argument("--vessel-model", choices=sorted(VESSEL_STRATEGY_MAP), default="auto")
    parser.add_argument("--fast", action="store_true", help="Использовать быстрые (менее точные) модели")
    parser.add_argument("--offline", action="store_true", help="Не обращаться к сети")
    parser.add_argument("--export-format", choices=["glb", "stl", "obj", "ply"], default="glb")
    return parser


def run_pipeline_cli(
    dicom_path: str | None = None,
    output_dir: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """CLI-обёртка над BrainModelPipeline.

    Может вызываться напрямую из Python (с явными dicom_path/output_dir/kwargs — удобно
    для тестов и скриптов) либо без аргументов, тогда используется sys.argv (для запуска
    как `python -m pipelines.brain_pipeline <dicom_path> <output_dir> [флаги]`).

    Args:
        dicom_path: Путь к DICOM-серии. None — брать из sys.argv.
        output_dir: Директория для экспорта. None — брать из sys.argv.
        **kwargs: modality/tumor_model/vessel_model/fast/offline/export_format —
            те же значения, что и одноимённые CLI-флаги (без "--" и дефисов).

    Returns:
        Результат BrainModelPipeline.process() (см. process_with_progress()).
    """
    parser = _build_cli_parser()

    if dicom_path is None:
        args = parser.parse_args()
    else:
        argv = [dicom_path, output_dir or "."]
        if kwargs.get("modality"):
            argv += ["--modality", kwargs["modality"]]
        if kwargs.get("tumor_model"):
            argv += ["--tumor-model", kwargs["tumor_model"]]
        if kwargs.get("vessel_model"):
            argv += ["--vessel-model", kwargs["vessel_model"]]
        if kwargs.get("fast"):
            argv.append("--fast")
        if kwargs.get("offline"):
            argv.append("--offline")
        if kwargs.get("export_format"):
            argv += ["--export-format", kwargs["export_format"]]
        args = parser.parse_args(argv)

    config = BrainPipelineConfig(
        modality_override=None if args.modality == "auto" else args.modality.upper(),
        tumor_model_strategy=TUMOR_STRATEGY_MAP[args.tumor_model],
        vessel_model_strategy=VESSEL_STRATEGY_MAP[args.vessel_model],
        fast_mode=args.fast,
        offline=args.offline,
        export_format=args.export_format,
        output_dir=Path(args.output_dir),
    )

    pipeline = BrainModelPipeline(config)
    result = pipeline.process(args.dicom_path)

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


if __name__ == "__main__":
    outcome = run_pipeline_cli()
    sys.exit(0 if outcome.get("status") == "ok" else 1)
