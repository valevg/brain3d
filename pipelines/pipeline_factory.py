"""Фабрика пайплайнов: определяет модальность исследования и создаёт нужный пайплайн.

Это единая точка входа, реализующая принцип "единый пайплайн обработки
с определением типа исследования" — вызывающему коду (API, desktop) не нужно
знать, КТ это или МРТ, достаточно передать путь к исследованию.
"""

from __future__ import annotations

from pathlib import Path

from core.config import PipelineConfig, get_pipeline_config
from core.exceptions import StudyLoadError, UnsupportedModalityError
from core.io.base_loader import BaseLoader
from core.io.dicom_loader import DicomLoader
from core.io.nifti_loader import NiftiLoader
from core.logging_config import get_logger
from models.registry import get_model_for_modality
from pipelines.base_pipeline import BasePipeline
from pipelines.ct_pipeline import CtPipeline
from pipelines.mri_pipeline import MriPipeline
from shared.enums import Modality
from shared.types import VolumeData

logger = get_logger(__name__)

_LOADERS: tuple[BaseLoader, ...] = (DicomLoader(), NiftiLoader())


def resolve_loader(path: Path) -> BaseLoader:
    """Находит подходящий загрузчик (DICOM/NIfTI) для указанного пути.

    Args:
        path: Путь к файлу или директории исследования.

    Returns:
        Первый загрузчик, для которого can_load(path) вернул True.

    Raises:
        StudyLoadError: Если ни один загрузчик не распознал формат.
    """
    for loader in _LOADERS:
        if loader.can_load(path):
            return loader
    raise StudyLoadError(f"Формат исследования не распознан: {path}")


def create_pipeline_for_volume(
    volume: VolumeData,
    device: str = "cpu",
    pipeline_config: PipelineConfig | None = None,
) -> BasePipeline:
    """Создаёт пайплайн, соответствующий модальности уже загруженного объёма.

    Args:
        volume: Загруженный объём (VolumeData.modality определяет выбор пайплайна).
        device: Устройство вычислений ("cpu" или "cuda").
        pipeline_config: Необязательная конфигурация; если не передана — читается
            из configs/config.yaml через get_pipeline_config().

    Returns:
        CtPipeline или MriPipeline с загруженной моделью.

    Raises:
        UnsupportedModalityError: Если модальность не КТ и не МРТ.
    """
    config = pipeline_config or get_pipeline_config()
    engine = get_model_for_modality(volume.modality, config, device=device)

    if volume.modality == Modality.CT:
        logger.info("Выбран пайплайн: CT")
        return CtPipeline.from_config(config, engine)
    if volume.modality == Modality.MRI:
        logger.info("Выбран пайплайн: MRI")
        return MriPipeline.from_config(config, engine)

    raise UnsupportedModalityError(f"Нет пайплайна для модальности {volume.modality}")


def create_pipeline(path: Path, device: str = "cpu") -> tuple[BasePipeline, VolumeData]:
    """Полная точка входа: определяет формат и модальность файла, возвращает готовый пайплайн.

    Args:
        path: Путь к DICOM-директории или NIfTI-файлу.
        device: Устройство вычислений ("cpu" или "cuda").

    Returns:
        Кортеж (пайплайн, загруженный объём) — объём передаётся отдельно, чтобы
        вызывающий код мог переиспользовать его (например, для превью до сегментации)
        без повторной загрузки с диска.
    """
    loader = resolve_loader(path)
    volume = loader.load(path)
    pipeline = create_pipeline_for_volume(volume, device=device)
    return pipeline, volume
