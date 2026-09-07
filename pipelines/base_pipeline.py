"""Базовый класс пайплайна: единая последовательность стадий для любой модальности."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from core.logging_config import get_logger
from core.mesh.mesh_generator import generate_meshes
from core.preprocessing import CTPreprocessor, MRIPreprocessor
from core.segmentation.inference import SegmentationEngine
from core.segmentation.postprocessing import clean_label_map
from shared.enums import PipelineStage
from shared.types import MeshAsset, VolumeData, VoxelSpacing

logger = get_logger(__name__)


class BasePipeline(ABC):
    """Оркестрирует полную обработку одного исследования от загрузки до мешей.

    Конкретные подклассы (CtPipeline, MriPipeline) отличаются только препроцессором
    (core.preprocessing.CTPreprocessor/MRIPreprocessor) и способом его вызова — сама
    последовательность стадий (PipelineStage) одинакова для обеих модальностей, что
    и требуется единым пайплайном проекта.
    """

    def __init__(
        self,
        preprocessor: CTPreprocessor | MRIPreprocessor,
        segmentation_engine: SegmentationEngine,
        target_spacing_mm: tuple[float, float, float],
        min_component_voxels: int = 50,
    ) -> None:
        self._preprocessor = preprocessor
        self._segmentation_engine = segmentation_engine
        self._target_spacing_mm = target_spacing_mm
        self._min_component_voxels = min_component_voxels

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Человекочитаемое имя пайплайна для логов (например, 'CT pipeline')."""

    @abstractmethod
    def _preprocess(self, volume: VolumeData) -> np.ndarray:
        """Вызывает preprocess_full() соответствующего препроцессора с нужными аргументами."""

    def run(self, volume: VolumeData) -> list[MeshAsset]:
        """Выполняет полный пайплайн для уже загруженного объёма.

        Args:
            volume: Сырой объём, полученный из core.io (DicomLoader/NiftiLoader).

        Returns:
            Список готовых к экспорту мешей (по одному на найденную структуру).

        Raises:
            MeshGenerationError: Если ни одна структура не была обнаружена.
        """
        logger.info("[%s] Стадия: %s", self.stage_name, PipelineStage.PREPROCESSING.value)
        preprocessed_voxels = self._preprocess(volume)
        preprocessed_volume = VolumeData(
            voxels=preprocessed_voxels,
            spacing=VoxelSpacing(*self._target_spacing_mm),
            origin=volume.origin,
            modality=volume.modality,
            metadata=volume.metadata,
        )

        logger.info("[%s] Стадия: %s", self.stage_name, PipelineStage.SEGMENTATION.value)
        segmentation_result = self._segmentation_engine.predict(preprocessed_volume)

        logger.info("[%s] Стадия: %s", self.stage_name, PipelineStage.POSTPROCESSING.value)
        segmentation_result.label_map = clean_label_map(
            segmentation_result.label_map, min_component_voxels=self._min_component_voxels
        )

        logger.info("[%s] Стадия: %s", self.stage_name, PipelineStage.MESH_GENERATION.value)
        meshes = generate_meshes(segmentation_result)

        logger.info("[%s] Стадия: %s", self.stage_name, PipelineStage.DONE.value)
        return meshes

    def run_from_path(
        self, path: Path, loader
    ) -> list[MeshAsset]:  # noqa: ANN001 — BaseLoader, избегаем цикл. импорта
        """Удобный метод: загружает исследование с диска и запускает полный пайплайн.

        Args:
            path: Путь к файлу/директории исследования.
            loader: Экземпляр BaseLoader (DicomLoader/NiftiLoader), уже проверивший can_load().

        Returns:
            Список мешей — см. run().
        """
        logger.info("[%s] Стадия: %s (%s)", self.stage_name, PipelineStage.LOADING.value, path)
        volume = loader.load(path)
        return self.run(volume)
