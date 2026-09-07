"""Конкретный пайплайн обработки МРТ-исследований."""

from __future__ import annotations

import numpy as np

from core.config import PipelineConfig
from core.preprocessing import MRIPreprocessor, PreprocessingConfig
from core.segmentation.inference import SegmentationEngine
from pipelines.base_pipeline import BasePipeline
from shared.types import VolumeData


class MriPipeline(BasePipeline):
    """Пайплайн для МРТ: z-score нормализация + общая логика BasePipeline."""

    def __init__(self, *args, mri_weighting: str = "T1ce", **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._mri_weighting = mri_weighting

    @property
    def stage_name(self) -> str:
        return "MRI pipeline"

    def _preprocess(self, volume: VolumeData) -> np.ndarray:
        preprocessor: MRIPreprocessor = self._preprocessor  # type: ignore[assignment]
        return preprocessor.preprocess_full(
            volume.voxels, volume.spacing.as_tuple(), modality=self._mri_weighting
        )

    @classmethod
    def from_config(
        cls, pipeline_config: PipelineConfig, segmentation_engine: SegmentationEngine
    ) -> "MriPipeline":
        """Создаёт MriPipeline, читая целевой spacing из config.yaml.

        Args:
            pipeline_config: Загруженная конфигурация пайплайнов.
            segmentation_engine: Уже инициализированная модель для МРТ (см. models.registry).

        Returns:
            Готовый к использованию MriPipeline.
        """
        mri_config = pipeline_config.mri
        preprocessing_config = PreprocessingConfig(target_spacing_mm=mri_config.target_spacing_mm)
        preprocessor = MRIPreprocessor(preprocessing_config)
        return cls(
            preprocessor=preprocessor,
            segmentation_engine=segmentation_engine,
            target_spacing_mm=mri_config.target_spacing_mm,
        )
