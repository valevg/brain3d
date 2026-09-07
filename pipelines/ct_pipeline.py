"""Конкретный пайплайн обработки КТ-исследований."""

from __future__ import annotations

import numpy as np

from core.config import PipelineConfig
from core.preprocessing import CTPreprocessor, PreprocessingConfig
from core.segmentation.inference import SegmentationEngine
from pipelines.base_pipeline import BasePipeline
from shared.types import VolumeData


class CtPipeline(BasePipeline):
    """Пайплайн для КТ: HU-оконная нормализация + общая логика BasePipeline."""

    @property
    def stage_name(self) -> str:
        return "CT pipeline"

    def _preprocess(self, volume: VolumeData) -> np.ndarray:
        preprocessor: CTPreprocessor = self._preprocessor  # type: ignore[assignment]
        return preprocessor.preprocess_full(volume.voxels, volume.spacing.as_tuple())

    @classmethod
    def from_config(
        cls, pipeline_config: PipelineConfig, segmentation_engine: SegmentationEngine
    ) -> "CtPipeline":
        """Создаёт CtPipeline, читая параметры окна HU и spacing из config.yaml.

        Args:
            pipeline_config: Загруженная конфигурация пайплайнов.
            segmentation_engine: Уже инициализированная модель для КТ (см. models.registry).

        Returns:
            Готовый к использованию CtPipeline.
        """
        ct_config = pipeline_config.ct
        preprocessing_config = PreprocessingConfig(
            target_spacing_mm=ct_config.target_spacing_mm,
            apply_hu_windowing=True,
            window_center=((ct_config.hu_min or -100.0) + (ct_config.hu_max or 300.0)) / 2.0,
            window_width=(ct_config.hu_max or 300.0) - (ct_config.hu_min or -100.0),
        )
        preprocessor = CTPreprocessor(preprocessing_config)
        return cls(
            preprocessor=preprocessor,
            segmentation_engine=segmentation_engine,
            target_spacing_mm=ct_config.target_spacing_mm,
        )
