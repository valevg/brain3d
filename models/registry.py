"""Реестр моделей: выбор весов и параметров инференса по модальности исследования."""

from __future__ import annotations

import torch

from core.config import PipelineConfig, SegmentationModelConfig
from core.exceptions import ModelNotFoundError, UnsupportedModalityError
from core.logging_config import get_logger
from core.segmentation.inference import SegmentationEngine
from models.unet3d import load_pretrained_unet3d
from shared.enums import Modality

logger = get_logger(__name__)


def _model_config_for(modality: Modality, pipeline_config: PipelineConfig) -> SegmentationModelConfig:
    """Возвращает конфигурацию модели (пути к весам, roi_size и т.д.) для модальности."""
    if modality == Modality.CT:
        return pipeline_config.ct_model
    if modality == Modality.MRI:
        return pipeline_config.mri_model
    raise UnsupportedModalityError(f"Нет конфигурации модели для модальности {modality}")


def get_model_for_modality(
    modality: Modality,
    pipeline_config: PipelineConfig,
    device: str = "cpu",
) -> SegmentationEngine:
    """Загружает и возвращает готовый к инференсу SegmentationEngine для указанной модальности.

    Args:
        modality: Modality.CT или Modality.MRI.
        pipeline_config: Конфигурация пайплайнов (из configs/config.yaml), содержащая
            пути к весам и гиперпараметры инференса для каждой модальности.
        device: Устройство вычислений ("cpu" или "cuda").

    Returns:
        Инициализированный SegmentationEngine с загруженными весами.

    Raises:
        ModelNotFoundError: Если файл весов не найден по указанному пути.
        UnsupportedModalityError: Если для модальности нет конфигурации.
    """
    model_config = _model_config_for(modality, pipeline_config)

    try:
        network = load_pretrained_unet3d(
            weights_path=model_config.weights_path,
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            device=device,
        )
    except FileNotFoundError as exc:
        raise ModelNotFoundError(
            f"Веса модели для {modality.value} не найдены: {model_config.weights_path}"
        ) from exc

    logger.info("Модель для модальности %s загружена с весами %s", modality.value, model_config.weights_path)

    return SegmentationEngine(
        model=network,
        roi_size=model_config.roi_size,
        sw_batch_size=model_config.sw_batch_size,
        overlap=model_config.overlap,
        device=device,
    )
