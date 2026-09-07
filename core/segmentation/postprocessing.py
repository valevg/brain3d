"""Постобработка карты меток: удаление мелкого шума, выбор наибольшей связной компоненты."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from core.logging_config import get_logger
from shared.types import LabelMap

logger = get_logger(__name__)


def clean_label_map(label_map: LabelMap, min_component_voxels: int = 50) -> LabelMap:
    """Удаляет мелкие несвязные компоненты для каждого класса, кроме фона (0).

    Нейросетевая сегментация иногда даёт единичные "шумовые" воксели вдали от основной
    структуры (например, ложные срабатывания опухоли на артефактах КТ). Такие компоненты
    портят последующую mesh-генерацию, поэтому отфильтровываются по объёму.

    Args:
        label_map: Исходная карта меток (Z, Y, X), 0 — фон.
        min_component_voxels: Минимальный размер связной компоненты в вокселях,
            компоненты меньшего размера удаляются (переводятся в фон).

    Returns:
        Очищенная карта меток той же формы и типа.
    """
    cleaned = np.zeros_like(label_map)

    for class_index in np.unique(label_map):
        if class_index == 0:
            continue

        class_mask = label_map == class_index
        labeled_components, num_components = ndimage.label(class_mask)

        if num_components == 0:
            continue

        component_sizes = ndimage.sum(class_mask, labeled_components, range(1, num_components + 1))
        for component_id, size in enumerate(component_sizes, start=1):
            if size >= min_component_voxels:
                cleaned[labeled_components == component_id] = class_index
            else:
                logger.debug(
                    "Удалена шумовая компонента класса %d размером %d вокселей",
                    class_index,
                    int(size),
                )

    return cleaned.astype(label_map.dtype)
