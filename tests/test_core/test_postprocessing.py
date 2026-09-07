"""Тесты постобработки карты меток (удаление шумовых компонент)."""

from __future__ import annotations

import numpy as np

from core.segmentation.postprocessing import clean_label_map


def test_clean_label_map_removes_small_component(synthetic_segmentation_result) -> None:
    cleaned = clean_label_map(synthetic_segmentation_result.label_map, min_component_voxels=50)

    # Шумовой воксель [1, 1, 1] должен быть удалён (переведён в фон)
    assert cleaned[1, 1, 1] == 0
    # Основная сфера должна сохраниться
    assert cleaned.sum() > 0
    assert set(np.unique(cleaned)) <= {0, 1}


def test_clean_label_map_keeps_large_component_intact(synthetic_segmentation_result) -> None:
    original_voxel_count = int((synthetic_segmentation_result.label_map == 1).sum())

    cleaned = clean_label_map(synthetic_segmentation_result.label_map, min_component_voxels=50)
    cleaned_voxel_count = int((cleaned == 1).sum())

    # Убрана только шумовая точка (1 воксель), основная сфера не пострадала
    assert cleaned_voxel_count == original_voxel_count - 1
