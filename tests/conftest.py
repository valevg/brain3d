"""Общие pytest-фикстуры: синтетические объёмы и карты сегментации для юнит-тестов."""

from __future__ import annotations

import numpy as np
import pytest

from shared.enums import Modality
from shared.types import SegmentationResult, VolumeData, VoxelSpacing


@pytest.fixture
def synthetic_ct_volume() -> VolumeData:
    """Небольшой синтетический КТ-объём со сферой высокой плотности в центре (имитация кости/опухоли)."""
    shape = (32, 64, 64)
    voxels = np.full(shape, fill_value=-1000.0, dtype=np.float32)  # воздух по фону

    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    sphere_mask = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= 10**2
    voxels[sphere_mask] = 150.0  # мягкая ткань/опухоль в HU

    return VolumeData(
        voxels=voxels,
        spacing=VoxelSpacing(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        modality=Modality.CT,
    )


@pytest.fixture
def synthetic_mri_volume() -> VolumeData:
    """Небольшой синтетический МРТ-объём с ненулевой тканью и фоном (для z-score нормализации)."""
    shape = (32, 64, 64)
    rng = np.random.default_rng(seed=42)
    voxels = np.zeros(shape, dtype=np.float32)

    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    brain_mask = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= 20**2
    voxels[brain_mask] = rng.normal(loc=500, scale=50, size=int(brain_mask.sum())).astype(np.float32)

    return VolumeData(
        voxels=voxels,
        spacing=VoxelSpacing(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        modality=Modality.MRI,
    )


@pytest.fixture
def synthetic_segmentation_result() -> SegmentationResult:
    """Карта меток с одной сферой класса 'мозг' (1) и одной мелкой шумовой точкой."""
    from shared.enums import StructureType

    shape = (32, 64, 64)
    label_map = np.zeros(shape, dtype=np.uint8)

    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    brain_mask = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= 15**2
    label_map[brain_mask] = 1

    label_map[1, 1, 1] = 1  # изолированный шумовой воксель того же класса

    return SegmentationResult(
        label_map=label_map,
        structures=[StructureType.BRAIN],
        spacing=VoxelSpacing(1.0, 1.0, 1.0),
    )
