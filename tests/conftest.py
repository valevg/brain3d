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


# --------------------------------------------------------------------------- #
# Синтетические данные для ВСЕХ типов структур (brain/tumor/vessel/hemorrhage)
# и обеих модальностей — позволяют реально прогнать mesh-генерацию, экспорт и
# постобработку на полной multi-label карте без нейросети и реальных снимков.
# --------------------------------------------------------------------------- #


def _sphere_label(
    label_map: np.ndarray,
    class_value: int,
    center: tuple[int, int, int],
    radius: float,
) -> None:
    """Заполняет сферическую область карты меток указанным классом."""
    z, y, x = np.ogrid[: label_map.shape[0], : label_map.shape[1], : label_map.shape[2]]
    mask = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= radius**2
    label_map[mask] = class_value


def _tube_label(
    label_map: np.ndarray,
    class_value: int,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    radius: float,
) -> None:
    """Заполняет цилиндрическую 'трубку' (имитация сосуда) указанным классом.

    Реализовано через покомпонентную раструбочку отрезка — даёт замкнутую
    поверхность, пригодную для marching cubes.
    """
    z, y, x = np.ogrid[: label_map.shape[0], : label_map.shape[1], : label_map.shape[2]]
    p0 = np.array(start, dtype=np.float64)
    p1 = np.array(end, dtype=np.float64)
    seg = p1 - p0
    seg_len_sq = float(np.dot(seg, seg))
    points = np.stack(np.broadcast_arrays(z, y, x), axis=-1).astype(np.float64)
    t = np.clip(((points - p0) @ seg) / max(seg_len_sq, 1e-12), 0.0, 1.0)
    closest = p0 + t[..., None] * seg
    dist_sq = ((points - closest) ** 2).sum(axis=-1)
    label_map[dist_sq <= radius**2] = class_value


@pytest.fixture
def all_structures_label_map() -> np.ndarray:
    """Multi-label карта (uint8) со ВСЕМИ четырьмя структурами.

    Индексы классов соответствуют CLASS_INDEX_TO_STRUCTURE из
    core/segmentation/inference.py: 1 — мозг, 2 — опухоль, 3 — сосуды,
    4 — кровоизлияние (плановая структура). Форма небольшая, чтобы
    marching cubes и trimesh выполнялись быстро, но каждая маска даёт
    валидную замкнутую поверхность.
    """
    from shared.enums import StructureType  # noqa: F401  (осмысленность индексов)

    label_map = np.zeros((32, 64, 64), dtype=np.uint8)
    _sphere_label(label_map, class_value=1, center=(16, 32, 32), radius=15)  # brain
    _sphere_label(label_map, class_value=2, center=(10, 24, 40), radius=5)  # tumor
    _tube_label(label_map, class_value=3, start=(16, 8, 20), end=(16, 56, 44), radius=2.5)  # vessel
    _sphere_label(label_map, class_value=4, center=(22, 44, 20), radius=4)  # hemorrhage
    return label_map


@pytest.fixture
def synthetic_all_structures_result(all_structures_label_map) -> SegmentationResult:
    """SegmentationResult со всеми поддерживаемыми структурами сразу."""
    from shared.enums import StructureType

    return SegmentationResult(
        label_map=all_structures_label_map,
        structures=[
            StructureType.BRAIN,
            StructureType.TUMOR,
            StructureType.VESSEL,
            StructureType.HEMORRHAGE,
        ],
        spacing=VoxelSpacing(1.0, 1.0, 1.0),
    )


@pytest.fixture(params=["CT", "MRI"])
def synthetic_volume_by_modality(request) -> VolumeData:
    """Один и тот же синтетический объём для обеих модальностей (параметризованный фикстура)."""
    modality = Modality(request.param)
    if modality is Modality.CT:
        return synthetic_ct_volume.__wrapped__()
    return synthetic_mri_volume.__wrapped__()
