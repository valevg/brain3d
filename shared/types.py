"""Общие типы данных (dataclass'ы, TypedDict, алиасы) для передачи данных между слоями."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from shared.enums import Modality, StructureType

VoxelArray = npt.NDArray[np.float32]
"""3D-массив вокселей (Z, Y, X) в float32."""

LabelMap = npt.NDArray[np.uint8]
"""3D-карта меток сегментации (Z, Y, X), значения — индексы классов."""


@dataclass(slots=True)
class VoxelSpacing:
    """Физический размер вокселя в миллиметрах."""

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Возвращает spacing в порядке (x, y, z)."""
        return (self.x, self.y, self.z)


@dataclass(slots=True)
class VolumeData:
    """Загруженный и приведённый к единому виду объём исследования.

    Attributes:
        voxels: 3D-массив интенсивностей (Z, Y, X).
        spacing: Физический размер вокселя в мм.
        origin: Координаты начала объёма в мировой системе координат (мм).
        modality: Модальность исследования (КТ или МРТ).
        metadata: Произвольные дополнительные метаданные (например, DICOM-теги).
    """

    voxels: VoxelArray
    spacing: VoxelSpacing
    origin: tuple[float, float, float]
    modality: Modality
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentationResult:
    """Результат работы модели сегментации.

    Attributes:
        label_map: Карта меток той же формы, что и исходный объём.
        structures: Список структур, реально присутствующих в label_map.
        spacing: Physical spacing, унаследованный от исходного объёма.
        probabilities: Необязательные карты вероятностей по классам (C, Z, Y, X).
    """

    label_map: LabelMap
    structures: list[StructureType]
    spacing: VoxelSpacing
    probabilities: npt.NDArray[np.float32] | None = None


@dataclass(slots=True)
class MeshAsset:
    """3D-меш одной структуры, готовый к экспорту/визуализации.

    Attributes:
        structure: Тип структуры, которой принадлежит меш.
        vertices: Массив вершин (N, 3) в мировых координатах (мм).
        faces: Массив треугольных граней (M, 3), индексы в vertices.
        color_rgba: Цвет структуры по умолчанию для рендера (0..1 на канал).
    """

    structure: StructureType
    vertices: npt.NDArray[np.float32]
    faces: npt.NDArray[np.int32]
    color_rgba: tuple[float, float, float, float]
