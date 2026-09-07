"""Общие перечисления, используемые во всех слоях приложения (core/models/api/desktop)."""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """Тип медицинского исследования."""

    CT = "CT"
    """Компьютерная томография."""

    MRI = "MRI"
    """Магнитно-резонансная томография."""

    @classmethod
    def from_dicom_tag(cls, modality_tag: str) -> "Modality":
        """Определяет модальность по DICOM-тегу (0008,0060) Modality.

        Args:
            modality_tag: Значение тега Modality из DICOM-файла (например, "CT", "MR").

        Returns:
            Соответствующее значение Modality.

        Raises:
            ValueError: Если тег не распознан.
        """
        normalized = modality_tag.strip().upper()
        mapping = {"CT": cls.CT, "MR": cls.MRI, "MRI": cls.MRI}
        if normalized not in mapping:
            raise ValueError(f"Неподдерживаемая модальность DICOM: {modality_tag!r}")
        return mapping[normalized]


class StructureType(str, Enum):
    """Анатомическая/патологическая структура, выделяемая сегментацией."""

    BRAIN = "brain"
    """Ткань мозга (полупрозрачная, серая)."""

    TUMOR = "tumor"
    """Опухоль (красный/желтый)."""

    VESSEL = "vessel"
    """Сосуды Виллизиева круга (синий)."""

    HEMORRHAGE = "hemorrhage"
    """Кровоизлияние — плановая структура (оранжевый), зарезервировано на будущее."""


class MRWeighting(str, Enum):
    """Тип взвешивания МР-последовательности."""

    T1 = "T1"
    T1CE = "T1ce"
    """T1 с контрастным усилением (наличие ContrastBolusAgent)."""

    T2 = "T2"
    FLAIR = "FLAIR"
    DWI = "DWI"
    UNKNOWN = "unknown"
    """Тип взвешивания не удалось определить по доступным DICOM-тегам."""


class FileFormat(str, Enum):
    """Поддерживаемые форматы входных/выходных файлов."""

    DICOM = "dicom"
    NIFTI = "nifti"
    GLB = "glb"
    STL = "stl"


class PipelineStage(str, Enum):
    """Стадии единого пайплайна обработки исследования."""

    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    SEGMENTATION = "segmentation"
    POSTPROCESSING = "postprocessing"
    MESH_GENERATION = "mesh_generation"
    DONE = "done"
