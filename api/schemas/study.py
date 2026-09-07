"""Схемы данных для эндпоинтов загрузки исследований и получения результатов сегментации."""

from __future__ import annotations

from pydantic import BaseModel, Field

from shared.enums import Modality, StructureType


class StudyUploadResponse(BaseModel):
    """Ответ после успешной загрузки исследования на сервер."""

    study_id: str = Field(description="Уникальный идентификатор загруженного исследования")
    modality: Modality
    num_slices: int | None = Field(default=None, description="Число срезов (для DICOM-серий)")


class SegmentationRequest(BaseModel):
    """Запрос на запуск сегментации ранее загруженного исследования."""

    study_id: str


class MeshInfo(BaseModel):
    """Метаданные одного сгенерированного меша."""

    structure: StructureType
    download_url: str
    vertex_count: int
    face_count: int


class SegmentationStatusResponse(BaseModel):
    """Статус и результаты обработки исследования."""

    study_id: str
    status: str = Field(description="pending | processing | done | failed")
    meshes: list[MeshInfo] = Field(default_factory=list)
    error: str | None = None
