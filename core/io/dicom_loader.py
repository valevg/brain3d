"""Загрузчик DICOM-серий (КТ/МРТ) на базе SimpleITK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from core.exceptions import InvalidDicomSeriesError, StudyLoadError
from core.io.base_loader import BaseLoader
from core.logging_config import get_logger
from shared.enums import Modality
from shared.types import VolumeData, VoxelSpacing

logger = get_logger(__name__)


class DicomLoader(BaseLoader):
    """Загружает DICOM-серию из директории и определяет модальность по DICOM-тегам."""

    def can_load(self, path: Path) -> bool:
        """Директория считается DICOM-серией, если в ней есть хотя бы один читаемый DICOM-файл."""
        if not path.is_dir():
            return False
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
        return len(series_ids) > 0

    def load(self, path: Path) -> VolumeData:
        """Загружает DICOM-серию из директории.

        Args:
            path: Директория, содержащая файлы одной DICOM-серии.

        Returns:
            VolumeData с интенсивностями, spacing и модальностью, определённой
            по тегу (0008,0060) первого файла серии.

        Raises:
            InvalidDicomSeriesError: Если серия не найдена или срезы несогласованы.
            StudyLoadError: При иной ошибке чтения.
        """
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
            if not series_ids:
                raise InvalidDicomSeriesError(f"В директории {path} не найдено DICOM-серий")

            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_ids[0])
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            image = reader.Execute()

            modality_tag = reader.GetMetaData(0, "0008|0060") if reader.HasMetaDataKey(0, "0008|0060") else "CT"
            modality = Modality.from_dicom_tag(modality_tag)

            voxels = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z, Y, X)
            spacing_x, spacing_y, spacing_z = image.GetSpacing()

            logger.info(
                "Загружена DICOM-серия: %s, форма=%s, модальность=%s",
                path,
                voxels.shape,
                modality.value,
            )

            return VolumeData(
                voxels=voxels,
                spacing=VoxelSpacing(x=spacing_x, y=spacing_y, z=spacing_z),
                origin=image.GetOrigin(),
                modality=modality,
                metadata={"source_path": str(path), "num_slices": str(len(file_names))},
            )
        except InvalidDicomSeriesError:
            raise
        except Exception as exc:  # noqa: BLE001 — любая ошибка чтения оборачивается доменным исключением
            raise StudyLoadError(f"Не удалось загрузить DICOM-серию из {path}: {exc}") from exc
