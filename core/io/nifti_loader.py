"""Загрузчик исследований в формате NIfTI (.nii, .nii.gz) на базе SimpleITK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from core.exceptions import StudyLoadError
from core.io.base_loader import BaseLoader
from core.logging_config import get_logger
from shared.enums import Modality
from shared.types import VolumeData, VoxelSpacing

logger = get_logger(__name__)

_NIFTI_SUFFIXES = (".nii", ".nii.gz")


class NiftiLoader(BaseLoader):
    """Загружает одиночный NIfTI-файл. Модальность указывается явно, т.к. NIfTI не хранит её."""

    def __init__(self, default_modality: Modality = Modality.MRI) -> None:
        """Инициализирует загрузчик.

        Args:
            default_modality: Модальность, используемая, если её нельзя определить
                из имени файла (например, "*_ct.nii.gz" -> CT, иначе MRI).
        """
        self._default_modality = default_modality

    def can_load(self, path: Path) -> bool:
        """Проверяет, что путь указывает на файл с расширением .nii или .nii.gz."""
        return path.is_file() and str(path).lower().endswith(_NIFTI_SUFFIXES)

    def load(self, path: Path) -> VolumeData:
        """Загружает NIfTI-файл.

        Args:
            path: Путь к .nii/.nii.gz файлу.

        Returns:
            VolumeData с интенсивностями и spacing из заголовка NIfTI.

        Raises:
            StudyLoadError: Если файл повреждён или не читается SimpleITK.
        """
        try:
            image = sitk.ReadImage(str(path))
            voxels = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z, Y, X)
            spacing_x, spacing_y, spacing_z = image.GetSpacing()
            modality = self._guess_modality(path)

            logger.info(
                "Загружен NIfTI-файл: %s, форма=%s, модальность=%s",
                path,
                voxels.shape,
                modality.value,
            )

            return VolumeData(
                voxels=voxels,
                spacing=VoxelSpacing(x=spacing_x, y=spacing_y, z=spacing_z),
                origin=image.GetOrigin(),
                modality=modality,
                metadata={"source_path": str(path)},
            )
        except Exception as exc:  # noqa: BLE001
            raise StudyLoadError(f"Не удалось загрузить NIfTI-файл {path}: {exc}") from exc

    def _guess_modality(self, path: Path) -> Modality:
        """Пытается определить модальность по имени файла, иначе возвращает default_modality."""
        name = path.name.lower()
        if "_ct" in name or name.startswith("ct"):
            return Modality.CT
        if "_mr" in name or "_mri" in name or name.startswith("mr"):
            return Modality.MRI
        return self._default_modality
