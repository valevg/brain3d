"""Абстрактный интерфейс загрузчика томографических данных."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from shared.types import VolumeData


class BaseLoader(ABC):
    """Единый интерфейс загрузки исследования независимо от формата и модальности.

    Конкретные реализации (DicomLoader, NiftiLoader) отвечают только за чтение
    файлов и приведение к общей структуре VolumeData; определение модальности
    и дальнейшая обработка выполняются выше по пайплайну.
    """

    @abstractmethod
    def can_load(self, path: Path) -> bool:
        """Проверяет, способен ли загрузчик обработать указанный путь.

        Args:
            path: Путь к файлу или директории с исследованием.

        Returns:
            True, если формат распознан этим загрузчиком.
        """

    @abstractmethod
    def load(self, path: Path) -> VolumeData:
        """Загружает исследование и приводит его к единой структуре VolumeData.

        Args:
            path: Путь к файлу (NIfTI) или директории с DICOM-серией.

        Returns:
            Загруженный объём с метаданными.

        Raises:
            StudyLoadError: Если данные повреждены или не читаются.
        """
