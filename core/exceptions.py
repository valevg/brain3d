"""Кастомные исключения ядра Brain3D AI.

Все исключения проекта наследуются от Brain3DError, что позволяет
API-слою единообразно перехватывать и транслировать ошибки в HTTP-ответы.
"""

from __future__ import annotations


class Brain3DError(Exception):
    """Базовое исключение для всех ошибок приложения Brain3D AI."""


class UnsupportedModalityError(Brain3DError):
    """Модальность исследования не поддерживается текущим пайплайном."""


class StudyLoadError(Brain3DError):
    """Ошибка загрузки исследования (DICOM/NIfTI) — повреждённые или неполные данные."""


class InvalidDicomSeriesError(StudyLoadError):
    """DICOM-серия не прошла валидацию (разные размеры срезов, пропуски и т.п.)."""


class MixedModalitySeriesError(InvalidDicomSeriesError):
    """В одной директории обнаружены DICOM-файлы разных модальностей (например, КТ и МРТ)."""


class SequenceNotFoundError(Brain3DError):
    """Не найдена МР-серия с запрошенным типом взвешивания (T1/T1ce/T2/FLAIR/DWI)."""


class PreprocessingError(Brain3DError):
    """Ошибка на этапе препроцессинга объёма (нормализация, ресемплинг)."""


class ModelNotFoundError(Brain3DError):
    """Веса модели для указанной модальности/задачи не найдены."""


class ModelDownloadError(Brain3DError):
    """Ошибка скачивания весов предобученной модели (нет сети, нет места на диске и т.п.)."""


class ChecksumMismatchError(ModelDownloadError):
    """Контрольная сумма скачанного файла весов не совпадает с ожидаемой — файл повреждён."""


class InferenceError(Brain3DError):
    """Ошибка на этапе инференса нейросети."""


class MeshGenerationError(Brain3DError):
    """Ошибка построения 3D-меша из карты сегментации (например, пустая маска)."""


class VesselSegmentationError(Brain3DError):
    """Ни один метод сегментации сосудов не дал пригодного результата."""


class ConfigurationError(Brain3DError):
    """Ошибка конфигурации приложения (некорректный config.yaml или .env)."""
