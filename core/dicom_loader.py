"""Расширенный загрузчик "сырых" DICOM-архивов с определением модальности и МР-взвешивания.

В отличие от core.io.dicom_loader.DicomLoader (минимальный загрузчик, используемый
единым пайплайном core.io.BaseLoader и предполагающий, что путь — это уже одна
чистая DICOM-серия), этот модуль решает более общую задачу подготовки данных:

    - директория может содержать НЕСКОЛЬКО серий (в т.ч. разных модальностей);
    - для МРТ заранее неизвестно, какое это взвешивание (T1/T1ce/T2/FLAIR/DWI);
    - нужен явный контроль над конвертацией в HU (КТ) и в NIfTI.

Типичный сценарий использования — ETL-скрипт, который разбирает сырой экспорт PACS
или архив вроде BraTS и готовит из него чистые NIfTI-файлы для core.io.NiftiLoader.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

from core.exceptions import (
    InvalidDicomSeriesError,
    MixedModalitySeriesError,
    SequenceNotFoundError,
    StudyLoadError,
    UnsupportedModalityError,
)
from core.logging_config import get_logger
from shared.enums import Modality, MRWeighting

logger = get_logger(__name__)

# Ключевые слова для определения взвешивания по человекочитаемым полям DICOM
# (SequenceName/ProtocolName/SeriesDescription). Порядок важен: более специфичные
# паттерны (FLAIR, DWI, T1ce) проверяются раньше общих (T1, T2), чтобы избежать
# ложного срабатывания T1 на строке вроде "T1ce_post_contrast".
_WEIGHTING_KEYWORDS: dict[MRWeighting, tuple[str, ...]] = {
    MRWeighting.FLAIR: ("flair",),
    MRWeighting.DWI: ("dwi", "diff", "trace", "adc", "b1000", "b0"),
    MRWeighting.T1CE: ("t1ce", "t1c", "t1_gd", "t1+c", "t1 post", "t1gd", "t1_ce"),
    MRWeighting.T1: ("t1",),
    MRWeighting.T2: ("t2",),
}
_WEIGHTING_KEYWORD_ORDER = (
    MRWeighting.FLAIR,
    MRWeighting.DWI,
    MRWeighting.T1CE,
    MRWeighting.T1,
    MRWeighting.T2,
)


# --------------------------------------------------------------------------- #
# Свободные функции: обнаружение и разбор DICOM-файлов на уровне директории
# --------------------------------------------------------------------------- #


def read_dicom_headers(directory: Path) -> dict[Path, pydicom.Dataset]:
    """Рекурсивно читает заголовки (без пиксельных данных) всех DICOM-файлов в директории.

    Файлы, которые не являются валидным DICOM (например, DICOMDIR, .txt, .json),
    молча пропускаются — это ожидаемая ситуация для "сырых" экспортов PACS.

    Args:
        directory: Директория для рекурсивного поиска DICOM-файлов.

    Returns:
        Словарь {путь_к_файлу: pydicom.Dataset (только заголовок)}.
    """
    headers: dict[Path, pydicom.Dataset] = {}
    for candidate in sorted(directory.rglob("*")):
        if not candidate.is_file():
            continue
        try:
            headers[candidate] = pydicom.dcmread(candidate, stop_before_pixels=True)
        except InvalidDicomError:
            continue
        except Exception as exc:  # noqa: BLE001 — повреждённый файл не должен ронять весь скан
            logger.warning("Не удалось прочитать заголовок %s: %s", candidate, exc)
    return headers


def group_datasets_by_series(headers: dict[Path, pydicom.Dataset]) -> dict[str, list[Path]]:
    """Группирует файлы по тегу SeriesInstanceUID.

    Args:
        headers: Результат read_dicom_headers().

    Returns:
        Словарь {series_instance_uid: [пути к файлам этой серии]}.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for file_path, dataset in headers.items():
        series_uid = str(getattr(dataset, "SeriesInstanceUID", "UNKNOWN"))
        groups[series_uid].append(file_path)
    return dict(groups)


def sort_files_by_slice_position(headers: dict[Path, pydicom.Dataset]) -> list[Path]:
    """Сортирует файлы одной серии по физическому положению среза вдоль нормали к плоскости.

    Использует проекцию ImagePositionPatient на нормаль к плоскости среза
    (векторное произведение row/column cosines из ImageOrientationPatient) —
    этот способ устойчив к пропускам срезов и не зависит от порядка InstanceNumber.

    Args:
        headers: Заголовки файлов ОДНОЙ серии (уже отфильтрованной по SeriesInstanceUID).

    Returns:
        Список путей, отсортированный от первого среза к последнему.
    """
    files = list(headers.keys())
    if not files:
        return files

    sample = headers[files[0]]
    orientation = getattr(sample, "ImageOrientationPatient", None)

    if orientation is None:
        # Нет геометрии ориентации — единственный доступный порядок это InstanceNumber.
        return sorted(files, key=lambda f: int(getattr(headers[f], "InstanceNumber", 0) or 0))

    row_cosine = np.array(orientation[0:3], dtype=np.float64)
    col_cosine = np.array(orientation[3:6], dtype=np.float64)
    slice_normal = np.cross(row_cosine, col_cosine)

    def _projection(file_path: Path) -> float:
        position = getattr(headers[file_path], "ImagePositionPatient", None)
        if position is None:
            return float(getattr(headers[file_path], "InstanceNumber", 0) or 0)
        return float(np.dot(slice_normal, np.array(position, dtype=np.float64)))

    return sorted(files, key=_projection)


def _to_float(value: Any) -> float | None:
    """Безопасно приводит DICOM-значение (может быть None/строкой/DSfloat) к float."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _detect_modality_from_headers(datasets: list[pydicom.Dataset]) -> Modality:
    """Определяет модальность серии по тегу (0008,0060) Modality с резервной эвристикой.

    Критерии из ТЗ:
        - КТ: Modality="CT", как правило присутствуют RescaleSlope/RescaleIntercept
          (перевод сырых значений в единицы Хаунсфилда).
        - МРТ: Modality="MR", как правило присутствуют ScanningSequence/EchoNumbers.

    Args:
        datasets: Заголовки файлов серии (используется первый, т.к. модальность
            одинакова для всех срезов одной серии).

    Returns:
        Modality.CT или Modality.MRI.

    Raises:
        UnsupportedModalityError: Если модальность не CT/MR и не определяется эвристикой.
    """
    dataset = datasets[0]
    modality_tag = str(getattr(dataset, "Modality", "")).strip().upper()

    if modality_tag == "CT":
        if not hasattr(dataset, "RescaleSlope") or not hasattr(dataset, "RescaleIntercept"):
            logger.warning("КТ-серия без RescaleSlope/RescaleIntercept — значения могут быть не в HU")
        return Modality.CT

    if modality_tag in ("MR", "MRI"):
        return Modality.MRI

    # Резервная эвристика на случай нестандартного/отсутствующего тега Modality.
    if hasattr(dataset, "ScanningSequence") or hasattr(dataset, "EchoNumbers"):
        logger.warning(
            "Modality=%r нестандартен, определено как MRI по ScanningSequence/EchoNumbers",
            modality_tag,
        )
        return Modality.MRI
    if hasattr(dataset, "RescaleSlope"):
        logger.warning("Modality=%r нестандартен, определено как CT по RescaleSlope", modality_tag)
        return Modality.CT

    raise UnsupportedModalityError(f"Не удалось определить модальность по тегу Modality={modality_tag!r}")


def _classify_mr_weighting(dataset: pydicom.Dataset) -> MRWeighting:
    """Определяет тип МР-взвешивания (T1/T1ce/T2/FLAIR/DWI) по тегам DICOM.

    Порядок определения:
        1. Ключевые слова в человекочитаемых полях (SequenceName/ProtocolName/
           SeriesDescription) — самый надёжный источник, когда они заполнены.
        2. Числовая эвристика по TR/TE/TI (RepetitionTime/EchoTime/InversionTime)
           и ScanningSequence — используется, если описание не дало результата.

    Args:
        dataset: Заголовок (первого файла) МР-серии.

    Returns:
        Наиболее вероятный MRWeighting; MRWeighting.UNKNOWN, если не удалось определить.
    """
    text_fields = " ".join(
        str(getattr(dataset, tag, "") or "") for tag in ("SequenceName", "ProtocolName", "SeriesDescription")
    ).lower()
    has_contrast = bool(getattr(dataset, "ContrastBolusAgent", "") or "")

    for weighting in _WEIGHTING_KEYWORD_ORDER:
        if any(keyword in text_fields for keyword in _WEIGHTING_KEYWORDS[weighting]):
            if weighting == MRWeighting.T1 and has_contrast:
                return MRWeighting.T1CE
            return weighting

    repetition_time = _to_float(getattr(dataset, "RepetitionTime", None))
    echo_time = _to_float(getattr(dataset, "EchoTime", None))
    inversion_time = _to_float(getattr(dataset, "InversionTime", None))
    scanning_sequence = set(getattr(dataset, "ScanningSequence", []) or [])
    diffusion_bvalue = _to_float(getattr(dataset, "DiffusionBValue", None))

    # FLAIR: инверсия-восстановление с длинным TI и длинным TR (классические Т2-FLAIR параметры).
    if inversion_time is not None and inversion_time > 1500 and repetition_time and repetition_time > 5000:
        return MRWeighting.FLAIR

    # DWI: эхо-планарная последовательность (EP) с ненулевым b-value.
    if "EP" in scanning_sequence and diffusion_bvalue:
        return MRWeighting.DWI

    if repetition_time is not None and echo_time is not None:
        if repetition_time < 800 and echo_time < 30:
            return MRWeighting.T1CE if has_contrast else MRWeighting.T1
        if repetition_time > 2000 and echo_time > 60:
            return MRWeighting.T2

    logger.debug(
        "Не удалось классифицировать МР-взвешивание: TR=%s, TE=%s, TI=%s",
        repetition_time,
        echo_time,
        inversion_time,
    )
    return MRWeighting.UNKNOWN


def _apply_ct_rescale(pixel_array: np.ndarray, dataset: pydicom.Dataset) -> np.ndarray:
    """Переводит сырые значения пикселей КТ в единицы Хаунсфилда (HU).

    Формула стандарта DICOM: HU = pixel_value * RescaleSlope + RescaleIntercept.

    Args:
        pixel_array: Сырой массив пикселей (dataset.pixel_array).
        dataset: Датасет среза с тегами RescaleSlope/RescaleIntercept.

    Returns:
        Массив float32 в единицах HU.
    """
    slope = _to_float(getattr(dataset, "RescaleSlope", 1.0)) or 1.0
    intercept = _to_float(getattr(dataset, "RescaleIntercept", 0.0)) or 0.0
    return pixel_array.astype(np.float32) * slope + intercept


def _geometry_from_datasets(
    sorted_datasets: list[pydicom.Dataset],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, ...]]:
    """Вычисляет (spacing, origin, direction) из тегов геометрии первых срезов серии.

    Direction — это 3x3 матрица направляющих косинусов, развёрнутая в 9 чисел
    построчно (тот же формат, что возвращает SimpleITK Image.GetDirection()),
    где столбцы матрицы — это направления осей i (по строкам), j (по столбцам)
    и k (между срезами) в системе координат пациента (LPS, как в самом DICOM).

    Args:
        sorted_datasets: Датасеты серии, ОТСОРТИРОВАННЫЕ по положению среза.

    Returns:
        Кортеж (spacing_xyz, origin_xyz, direction_9).
    """
    first = sorted_datasets[0]
    orientation = getattr(first, "ImageOrientationPatient", None) or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    row_cosine = np.array(orientation[0:3], dtype=np.float64)
    col_cosine = np.array(orientation[3:6], dtype=np.float64)
    slice_cosine = np.cross(row_cosine, col_cosine)

    pixel_spacing = getattr(first, "PixelSpacing", None) or (1.0, 1.0)
    spacing_x = float(pixel_spacing[1])  # PixelSpacing = [spacing между строками, между столбцами]
    spacing_y = float(pixel_spacing[0])

    if len(sorted_datasets) > 1:
        position_first = np.array(getattr(first, "ImagePositionPatient", (0.0, 0.0, 0.0)), dtype=np.float64)
        position_second = np.array(
            getattr(sorted_datasets[1], "ImagePositionPatient", (0.0, 0.0, 0.0)), dtype=np.float64
        )
        spacing_z = float(abs(np.dot(slice_cosine, position_second - position_first)))
        if spacing_z == 0.0:
            spacing_z = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    else:
        spacing_z = float(getattr(first, "SliceThickness", 1.0) or 1.0)

    origin = tuple(float(v) for v in getattr(first, "ImagePositionPatient", (0.0, 0.0, 0.0)))
    direction = (
        float(row_cosine[0]), float(col_cosine[0]), float(slice_cosine[0]),
        float(row_cosine[1]), float(col_cosine[1]), float(slice_cosine[1]),
        float(row_cosine[2]), float(col_cosine[2]), float(slice_cosine[2]),
    )  # fmt: skip

    return (spacing_x, spacing_y, spacing_z), origin, direction


def _lps_direction_to_nifti_affine(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    direction: tuple[float, ...],
) -> np.ndarray:
    """Строит affine-матрицу NIfTI (4x4, RAS+) из геометрии DICOM/ITK (LPS).

    DICOM и ITK используют систему координат LPS (Left-Posterior-Superior), а формат
    NIfTI — RAS+ (Right-Anterior-Superior). Поэтому знак первых двух осей (x, y)
    инвертируется — это стандартное преобразование LPS -> RAS, используемое во всех
    конвертерах DICOM->NIfTI (dcm2niix, dicom2nifti и т.п.).

    Args:
        origin: Мировые координаты первого воксела (мм), в LPS.
        spacing: Физический размер воксела (мм) по осям (x, y, z).
        direction: Направляющие косинусы (9 чисел), см. _geometry_from_datasets().

    Returns:
        Affine-матрица 4x4 в формате, ожидаемом nibabel.Nifti1Image.
    """
    direction_matrix = np.array(direction, dtype=np.float64).reshape(3, 3)
    scaled_direction = direction_matrix @ np.diag(spacing)

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = scaled_direction
    affine_lps[:3, 3] = origin

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ affine_lps


def _series_location(files: list[Path]) -> Path:
    """Возвращает "путь к серии": общую директорию файлов либо путь к первому файлу.

    Если все файлы серии лежат в одной директории (раскладка "одна серия — одна
    папка", типичная для экспортов PACS), возвращается эта директория — её можно
    напрямую передать в DICOMLoader.load_dicom_series(). Если файлы серии
    перемешаны с другими сериями в общей папке, возвращается путь к первому файлу.

    Args:
        files: Список путей файлов одной серии (обычно уже отсортированных).

    Returns:
        Директория серии или путь к первому файлу.
    """
    parents = {f.parent for f in files}
    if len(parents) == 1:
        return next(iter(parents))
    return sorted(files)[0]


# --------------------------------------------------------------------------- #
# DICOMLoader
# --------------------------------------------------------------------------- #


class DICOMLoader:
    """Загружает одну DICOM-серию, определяет её модальность и (для МРТ) взвешивание.

    Основной путь чтения пиксельных данных — SimpleITK (быстрый, корректно
    обрабатывает большинство transfer syntax, включая сжатые). При его сбое
    используется fallback на чистый pydicom с ручной сортировкой срезов по
    ImagePositionPatient и ручным построением геометрии.

    Пример:
        >>> loader = DICOMLoader()
        >>> voxels = loader.load_dicom_series("/data/patient001/ct_head")
        >>> loader.detect_modality()
        'CT'
        >>> info = loader.get_volume_info()
        >>> loader.convert_to_nifti("/data/patient001/ct_head.nii.gz")
    """

    def __init__(self, path: str | Path | None = None, series_uid: str | None = None) -> None:
        """Инициализирует загрузчик.

        Args:
            path: Директория с DICOM-серией (может содержать несколько серий —
                тогда потребуется указать series_uid). Можно не указывать здесь
                и передать напрямую в load_dicom_series().
            series_uid: SeriesInstanceUID серии для загрузки, если директория
                содержит несколько серий.
        """
        self.path: Path | None = Path(path) if path is not None else None
        self.series_uid: str | None = series_uid

        self._voxels: np.ndarray | None = None
        self._spacing: tuple[float, float, float] | None = None
        self._origin: tuple[float, float, float] | None = None
        self._direction: tuple[float, ...] | None = None
        self._modality: Modality | None = None
        self._mr_weighting: MRWeighting | None = None
        self._sitk_image: sitk.Image | None = None
        self._headers: dict[Path, pydicom.Dataset] = {}

    @property
    def modality(self) -> Modality | None:
        """Определённая модальность серии (None, если load_dicom_series ещё не вызывался)."""
        return self._modality

    @property
    def mr_weighting(self) -> MRWeighting | None:
        """Определённое МР-взвешивание (None для КТ или до загрузки)."""
        return self._mr_weighting

    def load_dicom_series(
        self,
        path: str | Path | None = None,
        series_uid: str | None = None,
    ) -> np.ndarray:
        """Загружает DICOM-серию из директории и возвращает 3D-массив интенсивностей.

        Args:
            path: Директория с серией. Если не указана, используется self.path.
            series_uid: SeriesInstanceUID для выбора конкретной серии из директории
                с несколькими сериями. Если не указан, используется self.series_uid.

        Returns:
            3D-массив float32 формы (Z, Y, X). Для КТ значения — в единицах HU.

        Raises:
            StudyLoadError: Если путь не указан.
            InvalidDicomSeriesError: Если серия не найдена, файлы не читаются,
                срезы имеют разное разрешение, либо найдено несколько серий одной
                модальности без явного указания series_uid.
            MixedModalitySeriesError: Если директория содержит серии разных
                модальностей (КТ и МРТ) без явного указания series_uid.
            UnsupportedModalityError: Если модальность серии не CT и не MR.
        """
        series_path = Path(path) if path is not None else self.path
        if series_path is None:
            raise StudyLoadError("Не указан путь к DICOM-серии")
        self.path = series_path

        target_series_uid = series_uid if series_uid is not None else self.series_uid

        headers = read_dicom_headers(series_path)
        if not headers:
            raise InvalidDicomSeriesError(f"В директории {series_path} не найдено DICOM-файлов")

        series_groups = group_datasets_by_series(headers)
        selected_uid = self._resolve_series_uid(series_groups, headers, target_series_uid)
        selected_files = series_groups[selected_uid]
        selected_headers = {f: headers[f] for f in selected_files}

        self._validate_consistent_geometry(selected_headers)

        sorted_files = sort_files_by_slice_position(selected_headers)
        sorted_datasets = [selected_headers[f] for f in sorted_files]

        self._modality = _detect_modality_from_headers(sorted_datasets)
        self._mr_weighting = (
            _classify_mr_weighting(sorted_datasets[0]) if self._modality == Modality.MRI else None
        )

        try:
            voxels = self._load_with_sitk(series_path, selected_uid)
        except Exception as exc:  # noqa: BLE001 — любая ошибка SimpleITK -> пробуем pydicom
            logger.warning(
                "SimpleITK не смог прочитать серию %s (%s); используется pydicom fallback", selected_uid, exc
            )
            voxels = self._load_with_pydicom(sorted_files, sorted_datasets)

        self.series_uid = selected_uid
        self._headers = selected_headers
        self._voxels = voxels

        logger.info(
            "Загружена DICOM-серия %s: форма=%s, модальность=%s%s",
            series_path,
            voxels.shape,
            self._modality.value,
            f", взвешивание={self._mr_weighting.value}" if self._mr_weighting else "",
        )
        return voxels

    def detect_modality(self) -> str:
        """Возвращает определённую модальность серии.

        Returns:
            "CT" или "MRI".

        Raises:
            StudyLoadError: Если серия ещё не загружена.
        """
        if self._modality is None:
            raise StudyLoadError("Модальность неизвестна — сначала вызовите load_dicom_series()")
        return self._modality.value

    def get_volume_info(self) -> dict[str, Any]:
        """Возвращает метаданные загруженного объёма.

        Returns:
            Словарь с ключами: shape, spacing, origin, direction, dtype, modality
            и (только для МРТ) mr_weighting.

        Raises:
            StudyLoadError: Если серия ещё не загружена.
        """
        if self._voxels is None or self._modality is None:
            raise StudyLoadError("Объём не загружен — сначала вызовите load_dicom_series()")

        info: dict[str, Any] = {
            "shape": tuple(self._voxels.shape),
            "spacing": self._spacing,
            "origin": self._origin,
            "direction": self._direction,
            "dtype": str(self._voxels.dtype),
            "modality": self._modality.value,
        }
        if self._mr_weighting is not None:
            info["mr_weighting"] = self._mr_weighting.value
        return info

    def convert_to_nifti(self, output_path: str | Path) -> Path:
        """Сохраняет загруженный объём в формате NIfTI (.nii/.nii.gz).

        Если объём был прочитан через SimpleITK (основной путь), используется
        встроенный ITK-writer (наиболее надёжный вариант). Если сработал pydicom
        fallback, affine-матрица строится вручную из геометрии DICOM-тегов и файл
        сохраняется через nibabel.

        Args:
            output_path: Путь к выходному файлу (.nii или .nii.gz).

        Returns:
            Путь к сохранённому файлу.

        Raises:
            StudyLoadError: Если серия ещё не загружена.
        """
        if self._voxels is None:
            raise StudyLoadError("Объём не загружен — сначала вызовите load_dicom_series()")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._sitk_image is not None:
            sitk.WriteImage(self._sitk_image, str(output_path))
        else:
            import nibabel as nib

            assert self._spacing is not None and self._origin is not None and self._direction is not None
            affine = _lps_direction_to_nifti_affine(self._origin, self._spacing, self._direction)
            data_xyz = np.asarray(self._voxels).transpose(2, 1, 0)  # (Z,Y,X) -> (X,Y,Z) для NIfTI
            nib.save(nib.Nifti1Image(data_xyz, affine), str(output_path))

        logger.info("Объём сохранён в NIfTI: %s", output_path)
        return output_path

    @staticmethod
    def normalize_mri_intensity(
        voxels: np.ndarray,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> np.ndarray:
        """Нормализует интенсивность МРТ клиппингом по перцентилям и приведением к [0, 1].

        В отличие от z-score нормализации в core.preprocessing.MriPreprocessor
        (используется непосредственно перед инференсом модели), этот метод
        предназначен для лёгкой нормализации "сырого" объёма сразу после загрузки
        (например, для превью) и устойчив к выбросам за счёт перцентильного клиппинга.

        Args:
            voxels: Исходный массив интенсивностей МРТ.
            lower_percentile: Нижний перцентиль клиппинга (0..100).
            upper_percentile: Верхний перцентиль клиппинга (0..100).

        Returns:
            Массив float32 в диапазоне [0, 1].
        """
        low, high = np.percentile(voxels, [lower_percentile, upper_percentile])
        if high <= low:
            return voxels.astype(np.float32)
        clipped = np.clip(voxels, low, high)
        return ((clipped - low) / (high - low)).astype(np.float32)

    def _resolve_series_uid(
        self,
        series_groups: dict[str, list[Path]],
        headers: dict[Path, pydicom.Dataset],
        target_series_uid: str | None,
    ) -> str:
        """Выбирает SeriesInstanceUID для загрузки, обрабатывая смешанные/множественные серии."""
        if target_series_uid is not None:
            if target_series_uid not in series_groups:
                raise InvalidDicomSeriesError(
                    f"Серия {target_series_uid} не найдена. Доступные серии: {list(series_groups)}"
                )
            return target_series_uid

        if len(series_groups) == 1:
            return next(iter(series_groups))

        modalities = {
            str(getattr(headers[files[0]], "Modality", "UNKNOWN")) for files in series_groups.values()
        }
        if len(modalities) > 1:
            raise MixedModalitySeriesError(
                f"В директории {self.path} обнаружены смешанные модальности: {sorted(modalities)}. "
                "Укажите series_uid явно."
            )
        raise InvalidDicomSeriesError(
            f"В директории {self.path} обнаружено {len(series_groups)} серий одной модальности. "
            f"Укажите series_uid из: {list(series_groups)}"
        )

    @staticmethod
    def _validate_consistent_geometry(headers: dict[Path, pydicom.Dataset]) -> None:
        """Проверяет, что все срезы серии имеют одинаковое разрешение (Rows/Columns)."""
        shapes = {
            (int(ds.Rows), int(ds.Columns))
            for ds in headers.values()
            if hasattr(ds, "Rows") and hasattr(ds, "Columns")
        }
        if len(shapes) > 1:
            raise InvalidDicomSeriesError(f"Срезы серии имеют разное разрешение: {shapes}")

    def _load_with_sitk(self, series_path: Path, series_uid: str) -> np.ndarray:
        """Читает пиксельные данные серии через SimpleITK/GDCM (основной путь)."""
        file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_path), series_uid, False, True)
        if not file_names:
            raise InvalidDicomSeriesError(f"SimpleITK не нашёл файлы серии {series_uid} в {series_path}")

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_names)
        image = reader.Execute()

        self._sitk_image = image
        self._spacing = image.GetSpacing()
        self._origin = image.GetOrigin()
        self._direction = image.GetDirection()
        return sitk.GetArrayFromImage(image).astype(np.float32)

    def _load_with_pydicom(
        self,
        sorted_files: list[Path],
        sorted_datasets: list[pydicom.Dataset],
    ) -> np.ndarray:
        """Читает пиксельные данные серии через чистый pydicom (fallback-путь)."""
        slices: list[np.ndarray] = []
        for file_path in sorted_files:
            full_dataset = pydicom.dcmread(file_path)
            pixel_array = full_dataset.pixel_array
            if self._modality == Modality.CT:
                pixel_array = _apply_ct_rescale(pixel_array, full_dataset)
            slices.append(pixel_array.astype(np.float32))

        voxels = np.stack(slices, axis=0)
        spacing, origin, direction = _geometry_from_datasets(sorted_datasets)
        self._spacing = spacing
        self._origin = origin
        self._direction = direction
        self._sitk_image = None
        return voxels


# --------------------------------------------------------------------------- #
# MRISeriesAnalyzer
# --------------------------------------------------------------------------- #


class MRISeriesAnalyzer:
    """Анализирует директорию с одной или несколькими МР-сериями и их взвешиванием.

    Используется, когда заранее неизвестно, какие последовательности (T1/T1ce/T2/
    FLAIR/DWI) присутствуют в исследовании — типичная ситуация для сырых экспортов
    PACS и датасетов вроде BraTS, где серии могут лежать как каждая в своей
    подпапке, так и все вперемешку в одной директории.
    """

    def analyze_series(self, path: str | Path) -> dict[str, dict[str, Any]]:
        """Разбирает директорию на отдельные серии и определяет параметры каждой.

        Args:
            path: Директория, содержащая одну или несколько DICOM-серий
                (рекурсивно, включая вложенные подпапки).

        Returns:
            Словарь {series_instance_uid: {modality, weighting, num_files,
            sequence_name, series_description, repetition_time_ms, echo_time_ms,
            contrast_agent, files}}.

        Raises:
            InvalidDicomSeriesError: Если в директории не найдено DICOM-файлов.
        """
        directory = Path(path)
        headers = read_dicom_headers(directory)
        if not headers:
            raise InvalidDicomSeriesError(f"В директории {directory} не найдено DICOM-файлов")

        series_groups = group_datasets_by_series(headers)

        result: dict[str, dict[str, Any]] = {}
        for series_uid, files in series_groups.items():
            sample = headers[files[0]]
            modality = str(getattr(sample, "Modality", "UNKNOWN")).strip().upper()
            weighting = _classify_mr_weighting(sample) if modality in ("MR", "MRI") else None

            result[series_uid] = {
                "modality": modality,
                "weighting": weighting.value if weighting else None,
                "num_files": len(files),
                "sequence_name": getattr(sample, "SequenceName", None),
                "series_description": getattr(sample, "SeriesDescription", None),
                "repetition_time_ms": _to_float(getattr(sample, "RepetitionTime", None)),
                "echo_time_ms": _to_float(getattr(sample, "EchoTime", None)),
                "contrast_agent": getattr(sample, "ContrastBolusAgent", None),
                "files": sorted(files),
            }
        return result

    def group_by_sequence(self, path: str | Path) -> dict[str, list[Path]]:
        """Группирует все DICOM-файлы директории по типу МР-взвешивания.

        Args:
            path: Директория с одной или несколькими сериями.

        Returns:
            Словарь {"T1": [...], "T2": [...], "FLAIR": [...], "T1ce": [...],
            "DWI": [...], "unknown": [...]} — списки файлов по каждому взвешиванию,
            присутствующему в директории.
        """
        analysis = self.analyze_series(path)
        grouped: dict[str, list[Path]] = defaultdict(list)
        for series_info in analysis.values():
            weighting = series_info["weighting"] or MRWeighting.UNKNOWN.value
            grouped[weighting].extend(series_info["files"])
        return dict(grouped)

    def select_best_sequence(self, path: str | Path, target: str = "T1ce") -> Path:
        """Находит наиболее полную серию с заданным типом взвешивания.

        При наличии нескольких серий одного взвешивания (например, повторное
        сканирование) выбирается серия с наибольшим числом срезов.

        Args:
            path: Директория с одной или несколькими сериями.
            target: Искомое взвешивание ("T1", "T1ce", "T2", "FLAIR", "DWI"),
                регистр не важен.

        Returns:
            Путь к директории серии (если все её файлы лежат в одной папке)
            либо путь к первому файлу серии (если серии перемешаны в общей папке).

        Raises:
            SequenceNotFoundError: Если серия с таким взвешиванием не найдена.
        """
        analysis = self.analyze_series(path)
        target_normalized = target.strip().lower()

        candidates = [
            info for info in analysis.values() if (info["weighting"] or "").lower() == target_normalized
        ]
        if not candidates:
            available = sorted({info["weighting"] for info in analysis.values() if info["weighting"]})
            raise SequenceNotFoundError(
                f"Серия с взвешиванием {target!r} не найдена в {path}. Доступны: {available}"
            )

        best = max(candidates, key=lambda info: info["num_files"])
        return _series_location(best["files"])


if __name__ == "__main__":
    # --- Пример использования: КТ ---
    logging.basicConfig(level=logging.INFO)

    ct_loader = DICOMLoader()
    ct_voxels = ct_loader.load_dicom_series("./data/examples/ct_head")
    print("КТ:", ct_loader.detect_modality(), ct_voxels.shape, ct_loader.get_volume_info())
    ct_loader.convert_to_nifti("./data/examples/ct_head.nii.gz")

    # --- Пример использования: МРТ с выбором нужной последовательности ---
    analyzer = MRISeriesAnalyzer()
    sequences = analyzer.group_by_sequence("./data/examples/mri_brain_raw")
    print("Найденные последовательности:", {k: len(v) for k, v in sequences.items()})

    t1ce_series_path = analyzer.select_best_sequence("./data/examples/mri_brain_raw", target="T1ce")
    mri_loader = DICOMLoader()
    mri_voxels = mri_loader.load_dicom_series(t1ce_series_path)
    print("МРТ:", mri_loader.detect_modality(), mri_loader.mr_weighting, mri_voxels.shape)
    mri_loader.convert_to_nifti("./data/examples/mri_brain_t1ce.nii.gz")
