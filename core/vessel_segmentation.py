"""Сегментация сосудов: иерархия готовых моделей с гарантированным CPU-фолбэком.

Приоритет методов (от лучшего к всегда работающему):
    1. Готовые DL-модели — MONAI Model Zoo (специализированная модель под сосуды мозга,
       если она есть в реестре — см. предупреждение в models.pretrained_models) и
       NVIDIA VISTA-3D (127+ классов, требует GPU/NIM или облачный API).
    2. TotalSegmentator — самый надёжный из готовых методов при наличии GPU: не требует
       собственного дообучения, работает "из коробки" на КТ (и МРТ — task="total_mr").
    3. Классический фильтр Франги (skimage) — не требует ни GPU, ни весов, ни сети;
       гарантированный fallback, если все остальные методы недоступны/не установлены.
    4. Пороговая обработка по HU — узкоспециализированный метод для контрастной КТ-
       ангиографии (йодный контраст даёт характерно высокую плотность).

VesselSegmenter скрывает эту иерархию за набором методов segment_with_*() и одним
адаптивным methodом segment_adaptive(), который пробует их в порядке приоритета и
останавливается на первом успешном — а не просто выбирает один метод и падает,
если он недоступен.

Модуль лежит в core/, но полагается на обёртки внешних систем из models.pretrained_models
(TotalSegmentatorWrapper, VISTA3DWrapper, MONAIModelZoo) — зависимость однонаправленная
(core.vessel_segmentation -> models.pretrained_models -> core.exceptions), циклов нет.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator
from scipy import ndimage
from skimage.filters import frangi

from core.exceptions import Brain3DError, ConfigurationError, VesselSegmentationError
from core.logging_config import get_logger
from models.pretrained_models import (
    ComputeResources,
    MONAIModelZoo,
    PretrainedModelRegistry,
    TotalSegmentatorWrapper,
    VISTA3DWrapper,
    detect_compute_resources,
    segment_vessels_frangi,
)

logger = get_logger(__name__)

MethodName = Literal[
    "totalsegmentator", "vista3d", "monai_vessel_model", "frangi", "threshold", "threshold+frangi"
]


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #


class VesselSegmentationConfig(BaseModel):
    """Конфигурация VesselSegmenter: какие методы включены и их параметры.

    Загружается из YAML через from_yaml() (см. пример в configs/vessel_segmentation.yaml)
    либо создаётся напрямую со значениями по умолчанию.
    """

    # --- Какие методы разрешены (учитывается segment_adaptive() и compare_methods()) ---
    enable_totalsegmentator: bool = True
    enable_vista3d: bool = Field(
        default=False, description="Требует API-ключ/локальный NIM — выкл. по умолчанию"
    )
    enable_monai_vessel_model: bool = Field(
        default=False, description="См. предупреждение в MONAIModelZoo про отсутствие готового bundle"
    )
    enable_frangi: bool = True
    enable_threshold: bool = True

    # --- TotalSegmentator ---
    totalsegmentator_fast: bool = False
    totalsegmentator_task: str = "headneck_bones_vessels"

    # --- Frangi ---
    frangi_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)

    # --- Пороговая сегментация (контрастная КТ-ангиография) ---
    threshold_min_hu: float = 100.0
    threshold_max_hu: float = 500.0

    # --- Постобработка ---
    min_component_size: int = 100
    morphology_iterations: int = 2
    smoothing_iterations: int = 3
    circle_of_willis_roi_dilation: int = 5

    # --- Кэширование результатов сегментации ---
    cache_enabled: bool = True
    result_cache_dir: Path = Field(default=Path.home() / ".cache" / "brain3d" / "vessel_segmentation")

    @field_validator("result_cache_dir", mode="after")
    @classmethod
    def _expand_user_in_cache_dir(cls, value: Path) -> Path:
        """Разворачивает "~" в абсолютный путь — pathlib.Path сам этого не делает,
        а YAML-конфиги (см. configs/vessel_segmentation.yaml) обычно пишут путь с "~"."""
        return value.expanduser()

    @classmethod
    def from_yaml(cls, path: Path | str) -> "VesselSegmentationConfig":
        """Загружает конфигурацию из YAML-файла.

        Args:
            path: Путь к YAML-файлу (см. configs/vessel_segmentation.yaml).

        Returns:
            Валидированный VesselSegmentationConfig.

        Raises:
            ConfigurationError: Если файл не найден или не проходит валидацию.
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise ConfigurationError(f"Файл конфигурации не найден: {yaml_path}")
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            return cls.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 — оборачиваем любую ошибку парсинга/валидации
            raise ConfigurationError(f"Некорректный vessel_segmentation.yaml ({yaml_path}): {exc}") from exc


# --------------------------------------------------------------------------- #
# Постобработка (общая для всех методов)
# --------------------------------------------------------------------------- #


def remove_small_components(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Удаляет связные компоненты маски размером меньше min_size вокселей.

    В отличие от выбора ЕДИНСТВЕННОЙ наибольшей компоненты (как при skull-stripping),
    здесь может остаться несколько компонент — сосудистая сеть физиологически состоит
    из множества раздельных ветвей, а не одного цельного объекта.

    Args:
        mask: Бинарная маска (Z, Y, X).
        min_size: Минимальный размер компоненты в вокселях.

    Returns:
        Очищенная бинарная маска uint8.
    """
    labeled, num_components = ndimage.label(mask)
    if num_components == 0:
        return mask.astype(np.uint8)

    sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for component_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            cleaned[labeled == component_id] = 1
    return cleaned


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заполняет внутренние полости маски (например, шум внутри просвета сосуда).

    Args:
        mask: Бинарная маска.

    Returns:
        Маска uint8 без внутренних полостей.
    """
    return ndimage.binary_fill_holes(mask).astype(np.uint8)


def smooth_vessels(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Сглаживает границы маски морфологическим открытием (erosion -> dilation).

    При небольшом числе итераций убирает "зубчатость" контура без существенной
    потери тонких сосудистых структур; при слишком большом — рискует стереть
    мелкие перфорантные сосуды целиком.

    Args:
        mask: Бинарная маска.
        iterations: Число итераций эрозии/дилатации.

    Returns:
        Сглаженная маска uint8.
    """
    smoothed = ndimage.binary_opening(mask.astype(bool), iterations=iterations)
    return smoothed.astype(np.uint8)


def connect_vessels(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Морфологическое закрытие (dilation -> erosion) для восстановления связности сосудов.

    Полезно, когда сосуд прерывается на 1-2 воксельных промежутках из-за шума или
    недосегментации отдельными методами (особенно Франги на границах структур).

    Использует ПОЛНУЮ 26-связность (generate_binary_structure(3, 3)), а не структуру
    по умолчанию (6-связность, только грани): у 1-воксельно тонкой трубки с разрывом
    точка разрыва не имеет истинных боковых соседей даже после дилатации, и erosion
    с 6-связным элементом всегда стирает её обратно — закрытие с минимальной структурой
    попросту не может соединить тонкие сосуды, только объёмные блобы.

    Args:
        mask: Бинарная маска.
        iterations: Число итераций дилатации/эрозии.

    Returns:
        Маска uint8 с восстановленной связностью.
    """
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    closed = ndimage.binary_closing(mask.astype(bool), structure=structure, iterations=iterations)
    return closed.astype(np.uint8)


def _combine_masks(masks: Iterable[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Логически объединяет (OR) несколько бинарных масок одной формы."""
    combined = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, mask.astype(np.uint8))
    return combined


@contextmanager
def _timed(method_name: str) -> Iterator[None]:
    """Логирует время выполнения метода сегментации (требование "логирование времени")."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Метод сегментации сосудов %r выполнен за %.2f с", method_name, elapsed)


#: Базовая эвристическая уверенность по методу — источник для _estimate_confidence().
#: DL-модели и TotalSegmentator доверяем больше, чем классической обработке изображений.
_METHOD_BASE_CONFIDENCE: dict[str, float] = {
    "totalsegmentator": 0.9,
    "vista3d": 0.9,
    "monai_vessel_model": 0.85,
    "threshold+frangi": 0.7,
    "threshold": 0.6,
    "frangi": 0.5,
}


def _estimate_confidence(mask: np.ndarray, method_name: str) -> float:
    """Эвристическая уверенность в результате метода (0..1).

    Базовый приоритет метода (см. _METHOD_BASE_CONFIDENCE) корректируется по
    правдоподобности объёма найденных сосудов: пустая маска — явный сбой (0.0),
    маска, занимающая более 30% объёма — тоже признак сбоя метода (сосуды физиологически
    занимают малую долю объёма), а не реальной сосудистой сети — уверенность снижается.

    Args:
        mask: Итоговая бинарная маска метода.
        method_name: Имя метода (см. MethodName).

    Returns:
        Число в диапазоне [0, 1].
    """
    base_confidence = _METHOD_BASE_CONFIDENCE.get(method_name, 0.5)
    voxel_fraction = float(mask.sum()) / mask.size if mask.size > 0 else 0.0

    if voxel_fraction == 0.0:
        return 0.0
    if voxel_fraction > 0.3:
        return base_confidence * 0.3
    return base_confidence


@dataclass(slots=True)
class MethodComparisonResult:
    """Результат одного метода в рамках compare_methods()."""

    method: str
    success: bool
    volume_voxels: int
    num_components: int
    largest_component_fraction: float
    elapsed_seconds: float
    error: str | None = None


# --------------------------------------------------------------------------- #
# VesselSegmenter
# --------------------------------------------------------------------------- #


class VesselSegmenter:
    """Сегментация сосудов через иерархию методов с общей постобработкой и кэшированием."""

    def __init__(
        self,
        config: VesselSegmentationConfig | None = None,
        registry: PretrainedModelRegistry | None = None,
    ) -> None:
        """Инициализирует сегментатор.

        Args:
            config: Конфигурация методов и постобработки. По умолчанию — VesselSegmentationConfig().
            registry: Реестр предобученных моделей (см. models.pretrained_models).
        """
        self.config = config or VesselSegmentationConfig()
        self.registry = registry or PretrainedModelRegistry()
        if self.config.cache_enabled:
            self.config.result_cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Метод 1: TotalSegmentator ------------------------------------------------ #

    def segment_with_totalsegmentator(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        task: str | None = None,
        roi_subset: list[str] | None = None,
    ) -> np.ndarray:
        """Сегментирует сосуды через TotalSegmentator — самый надёжный метод при наличии GPU.

        По умолчанию используется task="headneck_bones_vessels" (сонные/позвоночные
        артерии — ближе к сосудам мозга, чем список сосудов из "total"). Явный roi_subset
        (например, ['aorta', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left',
        'iliac_vena_right'] — сосуды туловища из задачи "total") переопределяет набор,
        если нужны именно эти структуры, а не сосуды головы/мозга.

        Args:
            volume: Объём (Z, Y, X).
            spacing: Spacing (x, y, z), мм.
            task: Задача TotalSegmentator; по умолчанию self.config.totalsegmentator_task.
            roi_subset: Явный список структур (работает только при fast=False).

        Returns:
            Объединённая бинарная маска найденных сосудов (Z, Y, X), uint8.

        Raises:
            VesselSegmentationError: Если ни один сосуд не найден.
            Brain3DError: Если totalsegmentator не установлен либо упал во время сегментации.
        """
        task = task or self.config.totalsegmentator_task
        cached = self._load_from_cache(volume, spacing, method="totalsegmentator", extra_key=task)
        if cached is not None:
            return cached

        with _timed("totalsegmentator"):
            wrapper = TotalSegmentatorWrapper(
                fast=self.config.totalsegmentator_fast, roi_subset=roi_subset, task=task
            )
            wrapper.segment(volume, spacing)
            vessel_masks = wrapper.get_vessel_masks()

        if not vessel_masks:
            raise VesselSegmentationError(f"TotalSegmentator (task={task!r}) не нашёл ни одного сосуда")

        combined = _combine_masks(vessel_masks.values(), volume.shape)
        self._save_to_cache(volume, spacing, method="totalsegmentator", mask=combined, extra_key=task)
        return combined

    # --- Метод 2: NVIDIA VISTA-3D -------------------------------------------------- #

    def segment_with_vista3d(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        api_key: str | None = None,
        mode: Literal["local", "cloud"] = "local",
        classes: list[str] | None = None,
    ) -> np.ndarray:
        """Сегментирует сосуды через NVIDIA VISTA-3D (NIM, локально или в облаке).

        Args:
            volume: Объём (Z, Y, X).
            spacing: Spacing (x, y, z), мм.
            api_key: Ключ для mode="cloud" (см. VISTA3DWrapper).
            mode: "local" (Docker) или "cloud" (управляемый API NVIDIA).
            classes: Запрашиваемые классы; по умолчанию ["brain", "hepatic_vessel"] —
                "brain" нужен, чтобы отделить сосудистые классы от маски мозга в ответе;
                конкретные имена сосудистых классов зависят от версии образа NIM.

        Returns:
            Объединённая бинарная маска сосудистых классов (без "brain").

        Raises:
            VesselSegmentationError: Если эндпоинт недоступен либо не вернул сосудов.
        """
        classes = classes or ["brain", "hepatic_vessel"]
        cached = self._load_from_cache(volume, spacing, method="vista3d", extra_key=",".join(classes))
        if cached is not None:
            return cached

        wrapper = VISTA3DWrapper(mode=mode, api_key=api_key)
        if not wrapper.is_available():
            raise VesselSegmentationError(f"VISTA-3D (mode={mode!r}) недоступен")

        with _timed("vista3d"):
            masks = wrapper.segment(volume, spacing, classes=classes)

        vessel_masks = {name: mask for name, mask in masks.items() if name != "brain"}
        if not vessel_masks:
            raise VesselSegmentationError("VISTA-3D не вернул ни одной сосудистой маски")

        combined = _combine_masks(vessel_masks.values(), volume.shape)
        self._save_to_cache(volume, spacing, method="vista3d", mask=combined, extra_key=",".join(classes))
        return combined

    # --- Метод 3: MONAI Model Zoo -------------------------------------------------- #

    def segment_with_monai_vessel_model(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: str = "cpu",
    ) -> np.ndarray:
        """Сегментирует сосуды мозга через специализированный bundle из MONAI Model Zoo.

        См. предупреждение в models.pretrained_models.MONAIModelZoo.load_brain_vessel_model:
        на момент написания в зоопарке нет отдельного bundle именно под сосуды мозга —
        метод либо использует такой bundle, если пользователь зарегистрировал его с
        реальным bundle_name, либо поднимет понятную ModelDownloadError.

        Args:
            volume: Объём (Z, Y, X), предпочтительно уже нормализованный (см. core.preprocessing).
            spacing: Spacing (x, y, z), мм.
            device: "cpu" или "cuda".

        Returns:
            Бинарная маска сосудов (Z, Y, X), uint8.

        Raises:
            Brain3DError: Если torch/monai не установлены, либо bundle недоступен.
        """
        del spacing  # зарезервировано: часть bundle'ов MONAI требует ресемплинг под свой spacing
        cached_key = f"{device}"
        cached = self._load_from_cache(
            volume, (1.0, 1.0, 1.0), method="monai_vessel_model", extra_key=cached_key
        )
        if cached is not None:
            return cached

        try:
            import torch
        except ImportError as exc:
            raise Brain3DError("Пакет torch не установлен. Установите: pip install torch") from exc

        zoo = MONAIModelZoo(self.registry, device=device)
        with _timed("monai_vessel_model"):
            model = zoo.load_brain_vessel_model()
            model.eval()
            with torch.no_grad():
                input_tensor = torch.from_numpy(np.asarray(volume, dtype=np.float32))
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)
                logits = model(input_tensor)
                probabilities = (
                    torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)[:, 1:2]
                )
                mask = (probabilities > 0.5).squeeze().to("cpu").numpy().astype(np.uint8)

        self._save_to_cache(
            volume, (1.0, 1.0, 1.0), method="monai_vessel_model", mask=mask, extra_key=cached_key
        )
        return mask

    # --- Метод 4: классический фильтр Франги (FALLBACK) --------------------------- #

    def segment_frangi(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sigmas: tuple[float, ...] | None = None,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Сегментирует сосуды классическим фильтром Франги — гарантированный CPU-фолбэк.

        Args:
            volume: Объём (Z, Y, X); рекомендуется нормализовать к [0, 1] заранее.
            spacing: Spacing (x, y, z), мм (см. models.pretrained_models.segment_vessels_frangi).
            sigmas: Масштабы фильтра; по умолчанию self.config.frangi_sigmas.
            threshold: Порог бинаризации; None — автоматический (Отсу).

        Returns:
            Бинарная маска сосудов (Z, Y, X), uint8.
        """
        sigmas = sigmas or self.config.frangi_sigmas
        with _timed("frangi"):
            mask = segment_vessels_frangi(volume, spacing=spacing, sigmas=sigmas, threshold=threshold)
        return mask

    # --- Метод 5: пороговая обработка (контрастная КТА) ---------------------------- #

    def segment_by_threshold(
        self,
        volume: np.ndarray,
        min_hu: float = 100.0,
        max_hu: float = 500.0,
    ) -> np.ndarray:
        """Пороговая сегментация контрастированных сосудов на КТ с йодным контрастом.

        Йодный контраст даёт заметно повышенную плотность (обычно 100-500+ HU в
        зависимости от фазы/концентрации) по сравнению с окружающими мягкими тканями
        (~20-80 HU), поэтому простой HU-порог с морфологической постобработкой часто
        достаточен для КТ-ангиографии — и заметно быстрее DL-методов.

        Args:
            volume: КТ-объём в единицах HU.
            min_hu: Нижняя граница диапазона контрастированных сосудов.
            max_hu: Верхняя граница диапазона.

        Returns:
            Бинарная маска после морфологической постобработки (Z, Y, X), uint8.
        """
        with _timed("threshold"):
            mask = ((volume >= min_hu) & (volume <= max_hu)).astype(np.uint8)
            mask = remove_small_components(mask, min_size=self.config.min_component_size)
            mask = fill_holes(mask)
            mask = connect_vessels(mask, iterations=self.config.morphology_iterations)
        return mask

    def _threshold_then_frangi(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
        """Комбинация для контрастной КТА: HU-порог для крупных сосудов + Франги для мелких."""
        threshold_mask = self.segment_by_threshold(
            volume, self.config.threshold_min_hu, self.config.threshold_max_hu
        )
        frangi_mask = self.segment_frangi(volume, spacing)
        combined = np.maximum(threshold_mask, frangi_mask)
        return remove_small_components(combined, min_size=self.config.min_component_size)

    # --- Адаптивный выбор метода ---------------------------------------------------- #

    def segment_adaptive(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        modality: str = "MRI",
        resources: ComputeResources | None = None,
        has_contrast: bool = False,
    ) -> tuple[np.ndarray, str, float]:
        """Пробует методы в порядке приоритета и останавливается на первом успешном.

        Порядок кандидатов (только включённые в конфиге пропускаются в список):
            1. modality="CT" и has_contrast=True -> "threshold+frangi".
            2. modality="MRA" -> "monai_vessel_model".
            3. Есть GPU -> "totalsegmentator", затем "vista3d".
            4. Всегда в конце -> "frangi" (гарантированный CPU-фолбэк).

        В отличие от жёсткого if/elif, здесь именно ИЕРАРХИЯ: если приоритетный метод
        для условий недоступен или падает во время выполнения (не установлен пакет,
        нет сети, GPU без места и т.п.), автоматически пробуется следующий по списку,
        вплоть до гарантированного Франги.

        Args:
            volume: Объём (Z, Y, X).
            spacing: Spacing (x, y, z), мм.
            modality: "CT" | "MRI" | "MRA" (свободная строка, не привязана к
                shared.enums.Modality — МРА это тип МР-последовательности, а не
                отдельная DICOM-модальность).
            resources: Доступные ресурсы; по умолчанию определяются автоматически
                (см. models.pretrained_models.detect_compute_resources).
            has_contrast: Признак контрастного усиления для КТ (например, по DICOM-тегу
                ContrastBolusAgent) — этот метод не пытается определить его сам.

        Returns:
            Кортеж (mask, method_used, confidence).

        Raises:
            VesselSegmentationError: Если ни один включённый в конфиге метод не сработал.
        """
        resources = resources or detect_compute_resources()
        modality_normalized = modality.strip().upper()

        attempts: list[tuple[str, Callable[[], np.ndarray]]] = []

        if modality_normalized == "CT" and has_contrast:
            attempts.append(("threshold+frangi", lambda: self._threshold_then_frangi(volume, spacing)))
        if modality_normalized == "MRA":
            attempts.append(
                ("monai_vessel_model", lambda: self.segment_with_monai_vessel_model(volume, spacing))
            )
        if resources.has_gpu:
            attempts.append(("totalsegmentator", lambda: self.segment_with_totalsegmentator(volume, spacing)))
            attempts.append(("vista3d", lambda: self.segment_with_vista3d(volume, spacing)))
        attempts.append(("frangi", lambda: self.segment_frangi(volume, spacing)))

        enabled_attempts = [(name, fn) for name, fn in attempts if self._is_method_enabled(name)]

        last_error: Exception | None = None
        for method_name, method_call in enabled_attempts:
            try:
                mask = method_call()
            except Brain3DError as exc:
                logger.warning(
                    "segment_adaptive: метод %r не сработал (%s), пробуем следующий", method_name, exc
                )
                last_error = exc
                continue

            confidence = _estimate_confidence(mask, method_name)
            logger.info("segment_adaptive: метод %r успешен (confidence=%.2f)", method_name, confidence)
            return mask, method_name, confidence

        raise VesselSegmentationError(
            f"Ни один включённый метод сегментации сосудов не сработал (последняя ошибка: {last_error})"
        )

    def _is_method_enabled(self, method_name: str) -> bool:
        """Проверяет, разрешён ли метод конфигурацией (self.config.enable_*)."""
        mapping = {
            "totalsegmentator": self.config.enable_totalsegmentator,
            "vista3d": self.config.enable_vista3d,
            "monai_vessel_model": self.config.enable_monai_vessel_model,
            "frangi": self.config.enable_frangi,
            "threshold": self.config.enable_threshold,
            "threshold+frangi": self.config.enable_threshold and self.config.enable_frangi,
        }
        return mapping.get(method_name, True)

    # --- Специализированная сегментация Виллизиева круга --------------------------- #

    def segment_circle_of_willis(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
        """Сегментирует Виллизиев круг комбинацией TotalSegmentator + Франги.

        Крупные артерии (внутренние сонные, базилярная и начальные сегменты мозговых
        артерий) TotalSegmentator способен выделить целиком или частично через задачу
        "headneck_bones_vessels". Мелкие перфорантные и соединительные артерии, которые
        TotalSegmentator не видит, дополняются откликом фильтра Франги — но только в
        расширенной окрестности уже найденных крупных сосудов, что заметно снижает число
        ложных срабатываний Франги на шуме вдали от круга.

        Args:
            volume: Объём (Z, Y, X), КТ или МРТ.
            spacing: Spacing (x, y, z), мм.

        Returns:
            Бинарная маска Виллизиева круга после постобработки (удаление мелких
            компонент, заполнение дыр, морфологическое закрытие для связности).

        Raises:
            VesselSegmentationError: Если не удалось выделить ни одного сосуда.
        """
        large_vessels_mask = np.zeros(volume.shape, dtype=np.uint8)
        try:
            large_vessels_mask = self.segment_with_totalsegmentator(
                volume, spacing, task="headneck_bones_vessels"
            )
        except Brain3DError as exc:
            logger.warning(
                "segment_circle_of_willis: TotalSegmentator недоступен (%s), используется только Frangi", exc
            )

        if large_vessels_mask.any():
            roi_mask = ndimage.binary_dilation(
                large_vessels_mask.astype(bool), iterations=self.config.circle_of_willis_roi_dilation
            )
        else:
            roi_mask = np.ones(volume.shape, dtype=bool)

        frangi_mask = self.segment_frangi(volume, spacing)
        frangi_mask_in_roi = frangi_mask * roi_mask.astype(np.uint8)

        combined = np.maximum(large_vessels_mask, frangi_mask_in_roi)
        if not combined.any():
            raise VesselSegmentationError("Не удалось выделить ни одного сосуда для Виллизиева круга")

        combined = remove_small_components(combined, min_size=self.config.min_component_size)
        combined = fill_holes(combined)
        combined = connect_vessels(combined, iterations=self.config.morphology_iterations)
        return combined

    # --- Vesselness map (для визуализации/отладки) --------------------------------- #

    def enhance_vesselness(self, volume: np.ndarray, sigmas: tuple[float, ...] | None = None) -> np.ndarray:
        """Возвращает непрерывную vesselness-карту (не бинарную маску).

        Args:
            volume: Объём (Z, Y, X).
            sigmas: Масштабы фильтра Франги; по умолчанию self.config.frangi_sigmas.

        Returns:
            Карта "трубчатости" (Z, Y, X), float32 — чем выше значение, тем более
            вероятно, что воксель принадлежит трубчатой (сосудистой) структуре.
        """
        sigmas = sigmas or self.config.frangi_sigmas
        with _timed("vesselness_enhancement"):
            vesselness = frangi(volume, sigmas=sigmas, black_ridges=False)
        return vesselness.astype(np.float32)

    # --- Сравнение методов ----------------------------------------------------------- #

    def compare_methods(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> list[MethodComparisonResult]:
        """Запускает все включённые в конфиге методы и сравнивает результаты.

        Полезно при разведочном анализе нового источника данных — какой метод вообще
        применим и даёт правдоподобный (не пустой, не занимающий половину объёма)
        результат, прежде чем закладывать его по умолчанию в segment_adaptive.

        Args:
            volume: Объём (Z, Y, X).
            spacing: Spacing (x, y, z), мм.

        Returns:
            Список MethodComparisonResult — по одному на каждый включённый метод,
            включая неудачные (success=False, error заполнен).
        """
        candidates: dict[str, Callable[[], np.ndarray]] = {
            "totalsegmentator": lambda: self.segment_with_totalsegmentator(volume, spacing),
            "vista3d": lambda: self.segment_with_vista3d(volume, spacing),
            "monai_vessel_model": lambda: self.segment_with_monai_vessel_model(volume, spacing),
            "frangi": lambda: self.segment_frangi(volume, spacing),
            "threshold": lambda: self.segment_by_threshold(
                volume, self.config.threshold_min_hu, self.config.threshold_max_hu
            ),
        }

        results: list[MethodComparisonResult] = []
        for method_name, method_call in candidates.items():
            if not self._is_method_enabled(method_name):
                continue

            start = time.perf_counter()
            try:
                mask = method_call()
            except Brain3DError as exc:
                elapsed = time.perf_counter() - start
                logger.info("compare_methods: %r не сработал за %.2fс: %s", method_name, elapsed, exc)
                results.append(
                    MethodComparisonResult(
                        method=method_name,
                        success=False,
                        volume_voxels=0,
                        num_components=0,
                        largest_component_fraction=0.0,
                        elapsed_seconds=elapsed,
                        error=str(exc),
                    )
                )
                continue

            elapsed = time.perf_counter() - start
            results.append(_build_comparison_result(method_name, mask, elapsed))

        return results

    # --- Кэширование результатов ----------------------------------------------------- #

    def _cache_key(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        method: str,
        extra_key: str,
    ) -> str:
        """Строит ключ кэша по содержимому объёма — исключает подмену результата при
        совпадении формы, но разном содержимом (что не покрыл бы ключ по одной форме)."""
        hasher = hashlib.sha256()
        hasher.update(np.ascontiguousarray(volume).tobytes())
        hasher.update(repr(spacing).encode("utf-8"))
        hasher.update(method.encode("utf-8"))
        hasher.update(extra_key.encode("utf-8"))
        return hasher.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.config.result_cache_dir / f"{key}.npz"

    def _load_from_cache(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        method: str,
        extra_key: str = "",
    ) -> np.ndarray | None:
        """Возвращает закэшированный результат метода, если кэш включён и он существует."""
        if not self.config.cache_enabled:
            return None
        path = self._cache_path(self._cache_key(volume, spacing, method, extra_key))
        if not path.exists():
            return None
        logger.info("Результат метода %r взят из кэша: %s", method, path)
        with np.load(path) as data:
            return data["mask"]

    def _save_to_cache(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        method: str,
        mask: np.ndarray,
        extra_key: str = "",
    ) -> None:
        """Сохраняет результат метода в кэш, если кэширование включено."""
        if not self.config.cache_enabled:
            return
        path = self._cache_path(self._cache_key(volume, spacing, method, extra_key))
        np.savez_compressed(path, mask=mask)
        logger.debug("Результат метода %r сохранён в кэш: %s", method, path)


def _build_comparison_result(method_name: str, mask: np.ndarray, elapsed: float) -> MethodComparisonResult:
    """Считает метрики объёма/связности одного успешного метода для compare_methods()."""
    labeled, num_components = ndimage.label(mask)
    volume_voxels = int(mask.sum())
    largest_fraction = 0.0
    if num_components > 0 and volume_voxels > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
        largest_fraction = float(np.max(sizes)) / volume_voxels

    return MethodComparisonResult(
        method=method_name,
        success=True,
        volume_voxels=volume_voxels,
        num_components=int(num_components),
        largest_component_fraction=largest_fraction,
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    # --- Пример использования: МРТ, только CPU -> Frangi ---
    import logging

    logging.basicConfig(level=logging.INFO)

    mri_volume = np.random.rand(32, 128, 128).astype(np.float32)
    segmenter = VesselSegmenter(VesselSegmentationConfig(enable_totalsegmentator=False, enable_vista3d=False))

    mask, method_used, confidence = segmenter.segment_adaptive(
        mri_volume,
        spacing=(1.0, 1.0, 1.0),
        modality="MRI",
        resources=detect_compute_resources(check_internet=False),
    )
    print(f"МРТ: метод={method_used}, confidence={confidence:.2f}, воксели сосудов={int(mask.sum())}")

    # --- Пример использования: контрастная КТ-ангиография ---
    cta_volume = np.random.uniform(-100, 50, size=(32, 128, 128)).astype(np.float32)
    cta_volume[10:20, 40:60, 40:60] = 300.0  # имитация контрастированного сосуда
    mask_ct, method_ct, confidence_ct = segmenter.segment_adaptive(
        cta_volume, spacing=(1.0, 1.0, 1.0), modality="CT", has_contrast=True
    )
    print(f"КТА: метод={method_ct}, confidence={confidence_ct:.2f}, воксели сосудов={int(mask_ct.sum())}")

    # --- Пример использования: сравнение методов на одном объёме ---
    comparison = segmenter.compare_methods(mri_volume, spacing=(1.0, 1.0, 1.0))
    for result in comparison:
        status = "OK" if result.success else f"FAILED ({result.error})"
        print(f"{result.method}: {status}, {result.elapsed_seconds:.2f}с")

    # --- Пример использования: vesselness-карта для визуализации ---
    vesselness_map = segmenter.enhance_vesselness(mri_volume)
    print("Vesselness map:", vesselness_map.shape, vesselness_map.dtype)
