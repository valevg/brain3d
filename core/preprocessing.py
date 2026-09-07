"""Препроцессинг объёмов КТ и МРТ перед подачей в 3D U-Net.

Заменяет собой прежний пакет core/preprocessing/{base,ct_preprocessing,mri_preprocessing}.py
единым модулем с более широким набором операций (HU-окна, ресемплинг, шумоподавление,
skull-stripping, коррекция неоднородности поля, объединение МР-модальностей в
многоканальный вход) и единой точкой входа UnifiedPreprocessor.

Все функции работают с "сырыми" объёмами np.ndarray формы (Z, Y, X) и spacing в
порядке (x, y, z) — том же соглашении, что и shared.types.VoxelSpacing.as_tuple() —
а не с VolumeData, чтобы препроцессинг можно было использовать и вне пайплайнов
(например, в ноутбуке при разведочном анализе данных).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from pydantic import BaseModel, Field
from scipy import ndimage
from skimage.filters import threshold_otsu

from core.exceptions import PreprocessingError, UnsupportedModalityError
from core.logging_config import get_logger
from shared.enums import Modality, MRWeighting

logger = get_logger(__name__)


class PreprocessingConfig(BaseModel):
    """Конфигурация препроцессинга с возможностью отключать отдельные шаги.

    Единая модель для CTPreprocessor и MRIPreprocessor — поля, не относящиеся к
    выбранной модальности (например, window_center для МРТ), просто игнорируются.
    """

    # --- Общие параметры ---
    target_spacing_mm: tuple[float, float, float] = Field(
        default=(1.0, 1.0, 1.0), description="Целевой spacing после ресемплинга (x, y, z), мм"
    )
    apply_resampling: bool = Field(default=True, description="Ресемплинг к target_spacing_mm")
    apply_normalization: bool = Field(
        default=True,
        description="Финальная нормализация: HU-окно (если оно выключено — min-max) для КТ, z-score для МРТ",
    )

    # --- КТ ---
    apply_hu_windowing: bool = Field(default=True, description="HU-окно как финальная нормализация КТ")
    window_center: float = 40.0
    window_width: float = 400.0
    apply_denoising: bool = Field(
        default=False, description="Медианный фильтр (по умолчанию выкл. — может размыть мелкие структуры)"
    )
    denoise_kernel_size: int = 3
    apply_skull_stripping_ct: bool = Field(
        default=False, description="Пороговый skull-stripping по HU (20..100)"
    )
    ct_skull_strip_hu_range: tuple[float, float] = (20.0, 100.0)

    # --- МРТ ---
    apply_bias_correction: bool = True
    apply_skull_stripping_mri: bool = True
    mri_weighting: str = Field(default="T1ce", description="T1 | T1ce | T2 | FLAIR — влияет на набор шагов")
    align_to_template_path: str | None = Field(
        default=None, description="Путь к атласу; None — регистрация пропускается"
    )


def _resample_volume(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    new_spacing: tuple[float, float, float],
) -> np.ndarray:
    """Ресемплирует объём (Z, Y, X) к новому spacing линейной интерполяцией (используется CT и MRI).

    spacing/new_spacing заданы в порядке (x, y, z) — как и всюду в проекте (см.
    shared.types.VoxelSpacing.as_tuple()) — а сам массив имеет порядок осей (Z, Y, X),
    поэтому множители масштабирования применяются в ОБРАТНОМ порядке индексов; перепутать
    их — самая частая ошибка при ручном ресемплинге медицинских объёмов.

    Args:
        volume: Исходный объём (Z, Y, X).
        spacing: Текущий spacing (x, y, z), мм.
        new_spacing: Целевой spacing (x, y, z), мм.

    Returns:
        Ресемплированный объём float32.
    """
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    new_spacing_zyx = (new_spacing[2], new_spacing[1], new_spacing[0])
    zoom_factors = tuple(s / ns for s, ns in zip(spacing_zyx, new_spacing_zyx))
    return ndimage.zoom(volume, zoom_factors, order=1).astype(np.float32)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Оставляет в бинарной маске только наибольшую связную компоненту (удаляет шум/артефакты стола)."""
    labeled, num_components = ndimage.label(mask)
    if num_components <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
    largest_component = int(np.argmax(sizes)) + 1
    return labeled == largest_component


class CTPreprocessor:
    """Препроцессор КТ-объёмов: HU-окно, ресемплинг, шумоподавление, пороговый skull-stripping."""

    #: Стандартные клинические HU-окна (center, width).
    WINDOW_PRESETS: dict[str, tuple[float, float]] = {
        "brain": (40.0, 80.0),
        "soft_tissue": (40.0, 400.0),
        "bone": (300.0, 1500.0),
    }

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Инициализирует препроцессор.

        Args:
            config: Конфигурация шагов препроцессинга. По умолчанию — PreprocessingConfig().
        """
        self.config = config or PreprocessingConfig()

    def apply_hounsfield_window(
        self,
        volume: np.ndarray,
        window_center: float = 40.0,
        window_width: float = 400.0,
    ) -> np.ndarray:
        """Клиппирует объём по HU-окну [center - width/2, center + width/2] и нормализует к [0, 1].

        Args:
            volume: Объём в единицах Хаунсфилда (HU).
            window_center: Центр окна (WC).
            window_width: Ширина окна (WW).

        Returns:
            Массив float32 в диапазоне [0, 1].
        """
        hu_min = window_center - window_width / 2.0
        hu_max = window_center + window_width / 2.0
        clipped = np.clip(volume, hu_min, hu_max)
        return ((clipped - hu_min) / (hu_max - hu_min)).astype(np.float32)

    def apply_window_preset(self, volume: np.ndarray, preset: str = "brain") -> np.ndarray:
        """Применяет один из стандартных клинических пресетов окна (brain/soft_tissue/bone).

        Args:
            volume: Объём в HU.
            preset: "brain" (40/80), "soft_tissue" (40/400) или "bone" (300/1500).

        Returns:
            Массив float32 в диапазоне [0, 1].

        Raises:
            PreprocessingError: Если имя пресета не распознано.
        """
        if preset not in self.WINDOW_PRESETS:
            raise PreprocessingError(
                f"Неизвестный пресет окна: {preset!r}. Доступны: {list(self.WINDOW_PRESETS)}"
            )
        center, width = self.WINDOW_PRESETS[preset]
        return self.apply_hounsfield_window(volume, center, width)

    def resample_to_isotropic(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        new_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Ресемплирует объём к новому (по умолчанию изотропному 1×1×1 мм) spacing.

        Args:
            volume: Исходный объём (Z, Y, X) в HU.
            spacing: Текущий spacing (x, y, z), мм.
            new_spacing: Целевой spacing (x, y, z), мм.

        Returns:
            Ресемплированный объём float32.
        """
        return _resample_volume(volume, spacing, new_spacing)

    def denoise_median(self, volume: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Подавляет шум медианным фильтром.

        Args:
            volume: Исходный объём.
            kernel_size: Размер окна фильтра (нечётное число, одинаковое по всем осям).

        Returns:
            Отфильтрованный объём float32.
        """
        return ndimage.median_filter(volume, size=kernel_size).astype(np.float32)

    def skull_stripping(
        self,
        volume: np.ndarray,
        hu_min: float = 20.0,
        hu_max: float = 100.0,
    ) -> np.ndarray:
        """Упрощённая пороговая сегментация мозга по HU-диапазону мягких тканей (20 < HU < 100).

        Это НЕ полноценный skull-stripping уровня FSL BET/HD-BET — просто отсекает
        кость (HU выше диапазона) и воздух/CSF/жир (HU ниже диапазона) порогом,
        оставляя наибольшую связную компоненту как приближение паренхимы мозга.
        Достаточно для типового препроцессинга перед сетью, но не для клинически
        точного построения маски черепа.

        Args:
            volume: Объём в HU.
            hu_min: Нижняя граница диапазона мягких тканей мозга.
            hu_max: Верхняя граница диапазона.

        Returns:
            Объём той же формы, где воксели вне маски заменены минимальным значением объёма.
        """
        mask = (volume > hu_min) & (volume < hu_max)
        mask = ndimage.binary_fill_holes(mask)
        mask = _keep_largest_component(mask)

        background_value = float(volume.min())
        return np.where(mask, volume, background_value).astype(np.float32)

    def normalize_to_0_1(self, volume: np.ndarray) -> np.ndarray:
        """Линейно нормализует объём к диапазону [0, 1] по его собственным min/max.

        В отличие от apply_hounsfield_window, не привязана к единицам HU — используется
        как запасной способ нормализации, если HU-окно отключено в конфигурации.

        Args:
            volume: Исходный объём.

        Returns:
            Массив float32 в диапазоне [0, 1] (нулевой массив, если объём константный).
        """
        vmin, vmax = float(volume.min()), float(volume.max())
        if vmax <= vmin:
            logger.warning("normalize_to_0_1: объём константный (min == max) — возвращены нули")
            return np.zeros_like(volume, dtype=np.float32)
        return ((volume - vmin) / (vmax - vmin)).astype(np.float32)

    def preprocess_full(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
        """Полный пайплайн препроцессинга КТ, управляемый self.config.

        Порядок шагов: ресемплинг -> шумоподавление -> skull-stripping -> нормализация
        (HU-окно, либо min-max, если окно отключено). Каждый шаг включается/выключается
        независимо через PreprocessingConfig.

        Args:
            volume: Сырой объём в HU, форма (Z, Y, X).
            spacing: Текущий spacing объёма (x, y, z), мм.

        Returns:
            Обработанный объём float32, готовый к подаче в модель сегментации.
        """
        result = volume

        if self.config.apply_resampling:
            result = self.resample_to_isotropic(result, spacing, self.config.target_spacing_mm)

        if self.config.apply_denoising:
            result = self.denoise_median(result, self.config.denoise_kernel_size)

        if self.config.apply_skull_stripping_ct:
            hu_min, hu_max = self.config.ct_skull_strip_hu_range
            result = self.skull_stripping(result, hu_min, hu_max)

        if not self.config.apply_normalization:
            return result

        if self.config.apply_hu_windowing:
            result = self.apply_hounsfield_window(result, self.config.window_center, self.config.window_width)
        else:
            result = self.normalize_to_0_1(result)

        return result


#: Каналы многоканального МР-входа в фиксированном порядке (важно для обученной модели).
_MULTIMODAL_ORDER: tuple[str, ...] = (
    MRWeighting.T1.value,
    MRWeighting.T1CE.value,
    MRWeighting.T2.value,
    MRWeighting.FLAIR.value,
)

#: Настройки препроцессинга по типу МР-взвешивания.
#: bias_correction для FLAIR по умолчанию выключен: N4 на длинных TI/TR последовательностях
#: (как FLAIR) менее стабилен и чаще усиливает шум, чем убирает реальную неоднородность поля.
_MRI_WEIGHTING_PRESETS: dict[str, dict[str, bool]] = {
    "T1": {"bias_correction": True, "skull_strip": True},
    "T1CE": {"bias_correction": True, "skull_strip": True},
    "T2": {"bias_correction": True, "skull_strip": True},
    "FLAIR": {"bias_correction": False, "skull_strip": True},
}


class MRIPreprocessor:
    """Препроцессор МР-объёмов: коррекция поля, skull-stripping, z-score, регистрация к атласу."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Инициализирует препроцессор.

        Args:
            config: Конфигурация шагов препроцессинга. По умолчанию — PreprocessingConfig().
        """
        self.config = config or PreprocessingConfig()

    def bias_field_correction(self, volume: np.ndarray, shrink_factor: int = 4) -> np.ndarray:
        """Корректирует низкочастотную неоднородность поля (bias field) алгоритмом N4ITK.

        Для скорости N4 выполняется на уменьшенной (в shrink_factor раз) копии объёма —
        стандартная практика N4ITK, т.к. алгоритм оценивает низкочастотное поле, которое
        не требует полного разрешения. Восстановленное логарифмическое поле затем
        применяется к объёму в исходном разрешении через GetLogBiasFieldAsImage.

        Args:
            volume: Исходный МР-объём (Z, Y, X).
            shrink_factor: Коэффициент даунсемплинга перед оценкой поля.

        Returns:
            Скорректированный объём float32 в исходном разрешении.
        """
        image = sitk.GetImageFromArray(volume.astype(np.float32))
        mask_image = sitk.OtsuThreshold(image, 0, 1, 200)

        # Ограничиваем коэффициент уменьшения так, чтобы РАЗМЕР ПОСЛЕ Shrink оставался
        # не меньше 2 по каждой оси: внутренняя B-spline сетка N4 не строится на оси
        # размером 1 и падает с ошибкой "Zero-valued spacing is not supported". Это
        # реальное ограничение самого N4ITK, а не только тестовых данных — встречается
        # на "тонких" объёмах (мало срезов после ресемплинга, 2D-подобные серии).
        per_axis_shrink = [max(1, min(shrink_factor, size // 2)) for size in image.GetSize()]
        shrunk_image = sitk.Shrink(image, per_axis_shrink)
        shrunk_mask = sitk.Shrink(mask_image, per_axis_shrink)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.Execute(shrunk_image, shrunk_mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(image)
        corrected_image = image / sitk.Exp(log_bias_field)

        return sitk.GetArrayFromImage(corrected_image).astype(np.float32)

    def skull_stripping_mri(self, volume: np.ndarray) -> np.ndarray:
        """Упрощённый BET-подобный skull-stripping: порог Отсу + наибольшая компонента.

        Не претендует на точность специализированных инструментов (FSL BET, HD-BET) —
        это быстрый пороговый метод, отделяющий ткань мозга от фона/CSF по гистограмме
        интенсивностей, с последующим удалением мелких несвязных фрагментов.

        Порог Отсу считается по ВСЕМУ объёму (включая нулевой фон), а не только по
        положительным вокселям: Отсу ищет разделение именно между двумя модами
        гистограммы (фон/ткань), и если вычислять его лишь по уже отфильтрованным
        тканевым значениям, метод теряет доступ к моде фона и может вернуть порог,
        не отделяющий вообще ничего (например, если ткань в объёме почти константна).

        Args:
            volume: МР-объём после (опциональной) коррекции поля.

        Returns:
            Объём той же формы, фон вне маски занулён.
        """
        if volume.max() <= volume.min():
            logger.warning("skull_stripping_mri: объём не содержит вариации интенсивности — пропущено")
            return volume.astype(np.float32)

        threshold = float(threshold_otsu(volume))
        mask = volume > threshold
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_erosion(mask, iterations=2)
        mask = ndimage.binary_dilation(mask, iterations=2)
        mask = _keep_largest_component(mask)

        return np.where(mask, volume, 0.0).astype(np.float32)

    def normalize_z_score(self, volume: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Z-score нормализация по ненулевым (тканевым) вокселям — стандарт для МРТ.

        Args:
            volume: Исходный объём.
            epsilon: Защита от деления на ноль при нулевой дисперсии.

        Returns:
            Массив float32 (среднее ~0, std ~1 по тканевым вокселям).
        """
        mask = volume > 0
        if not np.any(mask):
            logger.warning("normalize_z_score: объём не содержит положительных вокселей")
            return volume.astype(np.float32)

        mean = volume[mask].mean()
        std = volume[mask].std()
        return ((volume - mean) / (std + epsilon)).astype(np.float32)

    def align_to_template(
        self,
        volume: np.ndarray,
        template: np.ndarray | str | Path | None = None,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Регистрирует объём к атласу аффинным преобразованием (SimpleITK).

        Опциональный шаг: без template регистрация невозможна (нет эталона), поэтому
        объём возвращается без изменений с предупреждением в лог. Атлас (например,
        MNI152) должен быть предоставлен вызывающим кодом — модуль не поставляется
        с зашитым атласом.

        Args:
            volume: Объём для регистрации (Z, Y, X).
            template: Путь к файлу атласа (любой формат, читаемый SimpleITK) либо
                готовый np.ndarray. None — регистрация пропускается.
            spacing: Spacing объёма (x, y, z), используется для построения sitk.Image.

        Returns:
            Объём, зарегистрированный к сетке template, либо исходный volume при
            отсутствии template.
        """
        if template is None:
            logger.warning("align_to_template вызван без template — регистрация пропущена")
            return volume.astype(np.float32)

        moving_image = sitk.GetImageFromArray(volume.astype(np.float32))
        moving_image.SetSpacing(spacing)

        if isinstance(template, (str, Path)):
            fixed_image = sitk.ReadImage(str(template))
        else:
            fixed_image = sitk.GetImageFromArray(np.asarray(template, dtype=np.float32))
            fixed_image.SetSpacing(spacing)

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=100
        )
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = registration.Execute(
            sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
        )
        resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0)

        return sitk.GetArrayFromImage(resampled).astype(np.float32)

    def preprocess_full(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        modality: str = "T1ce",
    ) -> np.ndarray:
        """Полный пайплайн препроцессинга МРТ, управляемый self.config и типом взвешивания.

        Порядок шагов: ресемплинг -> коррекция поля (если предусмотрена пресетом
        взвешивания) -> skull-stripping -> z-score нормализация -> (опционально)
        регистрация к атласу.

        Args:
            volume: Сырой МР-объём, форма (Z, Y, X).
            spacing: Текущий spacing объёма (x, y, z), мм.
            modality: Тип взвешивания — "T1", "T1ce", "T2" или "FLAIR" (регистр не важен).
                Влияет на то, применяется ли коррекция поля (см. _MRI_WEIGHTING_PRESETS).

        Returns:
            Обработанный объём float32, готовый к подаче в модель сегментации.

        Raises:
            PreprocessingError: Если modality не входит в число известных взвешиваний.
        """
        preset_key = modality.strip().upper()
        preset = _MRI_WEIGHTING_PRESETS.get(preset_key)
        if preset is None:
            raise PreprocessingError(
                f"Неизвестное МР-взвешивание {modality!r}. Доступны: {list(_MRI_WEIGHTING_PRESETS)}"
            )

        result = volume

        if self.config.apply_resampling:
            result = _resample_volume(result, spacing, self.config.target_spacing_mm)

        if self.config.apply_bias_correction and preset["bias_correction"]:
            result = self.bias_field_correction(result)

        if self.config.apply_skull_stripping_mri and preset["skull_strip"]:
            result = self.skull_stripping_mri(result)

        if self.config.apply_normalization:
            result = self.normalize_z_score(result)

        if self.config.align_to_template_path:
            result = self.align_to_template(
                result, self.config.align_to_template_path, self.config.target_spacing_mm
            )

        return result

    def combine_multimodal(self, volumes: dict[str, np.ndarray]) -> np.ndarray:
        """Объединяет несколько МР-последовательностей в многоканальный вход для сети.

        Порядок каналов ФИКСИРОВАН (T1, T1ce, T2, FLAIR) независимо от порядка ключей
        во входном словаре — обученная модель ожидает каналы в строго определённом
        порядке. Все переданные объёмы должны быть уже приведены к одной форме
        (общий ресемплинг/регистрация выполняются заранее, до вызова этого метода).

        Args:
            volumes: Словарь {взвешивание: объём}. Ключи — значения shared.enums.MRWeighting
                ("T1", "T1ce", "T2", "FLAIR"); не обязательно передавать все четыре.

        Returns:
            Массив (C, Z, Y, X), где C — число переданных модальностей.

        Raises:
            PreprocessingError: Если volumes пуст, содержит неизвестный ключ взвешивания,
                либо объёмы имеют разную форму.
        """
        if not volumes:
            raise PreprocessingError("combine_multimodal: передан пустой словарь volumes")

        unknown_keys = set(volumes) - set(_MULTIMODAL_ORDER)
        if unknown_keys:
            raise PreprocessingError(
                f"Неизвестные ключи взвешивания: {sorted(unknown_keys)}. Ожидаются: {_MULTIMODAL_ORDER}"
            )

        shapes = {v.shape for v in volumes.values()}
        if len(shapes) > 1:
            raise PreprocessingError(f"Объёмы разных модальностей имеют разную форму: {shapes}")

        ordered_arrays = [volumes[key] for key in _MULTIMODAL_ORDER if key in volumes]
        return np.stack(ordered_arrays, axis=0).astype(np.float32)


class UnifiedPreprocessor:
    """Фабрика, скрывающая выбор CTPreprocessor/MRIPreprocessor за единым интерфейсом.

    Пример:
        >>> preprocessor = UnifiedPreprocessor(modality="CT")
        >>> processed = preprocessor.preprocess(ct_volume, spacing=(0.5, 0.5, 1.0))

        >>> preprocessor = UnifiedPreprocessor(modality="MRI")
        >>> processed = preprocessor.preprocess(mri_volume, spacing=(1.0, 1.0, 1.0), modality="T1ce")
    """

    def __init__(self, modality: str, config: PreprocessingConfig | None = None) -> None:
        """Инициализирует фабрику и создаёт нужный препроцессор.

        Args:
            modality: "CT" или "MR"/"MRI" (регистр не важен).
            config: Конфигурация препроцессинга, общая для CT и MRI веток.

        Raises:
            UnsupportedModalityError: Если modality не CT и не MR/MRI.
        """
        self.config = config or PreprocessingConfig()
        self._modality = self._normalize_modality(modality)
        self._impl: CTPreprocessor | MRIPreprocessor = (
            CTPreprocessor(self.config) if self._modality == Modality.CT else MRIPreprocessor(self.config)
        )

    @property
    def modality(self) -> Modality:
        """Модальность, выбранная при создании фабрики."""
        return self._modality

    @staticmethod
    def _normalize_modality(modality: str) -> Modality:
        normalized = modality.strip().upper()
        if normalized == "CT":
            return Modality.CT
        if normalized in ("MR", "MRI"):
            return Modality.MRI
        raise UnsupportedModalityError(f"Неизвестная модальность: {modality!r}")

    def preprocess(
        self, volume: np.ndarray, spacing: tuple[float, float, float], **kwargs: Any
    ) -> np.ndarray:
        """Запускает полный пайплайн препроцессинга, соответствующий выбранной модальности.

        Args:
            volume: Сырой объём (Z, Y, X).
            spacing: Текущий spacing (x, y, z), мм.
            **kwargs: Для МРТ можно передать modality="T1"/"T1ce"/"T2"/"FLAIR" (МР-взвешивание,
                не путать с CT/MRI модальностью верхнего уровня); по умолчанию берётся
                self.config.mri_weighting. Для КТ дополнительные аргументы не нужны.

        Returns:
            Обработанный объём float32.
        """
        if isinstance(self._impl, MRIPreprocessor):
            mri_weighting = kwargs.get("modality", self.config.mri_weighting)
            return self._impl.preprocess_full(volume, spacing, modality=mri_weighting)
        return self._impl.preprocess_full(volume, spacing)

    @classmethod
    def auto_detect_and_preprocess(
        cls,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        modality: str | None = None,
        config: PreprocessingConfig | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Определяет модальность (если не передана явно) и запускает соответствующий пайплайн.

        Автоопределение — грубая эвристика на случай, когда объём получен без DICOM-
        заголовка (например, "голый" NIfTI/ndarray): КТ практически всегда содержит
        выраженный воздух (HU ≈ -1000) по краям снимка, тогда как интенсивности МРТ
        неотрицательны. Если модальность уже известна (из DICOM-тега, см.
        shared.enums.Modality.from_dicom_tag), её следует передавать явно через
        modality — эвристика для этого случая не нужна и менее надёжна.

        Args:
            volume: Объём (Z, Y, X).
            spacing: Текущий spacing объёма (x, y, z), мм.
            modality: "CT"/"MR"/"MRI", либо None для автоопределения по интенсивностям.
            config: Конфигурация препроцессинга.
            **kwargs: Прокидывается в preprocess() (например, modality МР-взвешивания).

        Returns:
            Обработанный объём float32.
        """
        detected_modality = modality or cls._guess_modality(volume)
        if modality is None:
            logger.info("Модальность не указана явно, автоопределение: %s", detected_modality)
        return cls(detected_modality, config).preprocess(volume, spacing, **kwargs)

    @staticmethod
    def _guess_modality(volume: np.ndarray) -> str:
        """Грубая эвристика по интенсивностям: КТ содержит воздух (HU ~ -1000), МРТ — нет."""
        return "CT" if float(volume.min()) < -500.0 else "MRI"


if __name__ == "__main__":
    # --- Пример использования: КТ ---
    ct_volume = np.random.uniform(-1000, 300, size=(32, 128, 128)).astype(np.float32)
    ct_config = PreprocessingConfig(window_center=40.0, window_width=80.0, apply_skull_stripping_ct=True)
    ct_processed = CTPreprocessor(ct_config).preprocess_full(ct_volume, spacing=(0.5, 0.5, 1.0))
    print("КТ обработана:", ct_processed.shape, ct_processed.min(), ct_processed.max())

    # --- Пример использования: МРТ, объединение T1/T1ce/T2/FLAIR в 4-канальный вход ---
    mri_config = PreprocessingConfig(apply_bias_correction=False)  # ускоряем пример, N4 не быстрый
    mri_preprocessor = MRIPreprocessor(mri_config)

    raw_sequences = {
        weighting: np.random.uniform(0, 800, size=(32, 128, 128)).astype(np.float32)
        for weighting in ("T1", "T1ce", "T2", "FLAIR")
    }
    processed_sequences = {
        weighting: mri_preprocessor.preprocess_full(volume, spacing=(1.0, 1.0, 1.0), modality=weighting)
        for weighting, volume in raw_sequences.items()
    }
    multichannel_input = mri_preprocessor.combine_multimodal(processed_sequences)
    print("Многоканальный МР-вход:", multichannel_input.shape)  # (4, 32, 128, 128)

    # --- Пример использования: UnifiedPreprocessor с автоопределением модальности ---
    auto_processed = UnifiedPreprocessor.auto_detect_and_preprocess(ct_volume, spacing=(0.5, 0.5, 1.0))
    print("Автоопределение модальности:", auto_processed.shape)
