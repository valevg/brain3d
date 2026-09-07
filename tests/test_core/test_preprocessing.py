"""Тесты core/preprocessing.py: CTPreprocessor, MRIPreprocessor, UnifiedPreprocessor.

Используются небольшие синтетические объёмы (сфера высокой/низкой плотности на фоне) —
этого достаточно, чтобы реально прогнать всю логику (клиппинг, ресемплинг, пороговые
маски, N4-коррекцию, z-score, объединение каналов), не привязываясь к реальным
медицинским данным.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.exceptions import PreprocessingError, UnsupportedModalityError
from core.preprocessing import (
    CTPreprocessor,
    MRIPreprocessor,
    PreprocessingConfig,
    UnifiedPreprocessor,
)


def _make_sphere_volume(
    shape: tuple[int, int, int] = (8, 16, 16),
    background: float = -1000.0,
    foreground: float = 50.0,
    radius: int = 5,
) -> np.ndarray:
    """Синтетический объём: сфера постоянной интенсивности на однородном фоне."""
    volume = np.full(shape, fill_value=background, dtype=np.float32)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    mask = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= radius**2
    volume[mask] = foreground
    return volume


# --------------------------------------------------------------------------- #
# CTPreprocessor
# --------------------------------------------------------------------------- #


class TestCTPreprocessorWindowing:
    def test_apply_hounsfield_window_clips_and_normalizes(self) -> None:
        volume = np.array([-2000.0, -100.0, 40.0, 180.0, 5000.0], dtype=np.float32)
        preprocessor = CTPreprocessor()

        windowed = preprocessor.apply_hounsfield_window(volume, window_center=40.0, window_width=400.0)

        assert windowed.dtype == np.float32
        assert windowed.min() == pytest.approx(0.0)
        assert windowed.max() == pytest.approx(1.0)
        # center (40 HU) должен попасть точно в середину нормализованного диапазона
        assert windowed[2] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("preset", "expected_center", "expected_width"),
        [("brain", 40.0, 80.0), ("soft_tissue", 40.0, 400.0), ("bone", 300.0, 1500.0)],
    )
    def test_apply_window_preset_matches_manual_window(
        self, preset: str, expected_center: float, expected_width: float
    ) -> None:
        volume = np.linspace(-1000, 3000, num=50, dtype=np.float32)
        preprocessor = CTPreprocessor()

        from_preset = preprocessor.apply_window_preset(volume, preset=preset)
        manual = preprocessor.apply_hounsfield_window(volume, expected_center, expected_width)

        np.testing.assert_allclose(from_preset, manual)

    def test_apply_window_preset_raises_for_unknown_preset(self) -> None:
        preprocessor = CTPreprocessor()
        with pytest.raises(PreprocessingError):
            preprocessor.apply_window_preset(np.zeros((2, 2, 2), dtype=np.float32), preset="lung")


class TestCTPreprocessorResampling:
    def test_resample_to_isotropic_changes_shape_per_axis(self) -> None:
        volume = np.zeros((4, 8, 8), dtype=np.float32)
        preprocessor = CTPreprocessor()

        # spacing=(x=2.0, y=1.0, z=1.0) -> при апсемплинге к (1,1,1) должна вырасти
        # только ось X (последняя ось массива), Y и Z остаются без изменений.
        resampled = preprocessor.resample_to_isotropic(
            volume, spacing=(2.0, 1.0, 1.0), new_spacing=(1.0, 1.0, 1.0)
        )

        assert resampled.shape[0] == volume.shape[0]  # Z без изменений
        assert resampled.shape[1] == volume.shape[1]  # Y без изменений
        assert resampled.shape[2] == volume.shape[2] * 2  # X удвоена


class TestCTPreprocessorDenoiseAndNormalize:
    def test_denoise_median_removes_salt_noise(self) -> None:
        volume = np.zeros((5, 9, 9), dtype=np.float32)
        volume[2, 4, 4] = 1000.0  # одиночный выброс
        preprocessor = CTPreprocessor()

        denoised = preprocessor.denoise_median(volume, kernel_size=3)

        assert denoised[2, 4, 4] == pytest.approx(0.0)

    def test_normalize_to_0_1_range(self) -> None:
        volume = np.array([-500.0, 0.0, 500.0], dtype=np.float32)
        preprocessor = CTPreprocessor()

        normalized = preprocessor.normalize_to_0_1(volume)

        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_normalize_to_0_1_handles_constant_volume(self) -> None:
        volume = np.full((3, 3, 3), fill_value=42.0, dtype=np.float32)
        preprocessor = CTPreprocessor()

        normalized = preprocessor.normalize_to_0_1(volume)

        np.testing.assert_array_equal(normalized, np.zeros_like(volume))


class TestCTPreprocessorSkullStripping:
    def test_skull_stripping_keeps_sphere_and_removes_noise_component(self) -> None:
        volume = _make_sphere_volume(background=-1000.0, foreground=50.0, radius=5)
        # изолированный шумовой воксель того же HU-диапазона в другом углу объёма
        volume[0, 0, 0] = 50.0
        preprocessor = CTPreprocessor()

        stripped = preprocessor.skull_stripping(volume, hu_min=20.0, hu_max=100.0)

        assert stripped[4, 8, 8] == pytest.approx(50.0)  # центр сферы сохранён
        assert stripped[0, 0, 0] == pytest.approx(-1000.0)  # шумовой воксель удалён
        assert stripped[0, 1, 1] == pytest.approx(-1000.0)  # фон вне сферы занулён/заменён


class TestCTPreprocessorFullPipeline:
    def test_preprocess_full_default_config_returns_0_1_range(self) -> None:
        volume = _make_sphere_volume()
        preprocessor = CTPreprocessor(PreprocessingConfig(apply_resampling=False))

        result = preprocessor.preprocess_full(volume, spacing=(1.0, 1.0, 1.0))

        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_full_without_windowing_uses_min_max_normalize(self) -> None:
        volume = _make_sphere_volume()
        config = PreprocessingConfig(apply_resampling=False, apply_hu_windowing=False)
        preprocessor = CTPreprocessor(config)

        result = preprocessor.preprocess_full(volume, spacing=(1.0, 1.0, 1.0))

        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_preprocess_full_can_disable_normalization_entirely(self) -> None:
        volume = _make_sphere_volume()
        config = PreprocessingConfig(apply_resampling=False, apply_normalization=False)
        preprocessor = CTPreprocessor(config)

        result = preprocessor.preprocess_full(volume, spacing=(1.0, 1.0, 1.0))

        # Без нормализации сохраняются исходные HU-значения
        assert result.max() == pytest.approx(50.0)
        assert result.min() == pytest.approx(-1000.0)


# --------------------------------------------------------------------------- #
# MRIPreprocessor
# --------------------------------------------------------------------------- #


class TestMRIPreprocessorSkullStrippingAndNormalize:
    def test_skull_stripping_mri_keeps_sphere_and_zeroes_background(self) -> None:
        volume = _make_sphere_volume(background=0.0, foreground=500.0, radius=5)
        preprocessor = MRIPreprocessor()

        stripped = preprocessor.skull_stripping_mri(volume)

        assert stripped[4, 8, 8] == pytest.approx(500.0)
        assert stripped[0, 0, 0] == pytest.approx(0.0)

    def test_skull_stripping_mri_handles_all_zero_volume(self) -> None:
        volume = np.zeros((4, 4, 4), dtype=np.float32)
        preprocessor = MRIPreprocessor()

        result = preprocessor.skull_stripping_mri(volume)

        np.testing.assert_array_equal(result, volume)

    def test_normalize_z_score_zero_mean_on_tissue(self) -> None:
        volume = _make_sphere_volume(background=0.0, foreground=500.0, radius=5)
        preprocessor = MRIPreprocessor()

        normalized = preprocessor.normalize_z_score(volume)

        tissue_values = normalized[volume > 0]
        assert abs(tissue_values.mean()) < 1e-3


class TestMRIPreprocessorBiasFieldCorrection:
    def test_bias_field_correction_preserves_shape(self) -> None:
        volume = _make_sphere_volume(shape=(6, 12, 12), background=10.0, foreground=300.0, radius=3)
        preprocessor = MRIPreprocessor()

        corrected = preprocessor.bias_field_correction(volume)

        assert corrected.shape == volume.shape
        assert np.all(np.isfinite(corrected))


class TestMRIPreprocessorAlignToTemplate:
    def test_align_to_template_returns_unchanged_without_template(self) -> None:
        volume = _make_sphere_volume()
        preprocessor = MRIPreprocessor()

        result = preprocessor.align_to_template(volume, template=None)

        np.testing.assert_array_equal(result, volume.astype(np.float32))

    def test_align_to_template_runs_registration_and_preserves_shape(self) -> None:
        volume = _make_sphere_volume(shape=(6, 12, 12), background=0.0, foreground=500.0, radius=3)
        preprocessor = MRIPreprocessor()

        result = preprocessor.align_to_template(volume, template=volume, spacing=(1.0, 1.0, 1.0))

        assert result.shape == volume.shape
        assert np.all(np.isfinite(result))


class TestMRIPreprocessorFullPipeline:
    def test_preprocess_full_raises_for_unknown_weighting(self) -> None:
        volume = _make_sphere_volume()
        preprocessor = MRIPreprocessor()

        with pytest.raises(PreprocessingError):
            preprocessor.preprocess_full(volume, spacing=(1.0, 1.0, 1.0), modality="PD")

    def test_preprocess_full_skips_bias_correction_for_flair_even_if_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        volume = _make_sphere_volume(background=0.0, foreground=500.0)
        config = PreprocessingConfig(apply_resampling=False, apply_bias_correction=True)
        preprocessor = MRIPreprocessor(config)

        def _fail_if_called(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("bias_field_correction не должен вызываться для FLAIR")

        monkeypatch.setattr(preprocessor, "bias_field_correction", _fail_if_called)

        result = preprocessor.preprocess_full(volume, spacing=(1.0, 1.0, 1.0), modality="FLAIR")

        assert result.shape == volume.shape

    def test_preprocess_full_runs_bias_correction_for_t1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        volume = _make_sphere_volume(background=0.0, foreground=500.0)
        config = PreprocessingConfig(apply_resampling=False, apply_skull_stripping_mri=False)
        preprocessor = MRIPreprocessor(config)

        calls: list[bool] = []
        original = preprocessor.bias_field_correction

        def _spy(volume_arg: np.ndarray) -> np.ndarray:
            calls.append(True)
            return original(volume_arg)

        monkeypatch.setattr(preprocessor, "bias_field_correction", _spy)

        preprocessor.preprocess_full(volume, spacing=(1.0, 1.0, 1.0), modality="T1")

        assert calls == [True]


class TestCombineMultimodal:
    def test_combine_multimodal_stacks_in_fixed_order(self) -> None:
        shape = (4, 8, 8)
        volumes = {
            "FLAIR": np.full(shape, 4.0, dtype=np.float32),
            "T1": np.full(shape, 1.0, dtype=np.float32),
            "T2": np.full(shape, 3.0, dtype=np.float32),
            "T1ce": np.full(shape, 2.0, dtype=np.float32),
        }
        preprocessor = MRIPreprocessor()

        combined = preprocessor.combine_multimodal(volumes)

        assert combined.shape == (4, *shape)
        # Порядок каналов фиксирован: T1, T1ce, T2, FLAIR — независимо от порядка ключей в dict
        assert combined[0, 0, 0, 0] == pytest.approx(1.0)
        assert combined[1, 0, 0, 0] == pytest.approx(2.0)
        assert combined[2, 0, 0, 0] == pytest.approx(3.0)
        assert combined[3, 0, 0, 0] == pytest.approx(4.0)

    def test_combine_multimodal_allows_subset_of_channels(self) -> None:
        shape = (2, 4, 4)
        volumes = {"T1": np.zeros(shape, dtype=np.float32), "FLAIR": np.ones(shape, dtype=np.float32)}
        preprocessor = MRIPreprocessor()

        combined = preprocessor.combine_multimodal(volumes)

        assert combined.shape == (2, *shape)

    def test_combine_multimodal_raises_for_empty_dict(self) -> None:
        with pytest.raises(PreprocessingError):
            MRIPreprocessor().combine_multimodal({})

    def test_combine_multimodal_raises_for_unknown_key(self) -> None:
        with pytest.raises(PreprocessingError):
            MRIPreprocessor().combine_multimodal({"PD": np.zeros((2, 2, 2), dtype=np.float32)})

    def test_combine_multimodal_raises_for_shape_mismatch(self) -> None:
        with pytest.raises(PreprocessingError):
            MRIPreprocessor().combine_multimodal(
                {
                    "T1": np.zeros((2, 2, 2), dtype=np.float32),
                    "T2": np.zeros((3, 3, 3), dtype=np.float32),
                }
            )


# --------------------------------------------------------------------------- #
# UnifiedPreprocessor
# --------------------------------------------------------------------------- #


class TestUnifiedPreprocessor:
    @pytest.mark.parametrize("modality", ["CT", "ct"])
    def test_selects_ct_preprocessor(self, modality: str) -> None:
        unified = UnifiedPreprocessor(modality)
        assert isinstance(unified._impl, CTPreprocessor)  # noqa: SLF001

    @pytest.mark.parametrize("modality", ["MRI", "mr", "mri"])
    def test_selects_mri_preprocessor(self, modality: str) -> None:
        unified = UnifiedPreprocessor(modality)
        assert isinstance(unified._impl, MRIPreprocessor)  # noqa: SLF001

    def test_raises_for_unsupported_modality(self) -> None:
        with pytest.raises(UnsupportedModalityError):
            UnifiedPreprocessor("PET")

    def test_preprocess_dispatches_to_ct(self) -> None:
        volume = _make_sphere_volume()
        unified = UnifiedPreprocessor("CT", PreprocessingConfig(apply_resampling=False))

        result = unified.preprocess(volume, spacing=(1.0, 1.0, 1.0))

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_dispatches_to_mri_with_weighting_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        volume = _make_sphere_volume(background=0.0, foreground=500.0)
        config = PreprocessingConfig(apply_resampling=False)
        unified = UnifiedPreprocessor("MRI", config)

        received_modality: list[str] = []
        original = unified._impl.preprocess_full  # noqa: SLF001

        def _spy(volume_arg, spacing_arg, modality="T1ce"):  # noqa: ANN001
            received_modality.append(modality)
            return original(volume_arg, spacing_arg, modality=modality)

        monkeypatch.setattr(unified._impl, "preprocess_full", _spy)  # noqa: SLF001

        unified.preprocess(volume, spacing=(1.0, 1.0, 1.0), modality="T2")

        assert received_modality == ["T2"]

    def test_auto_detect_and_preprocess_picks_ct_for_low_hu_background(self) -> None:
        ct_like_volume = _make_sphere_volume(background=-1000.0, foreground=50.0)

        result = UnifiedPreprocessor.auto_detect_and_preprocess(
            ct_like_volume, spacing=(1.0, 1.0, 1.0), config=PreprocessingConfig(apply_resampling=False)
        )

        # КТ-ветка нормализует HU-окном в [0, 1]
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_auto_detect_and_preprocess_picks_mri_for_nonnegative_background(self) -> None:
        mri_like_volume = _make_sphere_volume(background=0.0, foreground=500.0)
        config = PreprocessingConfig(apply_resampling=False, apply_bias_correction=False)

        result = UnifiedPreprocessor.auto_detect_and_preprocess(
            mri_like_volume, spacing=(1.0, 1.0, 1.0), config=config
        )

        # МРТ-ветка возвращает z-score нормализованный объём (может уходить в отрицательные значения)
        assert result.shape == mri_like_volume.shape
        assert np.all(np.isfinite(result))
