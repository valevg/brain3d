"""Тесты core/vessel_segmentation.py.

Постобработка, конфигурация, кэш, оценка уверенности и сравнение методов проверяются
по-настоящему (numpy/scipy/skimage/pydantic/yaml — уже реальные зависимости проекта).
Внешние DL-обёртки (TotalSegmentator, VISTA-3D, MONAI) из models.pretrained_models
монкипатчатся прямо в пространстве имён core.vessel_segmentation (куда они
импортированы по имени) — так же, как их собственные тесты мокают totalsegmentator/
requests/monai на границе, здесь мокается сама обёртка, а проверяется логика
VesselSegmenter поверх неё: выбор метода, комбинирование масок, кэш, фолбэк.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.exceptions import Brain3DError, ConfigurationError, VesselSegmentationError
from core.vessel_segmentation import (
    MethodComparisonResult,
    VesselSegmentationConfig,
    VesselSegmenter,
    _build_comparison_result,
    _combine_masks,
    _estimate_confidence,
    connect_vessels,
    fill_holes,
    remove_small_components,
    smooth_vessels,
)
from models.pretrained_models import ComputeResources, PretrainedModelRegistry


def _make_tube_volume(shape: tuple[int, int, int] = (30, 30, 30), radius: int = 2) -> np.ndarray:
    """Синтетический объём с цилиндрической "трубкой" вдоль оси Z — для реальных тестов Франги."""
    volume = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[: shape[1], : shape[2]]
    tube_mask = (y - shape[1] // 2) ** 2 + (x - shape[2] // 2) ** 2 <= radius**2
    volume[:, tube_mask] = 1.0
    return volume


# --------------------------------------------------------------------------- #
# VesselSegmentationConfig
# --------------------------------------------------------------------------- #


class TestVesselSegmentationConfig:
    def test_defaults(self) -> None:
        config = VesselSegmentationConfig()

        assert config.enable_totalsegmentator is True
        assert config.enable_vista3d is False
        assert config.enable_frangi is True

    def test_expands_home_tilde_in_cache_dir(self) -> None:
        config = VesselSegmentationConfig(result_cache_dir="~/somewhere/cache")  # type: ignore[arg-type]

        assert "~" not in str(config.result_cache_dir)
        assert config.result_cache_dir.is_absolute()

    def test_from_yaml_loads_real_config_file(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "vessel_segmentation.yaml"

        config = VesselSegmentationConfig.from_yaml(config_path)

        assert config.totalsegmentator_task == "headneck_bones_vessels"
        assert config.frangi_sigmas == (1.0, 2.0, 3.0)

    def test_from_yaml_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            VesselSegmentationConfig.from_yaml(tmp_path / "does_not_exist.yaml")

    def test_from_yaml_raises_for_invalid_content(self, tmp_path: Path) -> None:
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("enable_totalsegmentator: 'not-a-bool-and-not-parseable-either: [}'")

        with pytest.raises(ConfigurationError):
            VesselSegmentationConfig.from_yaml(bad_config)


# --------------------------------------------------------------------------- #
# Постобработка
# --------------------------------------------------------------------------- #


class TestPostprocessing:
    def test_remove_small_components_keeps_multiple_large_components(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:3, 0:3, 0:3] = 1  # крупная компонента (27 вокселей)
        mask[8:10, 8:10, 8:10] = 1  # ещё одна крупная компонента (8 вокселей)
        mask[5, 5, 5] = 1  # шумовой одиночный воксель

        cleaned = remove_small_components(mask, min_size=5)

        assert cleaned[1, 1, 1] == 1
        assert cleaned[9, 9, 9] == 1
        assert cleaned[5, 5, 5] == 0  # шум удалён
        _, num_components = __import__("scipy.ndimage", fromlist=["label"]).label(cleaned)
        assert num_components == 2  # обе крупные компоненты сохранены — не как при skull-stripping

    def test_fill_holes_closes_internal_cavity(self) -> None:
        mask = np.ones((5, 5, 5), dtype=np.uint8)
        mask[2, 2, 2] = 0  # внутренняя полость

        filled = fill_holes(mask)

        assert filled[2, 2, 2] == 1

    def test_smooth_vessels_removes_thin_protrusion(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[3:7, 3:7, 3:7] = 1  # компактный блок
        mask[3, 3, 0] = 1  # тонкий одиночный "ус", не связанный с блоком по диагонали

        smoothed = smooth_vessels(mask, iterations=1)

        assert smoothed[3, 3, 0] == 0
        assert smoothed[4, 4, 4] == 1  # основной блок сохранён

    def test_connect_vessels_bridges_small_gap(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[4, 4, 0:4] = 1
        mask[4, 4, 5:9] = 1  # разрыв в один воксель на позиции 4

        connected = connect_vessels(mask, iterations=1)

        assert connected[4, 4, 4] == 1

    def test_combine_masks_logical_or(self) -> None:
        mask_a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        mask_b = np.array([[0, 0], [0, 1]], dtype=np.uint8)

        combined = _combine_masks([mask_a, mask_b], shape=(2, 2))

        np.testing.assert_array_equal(combined, np.array([[1, 0], [0, 1]], dtype=np.uint8))


class TestEstimateConfidence:
    def test_zero_for_empty_mask(self) -> None:
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        assert _estimate_confidence(mask, "totalsegmentator") == 0.0

    def test_base_confidence_for_plausible_volume(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:2, 0:2, 0:2] = 1  # 0.8% объёма — правдоподобно для сосудов

        assert _estimate_confidence(mask, "totalsegmentator") == pytest.approx(0.9)
        assert _estimate_confidence(mask, "frangi") == pytest.approx(0.5)

    def test_penalizes_implausibly_large_mask(self) -> None:
        mask = np.ones((10, 10, 10), dtype=np.uint8)  # 100% объёма — явно ошибка метода

        confidence = _estimate_confidence(mask, "totalsegmentator")

        assert confidence == pytest.approx(0.9 * 0.3)


class TestBuildComparisonResult:
    def test_computes_expected_metrics(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:2, 0:2, 0:2] = 1  # компонента размером 8
        mask[9, 9, 9] = 1  # компонента размером 1

        result = _build_comparison_result("frangi", mask, elapsed=1.23)

        assert result.method == "frangi"
        assert result.success is True
        assert result.volume_voxels == 9
        assert result.num_components == 2
        assert result.largest_component_fraction == pytest.approx(8 / 9)
        assert result.elapsed_seconds == 1.23


# --------------------------------------------------------------------------- #
# VesselSegmenter — методы, не требующие внешних систем
# --------------------------------------------------------------------------- #


class TestSegmentByThreshold:
    def test_isolates_contrast_range_and_cleans_noise(self) -> None:
        volume = np.full((10, 10, 10), fill_value=20.0, dtype=np.float32)
        volume[2:5, 2:5, 2:5] = 300.0  # контрастированный сосуд
        volume[9, 9, 9] = 300.0  # шумовой одиночный воксель в том же диапазоне

        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, min_component_size=5))
        mask = segmenter.segment_by_threshold(volume, min_hu=100.0, max_hu=500.0)

        assert mask[3, 3, 3] == 1
        assert mask[9, 9, 9] == 0


class TestSegmentFrangiAndVesselness:
    def test_segment_frangi_detects_tube(self) -> None:
        volume = _make_tube_volume()
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False))

        mask = segmenter.segment_frangi(volume, spacing=(1.0, 1.0, 1.0), sigmas=(1.0, 2.0))

        assert mask.sum() > 0

    def test_enhance_vesselness_returns_continuous_map(self) -> None:
        volume = _make_tube_volume()
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False))

        vesselness = segmenter.enhance_vesselness(volume, sigmas=(1.0, 2.0))

        assert vesselness.dtype == np.float32
        assert vesselness.shape == volume.shape
        assert vesselness.max() > 0.0
        assert set(np.unique(vesselness[:2, :2, :2])) != {0, 1}  # не бинарная карта


# --------------------------------------------------------------------------- #
# VesselSegmenter — кэширование
# --------------------------------------------------------------------------- #


class TestCaching:
    def test_totalsegmentator_uses_cache_on_second_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.vessel_segmentation as vs

        call_count = {"n": 0}

        class _FakeWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> dict:
                call_count["n"] += 1
                return {}

            def get_vessel_masks(self) -> dict[str, np.ndarray]:
                return {"internal_carotid_artery_left": np.ones((4, 4, 4), dtype=np.uint8)}

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FakeWrapper)

        config = VesselSegmentationConfig(cache_enabled=True, result_cache_dir=tmp_path)
        segmenter = VesselSegmenter(config)
        volume = np.zeros((4, 4, 4), dtype=np.float32)

        first = segmenter.segment_with_totalsegmentator(volume, spacing=(1.0, 1.0, 1.0))
        second = segmenter.segment_with_totalsegmentator(volume, spacing=(1.0, 1.0, 1.0))

        np.testing.assert_array_equal(first, second)
        assert call_count["n"] == 1  # второй вызов обслужен из кэша

    def test_cache_disabled_calls_method_every_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.vessel_segmentation as vs

        call_count = {"n": 0}

        class _FakeWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> dict:
                call_count["n"] += 1
                return {}

            def get_vessel_masks(self) -> dict[str, np.ndarray]:
                return {"aorta": np.ones((4, 4, 4), dtype=np.uint8)}

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FakeWrapper)

        config = VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path)
        segmenter = VesselSegmenter(config)
        volume = np.zeros((4, 4, 4), dtype=np.float32)

        segmenter.segment_with_totalsegmentator(volume, spacing=(1.0, 1.0, 1.0))
        segmenter.segment_with_totalsegmentator(volume, spacing=(1.0, 1.0, 1.0))

        assert call_count["n"] == 2


# --------------------------------------------------------------------------- #
# VesselSegmenter — обёртки TotalSegmentator/VISTA-3D (мокнутые на границе)
# --------------------------------------------------------------------------- #


class TestSegmentWithTotalSegmentator:
    def test_raises_when_no_vessels_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.vessel_segmentation as vs

        class _EmptyWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> dict:
                return {}

            def get_vessel_masks(self) -> dict[str, np.ndarray]:
                return {}

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _EmptyWrapper)
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))

        with pytest.raises(VesselSegmentationError):
            segmenter.segment_with_totalsegmentator(
                np.zeros((4, 4, 4), dtype=np.float32), spacing=(1.0, 1.0, 1.0)
            )

    def test_combines_multiple_vessel_masks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.vessel_segmentation as vs

        mask_a = np.zeros((4, 4, 4), dtype=np.uint8)
        mask_a[0, 0, 0] = 1
        mask_b = np.zeros((4, 4, 4), dtype=np.uint8)
        mask_b[3, 3, 3] = 1

        class _FakeWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> dict:
                return {}

            def get_vessel_masks(self) -> dict[str, np.ndarray]:
                return {"internal_carotid_artery_left": mask_a, "internal_carotid_artery_right": mask_b}

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FakeWrapper)
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))

        combined = segmenter.segment_with_totalsegmentator(
            np.zeros((4, 4, 4), dtype=np.float32), spacing=(1.0, 1.0, 1.0)
        )

        assert combined[0, 0, 0] == 1
        assert combined[3, 3, 3] == 1
        assert combined.sum() == 2


class TestSegmentWithVista3D:
    def test_raises_when_endpoint_unavailable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.vessel_segmentation as vs

        class _UnavailableWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def is_available(self) -> bool:
                return False

        monkeypatch.setattr(vs, "VISTA3DWrapper", _UnavailableWrapper)
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))

        with pytest.raises(VesselSegmentationError, match="недоступен"):
            segmenter.segment_with_vista3d(np.zeros((4, 4, 4), dtype=np.float32))

    def test_raises_when_no_vessel_classes_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.vessel_segmentation as vs

        class _BrainOnlyWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def is_available(self) -> bool:
                return True

            def segment(self, volume: np.ndarray, spacing: tuple, classes: list[str]) -> dict:
                return {"brain": np.ones((4, 4, 4), dtype=np.uint8)}

        monkeypatch.setattr(vs, "VISTA3DWrapper", _BrainOnlyWrapper)
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))

        with pytest.raises(VesselSegmentationError, match="сосудист"):
            segmenter.segment_with_vista3d(np.zeros((4, 4, 4), dtype=np.float32))

    def test_returns_combined_nonbrain_masks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.vessel_segmentation as vs

        class _FakeWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def is_available(self) -> bool:
                return True

            def segment(self, volume: np.ndarray, spacing: tuple, classes: list[str]) -> dict:
                vessel_mask = np.zeros((4, 4, 4), dtype=np.uint8)
                vessel_mask[1, 1, 1] = 1
                return {"brain": np.ones((4, 4, 4), dtype=np.uint8), "hepatic_vessel": vessel_mask}

        monkeypatch.setattr(vs, "VISTA3DWrapper", _FakeWrapper)
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))

        mask = segmenter.segment_with_vista3d(np.zeros((4, 4, 4), dtype=np.float32))

        assert mask[1, 1, 1] == 1
        assert mask.sum() == 1  # маска "brain" не должна попасть в результат


class TestSegmentWithMonaiVesselModel:
    def test_propagates_model_download_error_when_bundle_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MONAIModelZoo.load_brain_vessel_model поднимает ModelDownloadError ДО обращения
        к torch (bundle_name=None в реестре) — этот путь можно проверить без torch."""
        pytest.importorskip("monai", reason="Проверяем реальный путь через настоящий реестр")
        segmenter = VesselSegmenter(
            VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path),
            registry=PretrainedModelRegistry(cache_dir=tmp_path),
        )

        with pytest.raises(Brain3DError):
            segmenter.segment_with_monai_vessel_model(np.zeros((4, 4, 4), dtype=np.float32))


# --------------------------------------------------------------------------- #
# segment_adaptive
# --------------------------------------------------------------------------- #


class TestSegmentAdaptive:
    def test_ct_with_contrast_uses_threshold_then_frangi(self, tmp_path: Path) -> None:
        # min_component_size намеренно занижен: синтетический "сосуд" в тесте — всего 48
        # вокселей, дефолтные 100 удалили бы его при постобработке как шум.
        config = VesselSegmentationConfig(
            cache_enabled=False, result_cache_dir=tmp_path, min_component_size=10
        )
        segmenter = VesselSegmenter(config)
        volume = np.full((10, 20, 20), fill_value=20.0, dtype=np.float32)
        volume[3:6, 8:12, 8:12] = 300.0
        resources = ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=4)

        mask, method_used, confidence = segmenter.segment_adaptive(
            volume, spacing=(1.0, 1.0, 1.0), modality="CT", resources=resources, has_contrast=True
        )

        assert method_used == "threshold+frangi"
        assert mask.sum() > 0
        assert 0.0 < confidence <= 1.0

    def test_cpu_only_without_contrast_falls_back_to_frangi(self, tmp_path: Path) -> None:
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))
        volume = _make_tube_volume()
        resources = ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=4)

        _mask, method_used, _confidence = segmenter.segment_adaptive(
            volume, spacing=(1.0, 1.0, 1.0), modality="MRI", resources=resources
        )

        assert method_used == "frangi"

    def test_gpu_available_tries_totalsegmentator_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.vessel_segmentation as vs

        class _FakeWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple) -> dict:
                return {}

            def get_vessel_masks(self) -> dict[str, np.ndarray]:
                return {"internal_carotid_artery_left": np.ones((4, 4, 4), dtype=np.uint8)}

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FakeWrapper)
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))
        resources = ComputeResources(has_gpu=True, gpu_memory_gb=24.0, has_internet=True, cpu_count=16)

        _mask, method_used, _confidence = segmenter.segment_adaptive(
            np.zeros((4, 4, 4), dtype=np.float32),
            spacing=(1.0, 1.0, 1.0),
            modality="MRI",
            resources=resources,
        )

        assert method_used == "totalsegmentator"

    def test_falls_back_to_vista3d_when_totalsegmentator_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.vessel_segmentation as vs

        class _FailingTotalSegmentator:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple) -> dict:
                raise Brain3DError("totalsegmentator не установлен")

        class _WorkingVista3D:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def is_available(self) -> bool:
                return True

            def segment(self, volume: np.ndarray, spacing: tuple, classes: list[str]) -> dict:
                vessel_mask = np.zeros((4, 4, 4), dtype=np.uint8)
                vessel_mask[0, 0, 0] = 1
                return {"brain": np.ones((4, 4, 4), dtype=np.uint8), "hepatic_vessel": vessel_mask}

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FailingTotalSegmentator)
        monkeypatch.setattr(vs, "VISTA3DWrapper", _WorkingVista3D)
        # enable_vista3d выключен по умолчанию (требует API-ключ/NIM) — включаем явно для теста.
        config = VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path, enable_vista3d=True)
        segmenter = VesselSegmenter(config)
        resources = ComputeResources(has_gpu=True, gpu_memory_gb=48.0, has_internet=True, cpu_count=16)

        mask, method_used, _confidence = segmenter.segment_adaptive(
            np.zeros((4, 4, 4), dtype=np.float32),
            spacing=(1.0, 1.0, 1.0),
            modality="MRI",
            resources=resources,
        )

        assert method_used == "vista3d"
        assert mask[0, 0, 0] == 1

    def test_raises_when_all_enabled_methods_fail(self, tmp_path: Path) -> None:
        config = VesselSegmentationConfig(
            cache_enabled=False,
            result_cache_dir=tmp_path,
            enable_frangi=False,
            enable_threshold=False,
            enable_totalsegmentator=False,
            enable_vista3d=False,
            enable_monai_vessel_model=False,
        )
        segmenter = VesselSegmenter(config)
        resources = ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=4)

        with pytest.raises(VesselSegmentationError):
            segmenter.segment_adaptive(
                np.zeros((4, 4, 4), dtype=np.float32),
                spacing=(1.0, 1.0, 1.0),
                modality="MRI",
                resources=resources,
            )

    def test_respects_disabled_frangi_config(self, tmp_path: Path) -> None:
        config = VesselSegmentationConfig(
            cache_enabled=False,
            result_cache_dir=tmp_path,
            enable_frangi=False,
            enable_totalsegmentator=False,
            enable_vista3d=False,
        )
        segmenter = VesselSegmenter(config)
        resources = ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=4)

        with pytest.raises(VesselSegmentationError):
            segmenter.segment_adaptive(
                np.zeros((4, 4, 4), dtype=np.float32),
                spacing=(1.0, 1.0, 1.0),
                modality="MRI",
                resources=resources,
            )


# --------------------------------------------------------------------------- #
# segment_circle_of_willis
# --------------------------------------------------------------------------- #


class TestSegmentCircleOfWillis:
    def test_combines_totalsegmentator_and_frangi_within_roi(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.vessel_segmentation as vs

        shape = (20, 40, 40)
        large_vessel_mask = np.zeros(shape, dtype=np.uint8)
        large_vessel_mask[10, 20, 20] = 1  # "крупный сосуд", единичная точка для теста

        class _FakeWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple, task: str | None = None) -> dict:
                return {}

            def get_vessel_masks(self) -> dict[str, np.ndarray]:
                return {"internal_carotid_artery_left": large_vessel_mask}

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FakeWrapper)

        volume = _make_tube_volume(shape=shape, radius=1)
        # min_component_size занижен: тестовая "трубка" тоньше и короче реального сосуда,
        # дефолтные 100 могли бы удалить итоговую компоненту как шум.
        config = VesselSegmentationConfig(
            cache_enabled=False, result_cache_dir=tmp_path, min_component_size=1
        )
        segmenter = VesselSegmenter(config)

        mask = segmenter.segment_circle_of_willis(volume, spacing=(1.0, 1.0, 1.0))

        assert mask[10, 20, 20] == 1  # крупный сосуд включён
        assert mask.shape == shape

    def test_falls_back_to_frangi_only_when_totalsegmentator_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.vessel_segmentation as vs

        class _FailingWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple, task: str | None = None) -> dict:
                raise Brain3DError("недоступен")

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FailingWrapper)

        volume = _make_tube_volume()
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))

        mask = segmenter.segment_circle_of_willis(volume, spacing=(1.0, 1.0, 1.0))

        assert mask.sum() > 0

    def test_raises_when_nothing_found_at_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.vessel_segmentation as vs

        class _FailingWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple, task: str | None = None) -> dict:
                raise Brain3DError("недоступен")

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FailingWrapper)

        empty_volume = np.zeros((10, 10, 10), dtype=np.float32)
        segmenter = VesselSegmenter(VesselSegmentationConfig(cache_enabled=False, result_cache_dir=tmp_path))

        with pytest.raises(VesselSegmentationError):
            segmenter.segment_circle_of_willis(empty_volume, spacing=(1.0, 1.0, 1.0))


# --------------------------------------------------------------------------- #
# compare_methods
# --------------------------------------------------------------------------- #


class TestCompareMethods:
    def test_reports_success_and_failure_per_method(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import core.vessel_segmentation as vs

        class _FailingTotalSegmentator:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple) -> dict:
                raise Brain3DError("не установлен")

        monkeypatch.setattr(vs, "TotalSegmentatorWrapper", _FailingTotalSegmentator)

        config = VesselSegmentationConfig(
            cache_enabled=False,
            result_cache_dir=tmp_path,
            enable_vista3d=False,
            enable_monai_vessel_model=False,
        )
        segmenter = VesselSegmenter(config)
        volume = _make_tube_volume()

        results = segmenter.compare_methods(volume, spacing=(1.0, 1.0, 1.0))
        results_by_method = {r.method: r for r in results}

        assert results_by_method["totalsegmentator"].success is False
        assert "не установлен" in results_by_method["totalsegmentator"].error
        assert results_by_method["frangi"].success is True
        assert results_by_method["frangi"].volume_voxels > 0
        assert "vista3d" not in results_by_method  # отключён в конфиге
        assert "monai_vessel_model" not in results_by_method  # отключён в конфиге

    def test_all_results_are_method_comparison_result_instances(self, tmp_path: Path) -> None:
        config = VesselSegmentationConfig(
            cache_enabled=False,
            result_cache_dir=tmp_path,
            enable_totalsegmentator=False,
            enable_vista3d=False,
            enable_monai_vessel_model=False,
        )
        segmenter = VesselSegmenter(config)

        results = segmenter.compare_methods(_make_tube_volume(), spacing=(1.0, 1.0, 1.0))

        assert all(isinstance(r, MethodComparisonResult) for r in results)
        assert {r.method for r in results} == {"frangi", "threshold"}
