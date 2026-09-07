"""Тесты pipelines/brain_pipeline.py.

Стратегия тестирования смешанная:
    - ResourceDetector, ModelCacheManager, FallbackHandler, BrainPipelineConfig
      проверяются в основном по-настоящему (реальная файловая система, реальные
      объекты TotalSegmentatorWrapper/VISTA3DWrapper из models.pretrained_models,
      которые ничего не делают до вызова .segment()/.is_available()).
    - Тяжёлые внешние системы (реальный DICOM на диске, TotalSegmentator, VISTA-3D
      NIM, MONAI bundle) в этом окружении не установлены/не запущены — часть
      тестов сознательно ЭКСПЛУАТИРУЕТ это (например, tumor-фолбэк тестируется без
      единого мока: VISTA3DWrapper.is_available() сама вернёт False, потому что
      локального NIM-сервера нет, а load_tumor_model(strategy="brats_pretrained")
      сама поднимет ошибку, потому что monai не установлен).
    - process_with_progress() тестируется с моками на границе (DICOMLoader,
      TotalSegmentatorWrapper, MeshGenerator) — полный DICOM/TotalSegmentator стек
      здесь не поднять, но вся оркестрация вокруг них реальна.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.exceptions import Brain3DError, ChecksumMismatchError, MeshGenerationError
from models.pretrained_models import ComputeResources, PretrainedModelRegistry
from pipelines.brain_pipeline import (
    BrainModelPipeline,
    BrainPipelineConfig,
    FallbackHandler,
    ModelCacheManager,
    ResourceDetector,
    run_pipeline_cli,
)


def _make_tube_volume(shape: tuple[int, int, int] = (20, 20, 20), radius: int = 2) -> np.ndarray:
    volume = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[: shape[1], : shape[2]]
    tube_mask = (y - shape[1] // 2) ** 2 + (x - shape[2] // 2) ** 2 <= radius**2
    volume[:, tube_mask] = 1.0
    return volume


def _make_sphere_mask(shape: tuple[int, int, int] = (20, 20, 20), radius: int = 8) -> np.ndarray:
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = np.zeros(shape, dtype=np.uint8)
    mask[(z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= radius**2] = 1
    return mask


# --------------------------------------------------------------------------- #
# BrainPipelineConfig
# --------------------------------------------------------------------------- #


class TestBrainPipelineConfig:
    def test_defaults(self) -> None:
        config = BrainPipelineConfig()

        assert config.tumor_model_strategy == "auto"
        assert config.vessel_model_strategy == "auto"
        assert config.fast_mode is False
        assert config.offline is False
        assert config.export_format == "glb"

    def test_expands_tilde_in_output_dir(self) -> None:
        config = BrainPipelineConfig(output_dir="~/brain3d_output")  # type: ignore[arg-type]

        assert "~" not in str(config.output_dir)
        assert config.output_dir.is_absolute()


# --------------------------------------------------------------------------- #
# ResourceDetector
# --------------------------------------------------------------------------- #


class TestResourceDetector:
    def test_detect_gpu_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object):
            if name == "torch":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        info = ResourceDetector().detect_gpu()

        assert info == {"available": False, "memory_gb": None, "name": None}

    def test_detect_disk_space_returns_positive_int(self, tmp_path: Path) -> None:
        free_bytes = ResourceDetector().detect_disk_space(tmp_path)
        assert isinstance(free_bytes, int)
        assert free_bytes > 0

    def test_detect_internet_uses_is_online(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pipelines.brain_pipeline as bp

        monkeypatch.setattr(bp, "is_online", lambda: True)
        assert ResourceDetector().detect_internet() is True

        monkeypatch.setattr(bp, "is_online", lambda: False)
        assert ResourceDetector().detect_internet() is False

    def test_get_optimal_config_recommends_fast_and_offline_when_lacking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detector = ResourceDetector()
        monkeypatch.setattr(
            detector, "detect_gpu", lambda: {"available": False, "memory_gb": None, "name": None}
        )
        monkeypatch.setattr(detector, "detect_internet", lambda: False)
        monkeypatch.setattr(detector, "detect_ram", lambda: 0)
        monkeypatch.setattr(detector, "detect_disk_space", lambda path=".": 0)

        config = detector.get_optimal_config()

        assert config["recommended_fast_mode"] is True
        assert config["recommended_offline_mode"] is True


# --------------------------------------------------------------------------- #
# ModelCacheManager
# --------------------------------------------------------------------------- #


class TestModelCacheManager:
    def _populate_fake_model(self, registry: PretrainedModelRegistry, name: str, size_bytes: int) -> None:
        spec = registry.get_model(name)
        cache_path = registry.cache_dir / spec.source / spec.name
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / "weights.bin").write_bytes(b"0" * size_bytes)

    def test_get_cache_size_empty(self, tmp_path: Path) -> None:
        manager = ModelCacheManager(tmp_path)
        assert manager.get_cache_size() == 0

    def test_list_cached_models_reports_size_and_path(self, tmp_path: Path) -> None:
        registry = PretrainedModelRegistry(cache_dir=tmp_path)
        self._populate_fake_model(registry, "med3d_resnet18", 1024)
        manager = ModelCacheManager(tmp_path, registry)

        entries = manager.list_cached_models()

        assert len(entries) == 1
        assert entries[0]["name"] == "med3d_resnet18"
        assert entries[0]["size_bytes"] == 1024

    def test_clear_cache_removes_everything(self, tmp_path: Path) -> None:
        registry = PretrainedModelRegistry(cache_dir=tmp_path)
        self._populate_fake_model(registry, "med3d_resnet18", 512)
        manager = ModelCacheManager(tmp_path, registry)

        manager.clear_cache()

        assert manager.get_cache_size() == 0
        assert tmp_path.exists()  # директория кэша пересоздаётся, а не просто удаляется

    def test_manage_cache_removes_largest_models_first_until_within_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Реальных гигабайтных файлов на диске не создаём — только крошечные
        # placeholder-директории (чтобы было что реально удалить через shutil.rmtree),
        # а размеры для решения о вытеснении подменяем через list_cached_models().
        registry = PretrainedModelRegistry(cache_dir=tmp_path)
        one_gb = 1024**3
        self._populate_fake_model(registry, "totalsegmentator_total", 1)
        self._populate_fake_model(registry, "med3d_resnet18", 1)
        manager = ModelCacheManager(tmp_path, registry)

        fake_entries = [
            {
                "name": "totalsegmentator_total",
                "size_bytes": 6 * one_gb,
                "path": str(tmp_path / "totalsegmentator" / "totalsegmentator_total"),
            },
            {
                "name": "med3d_resnet18",
                "size_bytes": 2 * one_gb,
                "path": str(tmp_path / "med3d" / "med3d_resnet18"),
            },
        ]
        monkeypatch.setattr(manager, "list_cached_models", lambda: fake_entries)

        removed = manager.manage_cache(max_size_gb=5.0)

        assert removed == ["totalsegmentator_total"]
        assert not Path(fake_entries[0]["path"]).exists()
        assert Path(fake_entries[1]["path"]).exists()

    def test_prefetch_models_reports_success_and_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = PretrainedModelRegistry(cache_dir=tmp_path)
        manager = ModelCacheManager(tmp_path, registry)

        def _fake_download(name: str, *_args: object, **_kwargs: object) -> Path:
            if name == "totalsegmentator_total":
                return tmp_path
            raise Brain3DError("boom")

        monkeypatch.setattr(registry, "download_model", _fake_download)

        results = manager.prefetch_models(["totalsegmentator_total", "med3d_resnet18"])

        assert results == {"totalsegmentator_total": True, "med3d_resnet18": False}


# --------------------------------------------------------------------------- #
# FallbackHandler
# --------------------------------------------------------------------------- #


class TestFallbackHandler:
    def test_vista3d_falls_back_to_totalsegmentator(self) -> None:
        handler = FallbackHandler()
        assert handler.handle_model_failure("vista3d", Brain3DError("недоступен")) == "totalsegmentator"

    def test_totalsegmentator_falls_back_to_frangi(self) -> None:
        handler = FallbackHandler()
        assert handler.handle_model_failure("totalsegmentator", Brain3DError("не установлен")) == "frangi"

    def test_unknown_model_falls_back_to_frangi(self) -> None:
        handler = FallbackHandler()
        assert handler.handle_model_failure("something_unknown", Brain3DError("?")) == "frangi"

    def test_checksum_mismatch_triggers_redownload_and_returns_same_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = PretrainedModelRegistry(cache_dir=tmp_path)
        calls: list[tuple[str, bool]] = []

        def _fake_download(name: str, force: bool = False, **_kwargs: object) -> Path:
            calls.append((name, force))
            return tmp_path

        monkeypatch.setattr(registry, "download_model", _fake_download)
        handler = FallbackHandler(registry)

        result = handler.handle_model_failure("monai_brain_tumor_swinunetr", ChecksumMismatchError("corrupt"))

        assert result == "monai_brain_tumor_swinunetr"
        assert calls == [("monai_brain_tumor_swinunetr", True)]


# --------------------------------------------------------------------------- #
# BrainModelPipeline: detect_modality / select_models / download_required_models
# --------------------------------------------------------------------------- #


class TestDetectModality:
    def test_detect_modality_delegates_to_dicom_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pipelines.brain_pipeline as bp

        class _FakeLoader:
            def __init__(self) -> None:
                self.mr_weighting = SimpleNamespace(value="T1ce")

            def load_dicom_series(self, path: str) -> None:
                pass

            def detect_modality(self) -> str:
                return "MRI"

        monkeypatch.setattr(bp, "DICOMLoader", _FakeLoader)
        pipeline = BrainModelPipeline()

        assert pipeline.detect_modality("/fake/path") == "MRI"


class TestSelectModels:
    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        resources = ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=4)

        selected = pipeline.select_models("MRI", resources)

        assert set(selected) == {"brain", "tumor", "vessels"}
        assert selected["brain"] == "totalsegmentator"  # нет GPU 48GB+
        assert selected["vessels"] == "frangi"  # нет GPU

    def test_selects_vista3d_brain_for_large_gpu(self, tmp_path: Path) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        resources = ComputeResources(has_gpu=True, gpu_memory_gb=80.0, has_internet=True, cpu_count=16)

        selected = pipeline.select_models("MRI", resources)

        assert selected["brain"] == "vista3d"

    def test_offline_forces_no_internet_in_selection(self, tmp_path: Path) -> None:
        config = BrainPipelineConfig(offline=True)
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path)
        # даже если фактически интернет есть (has_internet=True передан явно), offline должен
        # переопределить его перед вызовом SmartModelSelector
        resources = ComputeResources(has_gpu=True, gpu_memory_gb=80.0, has_internet=True, cpu_count=16)

        selected = pipeline.select_models("MRI", resources)

        assert selected["brain"] == "totalsegmentator"  # vista3d запрещён offline-логикой
        assert selected["vessels"] == "frangi"  # offline-фоллбэк SmartModelSelector

    def test_explicit_strategy_override_bypasses_selector(self, tmp_path: Path) -> None:
        config = BrainPipelineConfig(
            tumor_model_strategy="totalsegmentator", vessel_model_strategy="totalsegmentator"
        )
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path)
        resources = ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=4)

        selected = pipeline.select_models("CT", resources)

        assert selected["tumor"] == "totalsegmentator"
        assert selected["vessels"] == "totalsegmentator"


class TestDownloadRequiredModels:
    def test_skips_models_without_registry_entry(self, tmp_path: Path) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)

        results = pipeline.download_required_models(["frangi", "vista3d", "threshold+frangi"])

        assert results == {"frangi": True, "vista3d": True, "threshold+frangi": True}

    def test_downloads_registry_backed_strategy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        calls: list[str] = []

        def _fake_download(name: str, *_args: object, **_kwargs: object) -> Path:
            calls.append(name)
            return tmp_path

        monkeypatch.setattr(pipeline.registry, "download_model", _fake_download)

        results = pipeline.download_required_models(["totalsegmentator"])

        assert results == {"totalsegmentator": True}
        assert calls == ["totalsegmentator_total"]

    def test_download_failure_reported_as_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        monkeypatch.setattr(
            pipeline.registry,
            "download_model",
            lambda *a, **kw: (_ for _ in ()).throw(Brain3DError("нет сети")),
        )

        results = pipeline.download_required_models(["totalsegmentator"])

        assert results == {"totalsegmentator": False}


# --------------------------------------------------------------------------- #
# Диспетчеры сегментации (частично реальные, без единого мока)
# --------------------------------------------------------------------------- #


class TestSegmentTumorFallbackChain:
    def test_totalsegmentator_strategy_returns_none_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        volume = _make_tube_volume()

        with caplog.at_level("WARNING"):
            result = pipeline._segment_tumor(volume, (1.0, 1.0, 1.0), "totalsegmentator")  # noqa: SLF001

        assert result is None
        assert any("не сегментирует опухоли" in message for message in caplog.messages)

    def test_vista3d_strategy_falls_back_to_totalsegmentator_then_none(self, tmp_path: Path) -> None:
        """Без единого мока: локального VISTA-3D NIM нет (is_available()->False), затем
        totalsegmentator-ветка тоже возвращает None (реальное ограничение — см. выше)."""
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        volume = _make_tube_volume()

        result = pipeline._segment_tumor(volume, (1.0, 1.0, 1.0), "vista3d")

        assert result is None

    def test_brats_pretrained_falls_back_when_monai_not_installed(self, tmp_path: Path) -> None:
        """Без мока: monai не установлен в этом окружении -> load_tumor_model поднимает
        Brain3DError -> фолбэк на totalsegmentator -> None (см. тест выше)."""
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        volume = _make_tube_volume()

        result = pipeline._segment_tumor(volume, (1.0, 1.0, 1.0), "brats_pretrained")

        assert result is None


class TestSegmentBrainFallbackChain:
    def test_totalsegmentator_failure_has_no_further_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pipelines.brain_pipeline as bp

        class _FailingWrapper:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple) -> dict:
                raise Brain3DError("не установлен")

        monkeypatch.setattr(bp, "TotalSegmentatorWrapper", _FailingWrapper)
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)

        result = pipeline._segment_brain(
            np.zeros((4, 4, 4), dtype=np.float32), (1.0, 1.0, 1.0), "totalsegmentator"
        )

        assert result is None

    def test_vista3d_failure_falls_back_to_totalsegmentator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pipelines.brain_pipeline as bp

        class _FailingVista3D:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def is_available(self) -> bool:
                return False

        class _WorkingTotalSegmentator:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def segment(self, volume: np.ndarray, spacing: tuple) -> dict:
                return {}

            def get_brain_mask(self) -> np.ndarray:
                return np.ones((4, 4, 4), dtype=np.uint8)

        monkeypatch.setattr(bp, "VISTA3DWrapper", _FailingVista3D)
        monkeypatch.setattr(bp, "TotalSegmentatorWrapper", _WorkingTotalSegmentator)
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)

        result = pipeline._segment_brain(np.zeros((4, 4, 4), dtype=np.float32), (1.0, 1.0, 1.0), "vista3d")

        np.testing.assert_array_equal(result, np.ones((4, 4, 4), dtype=np.uint8))


class TestSegmentVesselsReal:
    def test_cpu_only_uses_frangi_for_real(self, tmp_path: Path) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        volume = _make_tube_volume()
        resources = ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=4)

        mask, method, confidence = pipeline._segment_vessels(volume, (1.0, 1.0, 1.0), "MRI", resources)

        assert method == "frangi"
        assert mask is not None and mask.sum() > 0
        assert 0.0 < confidence <= 1.0


# --------------------------------------------------------------------------- #
# process_with_progress: полный оркестрированный прогон с моками на границе
# --------------------------------------------------------------------------- #


def _install_full_pipeline_mocks(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int, int] = (20, 20, 20)
) -> None:
    """Мокает DICOMLoader и методы TotalSegmentatorWrapper.

    Важно: патчим МЕТОДЫ реального класса TotalSegmentatorWrapper (segment/
    get_brain_mask), а не заменяем саму привязку имени класса в pipelines.brain_pipeline.
    load_tumor_model() (models.pretrained_models) строит TotalSegmentatorWrapper через
    СВОЮ собственную ссылку на класс — если подменить имя класса только в
    pipelines.brain_pipeline, isinstance-проверка в _segment_tumor() перестанет
    распознавать реальный объект, возвращённый load_tumor_model(), как
    TotalSegmentatorWrapper (тумор-пайплайн в этих тестах использует
    tumor_model_strategy="totalsegmentator" именно чтобы не тянуть torch/monai).
    """
    import pipelines.brain_pipeline as bp

    volume = _make_tube_volume(shape)

    class _FakeLoader:
        def __init__(self) -> None:
            self.mr_weighting = None

        def load_dicom_series(self, path: str) -> np.ndarray:
            return volume

        def detect_modality(self) -> str:
            return "MRI"

        def get_volume_info(self) -> dict:
            return {"spacing": (1.0, 1.0, 1.0)}

    def _fake_segment(self: object, volume: np.ndarray, spacing: tuple) -> dict:
        return {}

    def _fake_get_brain_mask(self: object) -> np.ndarray:
        return _make_sphere_mask(shape, radius=8)

    monkeypatch.setattr(bp, "DICOMLoader", _FakeLoader)
    monkeypatch.setattr(bp.TotalSegmentatorWrapper, "segment", _fake_segment)
    monkeypatch.setattr(bp.TotalSegmentatorWrapper, "get_brain_mask", _fake_get_brain_mask)


class TestProcessWithProgress:
    def test_happy_path_reports_expected_progress_and_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_full_pipeline_mocks(monkeypatch)
        config = BrainPipelineConfig(output_dir=tmp_path, tumor_model_strategy="totalsegmentator")
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path / "cache")

        progress_events: list[tuple[int, str]] = []
        result = pipeline.process_with_progress(
            "/fake/dicom", progress_callback=lambda p, m: progress_events.append((p, m))
        )

        assert result["status"] == "ok"
        assert [p for p, _ in progress_events] == [0, 10, 20, 50, 80, 100]
        assert result["loading"]["modality"] == "MRI"
        assert result["brain_segmentation"]["found"] is True
        assert result["vessel_segmentation"]["method"] == "frangi"
        assert (tmp_path / "scene.glb").exists()
        assert result["export"]["path"] == str(tmp_path / "scene.glb")

    def test_should_cancel_stops_before_next_stage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Отмена кооперативная: проверяется на ГРАНИЦЕ стадии, поэтому уже начатая
        стадия ("load") успевает завершиться, а следующая ("select_and_download") — нет."""
        _install_full_pipeline_mocks(monkeypatch)
        config = BrainPipelineConfig(output_dir=tmp_path, tumor_model_strategy="totalsegmentator")
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path / "cache")

        result = pipeline.process_with_progress("/fake/dicom", should_cancel=lambda: True)
        # Отмена сразу на первой же проверке (до стадии "load"):
        assert result["status"] == "failed"
        assert "отменена" in result["error"]
        assert "loading" not in result

    def test_should_cancel_after_first_stage_stops_before_second(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_full_pipeline_mocks(monkeypatch)
        config = BrainPipelineConfig(output_dir=tmp_path, tumor_model_strategy="totalsegmentator")
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path / "cache")

        call_count = {"n": 0}

        def _cancel_after_first_check() -> bool:
            call_count["n"] += 1
            return (
                call_count["n"] > 1
            )  # пропускаем проверку перед "load", отменяем перед "select_and_download"

        result = pipeline.process_with_progress("/fake/dicom", should_cancel=_cancel_after_first_check)

        assert result["status"] == "failed"
        assert "отменена" in result["error"]
        assert result["loading"]["status"] == "ok"  # стадия "load" успела завершиться
        assert "model_selection" not in result

    def test_process_is_equivalent_without_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_full_pipeline_mocks(monkeypatch)
        config = BrainPipelineConfig(output_dir=tmp_path, tumor_model_strategy="totalsegmentator")
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path / "cache")

        result = pipeline.process("/fake/dicom")

        assert result["status"] == "ok"

    def test_failure_sets_status_failed_and_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_full_pipeline_mocks(monkeypatch)
        config = BrainPipelineConfig(output_dir=tmp_path, tumor_model_strategy="totalsegmentator")
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path / "cache")

        def _raise_mesh_error(*_args: object, **_kwargs: object):
            raise MeshGenerationError("нет структур")

        monkeypatch.setattr(pipeline.mesh_generator, "create_full_brain_scene", _raise_mesh_error)

        result = pipeline.process("/fake/dicom")

        assert result["status"] == "failed"
        assert "нет структур" in result["error"]

    def test_resume_from_mesh_skips_reloading_dicom(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pipelines.brain_pipeline as bp

        load_call_count = {"n": 0}
        volume = _make_tube_volume()

        class _CountingLoader:
            def __init__(self) -> None:
                self.mr_weighting = None

            def load_dicom_series(self, path: str) -> np.ndarray:
                load_call_count["n"] += 1
                return volume

            def detect_modality(self) -> str:
                return "MRI"

            def get_volume_info(self) -> dict:
                return {"spacing": (1.0, 1.0, 1.0)}

        def _fake_segment(self: object, volume: np.ndarray, spacing: tuple) -> dict:
            return {}

        def _fake_get_brain_mask(self: object) -> np.ndarray:
            return _make_sphere_mask((20, 20, 20), radius=8)

        monkeypatch.setattr(bp, "DICOMLoader", _CountingLoader)
        monkeypatch.setattr(bp.TotalSegmentatorWrapper, "segment", _fake_segment)
        monkeypatch.setattr(bp.TotalSegmentatorWrapper, "get_brain_mask", _fake_get_brain_mask)

        config = BrainPipelineConfig(output_dir=tmp_path, tumor_model_strategy="totalsegmentator")
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path / "cache")

        # Первый прогон намеренно ломается на стадии mesh.
        original_create_full_brain_scene = pipeline.mesh_generator.create_full_brain_scene
        monkeypatch.setattr(
            pipeline.mesh_generator,
            "create_full_brain_scene",
            lambda *a, **kw: (_ for _ in ()).throw(MeshGenerationError("сбой")),
        )
        first_result = pipeline.process("/fake/dicom")
        assert first_result["status"] == "failed"
        assert load_call_count["n"] == 1

        # Чиним mesh-генерацию (возвращаем оригинальный метод) и продолжаем именно со
        # стадии "mesh", не перезагружая DICOM (DICOMLoader/TotalSegmentatorWrapper
        # остаются замоканными теми же патчами выше).
        monkeypatch.setattr(
            pipeline.mesh_generator, "create_full_brain_scene", original_create_full_brain_scene
        )

        second_result = pipeline.process_with_progress("/fake/dicom", resume_from="mesh")

        assert second_result["status"] == "ok"
        assert load_call_count["n"] == 1  # DICOM не перезагружался повторно
        assert "export" in second_result

    def test_resume_from_unknown_stage_raises(self, tmp_path: Path) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)

        with pytest.raises(Brain3DError):
            pipeline.process_with_progress("/fake/dicom", resume_from="not_a_real_stage")

    def test_resume_from_without_prior_state_raises(self, tmp_path: Path) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)

        with pytest.raises(Brain3DError):
            pipeline.process_with_progress("/fake/dicom", resume_from="mesh")


class TestGetModelInfo:
    def test_returns_info_for_selected_models_after_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_full_pipeline_mocks(monkeypatch)
        config = BrainPipelineConfig(output_dir=tmp_path, tumor_model_strategy="totalsegmentator")
        pipeline = BrainModelPipeline(config, model_cache_dir=tmp_path / "cache")

        pipeline.process("/fake/dicom")
        info = pipeline.get_model_info()

        assert set(info) == {"brain", "tumor", "vessels"}
        assert info["brain"]["strategy"] == "totalsegmentator"
        assert info["vessels"]["strategy"] == "frangi"

    def test_empty_before_any_run(self, tmp_path: Path) -> None:
        pipeline = BrainModelPipeline(model_cache_dir=tmp_path)
        assert pipeline.get_model_info() == {}


# --------------------------------------------------------------------------- #
# run_pipeline_cli
# --------------------------------------------------------------------------- #


class TestRunPipelineCli:
    def test_maps_cli_args_to_config_and_returns_process_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        import pipelines.brain_pipeline as bp

        captured_configs: list[BrainPipelineConfig] = []

        class _FakePipeline:
            def __init__(self, config: BrainPipelineConfig) -> None:
                captured_configs.append(config)

            def process(self, dicom_path: str) -> dict:
                return {"status": "ok", "dicom_path": dicom_path}

        monkeypatch.setattr(bp, "BrainModelPipeline", _FakePipeline)

        result = run_pipeline_cli(
            str(tmp_path / "dicom"),
            str(tmp_path / "out"),
            modality="ct",
            tumor_model="monai",
            vessel_model="totalseg",
            fast=True,
            offline=True,
            export_format="stl",
        )

        assert result == {"status": "ok", "dicom_path": str(tmp_path / "dicom")}
        config = captured_configs[0]
        assert config.modality_override == "CT"
        assert config.tumor_model_strategy == "brats_pretrained"
        assert config.vessel_model_strategy == "totalsegmentator"
        assert config.fast_mode is True
        assert config.offline is True
        assert config.export_format == "stl"

        printed = json.loads(capsys.readouterr().out)
        assert printed["status"] == "ok"

    def test_auto_modality_leaves_override_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pipelines.brain_pipeline as bp

        captured_configs: list[BrainPipelineConfig] = []

        class _FakePipeline:
            def __init__(self, config: BrainPipelineConfig) -> None:
                captured_configs.append(config)

            def process(self, dicom_path: str) -> dict:
                return {"status": "ok"}

        monkeypatch.setattr(bp, "BrainModelPipeline", _FakePipeline)

        run_pipeline_cli(str(tmp_path / "dicom"), str(tmp_path / "out"))

        assert captured_configs[0].modality_override is None
