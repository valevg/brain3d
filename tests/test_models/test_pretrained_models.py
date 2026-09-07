"""Тесты models/pretrained_models.py.

Реестр, детектор ресурсов, SmartModelSelector и фильтр Франги проверяются по-настоящему
(без моков — им достаточно numpy/scikit-image/stdlib). Обёртки над тяжёлыми внешними
системами (TotalSegmentator, MONAI bundle, VISTA-3D, torch/MONAI resnet) не устанавливаются
в тестовое окружение — их сетевые/GPU-вызовы монкипатчатся на границе (import внутри
функции), а проверяется собственная логика обёртки: сборка payload, разбор ответа,
преобразование геометрии, обработка ошибок.
"""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.exceptions import (
    Brain3DError,
    ChecksumMismatchError,
    ModelDownloadError,
    ModelNotFoundError,
)
from models import pretrained_models as pm

# --------------------------------------------------------------------------- #
# PretrainedModelRegistry
# --------------------------------------------------------------------------- #


class TestPretrainedModelRegistry:
    def test_list_available_models_returns_full_catalog(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        models = registry.list_available_models()

        assert len(models) == len(pm._MODEL_CATALOG)  # noqa: SLF001
        assert all(isinstance(spec, pm.ModelSpec) for spec in models)

    def test_list_available_models_filters_by_task(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        vessel_models = registry.list_available_models(task="vessel")

        assert vessel_models
        assert all(spec.task == "vessel" for spec in vessel_models)

    def test_get_model_returns_spec(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        spec = registry.get_model("totalsegmentator_total")

        assert spec.name == "totalsegmentator_total"
        assert spec.source == "totalsegmentator"

    def test_get_model_raises_for_unknown_name(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        with pytest.raises(ModelNotFoundError):
            registry.get_model("does_not_exist")

    def test_get_model_raises_for_mismatched_task(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        with pytest.raises(ModelNotFoundError):
            registry.get_model("totalsegmentator_total", task="vessel")

    def test_is_cached_false_for_fresh_cache_dir(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        assert registry.is_cached("totalsegmentator_total") is False

    def test_is_cached_true_after_populating_cache_dir(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)
        spec = registry.get_model("med3d_resnet18")
        cache_path = tmp_path / spec.source / spec.name
        cache_path.mkdir(parents=True)
        (cache_path / "weights.pt").write_bytes(b"fake weights")

        assert registry.is_cached("med3d_resnet18") is True

    def test_list_cached_models(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)
        spec = registry.get_model("med3d_resnet18")
        cache_path = tmp_path / spec.source / spec.name
        cache_path.mkdir(parents=True)
        (cache_path / "weights.pt").write_bytes(b"fake weights")

        cached = registry.list_cached_models()

        assert [s.name for s in cached] == ["med3d_resnet18"]

    def test_download_model_skips_if_already_cached(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)
        spec = registry.get_model("totalsegmentator_total")
        cache_path = tmp_path / spec.source / spec.name
        cache_path.mkdir(parents=True)
        (cache_path / "marker.txt").write_text("already here")

        result_path = registry.download_model("totalsegmentator_total")

        assert result_path == cache_path
        assert (cache_path / "marker.txt").exists()

    def test_download_model_totalsegmentator_is_noop_but_creates_dir(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        result_path = registry.download_model("totalsegmentator_total")

        assert result_path.exists()

    def test_download_model_monai_bundle_calls_bundle_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)
        calls: list[dict] = []

        def _fake_download(name: str, bundle_dir: str) -> None:
            calls.append({"name": name, "bundle_dir": bundle_dir})
            Path(bundle_dir).mkdir(parents=True, exist_ok=True)
            (Path(bundle_dir) / "model.pt").write_bytes(b"fake bundle weights")

        monkeypatch.setattr(
            pm,
            "_download_monai_bundle",
            lambda spec, target_dir: _fake_download(spec.bundle_name, str(target_dir)),
        )

        result_path = registry.download_model("monai_brain_tumor_swinunetr")

        assert calls == [{"name": "brats_mri_segmentation", "bundle_dir": str(result_path)}]
        assert (result_path / "model.pt").exists()

    def test_download_model_raises_for_monai_bundle_without_bundle_name(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        with pytest.raises(ModelDownloadError):
            registry.download_model("monai_brain_vessel")

    def test_download_model_verifies_checksum_when_specified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)
        spec = registry.get_model("totalsegmentator_total")
        weights_content = b"totally real weights"
        correct_sha256 = hashlib.sha256(weights_content).hexdigest()

        patched_spec = pm.ModelSpec(
            name=spec.name,
            source=spec.source,
            task=spec.task,
            description=spec.description,
            modalities=spec.modalities,
            sha256=correct_sha256,
        )
        monkeypatch.setitem(pm._MODEL_CATALOG, "totalsegmentator_total", patched_spec)  # noqa: SLF001

        cache_path = tmp_path / spec.source / spec.name
        cache_path.mkdir(parents=True)
        (cache_path / "weights.pt").write_bytes(weights_content)

        # Не должно бросить исключение — сумма совпадает
        registry.download_model("totalsegmentator_total")

    def test_verify_checksum_raises_on_mismatch(self, tmp_path: Path) -> None:
        weights_file = tmp_path / "weights.pt"
        weights_file.write_bytes(b"actual content")

        with pytest.raises(ChecksumMismatchError):
            pm._verify_checksum(weights_file, expected_sha256="0" * 64)  # noqa: SLF001

    def test_verify_checksum_passes_on_match(self, tmp_path: Path) -> None:
        content = b"actual content"
        weights_file = tmp_path / "weights.pt"
        weights_file.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()

        pm._verify_checksum(weights_file, expected_sha256=expected)  # noqa: SLF001 — не должно бросить


# --------------------------------------------------------------------------- #
# is_online
# --------------------------------------------------------------------------- #


class TestIsOnline:
    def test_returns_true_when_connection_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeConnection:
            def __enter__(self) -> "_FakeConnection":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        monkeypatch.setattr(pm.socket, "create_connection", lambda *a, **kw: _FakeConnection())

        assert pm.is_online() is True

    def test_returns_false_on_socket_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_args: object, **_kwargs: object):
            raise socket.timeout("no route")

        monkeypatch.setattr(pm.socket, "create_connection", _raise)

        assert pm.is_online() is False


# --------------------------------------------------------------------------- #
# TotalSegmentatorWrapper
# --------------------------------------------------------------------------- #


class TestTotalSegmentatorWrapper:
    def test_init_warns_when_roi_subset_used_with_fast(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            pm.TotalSegmentatorWrapper(fast=True, roi_subset=["brain"])

        assert any("roi_subset" in message for message in caplog.messages)

    def test_segment_raises_when_package_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object):
            if name == "totalsegmentator.python_api":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        wrapper = pm.TotalSegmentatorWrapper()
        with pytest.raises(Brain3DError, match="totalsegmentator не установлен"):
            wrapper.segment(np.zeros((4, 4, 4), dtype=np.float32), spacing=(1.0, 1.0, 1.0))

    def test_get_brain_mask_requires_segment_first(self) -> None:
        wrapper = pm.TotalSegmentatorWrapper()
        with pytest.raises(Brain3DError, match="Сначала вызовите segment"):
            wrapper.get_brain_mask()

    def test_get_brain_mask_and_bone_and_vessel_masks_after_fake_segment(self) -> None:
        wrapper = pm.TotalSegmentatorWrapper(task="total")
        wrapper._last_masks = {  # noqa: SLF001 — напрямую заполняем результат, минуя реальный вызов
            "brain": np.ones((2, 2, 2), dtype=np.uint8),
            "skull": np.ones((2, 2, 2), dtype=np.uint8),
            "vertebrae_L1": np.ones((2, 2, 2), dtype=np.uint8),
            "aorta": np.ones((2, 2, 2), dtype=np.uint8),
            "liver": np.zeros((2, 2, 2), dtype=np.uint8),
        }

        brain_mask = wrapper.get_brain_mask()
        bone_masks = wrapper.get_bone_masks()
        vessel_masks = wrapper.get_vessel_masks()

        np.testing.assert_array_equal(brain_mask, np.ones((2, 2, 2), dtype=np.uint8))
        assert set(bone_masks) == {"skull", "vertebrae_L1"}
        assert set(vessel_masks) == {"aorta"}
        assert "liver" not in bone_masks and "liver" not in vessel_masks

    def test_get_vessel_masks_uses_headneck_labels_for_that_task(self) -> None:
        wrapper = pm.TotalSegmentatorWrapper(task="headneck_bones_vessels")
        wrapper._last_masks = {  # noqa: SLF001
            "internal_carotid_artery_left": np.ones((2, 2, 2), dtype=np.uint8),
            "aorta": np.ones((2, 2, 2), dtype=np.uint8),  # не относится к headneck-списку
        }

        vessel_masks = wrapper.get_vessel_masks()

        assert set(vessel_masks) == {"internal_carotid_artery_left"}

    def test_segment_end_to_end_with_faked_totalsegmentator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Полный проход segment(): реальный nibabel round-trip, totalsegmentator замокан."""
        nib = pytest.importorskip("nibabel")

        def _fake_totalsegmentator(input, output, task, fast, roi_subset, device, ml, quiet):  # noqa: ANN001
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            source_image = nib.load(str(input))
            shape = source_image.shape
            mask_data = np.ones(shape, dtype=np.uint8)
            nib.save(nib.Nifti1Image(mask_data, source_image.affine), str(output_dir / "brain.nii.gz"))

        fake_module = SimpleNamespace(totalsegmentator=_fake_totalsegmentator)
        monkeypatch.setitem(__import__("sys").modules, "totalsegmentator.python_api", fake_module)
        monkeypatch.setitem(
            __import__("sys").modules, "totalsegmentator", SimpleNamespace(python_api=fake_module)
        )

        wrapper = pm.TotalSegmentatorWrapper(task="total")
        volume = np.random.rand(4, 6, 6).astype(np.float32)

        masks = wrapper.segment(volume, spacing=(1.0, 1.0, 1.0))

        assert set(masks) == {"brain"}
        assert masks["brain"].shape == volume.shape
        np.testing.assert_array_equal(masks["brain"], np.ones_like(volume, dtype=np.uint8))
        # get_brain_mask() должен теперь тоже работать без повторного вызова segment()
        np.testing.assert_array_equal(wrapper.get_brain_mask(), masks["brain"])


class TestAffineFromSpacing:
    def test_diagonal_matches_spacing_with_lps_to_ras_flip(self) -> None:
        affine = pm._affine_from_spacing((0.5, 1.0, 2.0))  # noqa: SLF001

        expected = np.diag([-0.5, -1.0, 2.0, 1.0])
        np.testing.assert_array_equal(affine, expected)


# --------------------------------------------------------------------------- #
# VISTA3DWrapper
# --------------------------------------------------------------------------- #


class TestVISTA3DWrapper:
    def test_cloud_mode_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        with pytest.raises(Brain3DError, match="api_key"):
            pm.VISTA3DWrapper(mode="cloud")

    def test_cloud_mode_accepts_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "secret-token")

        wrapper = pm.VISTA3DWrapper(mode="cloud")

        assert wrapper.api_key == "secret-token"

    def test_is_available_returns_false_on_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests = pytest.importorskip("requests")

        def _raise(*_args: object, **_kwargs: object):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", _raise)

        wrapper = pm.VISTA3DWrapper(mode="local")
        assert wrapper.is_available() is False

    def test_is_available_returns_true_on_ok_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests = pytest.importorskip("requests")

        monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(ok=True))

        wrapper = pm.VISTA3DWrapper(mode="local")
        assert wrapper.is_available() is True

    def test_segment_raises_brain3d_error_on_connection_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = pytest.importorskip("requests")

        def _raise(*_args: object, **_kwargs: object):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", _raise)

        wrapper = pm.VISTA3DWrapper(mode="local")
        with pytest.raises(Brain3DError, match="VISTA-3D недоступен"):
            wrapper.segment(np.zeros((2, 2, 2), dtype=np.float32), spacing=(1.0, 1.0, 1.0))

    def test_segment_parses_mocked_successful_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests = pytest.importorskip("requests")
        shape = (2, 3, 3)
        expected_mask = np.zeros(shape, dtype=np.uint8)
        expected_mask[0, 0, 0] = 1

        import base64

        fake_response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"masks": {"brain": base64.b64encode(expected_mask.tobytes()).decode("ascii")}},
        )
        monkeypatch.setattr(requests, "post", lambda *a, **kw: fake_response)

        wrapper = pm.VISTA3DWrapper(mode="local")
        masks = wrapper.segment(np.zeros(shape, dtype=np.float32), spacing=(1.0, 1.0, 1.0))

        np.testing.assert_array_equal(masks["brain"], expected_mask)

    def test_interactive_segmentation_includes_prompts_in_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = pytest.importorskip("requests")
        captured_payload: dict = {}

        def _fake_post(url: str, json: dict, headers: dict, timeout: float):  # noqa: ANN001
            captured_payload.update(json)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"masks": {}})

        monkeypatch.setattr(requests, "post", _fake_post)

        wrapper = pm.VISTA3DWrapper(mode="local")
        wrapper.interactive_segmentation(
            np.zeros((2, 2, 2), dtype=np.float32), points=[(1, 1, 1)], point_labels=[1]
        )

        assert captured_payload["prompts"] == {"points": [[1, 1, 1]], "point_labels": [1]}


# --------------------------------------------------------------------------- #
# Med3DLoader
# --------------------------------------------------------------------------- #


class TestMed3DLoader:
    def test_rejects_unsupported_depth(self) -> None:
        with pytest.raises(Brain3DError, match="не поддерживается"):
            pm.Med3DLoader(depth=101)

    def test_load_pretrained_encoder_raises_if_monai_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object):
            if name == "monai.networks.nets":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        loader = pm.Med3DLoader(depth=18)
        with pytest.raises(Brain3DError, match="monai не установлен"):
            loader.load_pretrained_encoder()


# --------------------------------------------------------------------------- #
# ComputeResources / SmartModelSelector
# --------------------------------------------------------------------------- #


class TestSmartModelSelector:
    def test_chooses_vista3d_for_large_gpu(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=True, gpu_memory_gb=80.0, has_internet=True, cpu_count=32)

        assert selector.choose_best_model("MRI", "tumor", resources) == "vista3d"

    def test_chooses_brats_pretrained_for_midrange_gpu_tumor_task(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=True, gpu_memory_gb=16.0, has_internet=True, cpu_count=8)

        assert selector.choose_best_model("MRI", "tumor", resources) == "brats_pretrained"

    def test_chooses_totalsegmentator_for_midrange_gpu_nontumor_task(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=True, gpu_memory_gb=16.0, has_internet=True, cpu_count=8)

        assert selector.choose_best_model("CT", "whole_body", resources) == "totalsegmentator"

    def test_chooses_frangi_for_cpu_only_vessel_task(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=8)

        assert selector.choose_best_model("MRI", "vessel", resources) == "frangi"

    def test_chooses_totalsegmentator_for_cpu_only_nonvessel_task(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=8)

        assert selector.choose_best_model("CT", "tumor", resources) == "totalsegmentator"

    def test_offline_prefers_frangi_for_vessel_task(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=True, gpu_memory_gb=80.0, has_internet=False, cpu_count=8)

        assert selector.choose_best_model("MRI", "vessel", resources) == "frangi"

    def test_offline_uses_cached_model_when_available(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)
        spec = registry.get_model("monai_brain_tumor_swinunetr")
        cache_path = tmp_path / spec.source / spec.name
        cache_path.mkdir(parents=True)
        (cache_path / "model.pt").write_bytes(b"cached")

        selector = pm.SmartModelSelector(registry)
        resources = pm.ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=False, cpu_count=8)

        assert selector.choose_best_model("MRI", "tumor", resources) == "monai_brain_tumor_swinunetr"

    def test_offline_falls_back_to_totalsegmentator_without_cache(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=False, cpu_count=8)

        assert selector.choose_best_model("MRI", "tumor", resources) == "totalsegmentator"

    def test_should_use_fast_mode_true_without_gpu(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=False, gpu_memory_gb=None, has_internet=True, cpu_count=8)

        assert selector.should_use_fast_mode(resources) is True

    def test_should_use_fast_mode_false_with_gpu(self, tmp_path: Path) -> None:
        selector = pm.SmartModelSelector(pm.PretrainedModelRegistry(cache_dir=tmp_path))
        resources = pm.ComputeResources(has_gpu=True, gpu_memory_gb=24.0, has_internet=True, cpu_count=8)

        assert selector.should_use_fast_mode(resources) is False


class TestDetectComputeResources:
    def test_returns_cpu_only_when_torch_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        original_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object):
            if name == "torch":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        monkeypatch.setattr(pm, "is_online", lambda: False)

        resources = pm.detect_compute_resources()

        assert resources.has_gpu is False
        assert resources.gpu_memory_gb is None
        assert resources.has_internet is False
        assert resources.cpu_count >= 1

    def test_check_internet_false_skips_online_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _should_not_be_called() -> bool:
            raise AssertionError("is_online не должен вызываться при check_internet=False")

        monkeypatch.setattr(pm, "is_online", _should_not_be_called)

        resources = pm.detect_compute_resources(check_internet=False)

        assert resources.has_internet is False


# --------------------------------------------------------------------------- #
# load_tumor_model / load_vessel_model
# --------------------------------------------------------------------------- #


class TestLoadTumorModel:
    def test_totalsegmentator_strategy_returns_wrapper(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        model = pm.load_tumor_model(strategy="totalsegmentator", registry=registry)

        assert isinstance(model, pm.TotalSegmentatorWrapper)
        assert model.task == "total"

    def test_vista3d_strategy_returns_wrapper(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        model = pm.load_tumor_model(strategy="vista3d", registry=registry)

        assert isinstance(model, pm.VISTA3DWrapper)
        assert model.mode == "local"

    def test_unknown_strategy_raises(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        with pytest.raises(Brain3DError, match="Неизвестная стратегия"):
            pm.load_tumor_model(strategy="nonsense", registry=registry)

    def test_auto_strategy_delegates_to_selector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)
        monkeypatch.setattr(
            pm.SmartModelSelector,
            "choose_best_model",
            lambda self, modality, task, available_resources=None: "totalsegmentator",
        )

        model = pm.load_tumor_model(strategy="auto", registry=registry)

        assert isinstance(model, pm.TotalSegmentatorWrapper)


class TestLoadVesselModel:
    def test_frangi_strategy_returns_callable(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        segmenter = pm.load_vessel_model(strategy="frangi", registry=registry)

        assert segmenter is pm.segment_vessels_frangi

    def test_totalsegmentator_strategy_uses_headneck_task(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        model = pm.load_vessel_model(strategy="totalsegmentator", registry=registry)

        assert isinstance(model, pm.TotalSegmentatorWrapper)
        assert model.task == "headneck_bones_vessels"

    def test_unknown_strategy_raises(self, tmp_path: Path) -> None:
        registry = pm.PretrainedModelRegistry(cache_dir=tmp_path)

        with pytest.raises(Brain3DError, match="Неизвестная стратегия"):
            pm.load_vessel_model(strategy="nonsense", registry=registry)


# --------------------------------------------------------------------------- #
# segment_vessels_frangi — реальный, без моков
# --------------------------------------------------------------------------- #


class TestSegmentVesselsFrangi:
    def test_detects_tube_like_structure(self) -> None:
        shape = (30, 30, 30)
        volume = np.zeros(shape, dtype=np.float32)
        # цилиндрическая "трубка" радиусом 2 вокселя вдоль оси Z — Франги реагирует именно
        # на структуры, тонкие в ДВУХ измерениях и вытянутые в третьем (в отличие от плоской
        # "пластины", тонкой лишь в одном измерении, которую фильтр должен подавлять).
        y, x = np.ogrid[: shape[1], : shape[2]]
        tube_mask = (y - 15) ** 2 + (x - 15) ** 2 <= 2**2
        volume[:, tube_mask] = 1.0

        mask = pm.segment_vessels_frangi(volume, spacing=(1.0, 1.0, 1.0), sigmas=(1.0, 2.0))

        assert mask.dtype == np.uint8
        assert mask.shape == volume.shape
        # хотя бы часть трубки должна быть распознана как сосудистая структура
        assert mask[:, tube_mask].sum() > 0

    def test_returns_all_zero_for_flat_volume(self) -> None:
        volume = np.zeros((3, 10, 10), dtype=np.float32)

        mask = pm.segment_vessels_frangi(volume)

        assert mask.sum() == 0

    def test_custom_threshold_is_respected(self) -> None:
        volume = np.zeros((3, 20, 20), dtype=np.float32)
        volume[:, :, 9:11] = 1.0

        mask_low_threshold = pm.segment_vessels_frangi(volume, threshold=0.0)
        mask_high_threshold = pm.segment_vessels_frangi(volume, threshold=10.0)

        assert mask_low_threshold.sum() >= mask_high_threshold.sum()
        assert mask_high_threshold.sum() == 0
