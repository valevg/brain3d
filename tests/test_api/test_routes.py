"""Тесты api/routes.py + api/fastapi_app.py.

Каждый тест получает изолированное окружение (tmp_path через переменные окружения
STORAGE_DIR/MODEL_CACHE_DIR + сброс lru_cache зависимостей и rate-limiter'а) —
тесты не делят реальный ~/.cache/brain3d или ./data между собой и с остальным проектом.

BackgroundTasks в FastAPI TestClient выполняются СИНХРОННО до возврата ответа —
поэтому после client.post(...) фоновая задача (скачивание модели, запуск пайплайна)
уже гарантированно завершена, что делает эти тесты детерминированными без опроса.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.exceptions import Brain3DError


def _make_tube_volume(shape: tuple[int, int, int] = (16, 16, 16), radius: int = 2) -> np.ndarray:
    volume = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[: shape[1], : shape[2]]
    tube_mask = (y - shape[1] // 2) ** 2 + (x - shape[2] // 2) ** 2 <= radius**2
    volume[:, tube_mask] = 1.0
    return volume


def _make_sphere_mask(shape: tuple[int, int, int] = (16, 16, 16), radius: int = 6) -> np.ndarray:
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = np.zeros(shape, dtype=np.uint8)
    mask[(z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= radius**2] = 1
    return mask


def _make_tiny_nifti(directory: Path, filename: str = "scan.nii.gz") -> Path:
    """Создаёт крошечный, но валидный NIfTI-файл — для реального прохода /studies/upload."""
    nib = pytest.importorskip("nibabel")
    volume = np.random.rand(4, 4, 4).astype(np.float32)
    path = directory / filename
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(path))
    return path


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Свежее FastAPI-приложение с изолированным storage_dir/model_cache_dir и сброшенным rate-limiter'ом."""
    from api.dependencies import get_cache_manager, get_model_registry, limiter
    from core.config import get_settings

    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "model_cache"))
    monkeypatch.setenv("UPLOAD_MAX_SIZE_MB", "10")

    get_settings.cache_clear()
    get_model_registry.cache_clear()
    get_cache_manager.cache_clear()
    limiter.reset()

    from api.fastapi_app import create_app

    with TestClient(create_app()) as client:
        yield client

    get_settings.cache_clear()
    get_model_registry.cache_clear()
    get_cache_manager.cache_clear()


def _install_pipeline_mocks(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int, int] = (16, 16, 16)
) -> None:
    """Мокает методы DICOMLoader/TotalSegmentatorWrapper НА КЛАССЕ (см. test_brain_pipeline.py:
    подмена самого класса ломает isinstance-проверку в BrainModelPipeline._segment_tumor)."""
    import pipelines.brain_pipeline as bp

    volume = _make_tube_volume(shape)

    def _fake_load_dicom_series(self, path):  # noqa: ANN001, ANN201
        return volume

    def _fake_detect_modality(self):  # noqa: ANN001
        return "MRI"

    def _fake_get_volume_info(self):  # noqa: ANN001
        return {"spacing": (1.0, 1.0, 1.0)}

    def _fake_segment(self, volume, spacing):  # noqa: ANN001
        return {}

    def _fake_get_brain_mask(self):  # noqa: ANN001
        return _make_sphere_mask(shape, radius=6)

    monkeypatch.setattr(bp.DICOMLoader, "load_dicom_series", _fake_load_dicom_series)
    monkeypatch.setattr(bp.DICOMLoader, "detect_modality", _fake_detect_modality)
    monkeypatch.setattr(bp.DICOMLoader, "get_volume_info", _fake_get_volume_info)
    monkeypatch.setattr(bp.TotalSegmentatorWrapper, "segment", _fake_segment)
    monkeypatch.setattr(bp.TotalSegmentatorWrapper, "get_brain_mask", _fake_get_brain_mask)


# --------------------------------------------------------------------------- #
# Health / upload (перенесены из старого api/routers, проверяем, что не сломались)
# --------------------------------------------------------------------------- #


class TestHealthAndUpload:
    def test_health_check(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_upload_study_returns_id_and_modality(self, api_client: TestClient, tmp_path: Path) -> None:
        nifti_path = _make_tiny_nifti(tmp_path)

        with nifti_path.open("rb") as handle:
            response = api_client.post(
                "/studies/upload", files={"file": ("scan.nii.gz", handle, "application/octet-stream")}
            )

        assert response.status_code == 201
        body = response.json()
        assert "study_id" in body
        assert body["modality"] == "MRI"

    def test_upload_rejects_oversized_file(self, api_client: TestClient) -> None:
        oversized_content = b"0" * (11 * 1024 * 1024)  # лимит теста — 10 МБ

        response = api_client.post(
            "/studies/upload", files={"file": ("big.nii", oversized_content, "application/octet-stream")}
        )

        assert response.status_code == 413


# --------------------------------------------------------------------------- #
# GET /api/models/available
# --------------------------------------------------------------------------- #


class TestListAvailableModels:
    def test_returns_all_catalog_models_with_expected_statuses(self, api_client: TestClient) -> None:
        response = api_client.get("/api/models/available")

        assert response.status_code == 200
        models = {m["name"]: m for m in response.json()["models"]}

        assert models["vista3d"]["status"] == "api_available"
        assert models["vista3d"]["size_gb"] == 0.0
        assert models["totalsegmentator_total"]["status"] == "not_downloaded"
        assert models["totalsegmentator_total"]["size_gb"] == pytest.approx(2.5)
        assert models["totalsegmentator_total"]["use_count"] == 0
        assert models["totalsegmentator_total"]["last_used"] is None

    def test_reflects_cached_model_as_loaded_with_real_size(self, api_client: TestClient) -> None:
        from api.dependencies import get_model_registry

        registry = get_model_registry()
        spec = registry.get_model("med3d_resnet18")
        cache_path = registry.cache_dir / spec.source / spec.name
        cache_path.mkdir(parents=True)
        # 10 МБ — не 2КБ: /api/models/available округляет size_gb до 3 знаков,
        # и доля ГБ для файла в пару килобайт после округления неотличима от 0.0.
        file_size_bytes = 10 * 1024 * 1024
        (cache_path / "weights.pt").write_bytes(b"0" * file_size_bytes)

        response = api_client.get("/api/models/available")
        model = next(m for m in response.json()["models"] if m["name"] == "med3d_resnet18")

        assert model["status"] == "loaded"
        assert model["size_gb"] == round(file_size_bytes / 1024**3, 3)


# --------------------------------------------------------------------------- #
# POST /api/models/download
# --------------------------------------------------------------------------- #


class TestDownloadModel:
    def test_unknown_model_returns_404(self, api_client: TestClient) -> None:
        response = api_client.post("/api/models/download", json={"model_name": "does_not_exist"})
        assert response.status_code == 404

    def test_successful_download_updates_status_to_loaded(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from api.dependencies import get_model_registry

        registry = get_model_registry()

        def _fake_download(name: str, **kwargs: object) -> Path:
            # Мок должен реально населять кэш файлом — иначе registry.is_cached()
            # (используемый /api/models/available для статуса "loaded") останется False,
            # т.к. он проверяет наличие файлов на диске, а не факт вызова download_model().
            spec = registry.get_model(name)
            path = registry.cache_dir / spec.source / spec.name
            path.mkdir(parents=True, exist_ok=True)
            (path / "weights.pt").write_bytes(b"0" * 1024)
            return path

        monkeypatch.setattr(registry, "download_model", _fake_download)

        response = api_client.post("/api/models/download", json={"model_name": "totalsegmentator_total"})

        assert response.status_code == 202
        body = response.json()
        assert "task_id" in body
        assert body["websocket_url"] == f"/ws/progress/{body['task_id']}"

        listing = api_client.get("/api/models/available").json()["models"]
        model = next(m for m in listing if m["name"] == "totalsegmentator_total")
        assert model["status"] == "loaded"

    def test_failed_download_sets_status_to_error(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from api.dependencies import get_model_registry

        registry = get_model_registry()

        def _always_fail(name: str, **kwargs: object):  # noqa: ANN001, ANN201
            raise Brain3DError("нет сети")

        monkeypatch.setattr(registry, "download_model", _always_fail)

        response = api_client.post("/api/models/download", json={"model_name": "med3d_resnet18"})
        assert response.status_code == 202

        listing = api_client.get("/api/models/available").json()["models"]
        model = next(m for m in listing if m["name"] == "med3d_resnet18")
        assert model["status"] == "error"


# --------------------------------------------------------------------------- #
# POST /api/detect-resources
# --------------------------------------------------------------------------- #


class TestDetectResources:
    def test_cpu_only_offline_recommends_frangi_for_vessels(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pipelines.brain_pipeline as bp

        monkeypatch.setattr(
            bp.ResourceDetector,
            "get_optimal_config",
            lambda self: {
                "gpu": {"available": False, "memory_gb": None, "name": None},
                "ram_gb": 16.0,
                "disk_free_gb": 50.0,
                "has_internet": True,
                "recommended_fast_mode": True,
                "recommended_offline_mode": False,
            },
        )

        response = api_client.post("/api/detect-resources")

        assert response.status_code == 200
        body = response.json()
        assert body["gpu_available"] is False
        assert body["ram_gb"] == 16.0
        assert "frangi" in body["recommended_models"]

    def test_large_gpu_recommends_vista3d(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pipelines.brain_pipeline as bp

        monkeypatch.setattr(
            bp.ResourceDetector,
            "get_optimal_config",
            lambda self: {
                "gpu": {"available": True, "memory_gb": 80.0, "name": "A100"},
                "ram_gb": 128.0,
                "disk_free_gb": 500.0,
                "has_internet": True,
                "recommended_fast_mode": False,
                "recommended_offline_mode": False,
            },
        )

        response = api_client.post("/api/detect-resources")
        body = response.json()

        assert body["gpu_model"] == "A100"
        assert "vista3d" in body["recommended_models"]


# --------------------------------------------------------------------------- #
# POST /api/process/{study_id} + статус + скачивание результата
# --------------------------------------------------------------------------- #


class TestProcessPipeline:
    def test_process_unknown_study_returns_404(self, api_client: TestClient) -> None:
        response = api_client.post("/api/process/does-not-exist", json={})
        assert response.status_code == 404

    def test_full_flow_upload_process_status_download(
        self, api_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_pipeline_mocks(monkeypatch)

        nifti_path = _make_tiny_nifti(tmp_path)
        with nifti_path.open("rb") as handle:
            upload_response = api_client.post(
                "/studies/upload", files={"file": ("scan.nii.gz", handle, "application/octet-stream")}
            )
        study_id = upload_response.json()["study_id"]

        process_response = api_client.post(f"/api/process/{study_id}", json={"export_format": "glb"})
        assert process_response.status_code == 202
        task_id = process_response.json()["task_id"]

        status_response = api_client.get(f"/api/pipeline/status/{task_id}")
        assert status_response.status_code == 200
        status_body = status_response.json()
        assert status_body["status"] == "done"
        assert status_body["progress_percent"] == 100
        assert status_body["selected_models"]["brain"] == "totalsegmentator"
        assert status_body["result"]["status"] == "ok"

        download_response = api_client.get(f"/api/process/{task_id}/download")
        assert download_response.status_code == 200
        assert len(download_response.content) > 0

    def test_status_unknown_task_returns_404(self, api_client: TestClient) -> None:
        response = api_client.get("/api/pipeline/status/does-not-exist")
        assert response.status_code == 404

    def test_download_before_ready_returns_404(self, api_client: TestClient) -> None:
        # Не запускаем /api/process — задачи с таким task_id ещё не существует.
        response = api_client.get("/api/process/nonexistent-task/download")
        assert response.status_code == 404


# --------------------------------------------------------------------------- #
# WebSocket /ws/progress/{task_id}
# --------------------------------------------------------------------------- #


class TestWebSocketProgress:
    def test_reports_events_and_closes_on_completion(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from api.dependencies import get_model_registry

        registry = get_model_registry()
        monkeypatch.setattr(registry, "download_model", lambda name, **kw: registry.cache_dir / name)

        response = api_client.post("/api/models/download", json={"model_name": "med3d_resnet18"})
        task_id = response.json()["task_id"]

        with api_client.websocket_connect(f"/ws/progress/{task_id}") as websocket:
            messages = []
            for _ in range(10):
                data = websocket.receive_json()
                messages.append(data)
                if data.get("status") in ("done", "failed"):
                    break

        assert any("Начало загрузки" in m["message"] for m in messages)
        assert messages[-1]["status"] == "done"


# --------------------------------------------------------------------------- #
# GET /api/cache/info, POST /api/cache/clear
# --------------------------------------------------------------------------- #


class TestCacheEndpoints:
    def test_cache_info_reports_populated_cache(self, api_client: TestClient) -> None:
        from api.dependencies import get_model_registry

        registry = get_model_registry()
        spec = registry.get_model("med3d_resnet18")
        cache_path = registry.cache_dir / spec.source / spec.name
        cache_path.mkdir(parents=True)
        # 10 МБ — не 4КБ: ответ округляет *_gb до 3 знаков, доля ГБ для пары
        # килобайт после округления неотличима от 0.0 (см. TestListAvailableModels).
        file_size_bytes = 10 * 1024 * 1024
        (cache_path / "weights.pt").write_bytes(b"0" * file_size_bytes)

        response = api_client.get("/api/cache/info")

        assert response.status_code == 200
        body = response.json()
        assert body["models"][0]["name"] == "med3d_resnet18"
        assert body["total_size_gb"] == round(file_size_bytes / 1024**3, 3)

    def test_cache_clear_removes_everything(self, api_client: TestClient) -> None:
        from api.dependencies import get_model_registry

        registry = get_model_registry()
        spec = registry.get_model("med3d_resnet18")
        cache_path = registry.cache_dir / spec.source / spec.name
        cache_path.mkdir(parents=True)
        file_size_bytes = 10 * 1024 * 1024
        (cache_path / "weights.pt").write_bytes(b"0" * file_size_bytes)

        clear_response = api_client.post("/api/cache/clear")

        assert clear_response.status_code == 200
        assert clear_response.json()["freed_gb"] == round(file_size_bytes / 1024**3, 3)
        assert api_client.get("/api/cache/info").json()["models"] == []


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


class TestRateLimiting:
    def test_cache_clear_is_rate_limited(self, api_client: TestClient) -> None:
        for _ in range(3):
            response = api_client.post("/api/cache/clear")
            assert response.status_code == 200

        limited_response = api_client.post("/api/cache/clear")
        assert limited_response.status_code == 429
