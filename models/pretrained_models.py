"""Интеграция готовых предобученных моделей вместо обучения с нуля.

Brain3D AI не обучает нейросети самостоятельно — вместо этого проект подбирает,
загружает и адаптирует уже обученные модели из открытых источников:

    - TotalSegmentator — главная модель, 104+ анатомические структуры (мозг, кости,
      часть сосудов), https://github.com/wasserth/TotalSegmentator;
    - MONAI Model Zoo — специализированные bundle-модели (например, Swin UNETR на BraTS),
      https://github.com/Project-MONAI/model-zoo;
    - NVIDIA VISTA-3D (NIM) — универсальная модель на 127+ классов, локально (Docker,
      нужен GPU ~48GB) или через облачный API;
    - Med3D/MedicalNet-style 3D-энкодеры — через встроенную поддержку pretrained-весов
      в monai.networks.nets.resnetXX, полезно для fine-tuning;
    - классический фильтр Франги (skimage) — запасной вариант для сосудов без GPU/сети.

SmartModelSelector и функции load_tumor_model()/load_vessel_model() скрывают этот
выбор за единым интерфейсом, подбирая конкретную модель под задачу, модальность
и доступные ресурсы (GPU/CPU/интернет).

Тяжёлые зависимости (torch, monai, totalsegmentator, requests) импортируются лениво,
внутри конкретных методов — это позволяет использовать реестр моделей, детектор
ресурсов и фильтр Франги вообще без установки torch/monai/totalsegmentator.
"""

from __future__ import annotations

import hashlib
import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from skimage.filters import frangi, threshold_otsu

from core.exceptions import (
    Brain3DError,
    ChecksumMismatchError,
    ModelDownloadError,
    ModelNotFoundError,
)
from core.logging_config import get_logger

logger = get_logger(__name__)

#: Локальный кэш весов Brain3D AI (метаданные реестра; сами тяжёлые библиотеки часто
#: держат собственный кэш рядом — например, TotalSegmentator использует ~/.totalsegmentator).
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "brain3d"


# --------------------------------------------------------------------------- #
# Реестр моделей
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class ModelSpec:
    """Метаданные одной модели реестра.

    Attributes:
        name: Уникальное имя модели в реестре.
        source: Источник — "totalsegmentator" | "monai_bundle" | "vista3d" | "med3d".
        task: Задача — "tumor" | "vessel" | "hemorrhage" | "whole_body" | "feature_extraction".
        description: Короткое описание для list_available_models().
        modalities: Модальности, для которых модель применима.
        min_gpu_memory_gb: Минимальный объём VRAM для комфортной работы; None — работает на CPU.
        bundle_name: Имя bundle в MONAI Model Zoo (только для source="monai_bundle").
        sha256: Ожидаемая контрольная сумма файла весов, если применимо и известно заранее.
        expected_size_gb: Приблизительный ожидаемый размер закэшированных весов, ГБ —
            для отображения в UI ДО скачивания (см. api.routes: GET /api/models/available).
            Оценка на глаз по публично известным размерам дистрибутивов; None — размер
            неизвестен заранее (например, "monai_brain_vessel" без реального bundle) или
            не применим (vista3d — облачный API/Docker-образ, веса не скачиваются нами).
    """

    name: str
    source: Literal["totalsegmentator", "monai_bundle", "vista3d", "med3d"]
    task: str
    description: str
    modalities: tuple[str, ...] = ("CT", "MRI")
    min_gpu_memory_gb: float | None = None
    bundle_name: str | None = None
    sha256: str | None = None
    expected_size_gb: float | None = None


_MODEL_CATALOG: dict[str, ModelSpec] = {
    "totalsegmentator_total": ModelSpec(
        name="totalsegmentator_total",
        source="totalsegmentator",
        task="whole_body",
        description=(
            "TotalSegmentator, задача 'total': 104+ структуры (органы, кости, часть "
            "сосудов туловища, мозг целиком)."
        ),
        modalities=("CT",),
        expected_size_gb=2.5,
    ),
    "totalsegmentator_total_mr": ModelSpec(
        name="totalsegmentator_total_mr",
        source="totalsegmentator",
        task="whole_body",
        description="TotalSegmentator, задача 'total_mr': аналог 'total' для МРТ.",
        modalities=("MRI",),
        expected_size_gb=2.5,
    ),
    "totalsegmentator_headneck_vessels": ModelSpec(
        name="totalsegmentator_headneck_vessels",
        source="totalsegmentator",
        task="vessel",
        description=(
            "TotalSegmentator, задача 'headneck_bones_vessels': сосуды и кости головы/шеи "
            "(сонные, позвоночные артерии — ближе к Виллизиеву кругу, чем список из 'total')."
        ),
        modalities=("CT",),
        expected_size_gb=1.0,
    ),
    "totalsegmentator_cerebral_bleed": ModelSpec(
        name="totalsegmentator_cerebral_bleed",
        source="totalsegmentator",
        task="hemorrhage",
        description=(
            "TotalSegmentator, задача 'cerebral_bleed': детекция внутричерепного "
            "кровоизлияния — задел под плановую структуру Brain3D AI."
        ),
        modalities=("CT",),
        expected_size_gb=0.5,
    ),
    "monai_brain_tumor_swinunetr": ModelSpec(
        name="monai_brain_tumor_swinunetr",
        source="monai_bundle",
        task="tumor",
        description=(
            "Swin UNETR, обучен на BraTS (вход T1/T1ce/T2/FLAIR, 3 подобласти опухоли). "
            "Также используется как мультимодальная модель — её вход уже 4-канальный."
        ),
        modalities=("MRI",),
        min_gpu_memory_gb=8.0,
        bundle_name="brats_mri_segmentation",
        expected_size_gb=1.2,
    ),
    "monai_brain_vessel": ModelSpec(
        name="monai_brain_vessel",
        source="monai_bundle",
        task="vessel",
        description=(
            "Заглушка под сосудистый bundle мозга: на момент написания в MONAI Model Zoo "
            "нет выделенного bundle именно под сосуды мозга/Виллизиев круг. Используйте "
            "totalsegmentator_headneck_vessels или Frangi (load_vessel_model). Оставлено "
            "для единообразия интерфейса и на случай появления такого bundle."
        ),
        modalities=("MRI",),
        min_gpu_memory_gb=8.0,
        bundle_name=None,
        expected_size_gb=None,
    ),
    "vista3d": ModelSpec(
        name="vista3d",
        source="vista3d",
        task="whole_body",
        description=(
            "NVIDIA VISTA-3D — универсальная модель на 127+ классов (включая мозг), "
            "распространяется как NIM: локально (Docker, GPU ~48GB) либо облачный API."
        ),
        modalities=("CT", "MRI"),
        min_gpu_memory_gb=48.0,
        expected_size_gb=0.0,
    ),
    "med3d_resnet18": ModelSpec(
        name="med3d_resnet18",
        source="med3d",
        task="feature_extraction",
        description=(
            "3D-ResNet18, предобученный на 23 медицинских датасетах (MedicalNet/Med3D), "
            "доступен через monai.networks.nets.resnet18(pretrained=True)."
        ),
        modalities=("CT", "MRI"),
        expected_size_gb=0.1,
    ),
}


class PretrainedModelRegistry:
    """Единый реестр доступных предобученных моделей и их локального кэша.

    Реестр хранит только МЕТАДАННЫЕ (ModelSpec) и решает, что и откуда загружать —
    сами веса в память загружают соответствующие обёртки (TotalSegmentatorWrapper,
    MONAIModelZoo, Med3DLoader). Кэш по умолчанию — ~/.cache/brain3d/<source>/<name>.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Инициализирует реестр.

        Args:
            cache_dir: Директория кэша. По умолчанию — ~/.cache/brain3d.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog = dict(_MODEL_CATALOG)

    def list_available_models(self, task: str | None = None) -> list[ModelSpec]:
        """Возвращает список моделей реестра.

        Args:
            task: Если задано, возвращаются только модели этой задачи.

        Returns:
            Список ModelSpec.
        """
        models = list(self._catalog.values())
        if task is not None:
            models = [spec for spec in models if spec.task == task]
        return models

    def list_cached_models(self) -> list[ModelSpec]:
        """Возвращает подмножество каталога, для которого веса уже есть в локальном кэше.

        Используется для offline-режима: если интернета нет, можно работать только
        с тем, что уже когда-то было загружено (см. SmartModelSelector).
        """
        return [spec for spec in self._catalog.values() if self.is_cached(spec.name)]

    def get_model(self, model_name: str, task: str | None = None) -> ModelSpec:
        """Возвращает метаданные модели по имени.

        Args:
            model_name: Имя модели из реестра.
            task: Если задано, проверяется, что модель зарегистрирована именно под эту задачу.

        Returns:
            ModelSpec.

        Raises:
            ModelNotFoundError: Если модель не найдена, либо не соответствует task.
        """
        if model_name not in self._catalog:
            raise ModelNotFoundError(
                f"Модель {model_name!r} не найдена в реестре. Доступны: {sorted(self._catalog)}"
            )
        spec = self._catalog[model_name]
        if task is not None and spec.task != task:
            raise ModelNotFoundError(
                f"Модель {model_name!r} зарегистрирована для задачи {spec.task!r}, а не {task!r}"
            )
        return spec

    def is_cached(self, model_name: str) -> bool:
        """Проверяет, есть ли для модели непустой кэш на диске."""
        spec = self.get_model(model_name)
        return self._is_populated(self._model_cache_path(spec))

    def download_model(
        self,
        model_name: str,
        cache_dir: Path | str | None = None,
        force: bool = False,
    ) -> Path:
        """Скачивает веса модели в локальный кэш (если их там ещё нет) и возвращает путь.

        Механизм скачивания зависит от source:
            - "totalsegmentator": веса скачивает сама библиотека при первом сегментировании
              (см. TotalSegmentatorWrapper.segment) — этот метод только логирует напоминание;
            - "monai_bundle": через monai.bundle.download(name=..., bundle_dir=...);
            - "med3d": веса скачивает MONAI при первом создании модели с pretrained=True
              (см. Med3DLoader.load_pretrained_encoder) — этот метод только логирует напоминание;
            - "vista3d": не применимо (облачный API либо Docker-образ) — возвращается
              путь-заглушка с предупреждением в лог.

        Args:
            model_name: Имя модели из реестра.
            cache_dir: Переопределяет self.cache_dir для этого вызова.
            force: Перезагрузить, даже если уже закэшировано.

        Returns:
            Путь к директории с весами в кэше.

        Raises:
            ModelNotFoundError: Если модель не найдена в реестре.
            ModelDownloadError: Если скачивание не удалось (нет сети, нет места, нет bundle_name).
            ChecksumMismatchError: Если у модели задан sha256 и он не совпал с фактическим.
        """
        spec = self.get_model(model_name)
        target_dir = Path(cache_dir) if cache_dir else self._model_cache_path(spec)
        target_dir.mkdir(parents=True, exist_ok=True)

        if not force and self._is_populated(target_dir):
            logger.info("Модель %s уже в кэше: %s", model_name, target_dir)
            return target_dir

        logger.info("Скачивание модели %s (источник: %s)...", model_name, spec.source)
        try:
            if spec.source == "monai_bundle":
                _download_monai_bundle(spec, target_dir)
            elif spec.source == "totalsegmentator":
                logger.info("TotalSegmentator скачивает веса самостоятельно при первом вызове segment()")
            elif spec.source == "med3d":
                logger.info(
                    "Веса Med3D/MedicalNet скачиваются MONAI при первом создании модели "
                    "(pretrained=True) — см. Med3DLoader.load_pretrained_encoder"
                )
            elif spec.source == "vista3d":
                logger.warning(
                    "VISTA-3D — облачный API или Docker-образ, download_model для него не применим"
                )
            else:  # pragma: no cover — защита на случай расширения Literal без обновления веток
                raise ModelDownloadError(f"Неизвестный источник модели: {spec.source}")
        except ModelDownloadError:
            raise
        except OSError as exc:
            raise ModelDownloadError(f"Не удалось скачать модель {model_name}: {exc}") from exc

        if spec.sha256 is not None:
            _verify_checksum(target_dir, spec.sha256)

        return target_dir

    def _model_cache_path(self, spec: ModelSpec) -> Path:
        return self.cache_dir / spec.source / spec.name

    @staticmethod
    def _is_populated(path: Path) -> bool:
        return path.exists() and path.is_dir() and any(path.iterdir())


def _download_monai_bundle(spec: ModelSpec, target_dir: Path) -> None:
    """Скачивает MONAI bundle через monai.bundle.download (импорт лениво, monai — тяжёлая зависимость)."""
    if spec.bundle_name is None:
        raise ModelDownloadError(
            f"Для модели {spec.name!r} не задан bundle_name — bundle в MONAI Model Zoo неизвестен"
        )
    try:
        from monai.bundle import download as bundle_download
    except ImportError as exc:
        raise ModelDownloadError("Пакет monai не установлен. Установите: pip install monai") from exc

    try:
        bundle_download(name=spec.bundle_name, bundle_dir=str(target_dir))
    except Exception as exc:  # noqa: BLE001 — сеть/диск/сервер MONAI hosting могут падать по-разному
        raise ModelDownloadError(f"Не удалось скачать bundle {spec.bundle_name!r}: {exc}") from exc


def _verify_checksum(path: Path, expected_sha256: str, chunk_size: int = 1024 * 1024) -> None:
    """Проверяет SHA-256 файла весов в директории (или самого файла, если path — файл).

    Args:
        path: Файл весов либо директория, в которой ищется единственный файл весов.
        expected_sha256: Ожидаемая контрольная сумма в hex.
        chunk_size: Размер блока чтения файла.

    Raises:
        ChecksumMismatchError: Если файл не найден либо сумма не совпадает.
    """
    target_file = path if path.is_file() else _find_weights_file(path)

    digest = hashlib.sha256()
    with target_file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    actual = digest.hexdigest()

    if actual != expected_sha256:
        raise ChecksumMismatchError(
            f"Контрольная сумма {target_file} не совпадает: ожидалось {expected_sha256}, получено {actual}"
        )
    logger.debug("Контрольная сумма %s подтверждена", target_file)


def _find_weights_file(directory: Path) -> Path:
    """Ищет файл весов (.pt/.pth/.ts) в директории для проверки контрольной суммы."""
    candidates = sorted([*directory.rglob("*.pt"), *directory.rglob("*.pth"), *directory.rglob("*.ts")])
    if not candidates:
        raise ChecksumMismatchError(f"В {directory} не найден файл весов для проверки контрольной суммы")
    return candidates[0]


def is_online(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """Быстрая проверка интернет-соединения TCP-подключением к публичному DNS.

    Не делает HTTP-запросов и не требует пакета requests — используется как лёгкая
    первая проверка перед скачиванием весов (см. SmartModelSelector, offline-режим).

    Args:
        host: Хост для проверки соединения (по умолчанию Google Public DNS).
        port: TCP-порт.
        timeout: Таймаут соединения в секундах.

    Returns:
        True, если соединение установлено, иначе False.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# TotalSegmentator — главная модель
# --------------------------------------------------------------------------- #


class TotalSegmentatorWrapper:
    """Обёртка над TotalSegmentator (https://github.com/wasserth/TotalSegmentator).

    TotalSegmentator сегментирует 104+ анатомические структуры на КТ (и МРТ — задача
    "total_mr"), включая мозг целиком и кости черепа; выделенных сосудов Виллизиева
    круга нет ни в одной задаче, но задача "headneck_bones_vessels" даёт сонные и
    позвоночные артерии, что значительно ближе к ним, чем список сосудов из "total"
    (аорта, подвздошные и т.п. — сосуды туловища, как и указано в задании на этот класс).
    Веса скачиваются самим TotalSegmentator при первом вызове (в ~/.totalsegmentator),
    отдельно от кэша PretrainedModelRegistry.
    """

    #: Метки костей из задачи "total" — используются для skull-stripping и общего
    #: удаления костных структур.
    BONE_LABEL_KEYWORDS: tuple[str, ...] = (
        "skull",
        "vertebrae",
        "rib",
        "clavicula",
        "scapula",
        "sternum",
        "femur",
        "hip",
        "sacrum",
        "humerus",
    )

    #: Сосуды из задачи "total" — преимущественно туловище, как в задании на этот класс.
    VESSEL_LABELS_TOTAL: tuple[str, ...] = (
        "aorta",
        "inferior_vena_cava",
        "portal_vein_and_splenic_vein",
        "iliac_artery_left",
        "iliac_artery_right",
        "iliac_vena_left",
        "iliac_vena_right",
        "pulmonary_vein",
        "brachiocephalic_trunk",
        "subclavian_artery_left",
        "subclavian_artery_right",
        "common_carotid_artery_left",
        "common_carotid_artery_right",
    )

    #: Сосуды из задачи "headneck_bones_vessels" — ближе к реальным сосудам, питающим
    #: Виллизиев круг, чем общий список из "total".
    VESSEL_LABELS_HEADNECK: tuple[str, ...] = (
        "common_carotid_artery_left",
        "common_carotid_artery_right",
        "internal_carotid_artery_left",
        "internal_carotid_artery_right",
        "vertebral_artery_left",
        "vertebral_artery_right",
    )

    def __init__(
        self,
        fast: bool = False,
        roi_subset: list[str] | None = None,
        task: str = "total",
        device: str = "cpu",
    ) -> None:
        """Инициализирует обёртку.

        Args:
            fast: Быстрый режим (модель на пониженном разрешении, ~3мм вместо ~1.5мм) —
                примерно в 5 раз быстрее ценой точности; рекомендуется на CPU.
            roi_subset: Ограничить сегментацию подмножеством структур. По документации
                TotalSegmentator работает только при fast=False — при fast=True игнорируется
                с предупреждением в лог.
            task: Задача TotalSegmentator: "total", "total_mr", "headneck_bones_vessels",
                "cerebral_bleed" и т.д. (полный список — totalsegmentator.map_to_binary).
            device: "cpu", "gpu" или "mps".
        """
        if roi_subset is not None and fast:
            logger.warning(
                "roi_subset игнорируется в fast-режиме TotalSegmentator (ограничение самой библиотеки)"
            )
        self.fast = fast
        self.roi_subset = roi_subset
        self.task = task
        self.device = device
        self._last_masks: dict[str, np.ndarray] | None = None

    def segment(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, np.ndarray]:
        """Сегментирует объём и возвращает бинарные маски по каждой найденной структуре.

        TotalSegmentator работает с файлами NIfTI, поэтому volume временно сохраняется
        на диск во временную директорию, обрабатывается, результат читается обратно,
        временные файлы удаляются автоматически.

        Args:
            volume: Объём (Z, Y, X).
            spacing: Spacing (x, y, z), мм. Ориентация/origin не учитываются (строится
                простая осевыровненная геометрия) — для большинства задач сегментации
                абсолютное положение в мировых координатах не важно.

        Returns:
            Словарь {имя_структуры: бинарная маска (Z, Y, X) uint8}.

        Raises:
            Brain3DError: Если пакет totalsegmentator не установлен, либо сегментация упала.
        """
        try:
            from totalsegmentator.python_api import totalsegmentator
        except ImportError as exc:
            raise Brain3DError(
                "Пакет totalsegmentator не установлен. Установите: pip install totalsegmentator"
            ) from exc

        import nibabel as nib

        with tempfile.TemporaryDirectory(prefix="brain3d_totalseg_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.nii.gz"
            output_dir = tmp_path / "output"

            affine = _affine_from_spacing(spacing)
            data_xyz = np.asarray(volume, dtype=np.float32).transpose(2, 1, 0)
            nib.save(nib.Nifti1Image(data_xyz, affine), str(input_path))

            logger.info(
                "Запуск TotalSegmentator: task=%s, fast=%s, roi_subset=%s, device=%s",
                self.task,
                self.fast,
                self.roi_subset,
                self.device,
            )
            try:
                totalsegmentator(
                    input=input_path,
                    output=output_dir,
                    task=self.task,
                    fast=self.fast,
                    roi_subset=self.roi_subset if not self.fast else None,
                    device=self.device,
                    ml=False,
                    quiet=True,
                )
            except Exception as exc:  # noqa: BLE001 — библиотека бросает разные типы ошибок
                raise Brain3DError(f"TotalSegmentator упал во время сегментации: {exc}") from exc

            masks: dict[str, np.ndarray] = {}
            for mask_file in sorted(output_dir.glob("*.nii.gz")):
                structure_name = mask_file.name[: -len(".nii.gz")]
                mask_image = nib.load(str(mask_file))
                mask_array = np.asarray(mask_image.dataobj, dtype=np.uint8).transpose(2, 1, 0)
                masks[structure_name] = mask_array

        self._last_masks = masks
        logger.info("TotalSegmentator: найдено %d структур", len(masks))
        return masks

    def get_brain_mask(self) -> np.ndarray:
        """Возвращает маску мозга целиком из последнего вызова segment() (метка "brain")."""
        return self._require_label("brain")

    def get_vessel_masks(self) -> dict[str, np.ndarray]:
        """Возвращает маски сосудов из последнего вызова segment().

        Набор зависит от self.task: "headneck_bones_vessels" даёт сосуды головы/шеи
        (ближе к Виллизиеву кругу), любая другая задача (в т.ч. "total") — общий список
        сосудов туловища (аорта, подвздошные и т.п.).

        Raises:
            Brain3DError: Если segment() ещё не вызывался.
        """
        masks = self._require_masks()
        vessel_names = (
            self.VESSEL_LABELS_HEADNECK if self.task == "headneck_bones_vessels" else self.VESSEL_LABELS_TOTAL
        )
        return {name: mask for name, mask in masks.items() if name in vessel_names}

    def get_bone_masks(self) -> dict[str, np.ndarray]:
        """Возвращает маски костей из последнего вызова segment() (для skull-stripping и др.).

        Raises:
            Brain3DError: Если segment() ещё не вызывался.
        """
        masks = self._require_masks()
        return {
            name: mask
            for name, mask in masks.items()
            if any(keyword in name for keyword in self.BONE_LABEL_KEYWORDS)
        }

    def _require_masks(self) -> dict[str, np.ndarray]:
        if self._last_masks is None:
            raise Brain3DError("Сначала вызовите segment(volume, spacing)")
        return self._last_masks

    def _require_label(self, name: str) -> np.ndarray:
        masks = self._require_masks()
        if name not in masks:
            raise Brain3DError(
                f"Структура {name!r} отсутствует в результатах последней сегментации "
                f"(задача={self.task!r}). Доступны: {sorted(masks)}"
            )
        return masks[name]


def _affine_from_spacing(spacing: tuple[float, float, float]) -> np.ndarray:
    """Строит простую диагональную affine-матрицу (RAS+) по spacing без учёта origin/ориентации.

    Используется там, где на вход подаётся только (volume, spacing) без полной геометрии
    (см. shared.types.VolumeData) — знак x/y инвертирован для перехода от внутреннего
    соглашения проекта (LPS-подобная ориентация по умолчанию) к RAS+, как того требует NIfTI.
    """
    return np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])


# --------------------------------------------------------------------------- #
# MONAI Model Zoo
# --------------------------------------------------------------------------- #


class MONAIModelZoo:
    """Загрузка bundle-моделей из MONAI Model Zoo (Project-MONAI/model-zoo).

    Имена bundle могут меняться между релизами зоопарка — перед использованием в проде
    сверьтесь со списком на https://github.com/Project-MONAI/model-zoo/tree/dev/models.
    "brats_mri_segmentation" на момент написания — реальный, существующий bundle (Swin
    UNETR, обучен на BraTS); имя для сосудов мозга в реестре — заглушка (см. ModelSpec
    "monai_brain_vessel"), т.к. специализированного bundle под эту структуру в зоопарке нет.
    """

    def __init__(self, registry: PretrainedModelRegistry | None = None, device: str = "cpu") -> None:
        """Инициализирует загрузчик.

        Args:
            registry: Реестр моделей (для скачивания/кэширования bundle).
            device: "cpu" или "cuda".
        """
        self.registry = registry or PretrainedModelRegistry()
        self.device = device

    def load_brain_tumor_model(self) -> Any:
        """Загружает предобученную модель сегментации опухолей мозга (Swin UNETR, BraTS).

        Returns:
            torch.nn.Module в режиме eval().
        """
        return self._load_bundle_model("monai_brain_tumor_swinunetr")

    def load_brain_vessel_model(self) -> Any:
        """Загружает предобученную модель сегментации сосудов мозга, если она есть в зоопарке.

        На момент написания в MONAI Model Zoo нет узкоспециализированного bundle именно
        под сосуды мозга/Виллизиев круг — практичнее TotalSegmentatorWrapper (task=
        "headneck_bones_vessels") или сегментация фильтром Франги (см. load_vessel_model()).
        Метод оставлен для единообразия интерфейса и поднимет понятную ошибку, объясняющую
        альтернативу, а не менее информативный ImportError/KeyError.

        Raises:
            ModelDownloadError: Всегда, пока в реестре нет реального bundle_name для этой модели.
        """
        return self._load_bundle_model("monai_brain_vessel")

    def load_multimodal_model(self) -> Any:
        """Загружает мультимодальную модель (вход T1+T1ce+T2+FLAIR, 4 канала).

        brats_mri_segmentation по построению уже 4-канальная (её штатный вход — ровно
        эти четыре последовательности), поэтому используется тот же bundle, что и для
        load_brain_tumor_model() — отдельной "мультимодальной" модели в зоопарке нет.
        """
        return self._load_bundle_model("monai_brain_tumor_swinunetr")

    def _load_bundle_model(self, model_name: str) -> Any:
        """Скачивает (при необходимости) и загружает bundle-модель как torch.nn.Module."""
        try:
            from monai.bundle import load as bundle_load
        except ImportError as exc:
            raise Brain3DError("Пакет monai не установлен. Установите: pip install monai") from exc

        spec = self.registry.get_model(model_name)
        bundle_dir = self.registry.download_model(model_name)

        try:
            model = bundle_load(
                name=spec.bundle_name,
                bundle_dir=str(bundle_dir.parent),
                source="monaihosting",
                map_location=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            raise Brain3DError(f"Не удалось загрузить bundle {spec.bundle_name!r}: {exc}") from exc

        logger.info("Bundle-модель %s загружена на устройство %s", model_name, self.device)
        return model


# --------------------------------------------------------------------------- #
# NVIDIA VISTA-3D (NIM)
# --------------------------------------------------------------------------- #


class VISTA3DWrapper:
    """HTTP-клиент NVIDIA VISTA-3D — универсальной 3D-модели сегментации (127+ классов).

    VISTA-3D распространяется как NVIDIA NIM (микросервис в Docker-контейнере с REST API),
    а не pip-пакет:
        - mode="local": контейнер запущен на своей машине (нужен GPU ~48GB VRAM для полной
          модели), эндпоинт обычно http://localhost:8000;
        - mode="cloud": управляемый эндпоинт NVIDIA (build.nvidia.com/api.nvidia.com),
          нужен api_key.

    Точная схема запроса/ответа зависит от версии образа NIM — перед реальным
    использованием сверьтесь с OpenAPI-схемой запущенного контейнера (обычно доступна
    на {endpoint}/docs) или официальной документацией NIM; формат payload ниже — это
    типовая схема NIM inference API (base64-изображение + JSON prompts), а не
    гарантированно неизменная спецификация конкретной версии.
    """

    DEFAULT_LOCAL_ENDPOINT = "http://localhost:8000"
    DEFAULT_CLOUD_ENDPOINT = "https://health.api.nvidia.com/v1/medical/vista-3d"

    def __init__(
        self,
        mode: Literal["local", "cloud"] = "local",
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        """Инициализирует клиент.

        Args:
            mode: "local" (Docker на своей машине) или "cloud" (управляемый API NVIDIA).
            api_key: API-ключ для mode="cloud". Если не передан, берётся из переменной
                окружения NVIDIA_API_KEY.
            endpoint: Переопределяет URL по умолчанию для выбранного режима.
            timeout_s: Таймаут HTTP-запросов сегментации (инференс может быть небыстрым).

        Raises:
            Brain3DError: Если mode="cloud" и api_key не передан и не найден в окружении.
        """
        self.mode = mode
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.endpoint = endpoint or (
            self.DEFAULT_LOCAL_ENDPOINT if mode == "local" else self.DEFAULT_CLOUD_ENDPOINT
        )
        self.timeout_s = timeout_s

        if mode == "cloud" and not self.api_key:
            raise Brain3DError("Для mode='cloud' требуется api_key (или переменная окружения NVIDIA_API_KEY)")

    def is_available(self) -> bool:
        """Проверяет доступность эндпоинта health-check запросом, без выполнения сегментации.

        Используется как первый шаг fallback-логики: если VISTA-3D недоступен, вызывающий
        код (см. SmartModelSelector) должен переключиться на локальные модели.

        Returns:
            True, если эндпоинт ответил успешно, иначе False (включая любые сетевые ошибки).
        """
        try:
            import requests
        except ImportError:
            logger.warning("Пакет requests не установлен — VISTA-3D считается недоступным")
            return False

        try:
            response = requests.get(f"{self.endpoint}/v1/health/ready", timeout=5.0)
            return response.ok
        except requests.RequestException as exc:
            logger.debug("VISTA-3D health-check не прошёл: %s", exc)
            return False

    def segment(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        classes: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Отправляет объём на сегментацию через VISTA-3D NIM.

        Args:
            volume: Объём (Z, Y, X).
            spacing: Spacing (x, y, z), мм.
            classes: Запрашиваемые классы, например ["brain", "hepatic_vessel"];
                None — полный поддерживаемый набор.

        Returns:
            Словарь {класс: бинарная маска (Z, Y, X) uint8}.

        Raises:
            Brain3DError: При ошибке запроса или недоступности эндпоинта — вызывающий код
                должен переключиться на локальную модель (см. SmartModelSelector).
        """
        payload = self._build_payload(volume, spacing, classes)
        return self._post(payload, volume.shape)

    def interactive_segmentation(
        self,
        volume: np.ndarray,
        points: list[tuple[int, int, int]],
        point_labels: list[int] | None = None,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict[str, np.ndarray]:
        """Интерактивная сегментация по точкам-подсказкам (аналог SAM для 3D медицинских данных).

        Args:
            volume: Объём (Z, Y, X).
            points: Координаты точек-подсказок (z, y, x).
            point_labels: Метки точек (1 — foreground, 0 — background); по умолчанию все foreground.
            spacing: Spacing (x, y, z), мм.

        Returns:
            {"interactive": бинарная маска (Z, Y, X) uint8}.

        Raises:
            Brain3DError: При ошибке запроса или недоступности эндпоинта.
        """
        labels = point_labels if point_labels is not None else [1] * len(points)
        payload = self._build_payload(volume, spacing, classes=None)
        payload["prompts"] = {"points": [list(p) for p in points], "point_labels": labels}
        return self._post(payload, volume.shape)

    def _post(self, payload: dict[str, Any], shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        try:
            import requests
        except ImportError as exc:
            raise Brain3DError("Пакет requests не установлен. Установите: pip install requests") from exc

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.post(
                f"{self.endpoint}/v1/vista3d/inference",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise Brain3DError(f"VISTA-3D недоступен (mode={self.mode}): {exc}") from exc

        return self._parse_response(response.json(), shape)

    @staticmethod
    def _build_payload(
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        classes: list[str] | None,
    ) -> dict[str, Any]:
        import base64

        payload: dict[str, Any] = {
            "image": base64.b64encode(np.asarray(volume, dtype=np.float32).tobytes()).decode("ascii"),
            "shape": list(volume.shape),
            "spacing": list(spacing),
        }
        if classes is not None:
            payload["classes"] = classes
        return payload

    @staticmethod
    def _parse_response(response_json: dict[str, Any], shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        import base64

        masks: dict[str, np.ndarray] = {}
        for class_name, encoded_mask in response_json.get("masks", {}).items():
            raw = base64.b64decode(encoded_mask)
            masks[class_name] = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        return masks


# --------------------------------------------------------------------------- #
# Med3D / MedicalNet-style энкодеры
# --------------------------------------------------------------------------- #


class Med3DLoader:
    """Загрузка предобученных 3D-энкодеров в стиле Med3D/MedicalNet через MONAI.

    Оригинальный проект Med3D (Chen et al., https://github.com/Tencent/MedicalNet)
    распространяет веса, предобученные на 23 медицинских датасетах, через Google Drive
    без стабильного программного доступа. MONAI переиздаёт эти же веса для части глубин
    ResNet3D через параметр pretrained=True встроенных конструкторов
    monai.networks.nets.resnetXX — это и используется здесь вместо ручного скачивания.
    """

    _SUPPORTED_DEPTHS: tuple[int, ...] = (10, 18, 34, 50)

    def __init__(self, depth: int = 18, device: str = "cpu") -> None:
        """Инициализирует загрузчик.

        Args:
            depth: Глубина ResNet3D (10, 18, 34 или 50 — глубины, для которых MONAI
                предоставляет веса Med3D/MedicalNet).
            device: "cpu" или "cuda".

        Raises:
            Brain3DError: Если depth не входит в число поддерживаемых.
        """
        if depth not in self._SUPPORTED_DEPTHS:
            raise Brain3DError(
                f"Глубина ResNet3D {depth} не поддерживается. Доступны: {self._SUPPORTED_DEPTHS}"
            )
        self.depth = depth
        self.device = device
        self._encoder: Any | None = None

    def load_pretrained_encoder(self) -> Any:
        """Загружает предобученный 3D-ResNet-энкодер (без классификационной головы).

        Returns:
            torch.nn.Module в режиме eval(), пригодный как feature extractor или как
            backbone для fine-tuning (например, энкодер собственного U-Net).

        Raises:
            Brain3DError: Если torch/monai не установлены, либо конструктор отклонил
                комбинацию параметров (сигнатура pretrained-весов может отличаться
                между версиями MONAI).
        """
        try:
            from monai.networks.nets import resnet
        except ImportError as exc:
            raise Brain3DError("Пакет monai не установлен. Установите: pip install monai") from exc

        builder: Callable[..., Any] = getattr(resnet, f"resnet{self.depth}")
        try:
            model = builder(pretrained=True, spatial_dims=3, n_input_channels=1, feed_forward=False)
        except Exception as exc:  # noqa: BLE001 — конкретные требуемые kwargs зависят от версии MONAI
            raise Brain3DError(
                f"Не удалось создать resnet{self.depth}(pretrained=True): {exc}. "
                "Комбинация параметров, требуемая для загрузки Med3D-весов, может отличаться "
                "между версиями MONAI — сверьтесь с текущей документацией monai.networks.nets.resnet."
            ) from exc

        model.to(self.device)
        model.eval()
        self._encoder = model
        logger.info("Med3D-энкодер resnet%d загружен на устройство %s", self.depth, self.device)
        return model

    def use_as_feature_extractor(self, volume: np.ndarray) -> np.ndarray:
        """Прогоняет объём через предобученный энкодер и возвращает вектор признаков.

        Энкодер загружается лениво при первом вызове, если ещё не был загружен явно
        через load_pretrained_encoder().

        Args:
            volume: Объём (Z, Y, X), желательно предварительно нормализованный
                (см. core.preprocessing).

        Returns:
            Вектор признаков (после глобального пулинга энкодера), np.ndarray формы (C,).
        """
        try:
            import torch
        except ImportError as exc:
            raise Brain3DError("Пакет torch не установлен. Установите: pip install torch") from exc

        if self._encoder is None:
            self.load_pretrained_encoder()

        with torch.no_grad():
            input_tensor = torch.from_numpy(np.asarray(volume, dtype=np.float32))
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            features = self._encoder(input_tensor)

        return features.squeeze(0).to("cpu").numpy()


# --------------------------------------------------------------------------- #
# Определение ресурсов и умный выбор модели
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ComputeResources:
    """Снимок доступных вычислительных ресурсов, используемый SmartModelSelector."""

    has_gpu: bool
    gpu_memory_gb: float | None
    has_internet: bool
    cpu_count: int


def detect_compute_resources(check_internet: bool = True) -> ComputeResources:
    """Определяет доступные ресурсы: GPU (и объём VRAM), интернет, число ядер CPU.

    Args:
        check_internet: Проверять ли интернет-соединение (~2 сек сетевой запрос).
            Выключите для быстрых вызовов там, где результат заведомо не важен.

    Returns:
        ComputeResources с текущим состоянием окружения.
    """
    has_gpu = False
    gpu_memory_gb: float | None = None
    try:
        import torch

        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        logger.debug("torch не установлен — GPU считается недоступным")

    return ComputeResources(
        has_gpu=has_gpu,
        gpu_memory_gb=gpu_memory_gb,
        has_internet=is_online() if check_internet else False,
        cpu_count=os.cpu_count() or 1,
    )


class SmartModelSelector:
    """Выбирает оптимальную стратегию модели под задачу, модальность и доступные ресурсы."""

    def __init__(self, registry: PretrainedModelRegistry | None = None) -> None:
        """Инициализирует селектор.

        Args:
            registry: Реестр моделей, используется для поиска закэшированных моделей
                в offline-сценарии.
        """
        self.registry = registry or PretrainedModelRegistry()

    def choose_best_model(
        self,
        modality: str,
        task: str,
        available_resources: ComputeResources | None = None,
    ) -> str:
        """Выбирает имя стратегии для заданных условий.

        Логика (в порядке приоритета):
            1. GPU >= 48GB -> "vista3d" (локально).
            2. GPU >= 12GB -> "brats_pretrained" для task="tumor" (MONAI Swin UNETR),
               иначе "totalsegmentator".
            3. Только CPU (или GPU меньше 12GB) -> "totalsegmentator" (в fast-режиме,
               см. should_use_fast_mode) для большинства задач; "frangi" для task="vessel",
               т.к. не требует ни GPU, ни загрузки весов.
            4. Нет интернета -> первая подходящая ЗАКЭШИРОВАННАЯ модель реестра, либо
               "frangi"/"totalsegmentator" как крайний случай (см. _offline_fallback).

        Args:
            modality: "CT" или "MRI" (пока не влияет на логику напрямую, но фиксирует
                контракт метода и учитывается при отборе кандидатов из реестра).
            task: "tumor" | "vessel" | "hemorrhage" | "whole_body" и т.п.
            available_resources: Заранее определённые ресурсы; если не переданы —
                определяются автоматически через detect_compute_resources().

        Returns:
            Имя стратегии — значение, подходящее для strategy= в load_tumor_model()/
            load_vessel_model(), либо имя модели из реестра в offline-режиме.
        """
        del modality  # зарезервировано для будущей модально-специфичной логики выбора
        resources = available_resources or detect_compute_resources()

        if not resources.has_internet:
            return self._offline_fallback(task)

        if resources.has_gpu and (resources.gpu_memory_gb or 0.0) >= 48.0:
            return "vista3d"

        if resources.has_gpu and (resources.gpu_memory_gb or 0.0) >= 12.0:
            return "brats_pretrained" if task == "tumor" else "totalsegmentator"

        return "frangi" if task == "vessel" else "totalsegmentator"

    def should_use_fast_mode(self, available_resources: ComputeResources | None = None) -> bool:
        """Определяет, нужен ли fast-режим TotalSegmentator (CPU-only окружения — да).

        Args:
            available_resources: Заранее определённые ресурсы; по умолчанию определяются
                без проверки интернета (это не нужно для решения о fast-режиме).

        Returns:
            True, если GPU недоступен.
        """
        resources = available_resources or detect_compute_resources(check_internet=False)
        return not resources.has_gpu

    def _offline_fallback(self, task: str) -> str:
        """Выбирает стратегию, когда интернета нет: закэшированная модель либо Frangi."""
        if task == "vessel":
            logger.info("Нет интернета — для сосудов используется Frangi (не требует сети)")
            return "frangi"

        for spec in self.registry.list_available_models(task=task):
            if self.registry.is_cached(spec.name):
                logger.info("Нет интернета — используется закэшированная модель: %s", spec.name)
                return spec.name

        logger.warning(
            "Нет интернета и нет подходящей закэшированной модели задачи %r — "
            "возвращается 'totalsegmentator', но она сработает только если её веса уже в кэше",
            task,
        )
        return "totalsegmentator"


# --------------------------------------------------------------------------- #
# Единые точки входа: load_tumor_model / load_vessel_model
# --------------------------------------------------------------------------- #


def load_tumor_model(
    strategy: str = "auto",
    device: str = "cpu",
    registry: PretrainedModelRegistry | None = None,
) -> Any:
    """Возвращает модель/обёртку для сегментации опухолей мозга согласно strategy.

    Args:
        strategy: "brats_pretrained" (MONAI Swin UNETR на BraTS) | "vista3d" (NVIDIA NIM) |
            "totalsegmentator" (общая модель, грубее для опухолей) | "auto"
            (решает SmartModelSelector по доступным ресурсам).
        device: "cpu" или "cuda" (используется для brats_pretrained/totalsegmentator).
        registry: Реестр моделей; по умолчанию создаётся новый с кэшем ~/.cache/brain3d.

    Returns:
        torch.nn.Module (strategy="brats_pretrained"), VISTA3DWrapper (strategy="vista3d")
        или TotalSegmentatorWrapper (strategy="totalsegmentator").

    Raises:
        Brain3DError: Если strategy не распознана.

    Example:
        >>> model = load_tumor_model(strategy="auto")
        >>> masks = model.segment(volume, spacing) if hasattr(model, "segment") else None
    """
    registry = registry or PretrainedModelRegistry()

    if strategy == "auto":
        strategy = SmartModelSelector(registry).choose_best_model(modality="MRI", task="tumor")
        logger.info("SmartModelSelector выбрал стратегию для опухолей: %s", strategy)

    if strategy == "brats_pretrained":
        return MONAIModelZoo(registry, device=device).load_brain_tumor_model()
    if strategy == "vista3d":
        return VISTA3DWrapper(mode="local")
    if strategy == "totalsegmentator":
        fast = SmartModelSelector(registry).should_use_fast_mode()
        return TotalSegmentatorWrapper(fast=fast, task="total", device=device)

    raise Brain3DError(f"Неизвестная стратегия load_tumor_model: {strategy!r}")


def load_vessel_model(
    strategy: str = "auto",
    device: str = "cpu",
    registry: PretrainedModelRegistry | None = None,
) -> Any:
    """Возвращает модель/функцию для сегментации сосудов согласно strategy.

    Args:
        strategy: "frangi" (классический фильтр, не требует GPU/сети/весов) |
            "totalsegmentator" (task="headneck_bones_vessels") | "vista3d" | "auto"
            (решает SmartModelSelector).
        device: "cpu" или "cuda" (используется для totalsegmentator/vista3d).
        registry: Реестр моделей.

    Returns:
        Для strategy="frangi" — сама функция segment_vessels_frangi (вызывается как
        frangi_fn(volume, spacing)); для остальных — TotalSegmentatorWrapper/VISTA3DWrapper.

    Raises:
        Brain3DError: Если strategy не распознана.

    Example:
        >>> segmenter = load_vessel_model(strategy="frangi")
        >>> vessel_mask = segmenter(volume, spacing=(1.0, 1.0, 1.0))
    """
    registry = registry or PretrainedModelRegistry()

    if strategy == "auto":
        strategy = SmartModelSelector(registry).choose_best_model(modality="MRI", task="vessel")
        logger.info("SmartModelSelector выбрал стратегию для сосудов: %s", strategy)

    if strategy == "frangi":
        return segment_vessels_frangi
    if strategy == "totalsegmentator":
        fast = SmartModelSelector(registry).should_use_fast_mode()
        return TotalSegmentatorWrapper(fast=fast, task="headneck_bones_vessels", device=device)
    if strategy == "vista3d":
        return VISTA3DWrapper(mode="local")

    raise Brain3DError(f"Неизвестная стратегия load_vessel_model: {strategy!r}")


def segment_vessels_frangi(
    volume: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
    threshold: float | None = None,
) -> np.ndarray:
    """Сегментирует трубчатые структуры (сосуды) классическим фильтром Франги.

    Единственная стратегия load_vessel_model(), не требующая ни GPU, ни скачивания
    весов, ни интернета — крайний вариант в offline/CPU-only сценариях
    (см. SmartModelSelector._offline_fallback).

    Args:
        volume: Объём (Z, Y, X); рекомендуется предварительно нормализовать к [0, 1]
            (см. core.preprocessing) — фильтр Франги чувствителен к масштабу интенсивностей.
        spacing: Spacing (x, y, z), мм. Зарезервировано для будущей калибровки sigmas в
            физических единицах — сейчас фильтр работает в вокселях.
        sigmas: Масштабы (в вокселях) для многомасштабного анализа — должны покрывать
            ожидаемый диапазон диаметров сосудов.
        threshold: Порог по отклику фильтра для бинаризации; None — автоматический порог
            Отсу по самому отклику.

    Returns:
        Бинарная маска сосудов (Z, Y, X), uint8.
    """
    del spacing  # см. docstring: пока не используется, зарезервировано на будущее

    vesselness = frangi(volume, sigmas=sigmas, black_ridges=False)

    if threshold is None:
        nonzero_response = vesselness[vesselness > 0]
        threshold = float(threshold_otsu(nonzero_response)) if nonzero_response.size > 0 else 0.0

    return (vesselness > threshold).astype(np.uint8)


if __name__ == "__main__":
    # --- Пример использования: реестр моделей ---
    registry = PretrainedModelRegistry()
    for model_spec in registry.list_available_models():
        print(f"{model_spec.name}: {model_spec.description}")

    # --- Пример использования: умный выбор модели под текущие ресурсы ---
    selector = SmartModelSelector(registry)
    resources = detect_compute_resources()
    tumor_strategy = selector.choose_best_model(modality="MRI", task="tumor", available_resources=resources)
    print("Выбранная стратегия для опухолей:", tumor_strategy)

    # --- Пример использования: сегментация сосудов без GPU (Frangi) ---
    synthetic_volume = np.random.rand(32, 64, 64).astype(np.float32)
    vessel_segmenter = load_vessel_model(strategy="frangi")
    vessel_mask = vessel_segmenter(synthetic_volume, spacing=(1.0, 1.0, 1.0))
    print("Сосудистая маска (Frangi):", vessel_mask.shape, "воксели сосудов:", int(vessel_mask.sum()))

    # --- Пример использования: TotalSegmentator (требует pip install totalsegmentator) ---
    # ct_volume = ...  # np.ndarray (Z, Y, X) в HU
    # wrapper = TotalSegmentatorWrapper(fast=True, task="total")
    # masks = wrapper.segment(ct_volume, spacing=(1.0, 1.0, 1.0))
    # brain_mask = wrapper.get_brain_mask()
    # bone_masks = wrapper.get_bone_masks()
