"""Генерация и сборка сцен 3D-мешей из масок готовых моделей сегментации.

В отличие от core.mesh.mesh_generator.generate_meshes() (используется единым
пайплайном pipelines.BasePipeline и работает с shared.types.SegmentationResult —
одной multi-label картой на несколько классов), этот модуль рассчитан на вход от
ГОТОВЫХ ВНЕШНИХ МОДЕЛЕЙ — в первую очередь models.pretrained_models.TotalSegmentatorWrapper,
которая возвращает СЛОВАРЬ {имя_структуры: бинарная маска}, а не единую карту меток.

Здесь же — то, что не нужно внутри пайплайна сегментации, но нужно для готовой
веб/desktop-визуализации: сборка сцены из множества структур со стандартными
медицинскими цветами, встраивание анонимизированных метаданных в GLB, экспорт в
несколько форматов и Level of Detail для прогрессивной загрузки в Three.js.

Примечание по децимации: trimesh.Trimesh.simplify_quadric_decimation в закреплённой
версии проекта (4.4.1) требует пакет open3d (тяжёлая необязательная зависимость) —
вместо него используется лёгкий fast-simplification напрямую.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fast_simplification
import numpy as np
import trimesh
from skimage import measure

from core.exceptions import MeshGenerationError
from core.logging_config import get_logger
from models.pretrained_models import TotalSegmentatorWrapper

logger = get_logger(__name__)

#: Стандартные медицинские цвета структур (RGBA, 0..255) — используются
#: create_scene_from_totalsegmentator()/create_full_brain_scene().
STRUCTURE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "brain": (200, 200, 220, 100),
    "tumor": (255, 100, 100, 255),
    "vessel": (100, 100, 255, 255),
    "bone": (255, 255, 255, 50),
    "ventricle": (100, 200, 255, 255),
    "hemorrhage": (255, 165, 0, 255),  # плановая структура — задел на будущее
}

#: Доля УДАЛЯЕМЫХ граней при децимации по структуре (0..1). Опухоль и сосуды
#: клинически значимы формой — децимируются меньше всего; кости и мозг — крупные
#: вспомогательные/контекстные поверхности, где агрессивная децимация не критична.
_CATEGORY_DECIMATION: dict[str, float] = {
    "brain": 0.30,
    "tumor": 0.10,
    "vessel": 0.20,
    "hemorrhage": 0.15,
    "bone": 0.40,
    "ventricle": 0.20,
}

_SCENE_FORMATS = ("glb", "obj")
_SINGLE_MESH_FORMATS = ("stl", "ply")


@dataclass(slots=True)
class SceneMetadata:
    """Анонимизированные метаданные исследования, встраиваемые в GLB-сцену.

    Attributes:
        patient_id_hash: Анонимизированный хэш/псевдоним пациента — НЕ реальный
            идентификатор. Анонимизация выполняется вызывающим кодом ДО передачи сюда.
        study_modality: "CT" | "MRI".
        study_date: Дата исследования в формате ISO 8601 (например, "2026-09-05").
        segmentation_models: Имена моделей/методов, которыми получены маски
            (например, ["totalsegmentator", "frangi"]).
        extra: Дополнительные произвольные JSON-совместимые поля.
    """

    patient_id_hash: str | None = None
    study_modality: str | None = None
    study_date: str | None = None
    segmentation_models: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализует метаданные в плоский JSON-совместимый словарь для scene.metadata."""
        return {
            "patient_id_hash": self.patient_id_hash,
            "study_modality": self.study_modality,
            "study_date": self.study_date,
            "segmentation_models": list(self.segmentation_models),
            **self.extra,
        }


class MeshGenerator:
    """Строит меши и сцены из масок сегментации (в т.ч. от TotalSegmentator и др.)."""

    def __init__(self, default_smoothing_iterations: int = 5) -> None:
        """Инициализирует генератор.

        Args:
            default_smoothing_iterations: Число итераций Laplacian-сглаживания по
                умолчанию для create_brain_mesh()/create_tumor_mesh() и т.п.
        """
        self.default_smoothing_iterations = default_smoothing_iterations

    # --- Базовая функциональность --------------------------------------------------- #

    def create_mesh_marching_cubes(
        self,
        mask: np.ndarray,
        level: float = 0.5,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        smooth: int = 5,
        decimate: float | None = None,
    ) -> trimesh.Trimesh:
        """Строит меш методом marching cubes из бинарной (или вероятностной) маски.

        Args:
            mask: Маска (Z, Y, X). Не обязательно строго бинарная — level задаёт
                изоповерхность (0.5 — стандартный порог для бинарных масок).
            level: Значение изоповерхности для marching cubes.
            spacing: Spacing (x, y, z), мм — приводит меш к физическим единицам.
            smooth: Число итераций Laplacian-сглаживания; 0 (или False) отключает.
            decimate: Доля удаляемых граней (0..1); None — без децимации.

        Returns:
            trimesh.Trimesh.

        Raises:
            MeshGenerationError: Если маска пуста (нет ни одного ненулевого вокселя).
        """
        if mask is None or not np.any(mask):
            raise MeshGenerationError("create_mesh_marching_cubes: маска пуста")

        spacing_zyx = (spacing[2], spacing[1], spacing[0])
        verts, faces, _normals, _values = measure.marching_cubes(mask, level=level, spacing=spacing_zyx)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

        if smooth:
            mesh = self.smooth_mesh(mesh, iterations=int(smooth))
        if decimate:
            mesh = self.decimate_mesh(mesh, target_reduction=decimate)

        logger.info(
            "Меш построен методом marching cubes: %d вершин, %d граней", len(mesh.vertices), len(mesh.faces)
        )
        return mesh

    def smooth_mesh(self, mesh: trimesh.Trimesh, iterations: int = 5) -> trimesh.Trimesh:
        """Сглаживает меш Laplacian-фильтром.

        Args:
            mesh: Исходный меш.
            iterations: Число итераций; 0 — вернуть меш без изменений.

        Returns:
            Сглаженный меш (новый объект) либо исходный, если iterations <= 0 или
            меш пуст (нет граней).
        """
        if iterations <= 0 or len(mesh.faces) == 0:
            return mesh
        return trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)

    def decimate_mesh(self, mesh: trimesh.Trimesh, target_reduction: float) -> trimesh.Trimesh:
        """Упрощает меш квадратичной децимацией (через fast-simplification).

        Args:
            mesh: Исходный меш.
            target_reduction: Доля УДАЛЯЕМЫХ граней (0..1), например 0.3 = убрать 30%.

        Returns:
            Упрощённый меш. Возвращается исходный меш без изменений, если у него
            меньше 4 граней (децимация ниже этого предела не имеет смысла).

        Raises:
            ValueError: Если target_reduction вне диапазона (0, 1).
        """
        if not 0.0 < target_reduction < 1.0:
            raise ValueError(f"target_reduction должен быть в (0, 1), получено {target_reduction}")
        if len(mesh.faces) <= 4:
            return mesh

        target_face_count = max(4, int(len(mesh.faces) * (1.0 - target_reduction)))
        simplified_vertices, simplified_faces = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=target_face_count
        )
        simplified = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces, process=False)

        logger.info(
            "Децимация: %d -> %d граней (цель: удалить %.0f%%)",
            len(mesh.faces),
            len(simplified.faces),
            target_reduction * 100,
        )
        return simplified

    # --- Меши по структурам (адаптировано под TotalSegmentator) -------------------- #

    def _create_structure_mesh(
        self,
        mask: np.ndarray | None,
        spacing: tuple[float, float, float],
        decimate: float,
        structure_name: str,
    ) -> trimesh.Trimesh | None:
        """Строит меш одной структуры либо возвращает None для отсутствующей/пустой маски."""
        if mask is None or not np.any(mask):
            logger.info("Структура %r: пустая маска — меш пропущен", structure_name)
            return None

        mesh = self.create_mesh_marching_cubes(
            mask, spacing=spacing, smooth=self.default_smoothing_iterations, decimate=decimate
        )
        logger.info(
            "Структура %r: меш построен (%d вершин, %d граней)",
            structure_name,
            len(mesh.vertices),
            len(mesh.faces),
        )
        return mesh

    def create_brain_mesh(
        self, brain_mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> trimesh.Trimesh | None:
        """Строит меш мозга (децимация 30% — крупная гладкая поверхность, детали не критичны)."""
        return self._create_structure_mesh(brain_mask, spacing, _CATEGORY_DECIMATION["brain"], "brain")

    def create_tumor_mesh(
        self, tumor_mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> trimesh.Trimesh | None:
        """Строит меш опухоли (децимация 10% — форма клинически значима, сохраняем детали)."""
        return self._create_structure_mesh(tumor_mask, spacing, _CATEGORY_DECIMATION["tumor"], "tumor")

    def create_vessel_mesh(
        self, vessel_mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> trimesh.Trimesh | None:
        """Строит меш сосудов (децимация 20%)."""
        return self._create_structure_mesh(vessel_mask, spacing, _CATEGORY_DECIMATION["vessel"], "vessel")

    def create_hemorrhage_mesh(
        self, hemorrhage_mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> trimesh.Trimesh | None:
        """Строит меш кровоизлияния — ЗАГОТОВКА под плановую структуру Brain3D AI.

        Децимация 15% выбрана как промежуточное значение между опухолью (10% — высокая
        клиническая значимость формы) и сосудами (20%), пока нет обученной модели
        сегментации кровоизлияний (см. models.pretrained_models,
        "totalsegmentator_cerebral_bleed" — ближайший готовый источник таких масок).
        """
        return self._create_structure_mesh(
            hemorrhage_mask, spacing, _CATEGORY_DECIMATION["hemorrhage"], "hemorrhage"
        )

    def create_bone_mesh(
        self, bone_mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> trimesh.Trimesh | None:
        """Строит меш костей черепа — опциональная контекстная структура (децимация 40%)."""
        return self._create_structure_mesh(bone_mask, spacing, _CATEGORY_DECIMATION["bone"], "bone")

    def create_ventricle_mesh(
        self, ventricle_mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> trimesh.Trimesh | None:
        """Строит меш желудочков мозга (децимация 20%)."""
        return self._create_structure_mesh(
            ventricle_mask, spacing, _CATEGORY_DECIMATION["ventricle"], "ventricle"
        )

    def _apply_structure_color(self, mesh: trimesh.Trimesh, category: str) -> trimesh.Trimesh:
        """Красит меш в стандартный медицинский цвет структуры (см. STRUCTURE_COLORS)."""
        color = STRUCTURE_COLORS.get(category, (200, 200, 200, 255))
        mesh.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.vertices), 1))
        return mesh

    # --- Сборка сцены из вывода TotalSegmentator ------------------------------------ #

    def _categorize_totalsegmentator_masks(
        self, segmentation_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Группирует "сырые" имена структур TotalSegmentator в категории визуализации.

        TotalSegmentator сегментирует 104+ структуры туловища (задача "total"), из
        которых для визуализации мозга релевантны единицы — остальные молча
        игнорируются (не попадают ни в одну категорию).
        """
        if not segmentation_dict:
            return {}

        sample_shape = next(iter(segmentation_dict.values())).shape
        categorized: dict[str, np.ndarray] = {}

        def _accumulate(category: str, mask: np.ndarray) -> None:
            if category not in categorized:
                categorized[category] = np.zeros(sample_shape, dtype=np.uint8)
            categorized[category] = np.maximum(categorized[category], mask.astype(np.uint8))

        for name, mask in segmentation_dict.items():
            if mask is None or not np.any(mask):
                continue

            if name == "brain":
                _accumulate("brain", mask)
            elif "ventricle" in name:
                _accumulate("ventricle", mask)
            elif "bleed" in name or "hemorrhage" in name:
                _accumulate("hemorrhage", mask)
            elif "tumor" in name:
                _accumulate("tumor", mask)
            elif any(keyword in name for keyword in TotalSegmentatorWrapper.BONE_LABEL_KEYWORDS):
                _accumulate("bone", mask)
            elif (
                name in TotalSegmentatorWrapper.VESSEL_LABELS_TOTAL
                or name in TotalSegmentatorWrapper.VESSEL_LABELS_HEADNECK
            ):
                _accumulate("vessel", mask)

        return categorized

    def create_scene_from_totalsegmentator(
        self,
        segmentation_dict: dict[str, np.ndarray],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict[str, trimesh.Trimesh]:
        """Строит цветную сцену мешей из словаря масок TotalSegmentator.

        Ключи segmentation_dict — имена структур в номенклатуре TotalSegmentator (см.
        models.pretrained_models.TotalSegmentatorWrapper), например "brain", "skull",
        "vertebrae_L1", "internal_carotid_artery_left". Они группируются в категории
        brain/vessel/bone/hemorrhage/tumor/ventricle по ключевым словам, красятся в
        стандартные медицинские цвета (STRUCTURE_COLORS) и децимируются по правилам
        _CATEGORY_DECIMATION. Структуры, не подошедшие ни под одну категорию,
        пропускаются без ошибки.

        Args:
            segmentation_dict: {имя_структуры: бинарная маска (Z, Y, X)}.
            spacing: Spacing (x, y, z), мм, общий для всех масок словаря.

        Returns:
            {категория: раскрашенный trimesh.Trimesh} — только для непустых категорий.
        """
        categorized_masks = self._categorize_totalsegmentator_masks(segmentation_dict)

        meshes: dict[str, trimesh.Trimesh] = {}
        for category, mask in categorized_masks.items():
            mesh = self._create_structure_mesh(mask, spacing, _CATEGORY_DECIMATION[category], category)
            if mesh is not None:
                meshes[category] = self._apply_structure_color(mesh, category)

        logger.info(
            "create_scene_from_totalsegmentator: построено %d структур: %s", len(meshes), sorted(meshes)
        )
        return meshes

    # --- Полная сцена из отдельных масок --------------------------------------------- #

    def create_full_brain_scene(
        self,
        brain_mask: np.ndarray,
        tumor_mask: np.ndarray | None = None,
        vessel_mask: np.ndarray | None = None,
        hemorrhage_mask: np.ndarray | None = None,
        bone_mask: np.ndarray | None = None,
        ventricle_mask: np.ndarray | None = None,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict[str, trimesh.Trimesh]:
        """Строит полную раскрашенную сцену из отдельно переданных масок структур мозга.

        Структуры с mask=None либо пустой (все нули) маской пропускаются без ошибки —
        типичная ситуация, например, когда кровоизлияние не найдено или его
        сегментация вообще не запускалась для конкретного пациента.

        Args:
            brain_mask: Маска мозга (обязательна).
            tumor_mask: Маска опухоли (опционально).
            vessel_mask: Маска сосудов (опционально).
            hemorrhage_mask: Маска кровоизлияния (опционально, задел на будущее).
            bone_mask: Маска костей черепа (опционально).
            ventricle_mask: Маска желудочков (опционально).
            spacing: Spacing (x, y, z), мм, общий для всех масок.

        Returns:
            {"brain": mesh, "tumor": mesh, ...} — только для присутствующих структур.
        """
        structure_masks: dict[str, np.ndarray | None] = {
            "brain": brain_mask,
            "tumor": tumor_mask,
            "vessel": vessel_mask,
            "hemorrhage": hemorrhage_mask,
            "bone": bone_mask,
            "ventricle": ventricle_mask,
        }

        meshes: dict[str, trimesh.Trimesh] = {}
        for category, mask in structure_masks.items():
            mesh = self._create_structure_mesh(mask, spacing, _CATEGORY_DECIMATION[category], category)
            if mesh is not None:
                meshes[category] = self._apply_structure_color(mesh, category)

        logger.info("create_full_brain_scene: построено %d структур: %s", len(meshes), sorted(meshes))
        return meshes

    # --- Экспорт сцены ---------------------------------------------------------------- #

    def build_scene(self, meshes_dict: dict[str, trimesh.Trimesh]) -> trimesh.Scene:
        """Собирает trimesh.Scene из словаря мешей (без экспорта на диск).

        Промежуточный шаг перед add_metadata_to_scene() и/или scene.export()/export_scene().

        Args:
            meshes_dict: {имя_структуры: trimesh.Trimesh}.

        Returns:
            trimesh.Scene с каждым мешом как именованным узлом.
        """
        scene = trimesh.Scene()
        for name, mesh in meshes_dict.items():
            scene.add_geometry(mesh, node_name=name, geom_name=name)
        return scene

    def add_metadata_to_scene(
        self, scene: trimesh.Scene, metadata: SceneMetadata | dict[str, Any]
    ) -> trimesh.Scene:
        """Встраивает анонимизированные метаданные исследования в сцену для экспорта в GLB.

        Проверено эмпирически: trimesh сохраняет scene.metadata в GLB как JSON (поле
        "extras") и корректно восстанавливает его при повторной загрузке файла.

        ВАЖНО: metadata не должна содержать прямых идентификаторов пациента (ФИО, номер
        карты и т.п.) — SceneMetadata.patient_id_hash предполагает уже анонимизированный
        хэш/псевдоним, сформированный ДО вызова этого метода средствами приложения,
        отвечающими за анонимизацию (вне этого модуля).

        Args:
            scene: Сцена (см. build_scene()), в которую встраиваются метаданные.
            metadata: SceneMetadata либо произвольный JSON-совместимый словарь.

        Returns:
            Та же сцена с обновлённым scene.metadata.
        """
        metadata_dict = metadata.to_dict() if isinstance(metadata, SceneMetadata) else dict(metadata)
        scene.metadata.update(metadata_dict)
        logger.info("В сцену добавлены метаданные: %s", sorted(metadata_dict))
        return scene

    def export_scene(
        self,
        meshes_dict: dict[str, trimesh.Trimesh],
        filepath: str | Path,
        format: str = "glb",
    ) -> Path | dict[str, Path]:
        """Экспортирует сцену мешей в указанный формат.

        GLB и OBJ поддерживают несколько именованных объектов в одном файле (цвета и
        прозрачность — только GLB) — сцена сохраняется целиком в filepath. STL и PLY —
        форматы одного меша без понятия сцены, поэтому для них каждая структура
        сохраняется в свой файл рядом с filepath (<имя_без_расширения>_<структура>.<расширение>).

        Args:
            meshes_dict: {имя_структуры: trimesh.Trimesh}, обычно результат
                create_full_brain_scene()/create_scene_from_totalsegmentator().
            filepath: Путь к выходному файлу (glb/obj) или путь-шаблон (stl/ply).
            format: "glb" (основной формат для Three.js) | "obj" | "stl" (3D-печать,
                только геометрия) | "ply".

        Returns:
            Path к единому файлу сцены (glb/obj), либо {структура: Path} для stl/ply.

        Raises:
            MeshGenerationError: Если meshes_dict пуст либо формат не поддерживается.
        """
        if not meshes_dict:
            raise MeshGenerationError("export_scene: meshes_dict пуст — нечего экспортировать")

        fmt = format.strip().lower()
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt in _SCENE_FORMATS:
            scene = self.build_scene(meshes_dict)
            scene.export(output_path, file_type=fmt)
            logger.info(
                "Сцена (%d структур) экспортирована в %s: %s", len(meshes_dict), fmt.upper(), output_path
            )
            return output_path

        if fmt in _SINGLE_MESH_FORMATS:
            exported_paths: dict[str, Path] = {}
            for name, mesh in meshes_dict.items():
                structure_path = output_path.parent / f"{output_path.stem}_{name}.{fmt}"
                mesh.export(structure_path, file_type=fmt)
                exported_paths[name] = structure_path
                logger.info("Структура %r экспортирована в %s: %s", name, fmt.upper(), structure_path)
            return exported_paths

        supported = _SCENE_FORMATS + _SINGLE_MESH_FORMATS
        raise MeshGenerationError(f"Формат экспорта {format!r} не поддерживается (доступны: {supported})")

    # --- Оптимизация под веб и Level of Detail --------------------------------------- #

    def optimize_for_web(self, mesh: trimesh.Trimesh, target_polygons: int = 100_000) -> trimesh.Trimesh:
        """Агрессивно упрощает меш для веб-визуализации, не опускаясь ниже target_polygons.

        Квадратичная децимация по построению приоритетно сохраняет геометрически
        значимые детали (высокую кривизну) и в первую очередь убирает грани на
        плоских участках — это и есть "адаптивность", отдельная эвристика поверх
        алгоритма не требуется.

        Args:
            mesh: Исходный меш.
            target_polygons: Целевое число граней; если у меша уже меньше — меш
                возвращается без изменений.

        Returns:
            Упрощённый (или исходный) меш.
        """
        if len(mesh.faces) <= target_polygons:
            return mesh

        simplified_vertices, simplified_faces = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=target_polygons
        )
        optimized = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces, process=False)
        logger.info("optimize_for_web: %d -> %d граней", len(mesh.faces), len(optimized.faces))
        return optimized

    def create_lod_meshes(
        self, mesh: trimesh.Trimesh, levels: list[float] | None = None
    ) -> dict[str, trimesh.Trimesh]:
        """Строит несколько версий меша с разной детализацией (Level of Detail).

        Используется для прогрессивной загрузки в Three.js: сначала подгружается
        низкодетализированная версия (быстрый первый рендер силуэта), а по мере
        приближения камеры подменяется на более детальный уровень.

        Args:
            mesh: Исходный (наиболее детальный) меш.
            levels: Доли от исходного числа граней для каждого уровня (1.0 — без
                изменений, 0.5 — половина граней и т.д.); по умолчанию [1.0, 0.5, 0.25].

        Returns:
            {"lod_100": mesh, "lod_50": mesh, "lod_25": mesh, ...} — ключ отражает
            процент от исходной детализации.

        Raises:
            ValueError: Если какой-либо уровень вне диапазона (0, 1].
        """
        levels = levels if levels is not None else [1.0, 0.5, 0.25]
        original_face_count = len(mesh.faces)

        lod_meshes: dict[str, trimesh.Trimesh] = {}
        for level in levels:
            if not 0.0 < level <= 1.0:
                raise ValueError(f"Уровень LOD должен быть в (0, 1], получено {level}")

            key = f"lod_{int(round(level * 100))}"
            if level >= 1.0 or original_face_count <= 4:
                lod_meshes[key] = mesh
                continue

            target_face_count = max(4, int(original_face_count * level))
            simplified_vertices, simplified_faces = fast_simplification.simplify(
                mesh.vertices, mesh.faces, target_count=target_face_count
            )
            lod_meshes[key] = trimesh.Trimesh(
                vertices=simplified_vertices, faces=simplified_faces, process=False
            )
            logger.info("LOD %s: %d -> %d граней", key, original_face_count, len(lod_meshes[key].faces))

        return lod_meshes


if __name__ == "__main__":
    # --- Пример: полная сцена из отдельных масок ---
    import logging

    logging.basicConfig(level=logging.INFO)

    shape = (60, 120, 120)
    brain_mask = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    brain_mask[(z - 30) ** 2 + (y - 60) ** 2 + (x - 60) ** 2 <= 40**2] = 1

    tumor_mask = np.zeros(shape, dtype=np.uint8)
    tumor_mask[(z - 25) ** 2 + (y - 70) ** 2 + (x - 70) ** 2 <= 8**2] = 1

    generator = MeshGenerator()
    scene_meshes = generator.create_full_brain_scene(brain_mask, tumor_mask=tumor_mask)
    print("Структуры сцены:", {name: len(mesh.faces) for name, mesh in scene_meshes.items()})

    scene = generator.build_scene(scene_meshes)
    metadata = SceneMetadata(
        patient_id_hash="anon-9f3a21",
        study_modality="MRI",
        study_date="2026-09-05",
        segmentation_models=["synthetic-example"],
    )
    generator.add_metadata_to_scene(scene, metadata)
    scene.export("./data/examples/full_brain_scene.glb", file_type="glb")

    # --- Пример: сцена из вывода TotalSegmentator ---
    fake_totalsegmentator_output = {
        "brain": brain_mask,
        "skull": np.zeros(shape, dtype=np.uint8),  # пустая маска — будет пропущена
        "internal_carotid_artery_left": tumor_mask,  # для примера переиспользуем форму
    }
    ts_scene = generator.create_scene_from_totalsegmentator(fake_totalsegmentator_output)
    print("Из TotalSegmentator:", sorted(ts_scene))

    # --- Пример: Level of Detail ---
    brain_mesh = scene_meshes["brain"]
    lods = generator.create_lod_meshes(brain_mesh, levels=[1.0, 0.5, 0.25])
    print("LOD:", {key: len(mesh.faces) for key, mesh in lods.items()})
