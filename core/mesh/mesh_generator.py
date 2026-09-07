"""Генерация 3D-мешей из карты сегментации методом marching cubes (skimage) + сглаживание (trimesh)."""

from __future__ import annotations

import fast_simplification
import numpy as np
import trimesh
from skimage import measure

from core.exceptions import MeshGenerationError
from core.logging_config import get_logger
from core.segmentation.inference import CLASS_INDEX_TO_STRUCTURE
from shared.enums import StructureType
from shared.types import MeshAsset, SegmentationResult, VoxelSpacing

logger = get_logger(__name__)

# Цвета структур по умолчанию (RGBA, 0..1) согласно ТЗ проекта.
DEFAULT_STRUCTURE_COLORS: dict[StructureType, tuple[float, float, float, float]] = {
    StructureType.BRAIN: (0.8, 0.8, 0.8, 0.35),  # полупрозрачный серый
    StructureType.TUMOR: (0.9, 0.15, 0.1, 1.0),  # красный
    StructureType.VESSEL: (0.1, 0.3, 0.9, 1.0),  # синий
    StructureType.HEMORRHAGE: (1.0, 0.55, 0.0, 1.0),  # оранжевый (плановая структура)
}


def generate_meshes(
    result: SegmentationResult,
    smoothing_iterations: int = 10,
    simplify_target_faces: int | None = 50_000,
) -> list[MeshAsset]:
    """Строит по одному мешу на каждую структуру, присутствующую в результате сегментации.

    Args:
        result: Результат сегментации (карта меток + список найденных структур).
        smoothing_iterations: Число итераций сглаживания Laplacian для каждого меша.
        simplify_target_faces: Целевое число граней после упрощения меша (None — без упрощения).
            Упрощение важно для веб-визуализации (Three.js) и производительности PyVista.

    Returns:
        Список MeshAsset, готовых к экспорту в GLB/STL.

    Raises:
        MeshGenerationError: Если ни одна структура не дала валидный меш.
    """
    meshes: list[MeshAsset] = []

    for class_index, structure in CLASS_INDEX_TO_STRUCTURE.items():
        if structure not in result.structures:
            continue

        binary_mask = (result.label_map == class_index).astype(np.uint8)
        if not np.any(binary_mask):
            continue

        try:
            mesh_asset = _mesh_from_binary_mask(
                binary_mask=binary_mask,
                spacing=result.spacing,
                structure=structure,
                smoothing_iterations=smoothing_iterations,
                simplify_target_faces=simplify_target_faces,
            )
            meshes.append(mesh_asset)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось построить меш для структуры %s: %s", structure.value, exc)

    if not meshes:
        raise MeshGenerationError("Ни для одной структуры не удалось построить меш")

    return meshes


def _mesh_from_binary_mask(
    binary_mask: np.ndarray,
    spacing: VoxelSpacing,
    structure: StructureType,
    smoothing_iterations: int,
    simplify_target_faces: int | None,
) -> MeshAsset:
    """Строит один MeshAsset из бинарной маски методом marching cubes."""
    # marching_cubes ожидает spacing в порядке осей массива (Z, Y, X)
    verts, faces, _normals, _values = measure.marching_cubes(
        binary_mask,
        level=0.5,
        spacing=(spacing.z, spacing.y, spacing.x),
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)

    if simplify_target_faces is not None and len(mesh.faces) > simplify_target_faces:
        # trimesh.Trimesh.simplify_quadric_decimation в установленной версии (4.4.1)
        # требует open3d (тяжёлая необязательная зависимость) — используем лёгкий
        # fast-simplification напрямую вместо него.
        simplified_vertices, simplified_faces = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=simplify_target_faces
        )
        mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces, process=False)

    color = DEFAULT_STRUCTURE_COLORS[structure]
    return MeshAsset(
        structure=structure,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        color_rgba=color,
    )
