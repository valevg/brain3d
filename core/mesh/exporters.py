"""Экспорт мешей в форматы GLB (для web/Three.js) и STL (для desktop/3D-печати)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from core.exceptions import MeshGenerationError
from core.logging_config import get_logger
from shared.enums import FileFormat
from shared.types import MeshAsset

logger = get_logger(__name__)

_SUPPORTED_FORMATS = {FileFormat.GLB, FileFormat.STL}


def export_mesh(mesh_asset: MeshAsset, output_path: Path, file_format: FileFormat) -> Path:
    """Сохраняет один MeshAsset в файл указанного формата.

    Args:
        mesh_asset: Меш структуры с вершинами, гранями и цветом.
        output_path: Целевой путь файла (директория будет создана при необходимости).
        file_format: FileFormat.GLB или FileFormat.STL.

    Returns:
        Путь к сохранённому файлу.

    Raises:
        MeshGenerationError: Если формат не поддерживается экспортом.
    """
    if file_format not in _SUPPORTED_FORMATS:
        raise MeshGenerationError(f"Экспорт в формат {file_format.value} не поддерживается")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.Trimesh(vertices=mesh_asset.vertices, faces=mesh_asset.faces, process=False)

    if file_format == FileFormat.GLB:
        # Цвет структуры сохраняется как vertex color — Three.js читает его из GLB напрямую.
        rgba_255 = np.array(mesh_asset.color_rgba, dtype=np.float64) * 255
        mesh.visual.vertex_colors = np.tile(rgba_255.astype(np.uint8), (len(mesh.vertices), 1))
        mesh.export(output_path, file_type="glb")
    else:
        mesh.export(output_path, file_type="stl")

    logger.info("Меш структуры %s сохранён: %s", mesh_asset.structure.value, output_path)
    return output_path


def export_meshes(
    mesh_assets: list[MeshAsset],
    output_dir: Path,
    file_format: FileFormat = FileFormat.GLB,
) -> list[Path]:
    """Сохраняет список мешей, по одному файлу на структуру, с именем по StructureType.

    Args:
        mesh_assets: Список мешей (обычно результат core.mesh.generate_meshes).
        output_dir: Директория для сохранения файлов.
        file_format: Формат экспорта, одинаковый для всех мешей.

    Returns:
        Список путей к сохранённым файлам, в том же порядке, что и mesh_assets.
    """
    extension = "glb" if file_format == FileFormat.GLB else "stl"
    return [
        export_mesh(asset, output_dir / f"{asset.structure.value}.{extension}", file_format)
        for asset in mesh_assets
    ]
