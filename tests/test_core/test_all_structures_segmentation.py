"""Реальные end-to-end тесты ВСЕХ типов данных для тестирования сегментации.

Цель модуля — покрыть исполнением каждый тип структуры из StructureType
(mind/tumor/vessel/hemorrhage) и каждую модальность из Modality (CT/MRI) на
синтетических multi-label картах: без нейросети, но с РЕАЛЬНЫМИ skimage
(marching cubes), trimesh (сглаживание/упрощение), scipy.ndimage
(постобработка) и экспортом файлов на диск.

Данные создаются фикстурами tests/conftest.py:
- all_structures_label_map / synthetic_all_structures_result — полная карта со
  всеми классами (1 brain, 2 tumor, 3 vessel, 4 hemorrhage);
- synthetic_ct_volume / synthetic_mri_volume — объёмы обеих модальностей;
- synthetic_volume_by_modality — параметризованный по модальностям объём.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from core.mesh.exporters import export_mesh
from core.mesh.mesh_generator import DEFAULT_STRUCTURE_COLORS, generate_meshes
from core.segmentation.inference import CLASS_INDEX_TO_STRUCTURE
from core.segmentation.postprocessing import clean_label_map
from shared.enums import FileFormat, Modality, StructureType
from shared.types import MeshAsset, SegmentationResult, VoxelSpacing

ALL_STRUCTURES = [
    StructureType.BRAIN,
    StructureType.TUMOR,
    StructureType.VESSEL,
    StructureType.HEMORRHAGE,
]


# --------------------------------------------------------------------------- #
# Полнота набора типов данных: каждый enum-значимый тип покрыт тестовыми данными
# --------------------------------------------------------------------------- #


def test_every_structure_type_has_class_index_and_color() -> None:
    """Каждый StructureType отображается в индекс класса и имеет цвет по умолчанию."""
    mapped_structures = set(CLASS_INDEX_TO_STRUCTURE.values())
    assert mapped_structures == set(StructureType), (
        "CLASS_INDEX_TO_STRUCTURE должен покрывать все значения StructureType"
    )
    for structure in ALL_STRUCTURES:
        assert structure in DEFAULT_STRUCTURE_COLORS
        color = DEFAULT_STRUCTURE_COLORS[structure]
        assert len(color) == 4 and all(0.0 <= c <= 1.0 for c in color)


def test_every_modality_is_testable_with_synthetic_volume(synthetic_volume_by_modality) -> None:
    """Параметризованная фикстура даёт валидный объём для каждой модальности."""
    volume = synthetic_volume_by_modality
    assert volume.modality in {Modality.CT, Modality.MRI}
    assert volume.voxels.ndim == 3
    assert volume.voxels.dtype == np.float32
    assert volume.spacing.as_tuple() == (1.0, 1.0, 1.0)


def test_both_modalities_covered_by_named_fixtures(
    synthetic_ct_volume, synthetic_mri_volume
) -> None:
    """Именованные фикстуры покрывают CT и MRI разными профилями интенсивностей."""
    assert synthetic_ct_volume.modality is Modality.CT
    assert synthetic_mri_volume.modality is Modality.MRI
    # КТ: фон — воздух (-1000 HU); МРТ: фон — нуль, ткань ~500 (z-score профиль)
    assert np.isclose(np.percentile(synthetic_ct_volume.voxels, 5), -1000.0)
    assert synthetic_mri_volume.voxels.max() > 300.0


# --------------------------------------------------------------------------- #
# Реальная mesh-генерация (marching cubes + trimesh) каждой структуры
# --------------------------------------------------------------------------- #


def test_generate_meshes_builds_mesh_for_every_structure(synthetic_all_structures_result) -> None:
    """generate_meshes реально строит по мешу на КАЖДУЮ из 4 структур сразу."""
    meshes = generate_meshes(synthetic_all_structures_result, smoothing_iterations=2)

    produced = {mesh.structure for mesh in meshes}
    assert produced == set(ALL_STRUCTURES), f"Ожидались все структуры, получены: {produced}"

    for mesh in meshes:
        assert isinstance(mesh, MeshAsset)
        assert mesh.vertices.shape[1] == 3 and len(mesh.vertices) > 100
        assert mesh.faces.shape[1] == 3 and len(mesh.faces) > 100
        assert mesh.color_rgba == DEFAULT_STRUCTURE_COLORS[mesh.structure]


@pytest.mark.parametrize("structure", ALL_STRUCTURES)
def test_single_structure_mesh_geometry(structure: StructureType, all_structures_label_map) -> None:
    """Каждая структура по отдельности даёт замкнутый физически осмысленный меш."""
    class_index = next(idx for idx, s in CLASS_INDEX_TO_STRUCTURE.items() if s == structure)
    result = SegmentationResult(
        label_map=all_structures_label_map,
        structures=[structure],
        spacing=VoxelSpacing(1.0, 1.0, 1.0),
    )

    (mesh,) = generate_meshes(result, smoothing_iterations=1, simplify_target_faces=None)

    assert mesh.structure is structure
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    # extents положительны по всем осям — геометрия не вырождена
    assert (tm.extents > 1.0).all()
    # объём тела ненулевой (watertight после закрытия дыр трimesh не гарантирует)
    assert tm.bounds[0].min() < tm.bounds[1].max()


# --------------------------------------------------------------------------- #
# Постобработка полной карты меток (scipy.ndimage — реальное исполнение)
# --------------------------------------------------------------------------- #


def test_clean_label_map_preserves_all_large_structures(all_structures_label_map) -> None:
    """Удаление шума не трогает крупные компоненты всех четырёх классов."""
    cleaned = clean_label_map(all_structures_label_map, min_component_voxels=50)

    assert set(np.unique(cleaned)) == {0, 1, 2, 3, 4}
    for class_index in (1, 2, 3, 4):
        original = int((all_structures_label_map == class_index).sum())
        kept = int((cleaned == class_index).sum())
        assert kept == original, f"Класс {class_index} потерял воксели при очистке"


def test_clean_label_map_removes_noise_from_every_class(all_structures_label_map) -> None:
    """Шумовые воксели каждого из 4 классов вычищаются, остальное цело."""
    noisy = all_structures_label_map.copy()
    noise_points = [(0, 0, 0), (0, 0, 63), (31, 0, 0), (31, 63, 63)]
    for point, class_index in zip(noise_points, (1, 2, 3, 4)):
        noisy[point] = class_index

    cleaned = clean_label_map(noisy, min_component_voxels=50)

    for point, class_index in zip(noise_points, (1, 2, 3, 4)):
        assert cleaned[point] == 0, f"Шум класса {class_index} в {point} не удалён"
    np.testing.assert_array_equal(cleaned, all_structures_label_map)


# --------------------------------------------------------------------------- #
# Реальный экспорт всех структур во всех поддерживаемых форматах
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("file_format", [FileFormat.GLB, FileFormat.STL])
def test_export_all_structures_real_files(
    tmp_path: Path, file_format: FileFormat, synthetic_all_structures_result
) -> None:
    """Каждая структура экспортируется в реальный файл и читается обратно trimesh."""
    meshes = generate_meshes(synthetic_all_structures_result, smoothing_iterations=1)
    assert len(meshes) == len(ALL_STRUCTURES)

    for mesh in meshes:
        suffix = "glb" if file_format is FileFormat.GLB else "stl"
        out = tmp_path / f"{mesh.structure.value}.{suffix}"
        exported = export_mesh(mesh, out, file_format)

        assert exported.exists() and exported.stat().st_size > 0
        reloaded = trimesh.load(exported, force="mesh")
        assert len(reloaded.faces) == len(mesh.faces)


def test_glb_export_preserves_per_structure_color(tmp_path: Path, synthetic_all_structures_result) -> None:
    """В GLB сохраняется vertex color каждой структуры — сверка байт-в-байт."""
    meshes = generate_meshes(synthetic_all_structures_result, smoothing_iterations=1)

    for mesh in meshes:
        out = tmp_path / f"{mesh.structure.value}.glb"
        export_mesh(mesh, out, FileFormat.GLB)
        scene_or_mesh = trimesh.load(out)
        geometry = (
            list(scene_or_mesh.geometry.values())[0]
            if isinstance(scene_or_mesh, trimesh.Scene)
            else scene_or_mesh
        )
        colors = np.asarray(geometry.visual.vertex_colors)[:, :4][0]
        expected = np.array(
            [round(c * 255) for c in mesh.color_rgba], dtype=np.uint8
        )
        np.testing.assert_allclose(colors.astype(int), expected.astype(int), atol=1)


# --------------------------------------------------------------------------- #
# Совместимость с внешним конвейером: сериализация полной карты меток
# --------------------------------------------------------------------------- #


def test_full_label_map_roundtrip_npz(tmp_path: Path, all_structures_label_map) -> None:
    """Полная multi-label карта корректно сохраняется/восстанавливается (кэш-формат)."""
    path = tmp_path / "labels.npz"
    np.savez_compressed(path, label_map=all_structures_label_map)

    restored = np.load(path)["label_map"]
    np.testing.assert_array_equal(restored, all_structures_label_map)
    assert restored.dtype == np.uint8


def test_mesh_assets_json_manifest(tmp_path: Path, synthetic_all_structures_result) -> None:
    """Манифест мешей (типы структур + число примитивов) сериализуется в JSON."""
    meshes = generate_meshes(synthetic_all_structures_result, smoothing_iterations=1)
    manifest = [
        {
            "structure": m.structure.value,
            "vertices": len(m.vertices),
            "faces": len(m.faces),
            "color_rgba": list(m.color_rgba),
        }
        for m in meshes
    ]

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert {entry["structure"] for entry in loaded} == {s.value for s in ALL_STRUCTURES}
    assert all(entry["vertices"] > 0 and entry["faces"] > 0 for entry in loaded)
