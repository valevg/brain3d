"""Тесты core/mesh_generator.py.

Все тесты используют реальные trimesh/skimage/fast-simplification — decimation,
marching cubes, GLB/OBJ/STL/PLY экспорт и метаданные проверяются end-to-end на
синтетических масках, без моков (в отличие от предыдущих модулей, здесь нет
внешних сетевых/GPU систем — вся логика исполняется локально).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from core.exceptions import MeshGenerationError
from core.mesh_generator import (
    STRUCTURE_COLORS,
    MeshGenerator,
    SceneMetadata,
)


def _make_sphere_mask(
    shape: tuple[int, int, int] = (30, 30, 30),
    radius: int = 10,
    center: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Синтетическая сферическая маска — даёт валидный замкнутый меш через marching cubes."""
    center = center or (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = np.zeros(shape, dtype=np.uint8)
    mask[(z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 <= radius**2] = 1
    return mask


# --------------------------------------------------------------------------- #
# Базовая функциональность
# --------------------------------------------------------------------------- #


class TestCreateMeshMarchingCubes:
    def test_builds_valid_mesh_from_sphere(self) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()

        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_raises_for_empty_mask(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        generator = MeshGenerator()

        with pytest.raises(MeshGenerationError):
            generator.create_mesh_marching_cubes(mask)

    def test_smooth_and_decimate_applied_when_requested(self) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()

        raw_mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)
        processed_mesh = generator.create_mesh_marching_cubes(mask, smooth=5, decimate=0.5)

        assert len(processed_mesh.faces) < len(raw_mesh.faces)

    def test_spacing_scales_mesh_extent(self) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()

        mesh_unit_spacing = generator.create_mesh_marching_cubes(mask, spacing=(1.0, 1.0, 1.0), smooth=0)
        mesh_double_spacing = generator.create_mesh_marching_cubes(mask, spacing=(2.0, 2.0, 2.0), smooth=0)

        extent_unit = mesh_unit_spacing.vertices.max(axis=0) - mesh_unit_spacing.vertices.min(axis=0)
        extent_double = mesh_double_spacing.vertices.max(axis=0) - mesh_double_spacing.vertices.min(axis=0)
        np.testing.assert_allclose(extent_double, extent_unit * 2.0, rtol=0.05)


class TestSmoothMesh:
    def test_returns_same_object_for_zero_iterations(self) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)

        result = generator.smooth_mesh(mesh, iterations=0)

        assert result is mesh

    def test_smoothing_preserves_vertex_and_face_count(self) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)

        smoothed = generator.smooth_mesh(mesh, iterations=5)

        assert len(smoothed.vertices) == len(mesh.vertices)
        assert len(smoothed.faces) == len(mesh.faces)


class TestDecimateMesh:
    def test_reduces_face_count_by_requested_fraction(self) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)
        original_faces = len(mesh.faces)

        decimated = generator.decimate_mesh(mesh, target_reduction=0.5)

        assert len(decimated.faces) < original_faces
        assert len(decimated.faces) == pytest.approx(original_faces * 0.5, rel=0.1)

    def test_raises_for_out_of_range_target_reduction(self) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)

        with pytest.raises(ValueError):
            generator.decimate_mesh(mesh, target_reduction=1.5)
        with pytest.raises(ValueError):
            generator.decimate_mesh(mesh, target_reduction=0.0)

    def test_tiny_mesh_returned_unchanged(self) -> None:
        tiny_mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False
        )
        generator = MeshGenerator()

        result = generator.decimate_mesh(tiny_mesh, target_reduction=0.5)

        assert result is tiny_mesh


# --------------------------------------------------------------------------- #
# Меши по структурам
# --------------------------------------------------------------------------- #


class TestStructureMeshMethods:
    @pytest.mark.parametrize(
        "method_name",
        [
            "create_brain_mesh",
            "create_tumor_mesh",
            "create_vessel_mesh",
            "create_hemorrhage_mesh",
            "create_bone_mesh",
            "create_ventricle_mesh",
        ],
    )
    def test_returns_mesh_for_nonempty_mask(self, method_name: str) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()

        mesh = getattr(generator, method_name)(mask)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.faces) > 0

    @pytest.mark.parametrize(
        "method_name",
        [
            "create_brain_mesh",
            "create_tumor_mesh",
            "create_vessel_mesh",
            "create_hemorrhage_mesh",
            "create_bone_mesh",
            "create_ventricle_mesh",
        ],
    )
    def test_returns_none_for_empty_mask_without_raising(self, method_name: str) -> None:
        empty_mask = np.zeros((10, 10, 10), dtype=np.uint8)
        generator = MeshGenerator()

        result = getattr(generator, method_name)(empty_mask)

        assert result is None

    def test_returns_none_for_none_mask(self) -> None:
        generator = MeshGenerator()
        assert generator.create_tumor_mesh(None) is None  # type: ignore[arg-type]

    def test_tumor_mesh_decimated_less_than_brain_mesh(self) -> None:
        """Опухоль (10% удаления) должна сохранять больше граней, чем мозг (30%) при равном исходнике."""
        mask = _make_sphere_mask(radius=12)
        generator = MeshGenerator(default_smoothing_iterations=0)

        raw = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)
        tumor_mesh = generator.create_tumor_mesh(mask)
        brain_mesh = generator.create_brain_mesh(mask)

        assert len(tumor_mesh.faces) > len(brain_mesh.faces)
        assert len(tumor_mesh.faces) < len(raw.faces)


# --------------------------------------------------------------------------- #
# create_scene_from_totalsegmentator
# --------------------------------------------------------------------------- #


class TestCreateSceneFromTotalSegmentator:
    def test_categorizes_and_colors_known_structures(self) -> None:
        shape = (30, 30, 30)
        brain_mask = _make_sphere_mask(shape, radius=10)
        skull_mask = _make_sphere_mask(shape, radius=12)
        vessel_mask = _make_sphere_mask(shape, radius=4, center=(20, 20, 20))

        segmentation_dict = {
            "brain": brain_mask,
            "skull": skull_mask,
            "internal_carotid_artery_left": vessel_mask,
            "liver": _make_sphere_mask(shape, radius=3),  # не относится ни к одной категории
        }

        generator = MeshGenerator()
        scene_meshes = generator.create_scene_from_totalsegmentator(segmentation_dict)

        assert set(scene_meshes) == {"brain", "bone", "vessel"}
        assert "liver" not in scene_meshes

        brain_colors = np.unique(scene_meshes["brain"].visual.vertex_colors, axis=0)
        assert len(brain_colors) == 1
        np.testing.assert_array_equal(brain_colors[0], np.array(STRUCTURE_COLORS["brain"], dtype=np.uint8))

    def test_empty_masks_are_skipped(self) -> None:
        segmentation_dict = {"brain": np.zeros((10, 10, 10), dtype=np.uint8)}
        generator = MeshGenerator()

        scene_meshes = generator.create_scene_from_totalsegmentator(segmentation_dict)

        assert scene_meshes == {}

    def test_empty_dict_returns_empty_scene(self) -> None:
        generator = MeshGenerator()
        assert generator.create_scene_from_totalsegmentator({}) == {}

    def test_combines_multiple_bone_labels_into_one_mesh(self) -> None:
        shape = (20, 20, 20)
        vertebra_1 = _make_sphere_mask(shape, radius=3, center=(5, 5, 5))
        vertebra_2 = _make_sphere_mask(shape, radius=3, center=(15, 15, 15))

        segmentation_dict = {"vertebrae_L1": vertebra_1, "vertebrae_L2": vertebra_2}
        generator = MeshGenerator()

        scene_meshes = generator.create_scene_from_totalsegmentator(segmentation_dict)

        assert set(scene_meshes) == {"bone"}
        # Комбинированная маска должна давать меш с вершинами вблизи ОБЕИХ сфер
        bounds = scene_meshes["bone"].bounds
        assert bounds[1][0] - bounds[0][0] > 8  # разброс по оси шире одной сферы радиуса 3


# --------------------------------------------------------------------------- #
# create_full_brain_scene
# --------------------------------------------------------------------------- #


class TestCreateFullBrainScene:
    def test_includes_only_provided_nonempty_structures(self) -> None:
        shape = (30, 30, 30)
        brain_mask = _make_sphere_mask(shape, radius=10)
        tumor_mask = _make_sphere_mask(shape, radius=4, center=(20, 20, 20))
        empty_hemorrhage_mask = np.zeros(shape, dtype=np.uint8)

        generator = MeshGenerator()
        scene_meshes = generator.create_full_brain_scene(
            brain_mask, tumor_mask=tumor_mask, hemorrhage_mask=empty_hemorrhage_mask
        )

        assert set(scene_meshes) == {"brain", "tumor"}

    def test_none_masks_are_skipped_without_error(self) -> None:
        shape = (20, 20, 20)
        brain_mask = _make_sphere_mask(shape, radius=8)
        generator = MeshGenerator()

        scene_meshes = generator.create_full_brain_scene(brain_mask)

        assert set(scene_meshes) == {"brain"}

    def test_all_structures_present(self) -> None:
        shape = (30, 30, 30)
        generator = MeshGenerator()
        masks = {
            "brain_mask": _make_sphere_mask(shape, radius=10),
            "tumor_mask": _make_sphere_mask(shape, radius=3, center=(20, 20, 20)),
            "vessel_mask": _make_sphere_mask(shape, radius=2, center=(5, 5, 5)),
            "hemorrhage_mask": _make_sphere_mask(shape, radius=2, center=(25, 5, 5)),
            "bone_mask": _make_sphere_mask(shape, radius=13),
            "ventricle_mask": _make_sphere_mask(shape, radius=2, center=(15, 15, 15)),
        }

        scene_meshes = generator.create_full_brain_scene(**masks)

        assert set(scene_meshes) == {"brain", "tumor", "vessel", "hemorrhage", "bone", "ventricle"}
        for category, mesh in scene_meshes.items():
            colors = np.unique(mesh.visual.vertex_colors, axis=0)
            np.testing.assert_array_equal(colors[0], np.array(STRUCTURE_COLORS[category], dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Сцена, метаданные и экспорт
# --------------------------------------------------------------------------- #


class TestBuildSceneAndMetadata:
    def test_build_scene_contains_named_geometries(self) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        scene_meshes = {"brain": generator.create_brain_mesh(mask)}

        scene = generator.build_scene(scene_meshes)

        assert set(scene.geometry.keys()) == {"brain"}

    def test_add_metadata_round_trips_through_glb(self, tmp_path: Path) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        scene = generator.build_scene({"brain": generator.create_brain_mesh(mask)})

        metadata = SceneMetadata(
            patient_id_hash="anon-1234",
            study_modality="MRI",
            study_date="2026-09-05",
            segmentation_models=["totalsegmentator", "frangi"],
        )
        generator.add_metadata_to_scene(scene, metadata)

        output_path = tmp_path / "scene.glb"
        scene.export(output_path, file_type="glb")

        reloaded = trimesh.load(output_path, file_type="glb")

        assert reloaded.metadata["patient_id_hash"] == "anon-1234"
        assert reloaded.metadata["study_modality"] == "MRI"
        assert reloaded.metadata["segmentation_models"] == ["totalsegmentator", "frangi"]

    def test_add_metadata_accepts_plain_dict(self) -> None:
        generator = MeshGenerator()
        scene = trimesh.Scene()

        generator.add_metadata_to_scene(scene, {"custom_field": "value"})

        assert scene.metadata["custom_field"] == "value"


class TestExportScene:
    def test_raises_for_empty_meshes_dict(self, tmp_path: Path) -> None:
        generator = MeshGenerator()
        with pytest.raises(MeshGenerationError):
            generator.export_scene({}, tmp_path / "scene.glb", format="glb")

    def test_raises_for_unsupported_format(self, tmp_path: Path) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        meshes = {"brain": generator.create_brain_mesh(mask)}

        with pytest.raises(MeshGenerationError):
            generator.export_scene(meshes, tmp_path / "scene.xyz", format="xyz")

    def test_glb_export_writes_single_file(self, tmp_path: Path) -> None:
        shape = (30, 30, 30)
        generator = MeshGenerator()
        meshes = {
            "brain": generator.create_brain_mesh(_make_sphere_mask(shape, radius=10)),
            "tumor": generator.create_tumor_mesh(_make_sphere_mask(shape, radius=3, center=(20, 20, 20))),
        }

        result_path = generator.export_scene(meshes, tmp_path / "scene.glb", format="glb")

        assert result_path == tmp_path / "scene.glb"
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_obj_export_writes_single_file(self, tmp_path: Path) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        meshes = {"brain": generator.create_brain_mesh(mask)}

        result_path = generator.export_scene(meshes, tmp_path / "scene.obj", format="OBJ")

        assert result_path.exists()

    def test_stl_export_writes_one_file_per_structure(self, tmp_path: Path) -> None:
        shape = (30, 30, 30)
        generator = MeshGenerator()
        meshes = {
            "brain": generator.create_brain_mesh(_make_sphere_mask(shape, radius=10)),
            "tumor": generator.create_tumor_mesh(_make_sphere_mask(shape, radius=3, center=(20, 20, 20))),
        }

        result_paths = generator.export_scene(meshes, tmp_path / "scene.stl", format="stl")

        assert set(result_paths) == {"brain", "tumor"}
        assert result_paths["brain"] == tmp_path / "scene_brain.stl"
        assert result_paths["brain"].exists()
        assert result_paths["tumor"].exists()

    def test_ply_export_writes_one_file_per_structure(self, tmp_path: Path) -> None:
        mask = _make_sphere_mask()
        generator = MeshGenerator()
        meshes = {"brain": generator.create_brain_mesh(mask)}

        result_paths = generator.export_scene(meshes, tmp_path / "scene.ply", format="ply")

        assert result_paths["brain"].exists()


# --------------------------------------------------------------------------- #
# Оптимизация под веб и LOD
# --------------------------------------------------------------------------- #


class TestOptimizeForWeb:
    def test_returns_unchanged_mesh_below_target(self) -> None:
        mask = _make_sphere_mask(radius=5)
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)

        result = generator.optimize_for_web(mesh, target_polygons=10**9)

        assert result is mesh

    def test_reduces_to_target_polygons(self) -> None:
        mask = _make_sphere_mask(radius=12)
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)
        target = len(mesh.faces) // 4

        optimized = generator.optimize_for_web(mesh, target_polygons=target)

        assert len(optimized.faces) <= len(mesh.faces)
        assert len(optimized.faces) == pytest.approx(target, rel=0.15)


class TestCreateLodMeshes:
    def test_default_levels_produce_decreasing_face_counts(self) -> None:
        mask = _make_sphere_mask(radius=12)
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)

        lods = generator.create_lod_meshes(mesh)

        assert set(lods) == {"lod_100", "lod_50", "lod_25"}
        assert lods["lod_100"] is mesh
        assert len(lods["lod_50"].faces) < len(lods["lod_100"].faces)
        assert len(lods["lod_25"].faces) < len(lods["lod_50"].faces)

    def test_custom_levels(self) -> None:
        mask = _make_sphere_mask(radius=10)
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)

        lods = generator.create_lod_meshes(mesh, levels=[0.8])

        assert set(lods) == {"lod_80"}
        assert len(lods["lod_80"].faces) < len(mesh.faces)

    def test_raises_for_invalid_level(self) -> None:
        mask = _make_sphere_mask(radius=8)
        generator = MeshGenerator()
        mesh = generator.create_mesh_marching_cubes(mask, smooth=0, decimate=None)

        with pytest.raises(ValueError):
            generator.create_lod_meshes(mesh, levels=[1.5])
        with pytest.raises(ValueError):
            generator.create_lod_meshes(mesh, levels=[0.0])
