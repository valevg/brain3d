"""Qt-виджет, встраивающий сцену PyVista для отображения мешей мозга/опухоли/сосудов."""

from __future__ import annotations

import numpy as np
from pyvistaqt import QtInteractor

from shared.types import MeshAsset


class PyVistaViewer(QtInteractor):
    """3D-вьюер на PyVista: каждая структура рендерится отдельным полупрозрачным меш-актором."""

    def __init__(self, parent=None) -> None:  # noqa: ANN001 — стандартный Qt parent
        super().__init__(parent)
        self.set_background("black")

    def display_meshes(self, meshes: list[MeshAsset]) -> None:
        """Очищает сцену и отрисовывает переданный список мешей.

        Args:
            meshes: Меши структур (мозг/опухоль/сосуды/кровоизлияние), обычно —
                результат pipelines.BasePipeline.run().
        """
        self.clear()

        for mesh_asset in meshes:
            polydata = self._to_pyvista_polydata(mesh_asset)
            r, g, b, a = mesh_asset.color_rgba
            self.add_mesh(
                polydata,
                color=(r, g, b),
                opacity=a,
                smooth_shading=True,
                name=mesh_asset.structure.value,
            )

        self.reset_camera()

    @staticmethod
    def _to_pyvista_polydata(mesh_asset: MeshAsset):  # noqa: ANN205 — возвращает pyvista.PolyData
        """Конвертирует MeshAsset (вершины + треугольные грани) в pyvista.PolyData."""
        import pyvista as pv

        num_faces = len(mesh_asset.faces)
        # PyVista ожидает плоский массив [3, i0, i1, i2, 3, i0, i1, i2, ...]
        faces_with_size = np.hstack(
            [np.full((num_faces, 1), 3, dtype=np.int64), mesh_asset.faces.astype(np.int64)]
        ).flatten()
        return pv.PolyData(mesh_asset.vertices, faces_with_size)
