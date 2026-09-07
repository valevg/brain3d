"""Главное окно desktop-приложения: панель управления + встроенный 3D-вьюер PyVista."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.exceptions import Brain3DError
from core.logging_config import get_logger
from desktop.viewer.pyvista_widget import PyVistaViewer
from pipelines.pipeline_factory import create_pipeline

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """Главное окно: кнопка загрузки исследования, статус обработки и 3D-вьюер."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Brain3D AI — Desktop")
        self.resize(1280, 800)

        self._viewer = PyVistaViewer(self)
        self._status_label = QLabel("Загрузите исследование (DICOM-папка или NIfTI-файл)")
        self._open_button = QPushButton("Открыть исследование…")
        self._open_button.clicked.connect(self._on_open_study_clicked)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self._open_button)
        controls_layout.addWidget(self._status_label, stretch=1)

        central_layout = QVBoxLayout()
        central_layout.addLayout(controls_layout)
        central_layout.addWidget(self._viewer, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

    def _on_open_study_clicked(self) -> None:
        """Обработчик кнопки открытия: выбор файла/папки, запуск пайплайна, рендер мешей."""
        path_str = QFileDialog.getExistingDirectory(self, "Выберите папку с DICOM-серией")
        if not path_str:
            path_str, _ = QFileDialog.getOpenFileName(
                self, "Или выберите NIfTI-файл", filter="NIfTI (*.nii *.nii.gz)"
            )
        if not path_str:
            return

        self._status_label.setText("Обработка… это может занять несколько минут")
        self._open_button.setEnabled(False)
        try:
            pipeline, volume = create_pipeline(Path(path_str))
            meshes = pipeline.run(volume)
            self._viewer.display_meshes(meshes)
            self._status_label.setText(f"Готово: найдено структур — {len(meshes)}")
        except Brain3DError as exc:
            logger.exception("Ошибка обработки исследования")
            QMessageBox.critical(self, "Ошибка обработки", str(exc))
            self._status_label.setText("Ошибка обработки — см. лог")
        finally:
            self._open_button.setEnabled(True)
