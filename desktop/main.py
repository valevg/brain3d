"""Точка входа desktop-приложения."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from core.config import get_settings
from core.logging_config import setup_logging
from desktop.ui.main_window import MainWindow


def main() -> int:
    """Запускает Qt-приложение и главное окно.

    Returns:
        Код завершения процесса (для sys.exit).
    """
    settings = get_settings()
    setup_logging(level=settings.log_level)

    app = QApplication(sys.argv)
    app.setApplicationName("Brain3D AI")

    window = MainWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
