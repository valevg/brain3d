"""Единая настройка логирования для всех точек входа проекта (api, desktop, pipelines)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Настраивает корневой логгер приложения.

    Args:
        level: Уровень логирования ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Необязательный путь к файлу лога. Если задан, логи дублируются в файл.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Очищаем ранее добавленные хендлеры, чтобы избежать дублирования
    # при повторном вызове (например, в тестах).
    root_logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Возвращает именованный логгер модуля.

    Args:
        name: Обычно ``__name__`` вызывающего модуля.

    Returns:
        Настроенный экземпляр Logger.
    """
    return logging.getLogger(name)
