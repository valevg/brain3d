"""Тесты DicomLoader: определение модальности и обработка отсутствующей серии."""

from __future__ import annotations

from pathlib import Path

from core.io.dicom_loader import DicomLoader


def test_can_load_returns_false_for_empty_directory(tmp_path: Path) -> None:
    loader = DicomLoader()
    assert loader.can_load(tmp_path) is False


def test_can_load_returns_false_for_file(tmp_path: Path) -> None:
    loader = DicomLoader()
    fake_file = tmp_path / "not_a_directory.txt"
    fake_file.write_text("test")

    assert loader.can_load(fake_file) is False
