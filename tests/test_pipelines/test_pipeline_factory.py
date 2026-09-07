"""Тесты фабрики пайплайнов: определение загрузчика и обработка неподдерживаемых форматов."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.exceptions import StudyLoadError
from pipelines.pipeline_factory import resolve_loader


def test_resolve_loader_raises_for_unknown_format(tmp_path: Path) -> None:
    unknown_file = tmp_path / "study.unknown"
    unknown_file.write_text("не медицинский файл")

    with pytest.raises(StudyLoadError):
        resolve_loader(unknown_file)


def test_resolve_loader_picks_nifti_by_extension(tmp_path: Path) -> None:
    nifti_file = tmp_path / "scan.nii.gz"
    nifti_file.write_bytes(b"\x1f\x8b\x00")  # заглушка, только для проверки расширения

    loader = resolve_loader(nifti_file)

    assert loader.__class__.__name__ == "NiftiLoader"
