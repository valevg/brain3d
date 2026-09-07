"""Тесты core/dicom_loader.py: определение модальности, МР-взвешивания, HU-конвертация,
сортировка срезов, валидация серий и MRISeriesAnalyzer.

Пиксельные/DICOM-заголовочные данные не читаются с реального диска — вместо этого
pydicom.dcmread монкипатчится на фейковую функцию, возвращающую заранее собранные
объекты SimpleNamespace с нужными DICOM-тегами. Это позволяет проверять логику
разбора тегов без необходимости генерировать валидные DICOM-байты, и при этом не
превращает тесты в проверку моков ради моков — вся бизнес-логика (сортировка,
классификация, HU-пересчёт, построение геометрии) исполняется по-настоящему.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydicom.errors import InvalidDicomError

import core.dicom_loader as dicom_loader_module
from core.dicom_loader import (
    DICOMLoader,
    MRISeriesAnalyzer,
    _apply_ct_rescale,
    _classify_mr_weighting,
    _detect_modality_from_headers,
    group_datasets_by_series,
    read_dicom_headers,
    sort_files_by_slice_position,
)
from core.exceptions import (
    InvalidDicomSeriesError,
    MixedModalitySeriesError,
    SequenceNotFoundError,
    UnsupportedModalityError,
)
from shared.enums import Modality, MRWeighting


def _fake_dataset(**tags: Any) -> SimpleNamespace:
    """Строит лёгкий объект-заглушку с DICOM-тегами как атрибутами (вместо pydicom.Dataset).

    getattr(ds, "Tag", default) работает с ним так же, как с настоящим Dataset,
    чего достаточно для всей логики core.dicom_loader — код нигде не проверяет
    isinstance(ds, pydicom.Dataset).
    """
    return SimpleNamespace(**tags)


def _make_fake_dcmread(
    headers_by_path: dict[Path, SimpleNamespace], full_by_path: dict[Path, SimpleNamespace] | None = None
):
    """Создаёт функцию-замену pydicom.dcmread, различающую header-only и полное чтение."""
    full_by_path = full_by_path or {}

    def _fake_dcmread(path: Path, stop_before_pixels: bool = False) -> SimpleNamespace:
        path = Path(path)
        if path.name == "not_dicom.txt":
            raise InvalidDicomError("не DICOM")
        if stop_before_pixels:
            return headers_by_path[path]
        return full_by_path.get(path, headers_by_path[path])

    return _fake_dcmread


# --------------------------------------------------------------------------- #
# read_dicom_headers / group_datasets_by_series / sort_files_by_slice_position
# --------------------------------------------------------------------------- #


class TestDirectoryScanning:
    def test_read_dicom_headers_skips_invalid_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dcm_file = tmp_path / "slice1.dcm"
        dcm_file.write_bytes(b"")
        junk_file = tmp_path / "not_dicom.txt"
        junk_file.write_text("not a dicom file")

        fake_header = _fake_dataset(SeriesInstanceUID="1.2.3", Modality="CT")
        monkeypatch.setattr(
            dicom_loader_module.pydicom, "dcmread", _make_fake_dcmread({dcm_file: fake_header})
        )

        headers = read_dicom_headers(tmp_path)

        assert list(headers.keys()) == [dcm_file]

    def test_group_datasets_by_series(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.dcm"
        file_b = tmp_path / "b.dcm"
        file_c = tmp_path / "c.dcm"
        headers = {
            file_a: _fake_dataset(SeriesInstanceUID="series-1"),
            file_b: _fake_dataset(SeriesInstanceUID="series-1"),
            file_c: _fake_dataset(SeriesInstanceUID="series-2"),
        }

        groups = group_datasets_by_series(headers)

        assert set(groups["series-1"]) == {file_a, file_b}
        assert groups["series-2"] == [file_c]

    def test_sort_files_by_slice_position_orders_by_projection(self, tmp_path: Path) -> None:
        file_top = tmp_path / "top.dcm"
        file_mid = tmp_path / "mid.dcm"
        file_bottom = tmp_path / "bottom.dcm"
        orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # аксиальная плоскость, нормаль вдоль +z

        headers = {
            file_top: _fake_dataset(
                ImageOrientationPatient=orientation, ImagePositionPatient=(0.0, 0.0, 20.0)
            ),
            file_mid: _fake_dataset(
                ImageOrientationPatient=orientation, ImagePositionPatient=(0.0, 0.0, 10.0)
            ),
            file_bottom: _fake_dataset(
                ImageOrientationPatient=orientation, ImagePositionPatient=(0.0, 0.0, 0.0)
            ),
        }

        ordered = sort_files_by_slice_position(headers)

        assert ordered == [file_bottom, file_mid, file_top]

    def test_sort_files_falls_back_to_instance_number_without_orientation(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.dcm"
        file_b = tmp_path / "b.dcm"
        headers = {
            file_a: _fake_dataset(InstanceNumber=2),
            file_b: _fake_dataset(InstanceNumber=1),
        }

        ordered = sort_files_by_slice_position(headers)

        assert ordered == [file_b, file_a]


# --------------------------------------------------------------------------- #
# Определение модальности
# --------------------------------------------------------------------------- #


class TestModalityDetection:
    def test_detects_ct(self) -> None:
        ds = _fake_dataset(Modality="CT", RescaleSlope=1.0, RescaleIntercept=-1024.0)
        assert _detect_modality_from_headers([ds]) == Modality.CT

    def test_detects_mri(self) -> None:
        ds = _fake_dataset(Modality="MR", ScanningSequence=["SE"])
        assert _detect_modality_from_headers([ds]) == Modality.MRI

    def test_fallback_heuristic_detects_mri_without_modality_tag(self) -> None:
        ds = _fake_dataset(Modality="", ScanningSequence=["GR"], EchoTime=90.0)
        assert _detect_modality_from_headers([ds]) == Modality.MRI

    def test_fallback_heuristic_detects_ct_without_modality_tag(self) -> None:
        ds = _fake_dataset(Modality="", RescaleSlope=1.0)
        assert _detect_modality_from_headers([ds]) == Modality.CT

    def test_raises_for_unsupported_modality(self) -> None:
        ds = _fake_dataset(Modality="PT")
        with pytest.raises(UnsupportedModalityError):
            _detect_modality_from_headers([ds])


# --------------------------------------------------------------------------- #
# Определение МР-взвешивания
# --------------------------------------------------------------------------- #


class TestMRWeightingClassification:
    def test_t1_by_keyword(self) -> None:
        ds = _fake_dataset(SeriesDescription="Ax T1 pre")
        assert _classify_mr_weighting(ds) == MRWeighting.T1

    def test_t1ce_by_keyword_with_contrast(self) -> None:
        ds = _fake_dataset(SeriesDescription="Ax T1 post", ContrastBolusAgent="Gadovist")
        assert _classify_mr_weighting(ds) == MRWeighting.T1CE

    def test_t1_becomes_t1ce_when_contrast_present(self) -> None:
        ds = _fake_dataset(SequenceName="t1_se", ContrastBolusAgent="Gadovist")
        assert _classify_mr_weighting(ds) == MRWeighting.T1CE

    def test_t2_by_keyword(self) -> None:
        ds = _fake_dataset(ProtocolName="T2 TSE axial")
        assert _classify_mr_weighting(ds) == MRWeighting.T2

    def test_flair_by_keyword(self) -> None:
        ds = _fake_dataset(SeriesDescription="Ax FLAIR")
        assert _classify_mr_weighting(ds) == MRWeighting.FLAIR

    def test_dwi_by_keyword(self) -> None:
        ds = _fake_dataset(SeriesDescription="DWI b1000")
        assert _classify_mr_weighting(ds) == MRWeighting.DWI

    def test_flair_by_numeric_heuristic_without_description(self) -> None:
        ds = _fake_dataset(RepetitionTime=9000.0, EchoTime=120.0, InversionTime=2500.0)
        assert _classify_mr_weighting(ds) == MRWeighting.FLAIR

    def test_t1_by_numeric_heuristic_without_description(self) -> None:
        ds = _fake_dataset(RepetitionTime=500.0, EchoTime=14.0)
        assert _classify_mr_weighting(ds) == MRWeighting.T1

    def test_t2_by_numeric_heuristic_without_description(self) -> None:
        ds = _fake_dataset(RepetitionTime=4000.0, EchoTime=100.0)
        assert _classify_mr_weighting(ds) == MRWeighting.T2

    def test_dwi_by_numeric_heuristic(self) -> None:
        ds = _fake_dataset(ScanningSequence=["EP"], DiffusionBValue=1000.0)
        assert _classify_mr_weighting(ds) == MRWeighting.DWI

    def test_unknown_when_no_signal_present(self) -> None:
        ds = _fake_dataset()
        assert _classify_mr_weighting(ds) == MRWeighting.UNKNOWN


# --------------------------------------------------------------------------- #
# HU-конвертация КТ
# --------------------------------------------------------------------------- #


class TestCtRescale:
    def test_applies_slope_and_intercept(self) -> None:
        pixel_array = np.array([[0, 1024, 2048]], dtype=np.int16)
        ds = _fake_dataset(RescaleSlope=1.0, RescaleIntercept=-1024.0)

        hu = _apply_ct_rescale(pixel_array, ds)

        np.testing.assert_allclose(hu, [[-1024.0, 0.0, 1024.0]])
        assert hu.dtype == np.float32

    def test_defaults_to_identity_without_rescale_tags(self) -> None:
        pixel_array = np.array([[10, 20]], dtype=np.int16)
        ds = _fake_dataset()

        hu = _apply_ct_rescale(pixel_array, ds)

        np.testing.assert_allclose(hu, [[10.0, 20.0]])


# --------------------------------------------------------------------------- #
# DICOMLoader: полный сценарий через pydicom fallback (SimpleITK намеренно ломается)
# --------------------------------------------------------------------------- #


def _write_ct_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Готовит директорию с фейковой 3-срезовой КТ-серией и форсирует pydicom fallback."""
    series_dir = tmp_path / "ct_series"
    series_dir.mkdir()

    orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    headers: dict[Path, SimpleNamespace] = {}
    full: dict[Path, SimpleNamespace] = {}

    for index in range(3):
        file_path = series_dir / f"slice_{index}.dcm"
        file_path.write_bytes(b"")
        common = dict(
            SeriesInstanceUID="ct-series-uid",
            Modality="CT",
            Rows=4,
            Columns=4,
            RescaleSlope=1.0,
            RescaleIntercept=-1024.0,
            ImageOrientationPatient=orientation,
            ImagePositionPatient=(0.0, 0.0, float(index) * 2.0),
            PixelSpacing=(0.5, 0.5),
            SliceThickness=2.0,
            InstanceNumber=index,
        )
        headers[file_path] = _fake_dataset(**common)
        full[file_path] = _fake_dataset(
            **common, pixel_array=np.full((4, 4), fill_value=1024 + index, dtype=np.int16)
        )

    monkeypatch.setattr(dicom_loader_module.pydicom, "dcmread", _make_fake_dcmread(headers, full))

    def _raise(*_args: Any, **_kwargs: Any) -> list[str]:
        raise RuntimeError("SimpleITK намеренно недоступен в этом тесте")

    monkeypatch.setattr(
        dicom_loader_module.sitk.ImageSeriesReader, "GetGDCMSeriesFileNames", staticmethod(_raise)
    )

    return series_dir


class TestDicomLoaderPydicomFallback:
    def test_load_ct_series_end_to_end(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        series_dir = _write_ct_series(tmp_path, monkeypatch)

        loader = DICOMLoader()
        voxels = loader.load_dicom_series(series_dir)

        assert voxels.shape == (3, 4, 4)
        # RescaleSlope=1, RescaleIntercept=-1024 применены к пиксельным значениям 1024/1025/1026
        np.testing.assert_allclose(voxels[:, 0, 0], [0.0, 1.0, 2.0])
        assert loader.detect_modality() == "CT"

        info = loader.get_volume_info()
        assert info["shape"] == (3, 4, 4)
        assert info["modality"] == "CT"
        assert "mr_weighting" not in info
        assert info["spacing"] == pytest.approx((0.5, 0.5, 2.0))
        assert info["origin"] == pytest.approx((0.0, 0.0, 0.0))

    def test_convert_to_nifti_uses_nibabel_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series_dir = _write_ct_series(tmp_path, monkeypatch)
        loader = DICOMLoader()
        loader.load_dicom_series(series_dir)

        output_path = tmp_path / "ct_series.nii.gz"
        result_path = loader.convert_to_nifti(output_path)

        assert result_path == output_path
        assert output_path.exists()

        import nibabel as nib

        saved = nib.load(str(output_path))
        assert saved.shape == (4, 4, 3)  # (X, Y, Z)
        np.testing.assert_allclose(saved.get_fdata()[0, 0, :], [0.0, 1.0, 2.0])


class TestDicomLoaderValidation:
    def test_raises_for_empty_directory(self, tmp_path: Path) -> None:
        loader = DICOMLoader()
        with pytest.raises(InvalidDicomSeriesError):
            loader.load_dicom_series(tmp_path)

    def test_raises_mixed_modality_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ct_file = tmp_path / "ct.dcm"
        mr_file = tmp_path / "mr.dcm"
        ct_file.write_bytes(b"")
        mr_file.write_bytes(b"")

        headers = {
            ct_file: _fake_dataset(SeriesInstanceUID="s-ct", Modality="CT", Rows=4, Columns=4),
            mr_file: _fake_dataset(SeriesInstanceUID="s-mr", Modality="MR", Rows=4, Columns=4),
        }
        monkeypatch.setattr(dicom_loader_module.pydicom, "dcmread", _make_fake_dcmread(headers))

        loader = DICOMLoader()
        with pytest.raises(MixedModalitySeriesError):
            loader.load_dicom_series(tmp_path)

    def test_raises_for_ambiguous_series_without_uid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_a = tmp_path / "a.dcm"
        file_b = tmp_path / "b.dcm"
        file_a.write_bytes(b"")
        file_b.write_bytes(b"")

        headers = {
            file_a: _fake_dataset(SeriesInstanceUID="series-1", Modality="MR", Rows=4, Columns=4),
            file_b: _fake_dataset(SeriesInstanceUID="series-2", Modality="MR", Rows=4, Columns=4),
        }
        monkeypatch.setattr(dicom_loader_module.pydicom, "dcmread", _make_fake_dcmread(headers))

        loader = DICOMLoader()
        with pytest.raises(InvalidDicomSeriesError):
            loader.load_dicom_series(tmp_path)

    def test_raises_for_inconsistent_slice_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_a = tmp_path / "a.dcm"
        file_b = tmp_path / "b.dcm"
        file_a.write_bytes(b"")
        file_b.write_bytes(b"")

        headers = {
            file_a: _fake_dataset(SeriesInstanceUID="s1", Modality="CT", Rows=4, Columns=4),
            file_b: _fake_dataset(SeriesInstanceUID="s1", Modality="CT", Rows=8, Columns=8),
        }
        monkeypatch.setattr(dicom_loader_module.pydicom, "dcmread", _make_fake_dcmread(headers))

        loader = DICOMLoader()
        with pytest.raises(InvalidDicomSeriesError):
            loader.load_dicom_series(tmp_path)

    def test_methods_raise_before_load(self) -> None:
        loader = DICOMLoader()
        with pytest.raises(Exception):
            loader.detect_modality()
        with pytest.raises(Exception):
            loader.get_volume_info()


# --------------------------------------------------------------------------- #
# DICOMLoader: успешный путь через SimpleITK (без форсирования fallback)
# --------------------------------------------------------------------------- #


class TestDicomLoaderSitkPath:
    def test_load_via_sitk_when_available(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import SimpleITK as sitk

        series_dir = tmp_path / "sitk_series"
        series_dir.mkdir()
        fake_file = series_dir / "slice_0.dcm"
        fake_file.write_bytes(b"")

        headers = {fake_file: _fake_dataset(SeriesInstanceUID="uid-1", Modality="CT", Rows=4, Columns=4)}
        monkeypatch.setattr(dicom_loader_module.pydicom, "dcmread", _make_fake_dcmread(headers))

        fake_image = sitk.GetImageFromArray(np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4))
        fake_image.SetSpacing((0.5, 0.5, 1.0))
        fake_image.SetOrigin((1.0, 2.0, 3.0))

        class _FakeImageSeriesReader:
            """Заглушка sitk.ImageSeriesReader: и статический метод, и инстанс-методы на одном классе,
            иначе замена класса на лямбду теряет доступ к GetGDCMSeriesFileNames как атрибуту класса."""

            @staticmethod
            def GetGDCMSeriesFileNames(*_args: Any, **_kwargs: Any) -> list[str]:
                return [str(fake_file)]

            def SetFileNames(self, _names: list[str]) -> None:
                pass

            def Execute(self) -> sitk.Image:
                return fake_image

        monkeypatch.setattr(dicom_loader_module.sitk, "ImageSeriesReader", _FakeImageSeriesReader)

        loader = DICOMLoader()
        voxels = loader.load_dicom_series(series_dir)

        assert voxels.shape == (2, 4, 4)
        assert loader.get_volume_info()["spacing"] == (0.5, 0.5, 1.0)
        assert loader.get_volume_info()["origin"] == (1.0, 2.0, 3.0)


# --------------------------------------------------------------------------- #
# MRISeriesAnalyzer
# --------------------------------------------------------------------------- #


class TestMRISeriesAnalyzer:
    def _prepare_mixed_mri_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        directory = tmp_path / "mri_raw"
        directory.mkdir()

        files_and_tags = [
            ("t1_1.dcm", dict(SeriesInstanceUID="s-t1", Modality="MR", SeriesDescription="Ax T1")),
            ("t1_2.dcm", dict(SeriesInstanceUID="s-t1", Modality="MR", SeriesDescription="Ax T1")),
            (
                "t1ce_1.dcm",
                dict(
                    SeriesInstanceUID="s-t1ce",
                    Modality="MR",
                    SeriesDescription="Ax T1 post",
                    ContrastBolusAgent="Gadovist",
                ),
            ),
            ("flair_1.dcm", dict(SeriesInstanceUID="s-flair", Modality="MR", SeriesDescription="Ax FLAIR")),
        ]

        headers: dict[Path, SimpleNamespace] = {}
        for filename, tags in files_and_tags:
            file_path = directory / filename
            file_path.write_bytes(b"")
            headers[file_path] = _fake_dataset(**tags)

        monkeypatch.setattr(dicom_loader_module.pydicom, "dcmread", _make_fake_dcmread(headers))
        return directory

    def test_analyze_series_reports_weighting_per_series(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = self._prepare_mixed_mri_directory(tmp_path, monkeypatch)

        analysis = MRISeriesAnalyzer().analyze_series(directory)

        assert analysis["s-t1"]["weighting"] == "T1"
        assert analysis["s-t1"]["num_files"] == 2
        assert analysis["s-t1ce"]["weighting"] == "T1ce"
        assert analysis["s-flair"]["weighting"] == "FLAIR"

    def test_group_by_sequence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        directory = self._prepare_mixed_mri_directory(tmp_path, monkeypatch)

        grouped = MRISeriesAnalyzer().group_by_sequence(directory)

        assert len(grouped["T1"]) == 2
        assert len(grouped["T1ce"]) == 1
        assert len(grouped["FLAIR"]) == 1

    def test_select_best_sequence_returns_series_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = self._prepare_mixed_mri_directory(tmp_path, monkeypatch)

        best = MRISeriesAnalyzer().select_best_sequence(directory, target="T1ce")

        assert best == directory

    def test_select_best_sequence_raises_when_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = self._prepare_mixed_mri_directory(tmp_path, monkeypatch)

        with pytest.raises(SequenceNotFoundError):
            MRISeriesAnalyzer().select_best_sequence(directory, target="DWI")
