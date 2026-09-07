"""Тесты общих перечислений (определение модальности по DICOM-тегу)."""

from __future__ import annotations

import pytest

from shared.enums import Modality


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("CT", Modality.CT), ("MR", Modality.MRI), ("mri", Modality.MRI), (" ct ", Modality.CT)],
)
def test_from_dicom_tag_recognizes_known_values(tag: str, expected: Modality) -> None:
    assert Modality.from_dicom_tag(tag) == expected


def test_from_dicom_tag_raises_for_unknown_value() -> None:
    with pytest.raises(ValueError):
        Modality.from_dicom_tag("PET")
