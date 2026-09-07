"""Загрузчики медицинских изображений (DICOM, NIfTI) с единым интерфейсом BaseLoader."""

from core.io.base_loader import BaseLoader
from core.io.dicom_loader import DicomLoader
from core.io.nifti_loader import NiftiLoader

__all__ = ["BaseLoader", "DicomLoader", "NiftiLoader"]
