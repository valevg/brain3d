"""Инференс модели сегментации и постобработка карты меток."""

from core.segmentation.inference import SegmentationEngine
from core.segmentation.postprocessing import clean_label_map

__all__ = ["SegmentationEngine", "clean_label_map"]
