"""Инференс 3D U-Net с sliding window (MONAI) для сегментации КТ/МРТ объёмов."""

from __future__ import annotations

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from core.exceptions import InferenceError
from core.logging_config import get_logger
from shared.enums import StructureType
from shared.types import LabelMap, SegmentationResult, VolumeData

logger = get_logger(__name__)

# Индекс класса в выходе модели -> тип структуры.
# Индекс 0 зарезервирован под фон и в structures не включается.
CLASS_INDEX_TO_STRUCTURE: dict[int, StructureType] = {
    1: StructureType.BRAIN,
    2: StructureType.TUMOR,
    3: StructureType.VESSEL,
    4: StructureType.HEMORRHAGE,
}


class SegmentationEngine(torch.nn.Module):
    """Обёртка над 3D U-Net, выполняющая sliding-window инференс на полном объёме.

    Args:
        model: Обученная torch.nn.Module (например, models.unet3d.build_unet3d(...)).
        roi_size: Размер окна инференса (D, H, W), должен совпадать с патчами обучения.
        sw_batch_size: Размер батча окон, подаваемых в модель одновременно.
        overlap: Доля перекрытия соседних окон (0..1), повышает качество на границах.
        device: Устройство вычислений ("cpu" или "cuda").
    """

    def __init__(
        self,
        model: torch.nn.Module,
        roi_size: tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._model = model.to(device).eval()
        self._roi_size = roi_size
        self._sw_batch_size = sw_batch_size
        self._overlap = overlap
        self._device = torch.device(device)

    @torch.no_grad()
    def predict(self, volume: VolumeData) -> SegmentationResult:
        """Выполняет сегментацию предобработанного объёма.

        Args:
            volume: Объём после препроцессинга (нормализованные интенсивности,
                нужный spacing). Ожидается форма voxels (Z, Y, X).

        Returns:
            SegmentationResult с картой меток той же формы, что и входной объём.

        Raises:
            InferenceError: При ошибке инференса (например, несовместимая форма входа).
        """
        try:
            input_tensor = torch.from_numpy(volume.voxels).float()
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self._device)  # (1, 1, Z, Y, X)

            logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=self._roi_size,
                sw_batch_size=self._sw_batch_size,
                predictor=self._model,
                overlap=self._overlap,
            )
            probabilities = torch.softmax(logits, dim=1)
            label_map: LabelMap = (
                probabilities.argmax(dim=1).squeeze(0).to("cpu").numpy().astype(np.uint8)
            )

            present_classes = sorted(int(c) for c in np.unique(label_map) if c != 0)
            structures = [
                CLASS_INDEX_TO_STRUCTURE[c] for c in present_classes if c in CLASS_INDEX_TO_STRUCTURE
            ]

            logger.info("Сегментация завершена: найдены структуры %s", [s.value for s in structures])

            return SegmentationResult(
                label_map=label_map,
                structures=structures,
                spacing=volume.spacing,
                probabilities=probabilities.squeeze(0).to("cpu").numpy().astype(np.float32),
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Ошибка инференса сегментации: {exc}") from exc
