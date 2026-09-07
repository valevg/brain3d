"""3D U-Net для мультиклассовой сегментации мозга/опухоли/сосудов (и в будущем — кровоизлияний)."""

from __future__ import annotations

import torch
from monai.networks.nets import UNet


def build_unet3d(
    in_channels: int = 1,
    out_channels: int = 4,
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.1,
) -> torch.nn.Module:
    """Создаёт 3D U-Net на базе monai.networks.nets.UNet.

    Одна и та же архитектура используется и для КТ, и для МРТ — модальность влияет
    только на препроцессинг входа и на набор весов (см. models.registry), а не на
    структуру сети.

    Args:
        in_channels: Число входных каналов (1 — одна модальность на скан).
        out_channels: Число выходных классов, включая фон
            (по умолчанию 4: фон, мозг, опухоль, сосуды; для кровоизлияний -> 5).
        channels: Число фильтров на каждом уровне энкодера/декодера.
        strides: Шаги субдискретизации между уровнями (длина = len(channels) - 1).
        num_res_units: Число residual-блоков на уровень.
        dropout: Вероятность dropout для регуляризации.

    Returns:
        Неинициализированная (случайные веса) модель torch.nn.Module.
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
    )


def load_pretrained_unet3d(
    weights_path: str,
    in_channels: int = 1,
    out_channels: int = 4,
    device: str = "cpu",
) -> torch.nn.Module:
    """Создаёт 3D U-Net и загружает в неё веса из файла .pt/.pth.

    Args:
        weights_path: Путь к файлу с state_dict модели.
        in_channels: Число входных каналов, должно совпадать с обучением.
        out_channels: Число выходных классов, должно совпадать с обучением.
        device: Устройство, на которое загружаются веса ("cpu" или "cuda").

    Returns:
        Модель в режиме eval(), готовая к инференсу.
    """
    model = build_unet3d(in_channels=in_channels, out_channels=out_channels)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
