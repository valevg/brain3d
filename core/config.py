"""Конфигурация приложения через pydantic-settings (.env) + config.yaml (параметры пайплайнов)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ModalityWindowConfig(BaseModel):
    """Параметры оконной обработки/нормализации интенсивностей для одной модальности."""

    hu_min: float | None = Field(default=None, description="Нижняя граница HU (только для КТ)")
    hu_max: float | None = Field(default=None, description="Верхняя граница HU (только для КТ)")
    normalize: Literal["z_score", "min_max", "hu_window"] = "z_score"
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    n4_bias_correction: bool = False


class SegmentationModelConfig(BaseModel):
    """Параметры модели сегментации для одной модальности."""

    weights_path: str
    in_channels: int = 1
    out_channels: int = 4
    roi_size: tuple[int, int, int] = (128, 128, 128)
    sw_batch_size: int = 4
    overlap: float = 0.5


class StructureColorConfig(BaseModel):
    """Цвет структуры для рендера (RGBA, 0..1)."""

    color_rgba: tuple[float, float, float, float]


class PipelineConfig(BaseModel):
    """Полная конфигурация пайплайнов, загружаемая из configs/config.yaml."""

    ct: ModalityWindowConfig
    mri: ModalityWindowConfig
    ct_model: SegmentationModelConfig
    mri_model: SegmentationModelConfig
    structure_colors: dict[str, StructureColorConfig]

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "PipelineConfig":
        """Загружает конфигурацию пайплайнов из YAML-файла.

        Args:
            path: Путь к config.yaml.

        Returns:
            Валидированный объект PipelineConfig.

        Raises:
            ConfigurationError: Если файл не найден или не проходит валидацию.
        """
        if not path.exists():
            raise ConfigurationError(f"Файл конфигурации не найден: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 — оборачиваем любую ошибку парсинга/валидации
            raise ConfigurationError(f"Некорректный config.yaml ({path}): {exc}") from exc


class Settings(BaseSettings):
    """Настройки окружения приложения, загружаемые из .env / переменных окружения."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Хранилище исследований и результатов
    storage_dir: Path = Path("./data")
    upload_max_size_mb: int = 512

    # Кэш весов предобученных моделей (models.pretrained_models.PretrainedModelRegistry)
    model_cache_dir: Path = Path.home() / ".cache" / "brain3d"

    # Вычисления
    device: Literal["cpu", "cuda"] = "cpu"

    # Путь к общему конфигу пайплайнов
    pipeline_config_path: Path = DEFAULT_CONFIG_PATH


@lru_cache
def get_settings() -> Settings:
    """Возвращает закэшированный экземпляр Settings (singleton в рамках процесса)."""
    return Settings()


@lru_cache
def get_pipeline_config() -> PipelineConfig:
    """Возвращает закэшированную конфигурацию пайплайнов из config.yaml."""
    return PipelineConfig.from_yaml(get_settings().pipeline_config_path)
