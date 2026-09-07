"""Оркестраторы пайплайна: единый DL-пайплайн (pipeline_factory) и пайплайн на готовых
предобученных моделях (brain_pipeline).

Пакет намеренно не делает eager-реэкспорт символов из своих подмодулей: pipeline_factory
требует torch/monai через models.registry, а brain_pipeline спроектирован так, чтобы
не тянуть эти тяжёлые зависимости на уровне импорта пакета — eager-импорт здесь свёл бы
это на нет. Импортируйте нужный подмодуль напрямую, например:
`from pipelines.pipeline_factory import create_pipeline`.
"""
