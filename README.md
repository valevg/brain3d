# Brain3D AI

Система 3D-визуализации КТ- и МРТ-снимков головного мозга с автоматической
сегментацией опухолей и сосудов Виллизиева круга. Архитектура рассчитана на
расширение под новые структуры (например, кровоизлияния) без переписывания
пайплайна.

## Ключевые возможности

- Единый пайплайн обработки для **КТ и МРТ** с автоматическим определением модальности
  (по DICOM-тегу `Modality` или по имени NIfTI-файла).
- Сегментация 3D U-Net (PyTorch + [MONAI](https://monai.io)) со sliding-window инференсом.
- Построение 3D-мешей (marching cubes + сглаживание) и экспорт в GLB/STL.
- Web-визуализация на Three.js и desktop-визуализация на PyQt5 + PyVista.
- FastAPI backend с загрузкой исследований, фоновой обработкой и отдачей мешей.

## Архитектура проекта

```
brain3d/
├── core/            # Ядро: загрузка (DICOM/NIfTI), препроцессинг, сегментация, mesh-генерация
│   ├── io/          # BaseLoader, DicomLoader, NiftiLoader
│   ├── preprocessing/  # CtPreprocessor (HU-окно), MriPreprocessor (z-score)
│   ├── segmentation/   # SegmentationEngine (sliding window), постобработка
│   └── mesh/           # marching cubes, экспорт GLB/STL
├── models/          # Архитектура 3D U-Net (MONAI) + реестр моделей по модальности
├── pipelines/       # Оркестраторы: BasePipeline, CtPipeline, MriPipeline, pipeline_factory
├── api/             # FastAPI backend (upload, segmentation, health)
├── web/             # Three.js фронтенд (Vite)
├── desktop/         # PyQt5 + PyVista desktop-приложение
├── shared/          # Общие enum'ы (Modality, StructureType) и типы данных
├── configs/         # config.yaml — параметры пайплайнов КТ/МРТ
├── requirements/    # Раздельные requirements по частям проекта
├── docker/          # Dockerfile.api, Dockerfile.web
├── tests/           # pytest-тесты (core, pipelines, api, shared)
└── .github/workflows/  # CI/CD (lint, test, docker build/publish)
```

Единый пайплайн реализован в `pipelines/pipeline_factory.py`: функция
`create_pipeline(path)` сама определяет формат файла и модальность исследования
и возвращает готовый `CtPipeline` или `MriPipeline` с одинаковым интерфейсом
`run(volume) -> list[MeshAsset]`.

## Целевые и плановые структуры

| Структура | Статус | Цвет (RGBA) |
|---|---|---|
| Мозг | текущая | серый, полупрозрачный `(0.8, 0.8, 0.8, 0.35)` |
| Опухоль | текущая | красный `(0.9, 0.15, 0.1, 1.0)` |
| Сосуды Виллизиева круга | текущая | синий `(0.1, 0.3, 0.9, 1.0)` |
| Кровоизлияние | плановая (резерв) | оранжевый `(1.0, 0.55, 0.0, 1.0)` |

Добавление кровоизлияний потребует: (1) переобучения 3D U-Net с 5 выходными
классами вместо 4, (2) добавления класса в `CLASS_INDEX_TO_STRUCTURE`
(`core/segmentation/inference.py`), цвет уже зарезервирован в `config.yaml`.

## Установка

### Требования

- Python 3.10+
- Node.js 20+ (для web/)
- Docker + Docker Compose (для деплоя)
- (опционально) CUDA-совместимый GPU для ускорения инференса

### 1. Клонирование и переменные окружения

```bash
git clone <repo-url> brain3d
cd brain3d
cp .env.example .env
```

### 2. Backend (API)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements/requirements-common.txt -r requirements/requirements-api.txt
uvicorn api.fastapi_app:app --reload
```

API будет доступен на `http://localhost:8000`, документация — на `/docs`.

### 3. Desktop-приложение

```bash
pip install -r requirements/requirements-common.txt -r requirements/requirements-desktop.txt
python -m desktop.main
```

### 4. Web-фронтенд

```bash
cd web
cp .env.example .env
npm install
npm run dev
```

### 5. Разработка (линтеры, тесты, pre-commit)

```bash
pip install -r requirements/requirements-common.txt -r requirements/requirements-api.txt -r requirements/requirements-dev.txt
pre-commit install
pytest
```

Все команды выше также доступны через `Makefile` — см. `make help`.

## Запуск через Docker

```bash
docker compose up --build
```

Поднимет `api` (порт `8000`) и `web` (порт `8080`). Веса моделей монтируются
из `./models/weights` (read-only) — положите туда `unet3d_ct.pt` и
`unet3d_mri.pt` перед запуском (пути задаются в `configs/config.yaml`).

## Конфигурация

- `.env` — окружение процесса (см. `.env.example`): хост/порт API, CORS,
  хранилище, устройство вычислений (`cpu`/`cuda`).
- `configs/config.yaml` — параметры пайплайнов: HU-окно и spacing для КТ,
  нормализация и spacing для МРТ, пути к весам моделей, цвета структур.

## Тестирование

```bash
make test        # pytest
make test-cov     # pytest + отчёт покрытия (htmlcov/)
make lint         # black --check + isort --check + flake8
make typecheck    # mypy
```

## CI/CD

`.github/workflows/ci.yml` на каждый push/PR прогоняет: lint (black/isort/
flake8/mypy) → pytest с покрытием → сборку web/ → пробную сборку Docker-образов.

`.github/workflows/docker-publish.yml` публикует образы `api`/`web` в GHCR
при пуше git-тега вида `v1.2.3`.

## Лицензия

Проект разрабатывается в рамках магистерской работы. Лицензия уточняется.
