.PHONY: help install install-dev install-api install-desktop install-web \
        run-api run-desktop run-web \
        lint format typecheck test test-cov \
        pre-commit-install pre-commit-run \
        docker-build docker-up docker-down docker-logs \
        clean

PYTHON ?= python
PIP ?= pip

help:
	@echo "Brain3D AI — доступные команды:"
	@echo "  make install            Установить core+api зависимости"
	@echo "  make install-dev        Установить зависимости для разработки (+ линтеры, тесты)"
	@echo "  make install-desktop    Установить зависимости desktop-приложения"
	@echo "  make install-web        Установить npm-зависимости web/"
	@echo "  make run-api            Запустить FastAPI backend (uvicorn, reload)"
	@echo "  make run-desktop        Запустить desktop-приложение (PyQt5)"
	@echo "  make run-web            Запустить dev-сервер web/ (Vite)"
	@echo "  make lint               black --check + isort --check + flake8"
	@echo "  make format             Автоформатирование black + isort"
	@echo "  make typecheck          mypy"
	@echo "  make test               pytest"
	@echo "  make test-cov           pytest с отчётом покрытия"
	@echo "  make pre-commit-install Установить git-хуки pre-commit"
	@echo "  make pre-commit-run     Прогнать все pre-commit хуки на всех файлах"
	@echo "  make docker-build       Собрать Docker-образы (api + web)"
	@echo "  make docker-up          Поднять docker compose (api + web)"
	@echo "  make docker-down        Остановить docker compose"
	@echo "  make clean              Удалить кэши Python (__pycache__, .pytest_cache и т.д.)"

install:
	$(PIP) install -r requirements/requirements-common.txt -r requirements/requirements-api.txt

install-dev:
	$(PIP) install -r requirements/requirements-common.txt \
	                -r requirements/requirements-api.txt \
	                -r requirements/requirements-dev.txt

install-desktop:
	$(PIP) install -r requirements/requirements-common.txt -r requirements/requirements-desktop.txt

install-web:
	cd web && npm install

run-api:
	uvicorn api.fastapi_app:app --reload --host 0.0.0.0 --port 8000

run-desktop:
	$(PYTHON) -m desktop.main

run-web:
	cd web && npm run dev

lint:
	black --check .
	isort --check-only .
	flake8 .

format:
	black .
	isort .

typecheck:
	mypy core models pipelines api shared

test:
	pytest

test-cov:
	pytest --cov --cov-report=term-missing --cov-report=html

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

clean:
	find . -type d -name "__pycache__" -not -path "./web/*" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
