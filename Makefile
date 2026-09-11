# VisionForge developer commands (Linux / WSL / CI).
# On Windows use scripts/vf.ps1 -- GNU make is not installed there.

PYTHON  := $(CURDIR)/.venv/bin/python
API_DIR := apps/api
WEB_DIR := apps/web

.PHONY: help setup infra-up infra-down infra-reset infra-status migrate \
        api web worker-cpu test test-integration lint format typecheck \
        contracts web-lint web-build compose-check check

help:
	@echo "setup infra-up infra-down infra-reset infra-status migrate"
	@echo "api web worker-cpu"
	@echo "check test test-integration lint format typecheck contracts"
	@echo "web-lint web-build compose-check"

setup:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install -e "$(API_DIR)[dev]"
	cd $(WEB_DIR) && pnpm install

infra-up:
	docker compose up -d
	$(MAKE) migrate

infra-down:
	docker compose down

infra-reset:
	docker compose down -v

infra-status:
	docker compose ps

migrate:
	cd $(API_DIR) && PYTHONPATH=src $(PYTHON) -m alembic upgrade head

api:
	cd $(API_DIR) && PYTHONPATH=src $(PYTHON) -m visionforge

web:
	cd $(WEB_DIR) && pnpm dev

worker-cpu:
	cd $(API_DIR) && PYTHONPATH=src $(PYTHON) -m celery \
	  -A visionforge.workers.cpu worker --pool=prefork --concurrency=2 -Q cpu -l info

test:
	cd $(API_DIR) && $(PYTHON) -m pytest -m "not integration and not gpu"

test-integration:
	cd $(API_DIR) && $(PYTHON) -m pytest -m integration

lint:
	cd $(API_DIR) && $(PYTHON) -m ruff check .
	cd $(API_DIR) && $(PYTHON) -m ruff format --check .

format:
	cd $(API_DIR) && $(PYTHON) -m ruff check --fix .
	cd $(API_DIR) && $(PYTHON) -m ruff format .

typecheck:
	cd $(API_DIR) && $(PYTHON) -m mypy

contracts:
	cd $(API_DIR) && PYTHONPATH=src $(CURDIR)/.venv/bin/lint-imports --config .importlinter

web-lint:
	cd $(WEB_DIR) && pnpm lint

web-build:
	cd $(WEB_DIR) && pnpm build

compose-check:
	docker compose config --quiet && echo "compose config valid"

check: lint typecheck contracts test web-lint web-build compose-check
