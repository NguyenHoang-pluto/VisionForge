# VisionForge developer commands (Linux / WSL / CI).
# On Windows use scripts/vf.ps1 -- GNU make is not installed there.

PYTHON  := $(CURDIR)/.venv/bin/python
API_DIR := apps/api
WEB_DIR := apps/web

.PHONY: help setup infra-up infra-down infra-reset infra-status migrate \
        api web worker-cpu worker-gpu worker-render e2e e2e-analysis e2e-edit \
        test test-integration lint format typecheck \
        contracts web-lint web-build compose-check check

help:
	@echo "setup infra-up infra-down infra-reset infra-status migrate"
	@echo "api web worker-cpu worker-gpu worker-render"
	@echo "e2e e2e-analysis e2e-edit"
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
	  -A visionforge.workers.cpu worker -n cpu@%%h --pool=prefork --concurrency=2 -Q cpu -l info

# --pool=solo is the process-level GPU mutex: one slot, one model, one
# inference at a time on a small card (ADR-0003, ADR-0008).
worker-gpu:
	cd $(API_DIR) && PYTHONPATH=src $(PYTHON) -m celery \
	  -A visionforge.workers.gpu worker -n gpu@%%h --pool=solo -Q gpu -l info

# Rendering is one long FFmpeg process per job, so one encode at a time gets
# the cores it can use and two renders never contend for the same scratch disk.
worker-render:
	cd $(API_DIR) && PYTHONPATH=src $(PYTHON) -m celery \
	  -A visionforge.workers.render worker -n render@%%h --pool=solo -Q render -l info

# The acceptance scripts drive a running stack, so they are not part of `check`.
e2e:            # Phase 2: needs the API and a cpu worker
	$(PYTHON) scripts/e2e_acceptance.py

e2e-analysis:   # Phase 3: needs the API and both cpu and gpu workers
	$(PYTHON) scripts/e2e_analysis.py

e2e-edit:       # Phase 4: needs the API and both cpu and render workers
	$(PYTHON) scripts/e2e_edit.py

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
