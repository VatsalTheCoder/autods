# Shortcuts for the commands you will run dozens of times.
# Run `make` on its own to list them.

.DEFAULT_GOAL := help
.PHONY: help up down restart logs ps build rebuild migrate migration shell-api shell-db test lint fmt check health clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---- Stack ------------------------------------------------------------------

up:  ## Start the whole stack in the background
	docker compose up -d
	@echo ""
	@echo "  UI      http://localhost:8501"
	@echo "  API     http://localhost:8000/docs"
	@echo "  MinIO   http://localhost:9001  (minioadmin / minioadmin)"

down:  ## Stop the stack (data is preserved)
	docker compose down

restart:  ## Restart every service
	docker compose restart

logs:  ## Follow logs from all services
	docker compose logs -f

ps:  ## Show container status
	docker compose ps

build:  ## Build images
	docker compose build

rebuild:  ## Rebuild images from scratch, ignoring the cache
	docker compose build --no-cache

# ---- Database ---------------------------------------------------------------

migrate:  ## Apply all pending migrations
	docker compose run --rm api alembic upgrade head

migration:  ## Generate a migration: make migration m="add jobs table"
	docker compose run --rm api alembic revision --autogenerate -m "$(m)"

shell-db:  ## Open a psql prompt
	docker compose exec postgres psql -U autods -d autods

shell-api:  ## Open a shell inside the api container
	docker compose exec api bash

# ---- Quality ----------------------------------------------------------------

test:  ## Run the test suite
	docker compose run --rm api pytest tests/ --cov=app --cov-report=term-missing

lint:  ## Lint and check formatting (same checks CI runs)
	docker compose run --rm api ruff check .
	docker compose run --rm api ruff format --check .

fmt:  ## Auto-fix formatting and lint issues
	docker compose run --rm api ruff format .
	docker compose run --rm api ruff check --fix .

check: lint test  ## Run everything CI runs

# Not part of `check`: this one calls a live model and takes minutes per
# dataset. It reports rather than passes -- read the table it prints.
sweep:  ## Run every dataset in data/examples through the pipeline: make sweep [paths=...]
	docker compose exec api python scripts/dataset_sweep.py $(or $(paths),data/examples) \
		--manifest scripts/sweep_manifest.json --out sweep_results

health:  ## Curl the health endpoint
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "API not reachable"

# ---- Cleanup ----------------------------------------------------------------

clean:  ## Stop the stack and DELETE all data volumes
	docker compose down -v
