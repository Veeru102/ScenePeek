.PHONY: up down logs migrate api-dev worker web-dev install test lint eval eval-real seed

N ?= 1

install:
	cd backend && uv sync --extra ml --extra dev
	cd frontend && npm install

up:
	docker compose up -d postgres minio minio-init api web

down:
	docker compose down

logs:
	docker compose logs -f api web

migrate:
	cd backend && uv run scenepeek migrate

api-dev:
	cd backend && uv run scenepeek api --reload

worker:
	cd backend && for i in $$(seq 1 $(N)); do uv run scenepeek worker & done; wait

web-dev:
	cd frontend && npm run dev

test:
	cd backend && uv run pytest -q

lint:
	cd backend && uv run ruff check scenepeek tests

eval:
	cd backend && uv run scenepeek eval run -c ../eval/configs/default.yaml

# Real-video ablations: every lane on/off + rerank + fusion method, then a paired-bootstrap comparison.
REAL_SET ?= real
REAL_VARIANTS = text_only no_visual no_ocr no_lexical no_rerank weighted
eval-real:
	cd backend && uv run scenepeek eval run -c ../eval/configs/$(REAL_SET).yaml
	cd backend && for v in $(REAL_VARIANTS); do uv run scenepeek eval run -c ../eval/configs/$(REAL_SET)_$$v.yaml; done
	cd backend && uv run scenepeek eval compare ../eval/reports/$(REAL_SET).json $(foreach v,$(REAL_VARIANTS),../eval/reports/$(REAL_SET)_$(v).json)

observability:
	docker compose --profile observability up -d prometheus grafana
