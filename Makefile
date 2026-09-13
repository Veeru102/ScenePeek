.PHONY: up down logs migrate api-dev worker web-dev install test lint eval seed

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

observability:
	docker compose --profile observability up -d prometheus grafana
