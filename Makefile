.PHONY: up down logs migrate api-dev worker web-dev install test lint eval eval-real seed

N ?= 1

install:
	cd backend && uv sync --extra ml --extra dev --extra bench
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
	cd backend && uv run scenepeek eval run -c ../eval/experiments/synthetic.yaml

# Ablation suite for one dataset (SET = human | auto | qvh_val): every lane on/off + rerank + fusion
# method, then a paired-bootstrap comparison. Specs live in eval/experiments/<SET>_<variant>.yaml.
SET ?= human
VARIANTS = text_only no_visual no_ocr no_lexical no_rerank rrf with_caption ocr_damped visual_damped
eval-suite:
	cd backend && uv run scenepeek eval run -c ../eval/experiments/$(SET).yaml
	cd backend && for v in $(VARIANTS); do uv run scenepeek eval run -c ../eval/experiments/$(SET)_$$v.yaml; done
	cd backend && uv run scenepeek eval compare ../eval/reports/$(SET).json $(foreach v,$(VARIANTS),../eval/reports/$(SET)_$(v).json)

# QVHighlights: annotations go in data/qvhighlights/ (moment_detr release), videos come via yt-dlp.
QVH_LIMIT ?= 300
qvh-import:
	cd backend && uv run scenepeek dataset import qvhighlights --dir ../data/qvhighlights --split val --limit $(QVH_LIMIT)
	cd backend && uv run scenepeek dataset fetch qvhighlights --split val

observability:
	docker compose --profile observability up -d prometheus grafana
