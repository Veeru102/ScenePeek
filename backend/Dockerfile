FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml ./

FROM base AS api
RUN uv sync --no-dev --no-install-project
COPY . .
RUN uv sync --no-dev
CMD ["scenepeek", "api", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker
RUN uv sync --no-dev --no-install-project --extra ml
COPY . .
RUN uv sync --no-dev --extra ml
CMD ["scenepeek", "worker"]
