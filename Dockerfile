FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV HF_HOME=/app/.hf_cache

# Dependencies first (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Code, then install the project itself
COPY src/ src/
COPY app/ app/
RUN uv sync --frozen --no-dev

# Download the embedding model at build time so the container runs offline
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

# Prebuilt index (run `gpu-dev-assist ingest` locally first)
COPY data/vector_store/ data/vector_store/

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]