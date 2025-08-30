# syntax=docker/dockerfile:1.6

FROM python:3.11-slim

# --- sane defaults & faster pip ---
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal system deps (add libgomp1 for OpenMP used by llama.cpp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- install deps (core only, CPU/glibc wheels) ---
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel setuptools && \
    # prefer manylinux wheels (glibc); fall back to normal if any pure wheel missing
    pip install --only-binary=:all: -r /app/requirements.txt || \
    pip install -r /app/requirements.txt

# --- OPTIONAL: RAG layer (CPU-only Torch) ---
# Build with:  --build-arg WITH_RAG=1
ARG WITH_RAG=0
COPY requirements.rag.txt /app/requirements.rag.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$WITH_RAG" = "1" ]; then \
      pip install --upgrade pip && \
      pip install --prefer-binary \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/requirements.rag.txt ; \
    fi

# --- project code (GUIs are inside src/) ---
COPY src/ /app/src/

# Create expected directories (models mounted at runtime)
RUN mkdir -p /app/data /app/models/gguf /app/models/baseline /app/rag_index

# If you want to bake assets into the image, uncomment these:
# COPY models/gguf/ /app/models/gguf/
# COPY models/baseline/ /app/models/baseline/
# COPY rag_index/ /app/rag_index/
# COPY data/ /app/data/

# Defaults for the agent (override with -e at runtime)
ENV CHAT_FORMAT=llama-2 \
    PROMPT_VERSION=v4

# ---- Runtime: CLI entrypoint ----
ENTRYPOINT ["python", "-m", "src.cli"]
