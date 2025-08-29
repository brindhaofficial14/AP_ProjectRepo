# ---- Base: slim Python with build toolchain for llama-cpp and chromadb ----
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps for building llama-cpp-python (CPU + OpenBLAS) and common libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-dev git curl \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Speed up CPU inference by linking against OpenBLAS at build time
ENV LLAMA_BLAS=1
ENV LLAMA_BLAS_VENDOR=OpenBLAS
# (Optional) tweak threading defaults at runtime
ENV OMP_NUM_THREADS=4
ENV N_THREADS=4
ENV N_CTX=4096
ENV N_GPU_LAYERS=0
ENV PROMPT_VERSION=v4
ENV CHAT_FORMAT=llama-2
ENV BASELINE_DIR=models/baseline
# resolves via src/model_resolver.py to models/gguf/llama-2-7b-chat.Q4_K_M.gguf
ENV LLM_MODEL=llama2  

WORKDIR /app

# Copy only requirement list first (layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project code
COPY src/ /app/src/


# Create optional dirs in the image (always succeeds)
RUN mkdir -p /app/data /app/rag_index /app/models/gguf /app/models/baseline

COPY models/ /app/models/
COPY rag_index/ /app/rag_index/
COPY data/      /app/data/

# Ensure model and baseline exist (you provide them before build)
# - Expected: models/gguf/<your>.gguf
# - Optional: models/baseline/tfidf.joblib (baseline; if missing, agent falls back safely)

# Default to CLI entrypoint so `docker run safety-agent "..."` just works
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["Is this prompt safe?"]
