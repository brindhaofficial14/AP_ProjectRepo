FROM python:3.11-slim

# for llama.cpp performance
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY src/ ./src/
COPY README.md .
COPY report.md .
# data/ and models/ are mounted or copied by you; keep empty in image

# default: CLI entry — accept a prompt from CMD
ENV PYTHONPATH=/app
CMD ["python", "-m", "src.cli"]
