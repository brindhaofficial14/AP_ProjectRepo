#!/usr/bin/env bash
set -euo pipefail

# Everything the user types after the image name is the prompt:
PROMPT="${*:-}"
if [ -z "$PROMPT" ]; then
  echo "Usage: docker run --rm safety-agent \"your prompt here\"" >&2
  exit 2
fi

# Settings via env (sensible defaults):
AGENT_VERSION="${AGENT_VERSION:-v2}"     # v2 by default; set v1 to use cli_v1
MODEL_ALIAS_OR_PATH="${LLM_MODEL_ALIAS:-llama2}"
N_CTX="${LLM_N_CTX:-4096}"
THREADS="${LLM_THREADS:-}"

# Auto-enable RAG if rag_index present or RAG_ENABLED=1
RAG_ENABLED="${RAG_ENABLED:-auto}"
RAG_FLAG=()  # empty array by default
if [ "$RAG_ENABLED" = "1" ] || [ "$RAG_ENABLED" = "true" ]; then
  RAG_FLAG=(--rag)
elif [ "$RAG_ENABLED" = "0" ] || [ "$RAG_ENABLED" = "false" ]; then
  RAG_FLAG=(--no-rag)
else
  # auto
  if [ -d "/app/rag_index" ]; then
    RAG_FLAG=(--rag)
  fi
fi

if [ "$AGENT_VERSION" = "v1" ]; then
  exec python -m src.cli_v1 --model "$MODEL_ALIAS_OR_PATH" --n_ctx "$N_CTX" ${THREADS:+--threads "$THREADS"} "${RAG_FLAG[@]}" "$PROMPT"
else
  exec python -m src.cli_v2 --model "$MODEL_ALIAS_OR_PATH" --n_ctx "$N_CTX" ${THREADS:+--threads "$THREADS"} "${RAG_FLAG[@]}" "$PROMPT"
fi
