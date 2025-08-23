## Project structure

prompt-safety-agent/
├─ src/
│ ├─ agent.py # LLM-powered classifier (JSON output, fallbacks)
│ ├─ prompts.py # System/user prompt templates + few-shot
│ ├─ utils.py # JSON parsing, rules, confidence calibration
│ ├─ evaluate.py # Eval scripts (metrics, JSONL outputs)
│ ├─ rag.py # (Optional) RAG index + retrieval
│ ├─ cli.py # CLI entrypoint
│ └─ model_resolver.py # Maps aliases (phi2/stablelm3b/llama2/mistral) → GGUF path
├─ data/ # Datasets (train/test CSVs)
│ └─ README.md # How to download/use the dataset
├─ models/
│ └─ gguf/ # Local GGUF model files (not tracked)
│ └─ README.md # Model choices & filenames
├─ notebooks/ # (Optional) EDA notebooks
├─ Dockerfile # Container image
├─ requirements.txt # Python deps
├─ report.md # Analysis & findings
└─ README.md # This file

# Prompt Safety Agent

Local, explainable classifier for prompt-injection detection. Includes:
- Baseline: TF-IDF + Logistic Regression
- LLM Agent: local 7B (GGUF via llama.cpp) with multi-step analysis, calibrated confidence, and fallbacks
- Dockerized CLI: `docker run safety-agent "prompt"`

## Dataset
- **Name**: Safe-Guard Prompt Injection
- **Source**: https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
- **Format**: CSV (`text`, `label`), ~8236 rows, ~70/30 safe/unsafe

### Download
```bash
python - <<'PY'
from datasets import load_dataset
import pandas as pd, os
ds = load_dataset("xTRam1/safe-guard-prompt-injection")
os.makedirs("data", exist_ok=True)
pd.DataFrame(ds["train"]).to_csv("data/train_full.csv", index=False)
PY
