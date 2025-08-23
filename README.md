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
