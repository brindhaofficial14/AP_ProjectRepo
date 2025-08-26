#python - <<'PY'
from evaluate import eval_all
# Use explicit model path to avoid alias mismatches
eval_all(
  test_csv="data/test.csv",
  model_path="models/gguf/llama-2-7b-chat.Q4_K_M.gguf",
  out_jsonl="eval_llama2.jsonl",
  use_rag=True
)
#PY
