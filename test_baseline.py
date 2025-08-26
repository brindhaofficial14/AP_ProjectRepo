#python - <<'PY'
from evaluate import train_baseline, eval_baseline
train_baseline("data/train.csv", "models/baseline.joblib")
eval_baseline("data/test.csv", "models/baseline.joblib", out_json="baseline_metrics.json")
print("Baseline metrics:\n", open("baseline_metrics.json").read())
#PY
