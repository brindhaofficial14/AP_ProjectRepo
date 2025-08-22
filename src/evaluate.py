# src/evaluate.py
import json, time, pandas as pd, numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from joblib import load
from .agent import PromptSafetyAgent
from .baseline import predict_baseline

def eval_all(test_csv: str, model_path: str, out_jsonl: str = "eval_outputs.jsonl", n_limit: int | None = None):
    df = pd.read_csv(test_csv)
    if n_limit: df = df.head(n_limit)
    agent = PromptSafetyAgent(model_path=model_path)
    rows, pred_b, pred_a, y_true = [], [], [], []

    t0 = time.time()
    for _, r in df.iterrows():
        text, y = str(r["text"]), int(r["label"])
        y_true.append(y)

        b = predict_baseline(text)
        a = agent.classify(text)
        rows.append({
            "text": text, "true": y,
            "baseline_label": b["label"], "baseline_score": b["score"],
            "agent": a
        })

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # metrics
    b_labels = [row["baseline_label"] for row in rows]
    a_labels = [row["agent"]["label"] for row in rows]
    a_confs  = [row["agent"]["confidence"] for row in rows]
    a_fallback = [row["agent"]["fallback_used"] for row in rows]
    a_latency = [row["agent"]["latency_ms"] for row in rows]

    print("== Baseline ==")
    print(classification_report(y_true, b_labels, digits=3))

    print("\n== Agent ==")
    p, r, f1, _ = precision_recall_fscore_support(y_true, a_labels, average="binary")
    acc = accuracy_score(y_true, a_labels)
    print(f"Accuracy: {acc:.3f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")
    print(f"Confidence: mean={np.mean(a_confs):.3f}, std={np.std(a_confs):.3f}")
    print(f"Fallback rate: {np.mean(a_fallback)*100:.1f}%")
    print(f"Avg latency: {np.mean(a_latency):.1f} ms")
    print(f"Processed {len(df)} prompts in {time.time()-t0:.1f}s")

def make_comparison_table(jsonl_path: str, out_csv: str = "comparisons.csv", k: int = 20):
    # pick interesting cases: disagreement, low confidence, wrong predictions
    rows = []
    data = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    # add simple scores
    for d in data:
        wrong = (d["true"] != d["agent"]["label"])
        disagree = (d["baseline_label"] != d["agent"]["label"])
        low_conf = d["agent"]["confidence"] < 0.5
        score = 1.0*wrong + 0.7*disagree + 0.5*low_conf
        rows.append((score, d))
    rows.sort(key=lambda x: x[0], reverse=True)
    sel = [d for _, d in rows[:k]]

    recs = []
    for d in sel:
        recs.append({
            "text": d["text"][:200].replace("\n"," "),
            "true": d["true"],
            "baseline_label": d["baseline_label"], "baseline_score": round(d["baseline_score"],3),
            "agent_label": d["agent"]["label"], "agent_score": round(d["agent"]["score"],3),
            "agent_confidence": round(d["agent"]["confidence"],3),
            "fallback_used": d["agent"]["fallback_used"],
            "explanation": d["agent"]["explanation"][:180]
        })
    pd.DataFrame(recs).to_csv(out_csv, index=False)
    print("Wrote", out_csv)
