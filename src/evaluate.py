# src/evaluate.py
# NOTE: This file preserves all existing logic and printouts,
# and adds an optional report bundle controlled by report_flag.

import os, json, time, datetime, pathlib, shutil
import pandas as pd, numpy as np  # std/json, timing, data wrangling, math
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)  # metrics
from joblib import load  # (imported if needed later)
from .agent import PromptSafetyAgent  # agent
from .baseline import predict_baseline  # baseline prediction

# Headless plotting for Docker/CI servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------
# Helpers for report mode
# -----------------------
def _now_tag() -> str:
    """Filesystem-friendly timestamp suffix."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str) -> str:
    """Create directory if missing; return path."""
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_txt(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _plot_hist(data, title, xlabel, out_path, bins=20):
    """Simple histogram plot."""
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_confusion(cm, labels, title, out_path):
    """Confusion matrix heatmap."""
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_roc(y_true, y_score, title, out_path):
    """ROC curve (works for the baseline since we have probabilities)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------
# Core evaluation (kept)
# -----------------------
def eval_all(
    test_csv: str,
    model_path: str,
    out_jsonl: str = "eval_outputs.jsonl",
    n_limit: int | None = None,
    use_rag: bool = True,
    rag_dir: str = "rag_index",
    report_flag: bool = False,
    report_dir_root: str = "reports",
):
    """Evaluate baseline & agent on a test split; write JSONL and print summary metrics.
       If report_flag=True, save metrics/graphs/CSV into reports/report_YYYYmmdd_HHMMSS/."""
    
    print(f"Loading test CSV from {test_csv}...",flush=True)
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} rows")
    
    if n_limit:
        df = df.head(n_limit)
        print(f"Limiting dataset to first {n_limit} rows",flush=True)
    else:
        n_limit = 1
        df = df.head(n_limit)
        print(f"n_limit not provided and hence setting n_limit to default value. Limiting dataset to first {n_limit} rows",flush=True)

    rag = None
    if use_rag:
        print(f"Initializing RAG index from directory {rag_dir}...")
        from .rag import RAGIndex
        rag = use_rag
        print("RAG initialized")

    print(f"Initializing PromptSafetyAgent with model {model_path}...")
    agent = PromptSafetyAgent(model_path=model_path, ragflag=rag, rag_k=3)
    print("Agent initialized")

    rows, y_true = [], []
    base_labels, base_scores = [], []
    agent_labels, agent_confs, agent_fallback, agent_latency = [], [], [], []

    t0 = time.time()
    print("Starting evaluation loop over test dataset...")
    
    for idx, r in enumerate(df.itertuples(index=False), start=1):
        text, y = str(r.text), int(r.label)
        y_true.append(y)

        print(f"\nEvaluating sample {idx}: {text[:50]}... (label={y})")
        b = predict_baseline(text)
        base_labels.append(int(b["label"]))
        base_scores.append(float(b["score"]))
        print(f"Baseline prediction: {b}")

        a = agent.classify(text)
        agent_labels.append(int(a["label"]))
        agent_confs.append(float(a.get("confidence", 0.5)))
        agent_fallback.append(bool(a.get("fallback_used", False)))
        agent_latency.append(int(a.get("latency_ms", 0)))
        print(f"Agent prediction: {a}")

        rows.append({
            "text": text, "true": y,
            "baseline_label": b["label"], "baseline_score": b["score"],
            "agent": a
        })

    print(f"\nWriting per-example outputs to {out_jsonl}...")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("JSONL writing completed")

    print("\n== Baseline Classification Report ==")
    print(classification_report(y_true, base_labels, digits=3, zero_division=0))

    print("\n== Agent Summary ==")
    p, r, f1, _ = precision_recall_fscore_support(y_true, agent_labels, average="binary", zero_division=0)
    acc = accuracy_score(y_true, agent_labels)
    print(f"Accuracy: {acc:.3f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")
    print(f"Confidence: mean={np.mean(agent_confs):.3f}, std={np.std(agent_confs):.3f}")
    print(f"Fallback rate: {np.mean(agent_fallback)*100:.1f}%")
    print(f"Avg latency: {np.mean(agent_latency):.1f} ms")
    print(f"Processed {len(df)} prompts in {time.time()-t0:.1f}s")

    if report_flag:
        stamp = _now_tag()
        out_dir = _ensure_dir(os.path.join(report_dir_root, f"report_{stamp}"))
        print(f"\nCreating report bundle in {out_dir}")

        try:
            shutil.copyfile(out_jsonl, os.path.join(out_dir, pathlib.Path(out_jsonl).name))
            jsonl_for_table = os.path.join(out_dir, pathlib.Path(out_jsonl).name)
            print("Copied JSONL into report folder")
        except Exception as e:
            print(f"Failed to copy JSONL: {e}")
            jsonl_for_table = out_jsonl

        comparisons_csv = os.path.join(out_dir, "comparisons.csv")
        make_comparison_table(jsonl_for_table, comparisons_csv, k=20)

        print("Saving baseline metrics...")
        baseline_report = classification_report(y_true, base_labels, digits=3, zero_division=0)
        b_p, b_r, b_f1, _ = precision_recall_fscore_support(y_true, base_labels, average="binary", zero_division=0)
        b_acc = accuracy_score(y_true, base_labels)
        _save_txt(baseline_report, os.path.join(out_dir, "baseline_report.txt"))
        _save_json({
            "accuracy": round(b_acc, 4),
            "precision": round(b_p, 4),
            "recall": round(b_r, 4),
            "f1": round(b_f1, 4),
        }, os.path.join(out_dir, "baseline_metrics.json"))
        print("Baseline metrics saved")

        print("Plotting baseline confusion matrix and ROC...")
        b_cm = confusion_matrix(y_true, base_labels, labels=[0, 1])
        _plot_confusion(b_cm, labels=["safe(0)", "unsafe(1)"],
                        title="Baseline Confusion Matrix",
                        out_path=os.path.join(out_dir, "baseline_confusion.png"))
        _plot_roc(y_true, base_scores,
                  title="Baseline ROC",
                  out_path=os.path.join(out_dir, "baseline_roc.png"))
        print("Baseline plots saved")

        print("Saving agent metrics...")
        a_p, a_r, a_f1, _ = precision_recall_fscore_support(y_true, agent_labels, average="binary", zero_division=0)
        a_acc = accuracy_score(y_true, agent_labels)
        conf_mean, conf_std = float(np.mean(agent_confs)), float(np.std(agent_confs))
        fallback_rate = float(np.mean(agent_fallback))
        latency_mean = float(np.mean(agent_latency)) if len(agent_latency) else 0.0

        _save_json({
            "accuracy": round(a_acc, 4),
            "precision": round(a_p, 4),
            "recall": round(a_r, 4),
            "f1": round(a_f1, 4),
            "confidence_mean": round(conf_mean, 4),
            "confidence_std": round(conf_std, 4),
            "fallback_rate": round(fallback_rate, 4),
            "latency_ms_mean": round(latency_mean, 1),
            "n_samples": len(df),
            "elapsed_sec": round(time.time() - t0, 1),
        }, os.path.join(out_dir, "agent_metrics.json"))
        print("Agent metrics saved")

        print("Plotting agent confusion, confidence, and latency histograms...")
        a_cm = confusion_matrix(y_true, agent_labels, labels=[0, 1])
        _plot_confusion(a_cm, labels=["safe(0)", "unsafe(1)"],
                        title="Agent Confusion Matrix",
                        out_path=os.path.join(out_dir, "agent_confusion.png"))
        _plot_hist(agent_confs,
                   "Agent Confidence Distribution",
                   "Confidence (0–1)",
                   os.path.join(out_dir, "agent_confidence_hist.png"),
                   bins=20)
        _plot_hist(agent_latency,
                   "Agent Latency Distribution",
                   "Latency (ms)",
                   os.path.join(out_dir, "agent_latency_hist.png"),
                   bins=20)
        print(f"Report bundle saved successfully → {out_dir}")


def make_comparison_table(jsonl_path: str, out_csv: str = "comparisons.csv", k: int = 20):
    """
    Build a CSV of interesting cases:
    - wrong predictions
    - baseline vs agent disagreements
    - low confidence
    """
    rows = []  # (score, record) tuples
    data = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]  # load JSONL

    # Score cases by importance (wrong > disagree > low_conf)
    for d in data:
        baseline_wrong = (d["true"] != d["baseline_label"])
        wrong = (d["true"] != d["agent"]["label"])               # model wrong?
        disagree = (d["baseline_label"] != d["agent"]["label"])  # disagreement?
        low_conf = d["agent"]["confidence"] < 0.5                # low conf?
        score = 1.0*wrong + 1.0*baseline_wrong + 0.7*disagree + 0.5*low_conf         # heuristic
        rows.append((score, d))                                   # keep both

    rows.sort(key=lambda x: x[0], reverse=True)  # sort by score
    sel = [d for _, d in rows[:k]]               # take top-k

    recs = []  # final records for CSV
    for d in sel:
         recs.append({
            "text": d["text"][:200].replace("\n"," "),
            "true": d["true"],
            "baseline_label": d["baseline_label"], "baseline_score": round(d["baseline_score"],3),
            "agent_label": d["agent"]["label"], "agent_score": round(d["agent"]["score"],3),
            "agent_confidence": round(d["agent"]["confidence"],3),
            "fallback_used": d["agent"]["fallback_used"],
            "intent_summary": str(d["agent"].get("intent_summary",""))[:200],
            "explanation": d["agent"]["explanation"][:180],
            "full_output": d["agent"]
        })
    pd.DataFrame(recs).to_csv(out_csv, index=False)  # write CSV
    print("Wrote", out_csv)  # info


# --------------
# Optional CLI for evaluate.py
# --------------
if __name__ == "__main__":
    print("Running Evaluate.py....", flush=True)
    import argparse
    from .model_resolver import resolve_model

    ap = argparse.ArgumentParser(description="Evaluate baseline + agent; optionally save reports/graphs.")
    ap.add_argument("--test_csv", required=True, help="Path to test CSV with columns text,label")
    # Choose model by alias OR provide explicit gguf path
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model alias (mistral|llama2|phi2|stablelm3b) or gguf filename in models/gguf")
    group.add_argument("--model_path", help="Absolute/relative path to a .gguf file")
    ap.add_argument("--out_jsonl", default="eval_outputs.jsonl", help="Per-example results (JSONL)")
    ap.add_argument("--n_limit", type=int, default=None, help="Evaluate only first N samples")
    ap.add_argument("--use_rag", action="store_true", help="Enable RAG if index exists")
    ap.add_argument("--rag_dir", default="rag_index", help="RAG persistence dir")
    ap.add_argument("--report", action="store_true", help="Save metrics & graphs into timestamped folder")
    ap.add_argument("--report_dir_root", default="reports", help="Where to create report_YYYYmmdd_HHMMSS")
    args = ap.parse_args()
   
    # Resolve the model path from alias or direct path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = resolve_model(args.model)

    print(f"Arguments passed: {args}",flush=True)
    eval_all(
        test_csv=args.test_csv,
        model_path=model_path,
        out_jsonl=args.out_jsonl,
        n_limit=args.n_limit,
        use_rag=args.use_rag,
        rag_dir=args.rag_dir,
        report_flag=args.report,
        report_dir_root=args.report_dir_root,
    )
