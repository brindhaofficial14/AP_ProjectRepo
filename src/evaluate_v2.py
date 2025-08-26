# src/evaluate.py  (only the changed/added parts shown; keep the rest)
import os, json, time, datetime, pathlib, shutil
import pandas as pd, numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from joblib import load
from .agent_v2 import PromptSafetyAgent
from .baseline import predict_baseline

# NEW: imports for rules and keywords
from .utils import rule_risk_score, rule_label, rules_explanation
from .prompts import UNSAFE_KEYWORDS

# ... keep helpers and plots as-is ...

import datetime, pathlib  # make sure these are imported

def _now_tag() -> str:
    """Filesystem-friendly timestamp suffix."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: str) -> str:
    """Create directory if missing; return path."""
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

def eval_all(
    test_csv: str,
    model_path: str,
    out_jsonl: str = "eval_outputs.jsonl",
    n_limit: int | None = None,
    use_rag: bool = True,
    rag_dir: str = "rag_index",
    report_flag: bool = False,
    report_dir_root: str = "reports",
    # NEW: where to write the policy trace CSV
    policy_csv: str = "decision_log.csv",
):
    print(f"Loading test CSV from {test_csv}...", flush=True)
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} rows")

    if n_limit:
        df = df.head(n_limit)
        print(f"Limiting dataset to first {n_limit} rows", flush=True)
    else:
        n_limit = 1
        df = df.head(n_limit)
        print("n_limit not provided and hence setting n_limit to default value. Limiting dataset to first 1 rows", flush=True)

    rag = None
    if use_rag:
        print(f"Initializing RAG index from directory {rag_dir}...")
        from .rag import RAGIndex
        rag = RAGIndex(persist_dir=rag_dir)
        print("RAG initialized")

    print(f"Initializing PromptSafetyAgent with model {model_path}...")
    agent = PromptSafetyAgent(model_path=model_path, ragflag=bool(rag), rag_k=3)
    print("Agent initialized")

    rows, y_true = [], []
    base_labels, base_scores = [], []
    agent_labels, agent_confs, agent_fallback, agent_latency = [], [], [], []

    # NEW: policy decision rows
    policy_rows = []

    t0 = time.time()
    print("Starting evaluation loop over test dataset...")

    for idx, r in enumerate(df.itertuples(index=False), start=1):
        text, y = str(r.text), int(r.label)
        y_true.append(y)

        print(f"\nEvaluating sample {idx}: {text[:50]}... (label={y})")
        # 1) Baseline
        b = predict_baseline(text)
        base_labels.append(int(b["label"]))
        base_scores.append(float(b["score"]))
        print(f"Baseline prediction: {b}")

        # 2) Agent (get LLM parts to drive policy)
        a = agent.classify(b,text)
        agent_labels.append(int(a["label"]))
        agent_confs.append(float(a.get("confidence", 0.5)))
        agent_fallback.append(bool(a.get("fallback_used", False)))
        agent_latency.append(int(a.get("latency_ms", 0)))
        print(f"Agent prediction: {a}")

        # Pull LLM primary parts (fallback to overall if not present)
        parts = a.get("parts", {})
        llm = parts.get("llm", {})
        llm_label = int(llm.get("label", a["label"]))
        llm_score = float(llm.get("score", a["score"]))
        llm_conf = float(llm.get("confidence", a.get("confidence", llm_score)))
        intent_summary = llm.get("intent_summary", a.get("intent_summary", ""))

        # Rules (for fallback)
        r_score = rule_risk_score(text, UNSAFE_KEYWORDS)
        r_label = rule_label(text, UNSAFE_KEYWORDS)
        r_expl = rules_explanation(text, UNSAFE_KEYWORDS)

        # Your policy:
        # Use LLM primary unless (true != llm_label) OR (llm_conf < 0.7)
        cond_mismatch = (y != llm_label)
        cond_lowconf = (llm_conf < 0.7)
        fallback_triggered = bool(cond_mismatch or cond_lowconf)

        if fallback_triggered:
            final_label = int(r_label)
            final_score = float(r_score)
            final_source = "rules_fallback"
            trigger_reasons = ";".join(
                [s for s in [
                    "mismatch_true_vs_llm" if cond_mismatch else "",
                    "llm_low_conf" if cond_lowconf else ""
                ] if s]
            )
            decision_trace = (
                f"fallback→rules because [{trigger_reasons}]. "
                f"llm(label={llm_label}, score={llm_score:.3f}, conf={llm_conf:.3f}); "
                f"rules(label={r_label}, score={r_score:.3f})."
            )
            explanation_full = f"{a.get('explanation','')}".strip()
            if r_expl:
                explanation_full = (explanation_full + " | " + r_expl).strip(" |")
        else:
            final_label = int(llm_label)
            final_score = float(llm_score)
            final_source = "llm_primary"
            trigger_reasons = "none"
            decision_trace = (
                f"use llm_primary (conf={llm_conf:.3f} and matches_true={not cond_mismatch}). "
                f"llm(label={llm_label}, score={llm_score:.3f})."
            )
            explanation_full = a.get("explanation", "")

        # Store per-example JSONL (original behavior)
        rows.append({
            "text": text, "true": y,
            "baseline_label": b["label"], "baseline_score": b["score"],
            "agent": a,
            # NEW: policy decision summary (also mirrored to CSV below)
            "policy_final_label": final_label,
            "policy_final_score": final_score,
            "policy_final_source": final_source,
            "policy_trigger": trigger_reasons,
            "policy_decision_trace": decision_trace,
        })

        # NEW: policy CSV row
        policy_rows.append({
            "idx": idx,
            "text": text[:300].replace("\n", " "),
            "true": y,
            "baseline_label": int(b["label"]),
            "baseline_score": float(b["score"]),
            "llm_label": llm_label,
            "llm_score": llm_score,
            "llm_confidence": llm_conf,
            "intent_summary": intent_summary,
            "agent_label": int(a["label"]),
            "agent_score": float(a["score"]),
            "agent_confidence": float(a.get("confidence", 0.5)),
            "rules_label": int(r_label),
            "rules_score": float(r_score),
            "fallback_triggered": fallback_triggered,
            "trigger_reasons": trigger_reasons,
            "final_label": final_label,
            "final_score": final_score,
            "final_source": final_source,
            "decision_trace": decision_trace,
            "explanation_full": explanation_full[:2000],
            "rules_explanation": r_expl[:1000],
        })

    print(f"\nWriting per-example outputs to {out_jsonl}...")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("JSONL writing completed")

    # NEW: write the policy CSV (your requested audit table)
    pd.DataFrame(policy_rows).to_csv(policy_csv, index=False, encoding="utf-8")
    print(f"Policy decision log written → {policy_csv}")

    # ---- keep your existing summary/plots/report code unchanged below ----
    print("\n== Baseline Classification Report ==")
    print(classification_report(y_true, base_labels, digits=3))

    print("\n== Agent Summary ==")
    p, r, f1, _ = precision_recall_fscore_support(y_true, agent_labels, average="binary")
    acc = accuracy_score(y_true, agent_labels)
    print(f"Accuracy: {acc:.3f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")
    print(f"Confidence: mean={np.mean(agent_confs):.3f}, std={np.std(agent_confs):.3f}")
    print(f"Fallback rate: {np.mean(agent_fallback)*100:.1f}%")
    print(f"Avg latency: {np.mean(agent_latency):.1f} ms")
    print(f"Processed {len(df)} prompts in {time.time()-t0:.1f}s")

    if report_flag:
        # ... unchanged reporting code ...
        # When creating report folder, also drop the policy CSV in there:
        stamp = _now_tag()
        out_dir = _ensure_dir(os.path.join(report_dir_root, f"report_{stamp}"))
        try:
            shutil.copyfile(policy_csv, os.path.join(out_dir, pathlib.Path(policy_csv).name))
        except Exception:
            pass
        # continue with your existing saved metrics/plots...

#CLI for for evaluate_v2.py
if __name__ == "__main__":
    import argparse
    from .model_resolver import resolve_model  # keep if you support --model aliases

    ap = argparse.ArgumentParser(
        description="Evaluate baseline + agent; write per-example JSONL and (optionally) a full report bundle."
    )
    ap.add_argument("--test_csv", required=True, help="Path to test CSV with columns text,label")

    # Choose model by alias OR provide explicit gguf path
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        help="Model alias (mistral|llama2|phi2|stablelm3b) or a gguf filename under models/gguf",
    )
    group.add_argument("--model_path", help="Absolute/relative path to a .gguf file")

    ap.add_argument("--out_jsonl", default="eval_outputs.jsonl", help="Per-example results (JSONL)")
    ap.add_argument("--n_limit", type=int, default=None, help="Evaluate only first N samples")
    ap.add_argument("--use_rag", action="store_true", help="Enable RAG if index exists")
    ap.add_argument("--rag_dir", default="rag_index", help="RAG persistence dir")
    ap.add_argument("--report", action="store_true", help="Save metrics & graphs into timestamped folder")
    ap.add_argument("--report_dir_root", default="reports", help="Where to create report_YYYYmmdd_HHMMSS")
    args = ap.parse_args()

    # Resolve model path from alias or direct path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = resolve_model(args.model)

    print(f"Arguments passed: {args}", flush=True)

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
