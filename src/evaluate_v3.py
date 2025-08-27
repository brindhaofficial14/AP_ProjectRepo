# src/evaluate_v3.py
import json, time, base64, io, math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Ensure headless plotting (no GUI needed on Windows/Linux)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
from collections import Counter

from .agent_v3 import PromptSafetyAgent

Json = Dict[str, Any]


# --------------------------- helpers ---------------------------

def _now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def _mk_report_dir(base: str = "reports") -> Path:
    p = Path(base) / f"eval_{_now_stamp()}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _metrics_binary(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "accuracy": float(acc)}

def _describe(values: List[float]) -> Dict[str, float]:
    if not values:
        values = [0.0]
    a = np.array(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
    }

def _pick_interesting_cases(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["baseline_label", "llm_label", "final_label", "true_label",
                "final_confidence", "fallback_used"]:
        if col not in df.columns:
            df[col] = np.nan

    cond = (
        (df["baseline_label"] != df["llm_label"])
        | (df["final_label"] != df["true_label"])
        | (df["final_confidence"] < 0.55)
        | (df["fallback_used"])
    )
    out = df.loc[cond].copy()
    if out.empty:
        return out

    out["rank"] = (
        (out["baseline_label"] != out["llm_label"]).astype(int) * 3
        + (out["final_label"] != out["true_label"]).astype(int) * 4
        + (out["fallback_used"]).astype(int) * 2
        + (0.55 - out["final_confidence"]).clip(lower=0)
    )
    out = out.sort_values("rank", ascending=False).head(k)

    def _note(row):
        bits = []
        if row["baseline_label"] != row["llm_label"]:
            bits.append("Baseline and LLM disagree")
        if row["final_label"] != row["true_label"]:
            bits.append("Final decision incorrect")
        if row["final_confidence"] < 0.55:
            bits.append(f"Low confidence ({row['final_confidence']:.2f})")
        if row["fallback_used"]:
            bits.append("Fallback triggered")
        return "; ".join(bits) + "."
    out["insight"] = out.apply(_note, axis=1)
    return out.drop(columns=["rank"])

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def _plot_confusion(y_true: List[int], y_pred: List[int], title: str) -> str:
    if not y_true:
        # Empty figure if no data
        fig = plt.figure()
        plt.title(title + " (no data)")
        return _fig_to_base64(fig)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    return _fig_to_base64(fig)

def _plot_hist(data: List[float], title: str, bins: int = 20, xlabel: str = "") -> str:
    fig = plt.figure()
    if data:
        plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel or title)
    plt.ylabel("Count")
    return _fig_to_base64(fig)

def _plot_pie(true_count: int, false_count: int, title: str) -> str:
    fig = plt.figure()
    total = max(1, true_count + false_count)
    plt.pie([true_count, false_count], labels=["True", "False"], autopct="%1.1f%%")
    plt.title(title + f" (n={total})")
    return _fig_to_base64(fig)

def _json_safe(o):
    """
    Make objects JSON-spec safe:
      - convert NaN/Inf to None
      - cast numpy scalars to Python types
      - recursively clean dicts/lists
    """
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if o is np.nan:  # rarely hit due to above
        return None
    return o

def _fmt_stage_line(name: str, op: Optional[Dict[str, Any]]) -> str:
    """Produce a compact, uniform one-liner per stage, matching console format."""
    if not op:
        return ""
    try:
        label = int(op.get("label", 0))
        score = float(op.get("score", 0.0))
        conf = float(op.get("confidence", 0.0))
        rec = str(op.get("recommendation", "")).strip()
        fb = bool(op.get("fallback_used", False))
        ms = op.get("_latency_ms", "-")
        src = op.get("_source", "?")
        exp = str(op.get("explanation", "")).replace("\n", " ").strip()
        if len(exp) > 200:
            exp = exp[:197] + "..."
        return (
            f"[{name:<9}] "
            f"label={label} score={score:.3f} conf={conf:.3f} "
            f"rec='{rec}' fb={fb} ms={ms} src={src} exp={exp}"
        )
    except Exception:
        return f"[{name:<9}] {op}"

def _write_html(
    out_html: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    df_summary: pd.DataFrame,
    df_cases: pd.DataFrame,
    y_true: List[int],
    y_base: List[int],
    y_final: List[int],
    confs: List[float],
    lat_ms: List[int],
    fallback_rate: float,
):
    img_cm_base = _plot_confusion(y_true, y_base, "Confusion Matrix — Baseline")
    img_cm_final = _plot_confusion(y_true, y_final, "Confusion Matrix — Final")
    img_conf = _plot_hist(confs, "Final Confidence Distribution", xlabel="confidence")
    img_lat = _plot_hist(lat_ms, "Latency (ms) Distribution", xlabel="ms")
    img_fallback = _plot_pie(
        int(round(fallback_rate * len(y_true))),
        len(y_true) - int(round(fallback_rate * len(y_true))),
        "Fallback Trigger Rate",
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8" />
<title>Prompt Safety Agent — Evaluation</title>
<style>
body {{ font-family: Arial, sans-serif; margin:24px; }}
h1,h2 {{ margin-top: 1.2em; }}
table {{ border-collapse: collapse; width:100%; margin: 12px 0; }}
th,td {{ border:1px solid #ddd; padding:8px; font-size:14px; }}
th {{ background:#f5f5f5; text-align:left; }}
img {{ max-width: 720px; width:100%; height:auto; border:1px solid #ddd; padding:4px; }}
.code {{ white-space: pre-wrap; background:#f8f8f8; padding:8px; border:1px solid #ddd; }}
small {{ color:#666; }}
</style></head>
<body>
<h1>Prompt Safety Agent — Evaluation</h1>

<h2>Configuration</h2>
<div class="code">{json.dumps(config, indent=2)}</div>

<h2>Metrics</h2>
<div class="code">{json.dumps(metrics, indent=2)}</div>

<h2>Summary</h2>
{df_summary.to_html(index=False)}

<h2>Interesting Cases (Top {len(df_cases)})</h2>
{df_cases.to_html(index=False)}

<h2>Plots</h2>
<h3>Confusion Matrix — Baseline</h3>
<img src="data:image/png;base64,{img_cm_base}" />
<h3>Confusion Matrix — Final</h3>
<img src="data:image/png;base64,{img_cm_final}" />
<h3>Final Confidence Distribution</h3>
<img src="data:image/png;base64,{img_conf}" />
<h3>Latency Distribution (ms)</h3>
<img src="data:image/png;base64,{img_lat}" />
<h3>Fallback Trigger Rate</h3>
<img src="data:image/png;base64,{img_fallback}" />

<hr />
<p><small>Generated at {time.strftime("%Y-%m-%d %H:%M:%S")}</small></p>
</body></html>
"""
    out_html.write_text(html, encoding="utf-8")


# --------------------------- main evaluation ---------------------------

def eval_all(
    test_csv: str,
    model_path: str,
    report_base_dir: str = "reports",
    n_limit: Optional[int] = None,
    ragflag: bool = True,
    rag_k: int = 3,
    baseline_model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate on test.csv and produce a timestamped report folder with:
      - eval_outputs.jsonl (full per-example details)
      - report.xlsx (multi-sheet: config, metrics_summary, per_example, interesting_cases, text_info, stage_logs)
      - report.html (tables + plots)
    """
    df = pd.read_csv(test_csv)
    if not {"text", "label"} <= set(df.columns):
        raise ValueError("CSV must contain columns: text, label (0/1).")
    if n_limit:
        df = df.head(int(n_limit)).copy()

    report_dir = _mk_report_dir(report_base_dir)
    jsonl_path = report_dir / "eval_outputs.jsonl"
    xlsx_path = report_dir / "report.xlsx"
    html_path = report_dir / "report.html"

    agent = PromptSafetyAgent(
        model_path=model_path,
        ragflag=ragflag,
        rag_k=rag_k,
        baseline_model_dir=baseline_model_dir,
    )

    rows: List[Json] = []
    stage_rows: List[Dict[str, Any]] = []  # NEW: for stage_logs sheet
    y_true: List[int] = []
    y_base: List[int] = []
    y_llm: List[int] = []
    y_final: List[int] = []
    confs_final: List[float] = []
    fallbacks: List[bool] = []
    latencies: List[int] = []

    t0 = time.time()
    for _, r in df.iterrows():
        text = str(r["text"])
        true_label = int(r["label"])

        # ---- Robust method calls (keep eval going even if one fails) ----
        try:
            base_op = agent.ClassifyUsingAlg(text)
        except Exception as e:
            base_op = {
                "label": 1, "score": 0.6, "confidence": 0.55,
                "fallback_used": True,
                "explanation": f"Baseline failed: {type(e).__name__}: {e}",
                "recommendation": "Proceed with caution.",
                "_source": "baseline_error", "_latency_ms": 0,
            }

        try:
            llm_op = agent.ClassifyUsingLLMModel(text)
        except Exception as e:
            llm_op = {
                "label": 1, "score": 0.6, "confidence": 0.55,
                "fallback_used": True,
                "explanation": f"LLM failed: {type(e).__name__}: {e}",
                "recommendation": "Proceed with caution.",
                "_source": "llm_error", "_latency_ms": 0,
            }

        used_fallback = False
        detail_ops: Dict[str, Json] = {"baseline": base_op, "llm": llm_op}

        if int(base_op.get("label", 0)) != int(llm_op.get("label", 0)):
            try:
                rag_op = agent.ClassifyBasedOnRagScore(text)
            except Exception as e:
                rag_op = {
                    "label": 1, "score": 0.5, "confidence": 0.5,
                    "fallback_used": True,
                    "explanation": f"RAG failed: {type(e).__name__}: {e}",
                    "recommendation": "Proceed with caution.",
                    "_source": "rag_error", "_latency_ms": 0,
                }
            try:
                rules_op = agent.ClassifyUsingRuleScoreOnUnSafeWords(text)
            except Exception as e:
                rules_op = {
                    "label": 1, "score": 0.5, "confidence": 0.5,
                    "fallback_used": True,
                    "explanation": f"Rules failed: {type(e).__name__}: {e}",
                    "recommendation": "Proceed with caution.",
                    "_source": "rules_error", "_latency_ms": 0,
                }
            try:
                reph_op = agent.ClassifyBasedOnRephrasedText(text)
            except Exception as e:
                reph_op = {
                    "label": 1, "score": 0.5, "confidence": 0.5,
                    "fallback_used": True,
                    "explanation": f"Rephrase failed: {type(e).__name__}: {e}",
                    "recommendation": "Proceed with caution.",
                    "_source": "rephrase_error", "_latency_ms": 0,
                }
            detail_ops.update({"rag": rag_op, "rules": rules_op, "rephrase": reph_op})
            used_fallback = True

        # ---- Majority vote across collected methods ----
        methods = list(detail_ops.keys())
        labels = [int(detail_ops[m].get("label", 0)) for m in methods]
        counts = Counter(labels)
        majority = int(counts.most_common(1)[0][0])

        maj_ops = [detail_ops[m] for m in methods if int(detail_ops[m].get("label", 0)) == majority]
        final_score = float(np.mean([float(op.get("score", 0.5)) for op in maj_ops])) if maj_ops else 0.5
        final_conf = float(np.mean([float(op.get("confidence", 0.5)) for op in maj_ops])) if maj_ops else 0.5
        best_idx = int(np.argmax([float(op.get("confidence", 0.0)) for op in maj_ops])) if maj_ops else 0
        final_expl = (maj_ops[best_idx].get("explanation", "Merged decision.") if maj_ops else "Merged decision.")

        if majority == 1 and final_score > 0.7:
            final_reco = "Block this prompt - suspected manipulation."
        elif majority == 1 and final_score > 0.3:
            final_reco = "Proceed with caution."
        else:
            final_reco = "Proceed."

        latency_ms = max((op.get("_latency_ms", 0) or 0) for op in detail_ops.values())

        record = {
            "text": text,
            "true_label": int(true_label),

            # Baseline
            "baseline_label": int(base_op.get("label", 0)),
            "baseline_score": float(base_op.get("score", 0.5)),
            "baseline_confidence": float(base_op.get("confidence", 0.5)),
            "baseline_explanation": str(base_op.get("explanation", "")),

            # LLM
            "llm_label": int(llm_op.get("label", 0)),
            "llm_score": float(llm_op.get("score", 0.5)),
            "llm_confidence": float(llm_op.get("confidence", 0.5)),
            "llm_explanation": str(llm_op.get("explanation", "")),

            # Optional branches (None = not triggered)
            "rag_label": int(detail_ops.get("rag", {}).get("label")) if "rag" in detail_ops else None,
            "rag_score": float(detail_ops.get("rag", {}).get("score")) if "rag" in detail_ops else None,
            "rag_confidence": float(detail_ops.get("rag", {}).get("confidence")) if "rag" in detail_ops else None,

            "rules_label": int(detail_ops.get("rules", {}).get("label")) if "rules" in detail_ops else None,
            "rules_score": float(detail_ops.get("rules", {}).get("score")) if "rules" in detail_ops else None,
            "rules_confidence": float(detail_ops.get("rules", {}).get("confidence")) if "rules" in detail_ops else None,

            "rephrase_label": int(detail_ops.get("rephrase", {}).get("label")) if "rephrase" in detail_ops else None,
            "rephrase_score": float(detail_ops.get("rephrase", {}).get("score")) if "rephrase" in detail_ops else None,
            "rephrase_confidence": float(detail_ops.get("rephrase", {}).get("confidence")) if "rephrase" in detail_ops else None,

            # Final
            "final_label": int(majority),
            "final_score": float(final_score),
            "final_confidence": float(final_conf),
            "final_recommendation": final_reco,
            "final_explanation": final_expl,

            "fallback_used": bool(used_fallback),
            "latency_ms": int(latency_ms),

            # Full detail for JSONL (kept spec-safe later)
            "ops_detail": detail_ops,
        }

        rows.append(record)
        y_true.append(int(true_label))
        y_base.append(int(base_op.get("label", 0)))
        y_llm.append(int(llm_op.get("label", 0)))
        y_final.append(int(majority))
        confs_final.append(float(final_conf))
        fallbacks.append(bool(used_fallback))
        latencies.append(int(latency_ms))

        # ---------- Stage logs row (human-readable, one column per method) ----------
        line_baseline = _fmt_stage_line("Baseline", base_op)
        line_llm      = _fmt_stage_line("LLM", llm_op)
        line_rag      = _fmt_stage_line("RAG", detail_ops.get("rag"))
        line_rules    = _fmt_stage_line("RuleScore", detail_ops.get("rules"))
        line_rephrase = _fmt_stage_line("Rephrase", detail_ops.get("rephrase"))
        final_op_view = {
            "label": record["final_label"],
            "score": record["final_score"],
            "confidence": record["final_confidence"],
            "fallback_used": record["fallback_used"],
            "explanation": record["final_explanation"],
            "recommendation": record["final_recommendation"],
            "_source": "final",
            "_latency_ms": None,
        }
        line_final = _fmt_stage_line("FINAL", final_op_view)
        stage_rows.append({
            "text": text,
            "Baseline": line_baseline,
            "LLM": line_llm,
            "RAG": line_rag,
            "RuleScore": line_rules,
            "Rephrase": line_rephrase,
            "FINAL": line_final,
        })

    wall_time = round(time.time() - t0, 2)

    # ---------------- JSONL (spec-safe) ----------------
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_json_safe(r), ensure_ascii=False, allow_nan=False) + "\n")

    # ---------------- Metrics ----------------
    base_metrics = _metrics_binary(y_true, y_base)
    llm_acc_raw = float(accuracy_score(y_true, y_llm)) if y_true else 0.0
    final_acc = float(accuracy_score(y_true, y_final)) if y_true else 0.0
    conf_summary = _describe(confs_final)
    fallback_rate = float(np.mean(np.array(fallbacks, dtype=float))) if fallbacks else 0.0
    avg_latency_ms = float(np.mean(np.array(latencies, dtype=float))) if latencies else 0.0

    df_summary = pd.DataFrame(
        [
            ["baseline_precision", base_metrics["precision"]],
            ["baseline_recall", base_metrics["recall"]],
            ["baseline_f1", base_metrics["f1"]],
            ["baseline_accuracy", base_metrics["accuracy"]],
            ["llm_accuracy_raw", llm_acc_raw],
            ["final_accuracy", final_acc],
            ["final_conf_mean", conf_summary["mean"]],
            ["final_conf_p10", conf_summary["p10"]],
            ["final_conf_p50", conf_summary["p50"]],
            ["final_conf_p90", conf_summary["p90"]],
            ["fallback_rate", fallback_rate],
            ["avg_latency_ms", avg_latency_ms],
            ["num_examples", len(rows)],
            ["wall_time_sec", wall_time],
        ],
        columns=["metric", "value"],
    )

    df_examples = pd.DataFrame(rows).drop(columns=["ops_detail"])
    df_cases = _pick_interesting_cases(df_examples, k=20)
    df_stagelogs = pd.DataFrame(stage_rows, columns=["text","Baseline","LLM","RAG","RuleScore","Rephrase","FINAL"])

    config = {
        "test_csv": str(Path(test_csv).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "baseline_model_dir": str(Path(baseline_model_dir).resolve()) if baseline_model_dir else None,
        "ragflag": bool(ragflag),
        "rag_k": int(rag_k),
        "n_limit": int(n_limit) if n_limit else None,
        "report_dir": str(report_dir.resolve()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": {
            "version": f"{np.__version__=}".split('=')[1] if hasattr(np, "__version__") else "unknown"
        }
    }
    df_config = pd.DataFrame(list(config.items()), columns=["key", "value"])

    # JSON details (truncated for Excel cells)
    df_textinfo = pd.DataFrame(
        {
            "text": [r["text"] for r in rows],
            "ops_detail_json": [json.dumps(_json_safe(r["ops_detail"]))[:32000] for r in rows],
        }
    )

    # ---------------- Excel (multi-sheet) ----------------
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        df_config.to_excel(xw, index=False, sheet_name="config")
        df_summary.to_excel(xw, index=False, sheet_name="metrics_summary")
        df_examples.to_excel(xw, index=False, sheet_name="per_example")
        if not df_cases.empty:
            df_cases.to_excel(xw, index=False, sheet_name="interesting_cases")
        df_textinfo.to_excel(xw, index=False, sheet_name="text_info")
        df_stagelogs.to_excel(xw, index=False, sheet_name="stage_logs")  # NEW

    # ---------------- HTML ----------------
    _write_html(
        out_html=html_path,
        config=config,
        metrics={
            "baseline": base_metrics,
            "llm_accuracy_raw": llm_acc_raw,
            "final_accuracy": final_acc,
            "confidence_summary": conf_summary,
            "fallback_rate": fallback_rate,
            "avg_latency_ms": avg_latency_ms,
            "num_examples": len(rows),
            "wall_time_sec": wall_time,
        },
        df_summary=df_summary,
        df_cases=df_cases[
            [
                "text", "true_label",
                "baseline_label", "baseline_score",
                "llm_label", "llm_score",
                "final_label", "final_score", "final_confidence",
                "fallback_used", "final_explanation", "insight",
            ]
        ] if not df_cases.empty else pd.DataFrame(columns=[
            "text","true_label","baseline_label","baseline_score","llm_label","llm_score",
            "final_label","final_score","final_confidence","fallback_used","final_explanation","insight"
        ]),
        # Build plot series directly from rows to avoid scope/name issues
        y_true=[int(r["true_label"]) for r in rows],
        y_base=[int(r["baseline_label"]) for r in rows],
        y_final=[int(r["final_label"]) for r in rows],
        confs=[float(r["final_confidence"]) for r in rows],
        lat_ms=[int(r["latency_ms"]) for r in rows],
        fallback_rate=fallback_rate,
    )

    report = {
        "report_dir": str(report_dir),
        "jsonl_path": str(jsonl_path),
        "excel_path": str(xlsx_path),
        "html_path": str(html_path),
        "baseline_metrics": base_metrics,
        "llm_accuracy_raw": llm_acc_raw,
        "final_accuracy": final_acc,
        "confidence_summary": conf_summary,
        "fallback_rate": fallback_rate,
        "avg_latency_ms": avg_latency_ms,
        "num_examples": len(rows),
        "wall_time_sec": wall_time,
    }

    print("\n=== EVALUATION SUMMARY ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate Prompt Safety Agent on test.csv and generate a full report.")
    ap.add_argument("--test_csv", required=True, help="Path to CSV with columns: text,label")
    ap.add_argument("--model_path", required=True, help="Path to local GGUF model for llama.cpp")
    ap.add_argument("--report_base_dir", default="reports", help="Base directory for timestamped report folder")
    ap.add_argument("--n_limit", type=int, default=None, help="Optional cap on number of rows")
    ap.add_argument("--ragflag", action="store_true", help="Enable RAG during evaluation")
    ap.add_argument("--rag_k", type=int, default=3, help="Top-K neighbors for RAG")
    ap.add_argument("--baseline_model_dir", default=None, help="Directory for baseline artifacts (TF-IDF, LR, etc.)")
    args = ap.parse_args()

    eval_all(
        test_csv=args.test_csv,
        model_path=args.model_path,
        report_base_dir=args.report_base_dir,
        n_limit=args.n_limit,
        ragflag=bool(args.ragflag),
        rag_k=int(args.rag_k),
        baseline_model_dir=args.baseline_model_dir,
    )
