# src/eval_ablation_v3.py
import re
import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.agent_v3 import PromptSafetyAgent


# ----------------------------- subset detector -----------------------------
# We focus on dataset-style scaffolding prompts (your false-positive hotspot).
SCAFFOLD_PAT = re.compile(
    r"(you will be given|your task is to|classify your answers|input:|original sentence:|paraphrase:|"
    r"\btense\b|\bnumber\b|\bvoice\b|\badverb\b|\bgender\b|\bsynonym\b)",
    flags=re.I,
)

def is_scaffold(text: str) -> bool:
    return bool(SCAFFOLD_PAT.search(text or ""))


# ----------------------------- evaluation helpers -----------------------------
def eval_one_version(version: str, model_path: str, rows: pd.DataFrame, temperature: float = 0.1) -> Dict[str, Any]:
    """
    Evaluate the pure LLM path for a given system/user template version ("v2" or "v3").
    This uses NO RAG (ragflag=False) so you keep the pure-LLM output.
    """
    agent = PromptSafetyAgent(
        model_path=model_path,
        version=version,
        verbose=False,
        ragflag=False,         # <- keep pure LLM (no retrieval, no fusion)
    )

    preds: List[int] = []
    scores: List[float] = []

    # Evaluate row-by-row (deterministic low-temp set within agent)
    for t in rows["text"]:
        out = agent.ClassifyUsingLLMModel(str(t))  # pure LLM call
        preds.append(int(out["label"]))
        scores.append(float(out["score"]))

    y_true = rows["label"].astype(int).tolist()
    acc = accuracy_score(y_true, preds)

    # We care most about SAFE=0 performance on this subset (reduce false positives)
    prec0, rec0, f10, _ = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], average="binary", pos_label=0
    )
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()

    return {
        "version": version,
        "acc": float(acc),
        "prec_safe": float(prec0),
        "rec_safe": float(rec0),
        "f1_safe": float(f10),
        "cm": cm,
        "preds": preds,
        "scores": scores,
    }


def confusion_df(cm: List[List[int]], labels: Tuple[str, str] = ("safe(0)", "unsafe(1)")) -> pd.DataFrame:
    return pd.DataFrame(
        cm,
        index=[f"true {labels[0]}", f"true {labels[1]}"],
        columns=[f"pred {labels[0]}", f"pred {labels[1]}"],
    )


def build_comparison_df(sub: pd.DataFrame, before: Dict[str, Any], after: Dict[str, Any]) -> pd.DataFrame:
    """
    Per-row comparison table: true label, V2 vs V3 scores/labels, and a small snippet.
    Flags rows where V3 fixed a V2 false positive ("FP→OK") or false negative ("FN→OK").
    """
    rows = []
    for i, row in sub.iterrows():
        y = int(row["label"])
        v2s, v2l = float(before["scores"][i]), int(before["preds"][i])
        v3s, v3l = float(after["scores"][i]),  int(after["preds"][i])

        if y == 0 and v2l == 1 and v3l == 0:
            fix = "FP→OK"
        elif y == 1 and v2l == 0 and v3l == 1:
            fix = "FN→OK"
        else:
            fix = ""

        snippet = " ".join(str(row["text"]).split())
        if len(snippet) > 260:
            snippet = snippet[:257] + "..."

        rows.append({
            "text_snippet": snippet,
            "true_label": y,
            "v2_score": v2s, "v2_label": v2l,
            "v3_score": v3s, "v3_label": v3l,
            "fix_flag": fix
        })
    return pd.DataFrame(rows)


# ----------------------------- main script -----------------------------
def main():
    ap = argparse.ArgumentParser("Ablation: System Prompt V2 → V3")
    ap.add_argument("--csv", required=True, help="CSV with columns: text,label")
    ap.add_argument("--model_path", required=True, help="Path to your local *.gguf model")
    ap.add_argument("--before-version", default="v2", help="Template version for BEFORE (default: v2)")
    ap.add_argument("--after-version",  default="v3", help="Template version for AFTER (default: v3)")
    ap.add_argument("--subset", choices=["scaffold", "all"], default="scaffold",
                    help="Evaluate on scaffolding subset or the full dataset (default: scaffold)")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for speed")
    ap.add_argument("--out-xlsx", default=None, help="Write results to an Excel file (e.g., reports/ablation_v3.xlsx)")
    ap.add_argument("--out-html", default=None, help="Write results to an HTML file (e.g., reports/ablation_v3.html)")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.csv)[["text", "label"]].dropna()
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)

    # Subset selection
    if args.subset == "scaffold":
        sub = df[df["text"].apply(is_scaffold)].reset_index(drop=True)
        if sub.empty:
            print("[warn] No scaffolding prompts detected; falling back to full dataset.")
            sub = df.copy()
    else:
        sub = df.copy()

    print(f"[subset] evaluating n={len(sub)} rows (subset='{args.subset}') from total={len(df)}")

    # Evaluate before/after
    before = eval_one_version(args.before_version, args.model_path, sub)
    after  = eval_one_version(args.after_version,  args.model_path, sub)

    # Print summary
    def round_dict(d):
        return {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in d.items()
            if k not in ("preds", "scores")
        }

    print("\n== BEFORE ({}) ==".format(args.before_version))
    print(json.dumps(round_dict(before), indent=2))
    print("ConfusionMatrix [ [TP(safe=0), FP], [FN, TN] ]:", before["cm"])

    print("\n== AFTER  ({}) ==".format(args.after_version))
    print(json.dumps(round_dict(after), indent=2))
    print("ConfusionMatrix [ [TP(safe=0), FP], [FN, TN] ]:", after["cm"])

    # Build DataFrames for export
    metrics_df = pd.DataFrame([
        {"version": args.before_version, "acc": before["acc"], "prec_safe": before["prec_safe"], "rec_safe": before["rec_safe"], "f1_safe": before["f1_safe"]},
        {"version": args.after_version,  "acc": after["acc"],  "prec_safe": after["prec_safe"],  "rec_safe": after["rec_safe"],  "f1_safe": after["f1_safe"]},
    ])
    cm_before_df = confusion_df(before["cm"])
    cm_after_df  = confusion_df(after["cm"])
    comp_df = build_comparison_df(sub, before, after)

    # Show up to 2 concrete fixes in console
    fixes = comp_df[comp_df["fix_flag"].isin(["FP→OK", "FN→OK"])].head(2)
    if not fixes.empty:
        print("\n== Concrete fixes (Before→After) ==")
        for i, row in fixes.iterrows():
            print(f"\nSnippet: {row['text_snippet']}")
            print(f"Before score/label: {row['v2_score']:.3f} / {row['v2_label']}  →  After score/label: {row['v3_score']:.3f} / {row['v3_label']}  ({row['fix_flag']})")
    else:
        print("\n(no flips found in this subset; expand subset size or verify labels)")

    # Optional: write Excel
    if args.out_xlsx:
        os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)
        with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as xw:
            metrics_df.to_excel(xw, sheet_name="metrics", index=False)
            cm_before_df.to_excel(xw, sheet_name="cm_before")
            cm_after_df.to_excel(xw,  sheet_name="cm_after")
            comp_df.to_excel(xw,     sheet_name="per_row_compare", index=False)
        print(f"[saved] Excel report → {args.out_xlsx}")

    # Optional: write HTML
    if args.out_html:
        os.makedirs(os.path.dirname(args.out_html), exist_ok=True)

        def section(title: str, html_table: str) -> str:
            return f"<h2>{title}</h2>\n{html_table}"

        html_parts: List[str] = []
        html_parts.append("<html><head><meta charset='utf-8'><title>Ablation Report</title>")
        html_parts.append("<style>body{font:14px/1.5 system-ui, sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
                          "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px}"
                          "th{background:#f6f7f9;text-align:left}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style>")
        html_parts.append("</head><body>")
        html_parts.append(f"<h1>Ablation: {args.before_version} → {args.after_version}</h1>")
        html_parts.append(section("Subset",
                                  f"<p>Evaluated <b>{len(sub)}</b> rows (subset='{args.subset}') from total <b>{len(df)}</b>.</p>"))
        html_parts.append(section("Metrics", metrics_df.round(3).to_html(index=False)))
        html_parts.append(section("Confusion Matrix (Before)", cm_before_df.to_html(index=True)))
        html_parts.append(section("Confusion Matrix (After)",  cm_after_df.to_html(index=True)))
        html_parts.append(section("Per-row comparison", comp_df.to_html(index=False)))
        html_parts.append("</body></html>")

        with open(args.out_html, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))
        print(f"[saved] HTML report → {args.out_html}")


if __name__ == "__main__":
    main()


## Evaluate scaffolding subset; save both Excel & HTML
#python -m src.eval_ablation_v3 --model_path "models/gguf/llama-2-7b-chat.Q4_K_M.gguf"   --csv "data/safe_guard_eval.csv" --subset scaffold   --out-xlsx "reports/ablation_v3.xlsx"   --out-html "reports/ablation_v3.html" --limit 20
