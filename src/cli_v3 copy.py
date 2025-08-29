# src/cli.py
import os, sys, json, argparse, time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from src.agent_v3 import PromptSafetyAgent
from src.rag import RAGIndex
from src.evaluate_v3 import eval_all as eval_all_v3

Json = Dict[str, Any]


def _print_json(obj: Json):
    print(json.dumps(obj, ensure_ascii=False))


def _load_agent(args) -> PromptSafetyAgent:
    return PromptSafetyAgent(
        model_path=args.model_path,
        version=args.version,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
        ragflag=not args.no_rag,
        rag_k=args.rag_k,
        n_threads=args.n_threads,
        use_fallback=args.use_fallback,
        baseline_model_dir=args.baseline_model_dir,
        verbose=not args.quiet,
        chat_format=args.chat_format,
        rag_mode=args.rag_mode,
        rag_alpha=args.rag_alpha,
    )


# ---------- subcommands ----------

def cmd_classify(args):
    agent = _load_agent(args)

    if args.prompt is None and not args.stdin:
        print("error: provide --prompt or --stdin", file=sys.stderr)
        sys.exit(2)

    text = args.prompt
    if args.stdin:
        text = sys.stdin.read().strip()

    op = agent.FormFinalOutput(text)
    _print_json(op)


def cmd_batch(args):
    """
    Batch classify a CSV with column 'text'.
    Writes out JSONL incrementally so partial progress is preserved if interrupted.
    """
    agent = _load_agent(args)

    inp = Path(args.in_csv)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")
    if args.limit:
        df = df.head(int(args.limit))

    # Resume: collect already processed texts (by index)
    processed_idx = set()
    if args.resume and out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                    processed_idx.add(rec.get("_row_idx", -1))
                except Exception:
                    continue

    n = len(df)
    t0 = time.time()
    with out_jsonl.open("a", encoding="utf-8") as f:
        for i, row in df.iterrows():
            if i in processed_idx:
                continue
            text = str(row["text"])
            try:
                op = agent.FormFinalOutput(text)
                op["_row_idx"] = int(i)
                op["_source_text"] = text if args.include_text else None
                f.write(json.dumps(op, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                dummy = {
                    "label": 1, "score": 0.6, "confidence": 0.55,
                    "fallback_used": True,
                    "explanation": f"Batch error at row {i}: {type(e).__name__}: {e}",
                    "_row_idx": int(i), "_source_text": text if args.include_text else None,
                    "_source": "batch_error",
                }
                f.write(json.dumps(dummy, ensure_ascii=False) + "\n")
                f.flush()

    took = round(time.time() - t0, 2)
    _print_json({"ok": True, "count": n, "saved": str(out_jsonl), "seconds": took})


def cmd_build_rag(args):
    """
    Build (or update) the persistent RAG index from a CSV with columns: text,label.
    """
    rag = RAGIndex(persist_dir=args.persist_dir, model_name=args.emb_model)
    rag.build_from_csv(args.csv, text_col=args.text_col, label_col=args.label_col, batch_size=args.batch_size)
    _print_json({"ok": True, "persist_dir": args.persist_dir})


def cmd_query_rag(args):
    rag = RAGIndex(persist_dir=args.persist_dir, model_name=args.emb_model)
    hits = rag.query(args.text, k=args.k)
    _print_json({"k": args.k, "neighbors": hits})


def cmd_evaluate(args):
    """
    Wraps evaluate_v3.eval_all to generate the full HTML/Excel/JSONL report.
    """
    rep = eval_all_v3(
        test_csv=args.test_csv,
        model_path=args.model_path,
        report_base_dir=args.report_base_dir,
        n_limit=args.n_limit,
        ragflag=not args.no_rag,
        rag_k=args.rag_k,
        baseline_model_dir=args.baseline_model_dir,
    )
    _print_json(rep)


# ---------- parser ----------

def build_parser():
    p = argparse.ArgumentParser(prog="safety-agent", description="Prompt Safety Agent CLI")
    p.add_argument("--model-path", required=True, help="Path to local GGUF model (llama.cpp)")
    p.add_argument("--version", default="v4", help="Prompt template version (e.g., v2/v3/v4)")
    p.add_argument("--n_ctx", type=int, default=4096)
    p.add_argument("--n_gpu_layers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_threads", type=int, default=0)
    p.add_argument("--chat_format", default=None, help='Optional chat format (e.g., "mistral-instruct", "llama-2")')
    p.add_argument("--baseline_model_dir", default=None, help="TF-IDF/LR baseline dir (defaults to models/baseline)")
    p.add_argument("--quiet", action="store_true", help="Reduce logging")

    # RAG options
    p.add_argument("--no_rag", action="store_true", help="Disable RAG paths (default: enabled)")
    p.add_argument("--rag_mode", default="fallback", choices=["fallback", "assist", "fewshot", "hybrid"])
    p.add_argument("--rag_k", type=int, default=3)
    p.add_argument("--rag_alpha", type=float, default=0.6, help="RAG weight when fusing scores (0..1)")
    p.add_argument("--use_fallback", action="store_true", help="Force fallback path (debugging)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # classify
    sc = sub.add_parser("classify", help="Classify a single prompt")
    sc.add_argument("--prompt", help="Prompt text; omit to read from --stdin")
    sc.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    sc.set_defaults(func=cmd_classify)

    # batch
    sb = sub.add_parser("batch", help="Classify an input CSV (column 'text'); write JSONL incrementally")
    sb.add_argument("--in-csv", required=True)
    sb.add_argument("--out-jsonl", required=True)
    sb.add_argument("--limit", type=int, default=None)
    sb.add_argument("--resume", action="store_true", help="Append and skip already processed rows")
    sb.add_argument("--include-text", action="store_true", help="Copy source text into the JSONL (bigger file)")
    sb.set_defaults(func=cmd_batch)

    # rag build
    srb = sub.add_parser("build-rag", help="Build/refresh the RAG index from CSV (text,label)")
    srb.add_argument("--csv", required=True)
    srb.add_argument("--persist-dir", default="rag_index")
    srb.add_argument("--emb-model", default="sentence-transformers/all-MiniLM-L6-v2")
    srb.add_argument("--text-col", default="text")
    srb.add_argument("--label-col", default="label")
    srb.add_argument("--batch-size", type=int, default=256)
    srb.set_defaults(func=cmd_build_rag)

    # rag query
    srq = sub.add_parser("query-rag", help="Query the RAG index for nearest neighbors")
    srq.add_argument("--text", required=True)
    srq.add_argument("-k", type=int, default=5)
    srq.add_argument("--persist-dir", default="rag_index")
    srq.add_argument("--emb-model", default="sentence-transformers/all-MiniLM-L6-v2")
    srq.set_defaults(func=cmd_query_rag)

    # evaluate
    se = sub.add_parser("evaluate", help="Run full evaluation and generate HTML/Excel/JSONL report")
    se.add_argument("--test-csv", required=True)
    se.add_argument("--report_base_dir", default="reports")
    se.add_argument("--n_limit", type=int, default=None)
    se.add_argument("--baseline_model_dir", default=None)
    se.set_defaults(func=cmd_evaluate)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
