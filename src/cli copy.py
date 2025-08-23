# src/cli.py
import os, sys, json, argparse
from .agent import PromptSafetyAgent
from .model_resolver import resolve_model

def main():
    ap = argparse.ArgumentParser(description="Prompt Safety Agent CLI")
    ap.add_argument("text", nargs="?", help="Prompt text to classify")
    ap.add_argument("--model", "-m", default=os.getenv("LLM_MODEL_ALIAS", ""),
                    help="Model alias (mistral|llama2|phi2|stablelm3b) or a .gguf path or filename in models/gguf")
    ap.add_argument("--n_ctx", type=int, default=int(os.getenv("LLM_N_CTX", "4096")), help="Context window")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for llama.cpp (default: auto)")
    ap.add_argument("--rag", action="store_true", help="Enable RAG if index exists")
    ap.add_argument("--rag_k", type=int, default=3, help="Retrieved neighbors if RAG is enabled")
    args = ap.parse_args()

    if not args.text:
        print('Usage: python -m src.cli --model phi2 "your prompt"', file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model(args.model)
    rag = None
    if args.rag:
        try:
            from .rag import RAGIndex
            rag = RAGIndex(persist_dir="rag_index")
        except Exception:
            rag = None

    agent = PromptSafetyAgent(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_threads=args.threads,
        rag=rag,
        rag_k=args.rag_k
    )
    out = agent.classify(args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
