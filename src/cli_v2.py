# src/cli_v2.py
import os, sys, json, argparse
from .model_resolver import resolve_model
from .agent_v2 import PromptSafetyAgent  # v2 uses ragflag=True/False

def main():
    ap = argparse.ArgumentParser(description="Prompt Safety CLI (v2)")
    ap.add_argument("text", help="Prompt text to classify")
    ap.add_argument("--model", "-m", default=os.getenv("LLM_MODEL_ALIAS", "llama2"),
                    help="Model alias or .gguf path")
    ap.add_argument("--n_ctx", type=int, default=int(os.getenv("LLM_N_CTX", "4096")))
    ap.add_argument("--threads", type=int, default=None)
    # if --rag omitted, auto-enable when rag_index/ exists
    ap.add_argument("--rag", action="store_true", help="Force-enable RAG")
    ap.add_argument("--no-rag", action="store_true", help="Force-disable RAG")
    ap.add_argument("--rag_k", type=int, default=3)
    args = ap.parse_args()

    model_path = resolve_model(args.model)
    ragflag = args.rag or (os.path.isdir("rag_index") and not args.no_rag)

    agent = PromptSafetyAgent(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_threads=args.threads,
        ragflag=ragflag,
        rag_k=args.rag_k,
    )
    out = agent.classify(args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
