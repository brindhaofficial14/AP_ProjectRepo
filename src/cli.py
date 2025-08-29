# src/cli.py
import argparse, json, os, sys

from src.agent_v3 import PromptSafetyAgent
from src.model_resolver import resolve_model

def main():
    ap = argparse.ArgumentParser(description="Prompt Safety Agent (local)")
    ap.add_argument("prompt", nargs="?", help="Text to evaluate for prompt-injection risk")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "llama2"),
                    help="Model alias (llama2/mistral/phi2/stablelm3b) or GGUF path")
    ap.add_argument("--rag", action="store_true", help="Enable RAG fallback")
    ap.add_argument("--no-rag", dest="rag", action="store_false")
    ap.set_defaults(rag=False)
    ap.add_argument("--baseline-dir", default=os.environ.get("BASELINE_DIR", "models/baseline"),
                    help="Directory containing tfidf.joblib")
    args = ap.parse_args()

    if not args.prompt:
        print('Usage: docker run safety-agent "Your prompt here"', file=sys.stderr)
        sys.exit(64)

    model_path = resolve_model(args.model, base_dir="models/gguf")
    agent = PromptSafetyAgent(
        model_path=model_path,
        baseline_model_dir=args.baseline_dir,
        ragflag=args.rag,
        verbose=False,
        n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "0")),
        n_ctx=int(os.environ.get("N_CTX", "4096")),
        n_threads=int(os.environ.get("N_THREADS", "4")),
        version=os.environ.get("PROMPT_VERSION", "v4"),
        chat_format=os.environ.get("CHAT_FORMAT", "llama-2"),
    )
    out = agent.FormFinalOutput(args.prompt)

    # minimal JSON to stdout
    result = {
        "label": out.get("label"),
        "score": out.get("score"),
        "confidence": out.get("confidence"),
        "fallback_used": out.get("fallback_used"),
        "explanation": out.get("explanation"),
        "recommendation": out.get("recommendation"),
    }
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
