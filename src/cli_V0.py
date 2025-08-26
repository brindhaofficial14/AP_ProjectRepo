# src/cli.py
import os, sys, json, argparse
from importlib import import_module
from .model_resolver import resolve_model  # map alias/path to absolute GGUF
from .agent_v2 import PromptSafetyAgent as PromptSafetyAgentV2
from .agent import PromptSafetyAgent as PromptSafetyAgentV1

def _load_agent_class(agent_impl: str):
    """
    Dynamically load the requested agent implementation.
    Returns (impl_tag, AgentClass)
    """
    impl = agent_impl.strip().lower()
    if impl in ("v2", "agent_v2", "new"):
        # Prefer agent_v2.PromptSafetyAgent; fall back to agent.PromptSafetyAgent if not present.
        try:
            mod = import_module("src.agent_v2", package=__package__)
            AgentClass = getattr(mod, "PromptSafetyAgent")
            return "v2", AgentClass
        except Exception:
            # Fall back gracefully if agent_v2 isn't available
            mod = import_module("src.agent", package=__package__)
            AgentClass = getattr(mod, "PromptSafetyAgent")
            return "v2", AgentClass
    else:
        # v1 (original) from agent.py
        mod = import_module(".agent", package=__package__)
        AgentClass = getattr(mod, "PromptSafetyAgent")
        return "v1", AgentClass


def _maybe_build_rag(use_rag: bool):
    if not use_rag:
        return None
    try:
        from .rag import RAGIndex
        return RAGIndex(persist_dir="rag_index")
    except Exception:
        return None  # fail-quiet


def main():
    """
    CLI entrypoint.

    Examples:
      - Classify with v1 agent:
          python -m src.cli --agent_impl v1 --model llama2 "your prompt"

      - Classify with v2 agent:
          python -m src.cli --agent_impl v2 --model llama2 "your prompt"

      - Read text from stdin:
          echo "prompt here" | python -m src.cli --agent_impl v2 --model llama2 --stdin
    """
    ap = argparse.ArgumentParser(description="Prompt Safety Agent CLI")
    ap.add_argument("text", nargs="?", help="Prompt text to classify (optional when --stdin is used)")
    ap.add_argument("--stdin", action="store_true", help="Read the prompt text from STDIN")

    # Choose which agent implementation to use
    ap.add_argument(
        "--agent_impl",
        choices=["v1", "v2"],
        default=os.getenv("AGENT_IMPL", "v2"),
        help="Select agent implementation: v1 uses src/agent.py; v2 uses src/agent_v2.py (default: v2)",
    )

    # Model selection (alias or explicit .gguf path handled by resolve_model)
    ap.add_argument(
        "--model", "-m", default=os.getenv("LLM_MODEL_ALIAS", ""),
        help="Model alias (mistral|llama2|phi2|stablelm3b) or a .gguf path or filename in models/gguf"
    )

    # Perf / runtime options
    ap.add_argument("--n_ctx", type=int, default=int(os.getenv("LLM_N_CTX", "4096")), help="Context window")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for llama.cpp (default: auto)")

    # Retrieval options
    ap.add_argument("--rag", action="store_true", help="Enable RAG if index exists")
    ap.add_argument("--rag_k", type=int, default=3, help="Retrieved neighbors if RAG is enabled")

    args = ap.parse_args()

    # Acquire text (positional or stdin)
    prompt_text = args.text
    if args.stdin:
        stdin_text = sys.stdin.read()
        prompt_text = (stdin_text or "").strip()
    if not prompt_text:
        print('Usage: python -m src.cli --agent_impl v2 --model llama2 "your prompt"', file=sys.stderr)
        sys.exit(1)

    # Resolve model path from alias / path
    model_path = resolve_model(args.model)

    # Load requested agent implementation
    impl_tag, AgentClass = _load_agent_class(args.agent_impl)

    # Instantiate agent, adapting to each impl's constructor signature
    # v1 agent expects: rag (RAGIndex|None) passed in; v2 agent typically uses ragflag
    agent = None
    if impl_tag == "v2":
        # Try v2 signature first: ragflag + rag_k
        try:
            agent = AgentClass(
                model_path=model_path,
                n_ctx=args.n_ctx,
                n_threads=args.threads,
                ragflag=bool(args.rag),
                rag_k=args.rag_k,
            )
        except TypeError:
            # Fall back to v1-style signature if v2 not available
            rag = _maybe_build_rag(args.rag)
            agent = AgentClass(
                model_path=model_path,
                n_ctx=args.n_ctx,
                n_threads=args.threads,
                rag=rag,
                rag_k=args.rag_k,
            )
    else:
        # v1: pass a RAGIndex or None
        rag = _maybe_build_rag(args.rag)
        agent = AgentClass(
            model_path=model_path,
            n_ctx=args.n_ctx,
            n_threads=args.threads,
            rag=rag,
            rag_k=args.rag_k,
        )

    # Run classification and print JSON
    out = agent.classify(prompt_text)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
