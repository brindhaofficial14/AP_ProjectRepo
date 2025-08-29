# src/cli.py
import os, sys, json, builtins
from typing import Any, Dict, Optional
from pathlib import Path

from src.agent_v3 import PromptSafetyAgent

Json = Dict[str, Any]

class _StderrPrint:
    """Route prints to stderr so stdout stays clean JSON."""
    def __enter__(self):
        self._orig = builtins.print
        def _p(*args, **kwargs):
            kwargs.setdefault("file", sys.stderr)
            return self._orig(*args, **kwargs)
        builtins.print = _p
        return self
    def __exit__(self, exc_type, exc, tb):
        builtins.print = self._orig

def _read_prompt() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return sys.stdin.read().strip()

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "") else default

def _resolve_model_path() -> str:
    """Best-effort resolver that works both locally and in Docker."""
    # 1) Explicit env wins
    env_mp = _env("MODEL_PATH")
    if env_mp and Path(env_mp).exists():
        return env_mp

    # 2) Common local repo paths
    repo_root = Path(__file__).resolve().parent.parent
    local_candidates = [
        repo_root / "models" / "gguf" / "llama-2-7b-chat.Q4_K_M.gguf",
        repo_root / "models" / "gguf" / "model.gguf",
    ]
    for p in local_candidates:
        if p.exists():
            return str(p)

    # 3) Docker defaults
    docker_candidates = [
        Path("/app/models/gguf/llama-2-7b-chat.Q4_K_M.gguf"),
        Path("/app/models/gguf/model.gguf"),
    ]
    for p in docker_candidates:
        if p.exists():
            return str(p)

    # 4) If none exist, fall back to Docker default path (useful for a mounted volume)
    return "/app/models/gguf/llama-2-7b-chat.Q4_K_M.gguf"

def _emit_fallback_error(msg: str) -> None:
    """Emit schema-correct JSON indicating a fallback/error condition."""
    err = {
        "label": 1,
        "score": 0.60,
        "confidence": 0.55,
        "fallback_used": True,
        "explanation": msg,
        "recommendation": "Proceed with caution",
    }
    print(json.dumps(err, ensure_ascii=False))

def main():
    prompt = _read_prompt()
    if not prompt:
        print(json.dumps({"error": "No prompt supplied"}))
        sys.exit(2)

    # Defaults are container-friendly; override with envs if needed.
    model_path   = _env("MODEL_PATH") or _resolve_model_path()
    baseline_dir = _env("BASELINE_DIR", "/app/models/baseline")
    chat_format  = _env("CHAT_FORMAT", "llama-2")  # e.g., "mistral-instruct", "llama-2"
    version      = _env("PROMPT_VERSION", "v2")
    ragflag      = _env("RAGFLAG", "false").lower() in ("1","true","yes","on")
    rag_mode     = _env("RAG_MODE", "fallback")
    rag_alpha    = float(_env("RAG_ALPHA", "0.6") or 0.6)
    n_threads    = int(_env("N_THREADS", str(os.cpu_count() or 2)))
    n_ctx        = int(_env("N_CTX", "4096"))
    n_gpu_layers = int(_env("N_GPU_LAYERS", "0"))
    verbose      = _env("VERBOSE", "false").lower() in ("1","true","yes","on")

    # Proactive check so we can emit a clean JSON fallback if missing
    if not Path(model_path).exists():
        _emit_fallback_error(f"Model not found at MODEL_PATH='{model_path}'. "
                             f"Set MODEL_PATH or mount models to /app/models/gguf.")
        sys.exit(2)

    try:
        agent = PromptSafetyAgent(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=0,
            version=version,
            ragflag=ragflag,
            rag_k=3,
            n_threads=n_threads,
            use_fallback=False,
            baseline_model_dir=baseline_dir,
            verbose=verbose,
            chat_format=chat_format,
            rag_mode=rag_mode,
            rag_alpha=rag_alpha,
        )
    except Exception as e:
        _emit_fallback_error(f"Agent init error: {type(e).__name__}: {e}")
        sys.exit(2)

    with _StderrPrint():
        try:
            op: Json = agent.FormFinalOutput(prompt)
        except Exception as e:
            _emit_fallback_error(f"Inference error: {type(e).__name__}: {e}")
            sys.exit(2)

    result = {
        "label": int(op.get("label", 1 if float(op.get("score", 0.5)) >= 0.5 else 0)),
        "score": float(op.get("score", 0.5)),
        "confidence": float(op.get("confidence", 0.5)),
        "fallback_used": bool(op.get("fallback_used", False)),
        "explanation": str(op.get("explanation", "")),
        "recommendation": str(op.get("recommendation", "")),
    }
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
