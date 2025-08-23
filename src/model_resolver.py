# src/model_resolver.py
import os

# Aliases → filenames you already downloaded to models/gguf/
MODEL_REGISTRY = {
    "mistral":      "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "llama2":       "llama-2-7b-chat.Q4_K_M.gguf",
    "phi2":         "phi-2.Q4_K_M.gguf",
    "stablelm3b":   "stablelm-zephyr-3b.Q4_K_M.gguf",
}

def resolve_model(model_or_path: str, base_dir: str = "models/gguf") -> str:
    """
    Accepts:
      - alias in MODEL_REGISTRY (case-insensitive)
      - a bare filename in base_dir
      - an absolute/relative path to a .gguf file
    Returns absolute path to the .gguf, or raises FileNotFoundError.
    """
    if not model_or_path:
        # fall back to env var or default mistral
        env = os.getenv("LLM_MODEL_PATH")
        if env and os.path.exists(env):
            return os.path.abspath(env)
        model_or_path = "mistral"

    key = model_or_path.lower()

    # 1) Alias
    if key in MODEL_REGISTRY:
        candidate = os.path.join(base_dir, MODEL_REGISTRY[key])
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        raise FileNotFoundError(f"Alias '{key}' maps to '{candidate}', but file not found.")

    # 2) Bare filename in base_dir
    candidate = os.path.join(base_dir, model_or_path)
    if os.path.exists(candidate) and candidate.endswith(".gguf"):
        return os.path.abspath(candidate)

    # 3) Direct path
    if os.path.exists(model_or_path) and model_or_path.endswith(".gguf"):
        return os.path.abspath(model_or_path)

    raise FileNotFoundError(
        f"Could not resolve model '{model_or_path}'. "
        f"Known aliases: {', '.join(MODEL_REGISTRY)}. "
        f"Looked in: {os.path.abspath(base_dir)}"
    )
