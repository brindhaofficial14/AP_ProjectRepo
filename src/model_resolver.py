# src/model_resolver.py
import os  # filesystem, env

# Map friendly aliases → filenames you've placed in models/gguf/
MODEL_REGISTRY = {
    "mistral":      "mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Mistral 7B Instruct
    "llama2":       "llama-2-7b-chat.Q4_K_M.gguf",           # Llama-2 7B Chat
    "phi2":         "phi-2.Q4_K_M.gguf",                     # Phi-2 2.7B
    "stablelm3b":   "stablelm-zephyr-3b.Q4_K_M.gguf",        # StableLM Zephyr 3B
}

def resolve_model(model_or_path: str, base_dir: str = "models/gguf") -> str:
    """
    Resolve a model argument to an absolute .gguf path.

    Accepts:
      - alias in MODEL_REGISTRY (case-insensitive)
      - a bare filename expected under base_dir
      - an absolute/relative path to a .gguf file

    Returns:
      Absolute path to the .gguf, or raises FileNotFoundError.
    """
    if not model_or_path:
        # Fallback to environment variable or default alias 'mistral'
        env = os.getenv("LLM_MODEL_PATH")
        if env and os.path.exists(env):
            return os.path.abspath(env)  # honor env override
        model_or_path = "mistral"  # default alias if nothing given

    key = model_or_path.lower()  # normalize

    # 1) Alias lookup
    if key in MODEL_REGISTRY:
        candidate = os.path.join(base_dir, MODEL_REGISTRY[key])  # construct path under base_dir
        if os.path.exists(candidate):
            return os.path.abspath(candidate)  # resolved
        raise FileNotFoundError(f"Alias '{key}' maps to '{candidate}', but file not found.")  # helpful error

    # 2) Bare filename under base_dir
    candidate = os.path.join(base_dir, model_or_path)  # try filename inside base_dir
    if os.path.exists(candidate) and candidate.endswith(".gguf"):
        return os.path.abspath(candidate)  # resolved

    # 3) Direct path provided
    if os.path.exists(model_or_path) and model_or_path.endswith(".gguf"):
        return os.path.abspath(model_or_path)  # resolved

    # If all fail, explain what was tried
    raise FileNotFoundError(
        f"Could not resolve model '{model_or_path}'. "
        f"Known aliases: {', '.join(MODEL_REGISTRY)}. "
        f"Looked in: {os.path.abspath(base_dir)}"
    )
