# src/cli.py
import sys, json, os
from .agent import PromptSafetyAgent

def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.cli "your prompt"')
        sys.exit(1)
    prompt = sys.argv[1]

    model_path = os.environ.get("LLM_MODEL_PATH", "models/gguf/mistral-7b-instruct.Q4_K_M.gguf")
    agent = PromptSafetyAgent(model_path=model_path)
    out = agent.classify(prompt)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
