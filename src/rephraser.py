# src/rephraser.py
import json, re
from typing import List, Optional

PARAPHRASE_SYSTEM = """You rewrite user prompts for a safety classifier.
- Preserve meaning; remove ambiguity; keep names & entities.
- Output JSON list of distinct rephrasings (2-5). No explanations."""

PARAPHRASE_USER_TMPL = """Rewrite the following prompt into {k} distinct paraphrases.
Keep semantics, make implicit intents explicit, avoid slang.

PROMPT:
---
{prompt}
---
Return ONLY a JSON list of strings.
"""

# very small regex fallback “reword” map
_FALLBACK_MAP = [
    (r"\bignore\b", "disregard"),
    (r"\bignoring\b", "disregarding"),
    (r"\bcontent policies?\b", "safety policies"),
    (r"\bguardrails?\b", "safety rules"),
    (r"\bno rules?\b", "without restrictions"),
]

def _fallback_variants(text: str, k: int) -> List[str]:
    out = set()
    for pat, rep in _FALLBACK_MAP:
        v = re.sub(pat, rep, text, flags=re.IGNORECASE)
        if v != text:
            out.add(v)
        if len(out) >= k:
            break
    if not out:
        out.add(text)  # at least echo original once
    return list(out)[:k]

class Rephraser:
    def __init__(self, llm):
        # reuse your existing llama_cpp.Llama instance
        self.llm = llm

    def generate(self, prompt: str, k: int = 3) -> List[str]:
        try:
            msgs = [
                {"role": "system", "content": PARAPHRASE_SYSTEM},
                {"role": "user", "content": PARAPHRASE_USER_TMPL.format(prompt=prompt, k=k)},
            ]
            out = self.llm.create_chat_completion(messages=msgs, temperature=0.2, max_tokens=256)
            text = out["choices"][0]["message"]["content"]
            variants = json.loads(text)
            # clean & dedupe
            seen, clean = set(), []
            for v in variants:
                v = re.sub(r"\s+", " ", str(v)).strip()
                if v and v not in seen:
                    seen.add(v); clean.append(v)
            return clean[:k] if clean else _fallback_variants(prompt, k)
        except Exception:
            return _fallback_variants(prompt, k)
