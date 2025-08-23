# src/agent.py
import time, json
from typing import Dict, Any
from llama_cpp import Llama
from .prompts import system_prompt, build_user_prompt, FEWSHOT, UNSAFE_KEYWORDS
from .utils import safe_json_parse, rule_risk_score, rule_label, calibrate_confidence
from .rag import RAGIndex

class PromptSafetyAgent:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0, seed: int = 0, version: str = "v2", rag: RAGIndex | None = None, rag_k: int = 3):
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, seed=seed, verbose=False)
        self.version = version
        self.rag = rag
        self.rag_k = rag_k

    def _call_llm(self, prompt: str,original_text: str) -> Dict[str, Any]:
        # chat-style with system + few-shot
        messages = [{"role":"system","content":system_prompt(self.version)}]
        # RAG context 
        if self.rag:
            sims = self.rag.query(original_text, k=self.rag_k)
            # compact, labeled examples for the model to see
            ctx_lines = [f"- (label={e['label']}) {e['text'][:180].replace('\n',' ')}" for e in sims]
            ctx = "Similar labeled examples:\n" + "\n".join(ctx_lines) + \
                  "\nUse as references; still judge the new prompt independently."
            messages.append({"role":"system","content": ctx})

        for eg in FEWSHOT:
            messages.append({"role":"user","content": eg["input"]})
            messages.append({"role":"assistant","content": json.dumps(eg["json"])})
        messages.append({"role":"user","content": prompt})

        out = self.llm.create_chat_completion(messages=messages, temperature=0.1, max_tokens=256)
        text = out["choices"][0]["message"]["content"]
        return safe_json_parse(text)

    def classify(self, text: str, use_fallback: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        # 1) Primary LLM pass
        user = build_user_prompt(text, version=self.version)
        try:
            primary = self._call_llm(user,original_text=text)
            primary["fallback_used"] = False
        except Exception as e:
            primary = {"label": 1, "score": 0.7, "confidence": 0.4, "fallback_used": True,
                       "explanation": f"Primary LLM parse error: {e}", "recommendation": "Proceed with caution"}

        # 2) Rule heuristic
        r_score = rule_risk_score(text, UNSAFE_KEYWORDS)
        r_label = rule_label(text, UNSAFE_KEYWORDS)

        if self.rag:
            sims = self.rag.query(text, k=self.rag_k)
            vote = sum(e["label"] for e in sims) / max(1, len(sims))  # fraction unsafe
            agree = (primary["label"] == (1 if vote >= 0.5 else 0))
            # small calibration bump if LLM agrees with neighbors
            final_conf = final.get("confidence", 0.5)
            if agree:
                final["confidence"] = min(1.0, final_conf + 0.08)

        # 3) Low-confidence fallback: rephrase + rules ensemble
        final = dict(primary)
        agree = (primary["label"] == r_label)
        final["confidence"] = calibrate_confidence(primary.get("confidence", 0.5), r_score, agree)

        if use_fallback and final["confidence"] < 0.5:
            rephrased_user = user.replace("Prompt:", "Rephrased prompt:").replace("---\n", "---\n(Paraphrase for clarity)\n")
            try:
                alt = self._call_llm(rephrased_user)
                # Ensemble: average scores; majority label
                ens_score = 0.5 * primary["score"] + 0.5 * alt["score"]
                ens_label = 1 if sum([primary["label"], alt["label"], r_label]) >= 2 else 0
                explanation = f"{primary.get('explanation','').strip()} | Alt:{alt.get('explanation','').strip()}"
                final.update({
                    "label": ens_label,
                    "score": ens_score,
                    "explanation": explanation[:300],
                    "fallback_used": True
                })
                final["confidence"] = calibrate_confidence(max(primary["confidence"], alt["confidence"]), r_score, ens_label==r_label)
            except Exception:
                # If alt fails, at least blend rules
                final["score"] = 0.7 * primary["score"] + 0.3 * r_score
                if final["score"] >= 0.5: final["label"] = 1
                final["fallback_used"] = True

        # 4) Normalize recommendation
        if final["label"] == 1:
            final["recommendation"] = "Block this prompt - suspected manipulation."
        elif final["score"] > 0.3:
            final["recommendation"] = "Proceed with caution."
        else:
            final["recommendation"] = "Proceed."

        final["latency_ms"] = int((time.time() - t0) * 1000)
        return final
