# src/agent.py
import time, json
from typing import Dict, Any
from llama_cpp import Llama
from .prompts import system_prompt, build_user_prompt, FEWSHOT, UNSAFE_KEYWORDS
from .utils import safe_json_parse, rule_risk_score, rule_label, calibrate_confidence, enforce_consistency, rules_explanation
from .rag import RAGIndex


class PromptSafetyAgent:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0, seed: int = 0,
                 version: str = "v2", rag: RAGIndex | None = None, rag_k: int = 3,
                 n_threads: int | None = None):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_threads=n_threads or 0,  # 0 = auto
            verbose=False
        )
        self.version = version
        self.rag = rag
        self.rag_k = rag_k

    def _call_llm(self, prompt: str, original_text: str) -> Dict[str, Any]:
        # chat-style with system + few-shot
        messages = [{"role": "system", "content": system_prompt(self.version)}]

        # RAG context (optional)
        if self.rag:
            sims = self.rag.query(original_text, k=self.rag_k)
            # Avoid backslashes inside f-string expressions by precomputing the snippet
            ctx_lines = []
            for e in sims:
                text_snip = " ".join(str(e.get("text", ""))[:180].splitlines())
                lab = int(e.get("label", 0))
                ctx_lines.append(f"- (label={lab}) {text_snip}")
            if ctx_lines:
                ctx = "Similar labeled examples:\n" + "\n".join(ctx_lines) + \
                      "\nUse as references; still judge the new prompt independently."
                messages.append({"role": "system", "content": ctx})

        # Few-shot
        for eg in FEWSHOT:
            messages.append({"role": "user", "content": eg["input"]})
            messages.append({"role": "assistant", "content": json.dumps(eg["json"])})

        messages.append({"role": "user", "content": prompt})

        out = self.llm.create_chat_completion(messages=messages, temperature=0.1, max_tokens=256)
        text = out["choices"][0]["message"]["content"]
        return safe_json_parse(text)

    def classify(self, text: str, use_fallback: bool = True) -> Dict[str, Any]:
        t0 = time.time()

        # 1) Primary LLM pass
        user = build_user_prompt(text, version=self.version)
        try:
            primary = self._call_llm(user, original_text=text)
            primary["fallback_used"] = False
        except Exception as e:
            # Rule-only fallback instead of defaulting to unsafe
            r = rule_risk_score(text, UNSAFE_KEYWORDS)
            primary = {
                "label": 1 if r >= 0.5 else 0,
                "score": float(r),
                "confidence": 0.35,
                "fallback_used": True,
                "explanation": f"Primary LLM parse error: {e}. Falling back to rules. " + rules_explanation(text, UNSAFE_KEYWORDS),
                "recommendation": "Proceed with caution" if r < 0.5 else "Block this prompt - suspected manipulation."
            }

        # 2) Rules
        r_score = rule_risk_score(text, UNSAFE_KEYWORDS)
        r_label = rule_label(text, UNSAFE_KEYWORDS)

        # Initialize 'final'
        final = dict(primary)

        # Optional: RAG confidence nudge
        if self.rag:
            try:
                sims = self.rag.query(text, k=self.rag_k)
                if sims:
                    vote = sum(e["label"] for e in sims) / len(sims)  # fraction unsafe
                    agree_with_rag = (primary["label"] == (1 if vote >= 0.5 else 0))
                    if agree_with_rag:
                        final["confidence"] = min(1.0, final.get("confidence", 0.5) + 0.08)
            except Exception:
                pass  # fail-quiet

        # Calibrate confidence with rules agreement
        agree_rules = (primary["label"] == r_label)
        final["confidence"] = calibrate_confidence(
            final.get("confidence", primary.get("confidence", 0.5)),
            r_score,
            agree_rules
        )
        final = enforce_consistency(final)

        # 3) Low-confidence fallback: rephrase + ensemble
        if use_fallback and final["confidence"] < 0.5:
            rephrased_user = user.replace("Prompt:", "Rephrased prompt:").replace(
                "---\n", "---\n(Paraphrase for clarity)\n"
            )
            try:
                alt = self._call_llm(rephrased_user, original_text=text)

                ens_score = 0.5 * primary["score"] + 0.5 * alt["score"]
                ens_label = 1 if sum([primary["label"], alt["label"], r_label]) >= 2 else 0
                explanation = f"{primary.get('explanation','').strip()} | Alt:{alt.get('explanation','').strip()}"

                final.update({
                    "label": ens_label,
                    "score": ens_score,
                    "explanation": explanation[:300],
                    "fallback_used": True
                })
                final["confidence"] = calibrate_confidence(
                    max(primary.get("confidence", 0.5), alt.get("confidence", 0.5)),
                    r_score,
                    ens_label == r_label
                )
                final = enforce_consistency(final)
            except Exception:
                # If alt fails, blend rules with primary
                final["score"] = 0.7 * primary["score"] + 0.3 * r_score
                if final["score"] >= 0.5:
                    final["label"] = 1
                final["fallback_used"] = True
                final = enforce_consistency(final)

        if not final.get("explanation"):
            final["explanation"] = rules_explanation(text, UNSAFE_KEYWORDS)

        # 4) Normalize recommendation (after one last consistency check)
        final = enforce_consistency(final)
        if final["label"] == 1:
            final["recommendation"] = "Block this prompt - suspected manipulation."
        elif final.get("score", 0.0) > 0.3:
            final["recommendation"] = "Proceed with caution."
        else:
            final["recommendation"] = "Proceed."

        final["latency_ms"] = int((time.time() - t0) * 1000)
        return final