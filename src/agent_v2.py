# src/agent.py
import time, json
from typing import Dict, Any
from llama_cpp import Llama
from .prompts import system_prompt, build_user_prompt, FEWSHOT, UNSAFE_KEYWORDS
from .utils import (
    safe_json_parse,
    rule_risk_score,
    rule_label,
    calibrate_confidence,
    enforce_consistency,
    rules_explanation,
    scenario_features,
    scenario_weights,
)
from .rag import RAGIndex
from .rephraser import Rephraser
from .baseline import predict_baseline

class PromptSafetyAgent:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        seed: int = 0,
        version: str = "v2",
        ragflag: bool = True,
        rag_k: int = 3,
        n_threads: int | None = None,
        use_fallback: bool = True,        # <- keep this
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_threads=n_threads or 0,
            verbose=False,
        )
        self.version = version
        self.ragflag = ragflag
        self.rag_k = rag_k
        self.rephraser = Rephraser(self.llm)
        self.use_fallback = use_fallback

    def _call_llm(self, prompt: str, original_text: str) -> Dict[str, Any]:
        # chat-style with system + few-shot
        messages = [{"role": "system", "content": system_prompt(self.version)}]

        # RAG context (optional)
        if self.ragflag:
            rag = RAGIndex(persist_dir="rag_index")
            sims = rag.query(original_text, k=self.rag_k)
            # Avoid backslashes inside f-string expressions by precomputing the snippet
            ctx_lines = []
            for e in sims:
                text_snip = " ".join(str(e.get("text", ""))[:180].splitlines())
                lab = int(e.get("label", 0))
                ctx_lines.append(f"- (label={lab}) {text_snip}")
            if ctx_lines:
                ctx = (
                    "Similar labeled examples:\n"
                    + "\n".join(ctx_lines)
                    + "\nUse as references; still judge the new prompt independently."
                )
                messages.append({"role": "system", "content": ctx})

        # Few-shot
        for eg in FEWSHOT:
            messages.append({"role": "user", "content": eg["input"]})
            messages.append({"role": "assistant", "content": json.dumps(eg["json"])})

        messages.append({"role": "user", "content": prompt})

        out = self.llm.create_chat_completion(
            messages=messages, temperature=0.1, max_tokens=256
        )
        text = out["choices"][0]["message"]["content"]
        print(f"model output text:{text}")
        return safe_json_parse(text)

    def _summarize_intent(self, text: str) -> str:
        """
        Lightweight helper: if the main JSON doesn't carry `intent_summary`,
        ask the LLM for a <=15-word neutral description (no JSON required).
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise safety analyst. "
                        "In <=5 words, neutrally summarize the user's apparent intent/purpose. "
                        "No JSON, just plain text."
                    ),
                },
                {"role": "user", "content": text},
            ]
            out = self.llm.create_chat_completion(
                messages=messages, temperature=0.1, max_tokens=32
            )
            return str(out["choices"][0]["message"]["content"]).strip()[:200]
        except Exception:
            return ""

    def _classify_once(self, text: str) -> Dict[str, Any]:
        """Single LLM classification pass on a specific text (requests `intent_summary`)."""
        user = build_user_prompt(text, version=self.version)
        # Nudge the model to include an intent summary in its JSON (backward-compatible)
        user += (
            "\n\nAdditionally, include an 'intent_summary' field (<=15 words) that "
            "neutrally describes the user's apparent intent."
        )
        return self._call_llm(user, original_text=text)
    
    def Run_Fallback(self,t0: float,fb_reason: str, text: str,primary: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Robust fallback that uses rules (+ optional RAG/rephrase) even if `primary` is missing
        or lacks fields. Returns a full `final` dict.
        """
        # Safe primary skeleton
        base_primary: Dict[str, Any] = dict(primary or {})
        base_primary.setdefault("label", 0)
        base_primary.setdefault("score", 0.0)
        base_primary.setdefault("confidence", 0.0)
        base_primary["fallback_used"] = True

        # ---- Rules ----
        r_score = rule_risk_score(text, UNSAFE_KEYWORDS)
        r_label = rule_label(text, UNSAFE_KEYWORDS)
        print(f"Rules Risk Score:{r_score}; Rules Label:{r_label}")

        # ---- RAG vote ----
        rag_vote = None
        sims = []
        if self.ragflag:
            rag = RAGIndex(persist_dir="rag_index")
            try:
                sims = rag.query(text, k=self.rag_k)
                if sims:
                    rag_vote = sum(e["label"] for e in sims) / len(sims)  # 0..1
            except Exception:
                pass
        print(f"Sims is {str(sims)}. ragflag={self.ragflag}. Rag vote={rag_vote}")

        # ---- Ensure intent_summary ----
        intent_summary = base_primary.get("intent_summary") or self._summarize_intent(text)

        # ---- Confidence calibration vs rules ----
        agree_rules = (int(base_primary.get("label", 0)) == r_label)
        conf0 = float(base_primary.get("confidence", base_primary.get("score", 0.0)))
        conf_cal = calibrate_confidence(conf0, r_score, agree_rules)

        # ---- Scenario weights ----
        feats = scenario_features(text)
        W = scenario_weights(feats, rag_vote)

        # ---- Optional rephrase set (only if allowed & helpful) ----
        re_scores, re_labels, variants = [], [], []
        if self.use_fallback and (conf_cal < 0.7 or feats.get("ambiguous", 0.0) > 0.0):
            try:
                variants = self.rephraser.generate(text, k=3)
            except Exception:
                variants = []
            print(f"Rephrased variants:{str(variants)}")
            for v in variants:
                try:
                    alt = self._classify_once(v)
                    re_scores.append(float(alt.get("score", 0.0)))
                    re_labels.append(int(alt.get("label", 0)))
                except Exception:
                    pass
        print(f"Rephrase scores:{re_scores} labels:{re_labels}")

        # ---- Blend & vote ----
        llm_score = float(base_primary.get("score", 0.0))
        rules_score = float(r_score)
        rephrase_score = (sum(re_scores) / len(re_scores)) if re_scores else llm_score
        rag_score = float(rag_vote) if rag_vote is not None else 0.0

        blended = (
            W["llm"] * llm_score
            + W["rules"] * rules_score
            + W["rephrase"] * rephrase_score
            + W["rag"] * rag_score
        )

        votes = [int(base_primary.get("label", 0)), int(r_label)]
        if re_labels:
            votes += [int(x) for x in re_labels]
        if rag_vote is not None:
            votes.append(1 if rag_vote >= 0.5 else 0)
        ens_label = 1 if sum(votes) >= (len(votes) / 2 + 1e-6) else 0

        # ---- Build explanation ----
        avg_re = (sum(re_scores) / len(re_scores)) if re_scores else None
        weights_str = (
            f"weights={{'llm':{W['llm']:.2f}, 'rules':{W['rules']:.2f}, "
            f"'rephrase':{W['rephrase']:.2f}, 'rag':{W['rag']:.2f}}}"
        )
        expl_parts = [
            f"Fallback: {fb_reason}",
            f"LLM:{llm_score:.2f}",
            f"Rules:{rules_score:.2f}",
            f"Rephrases:{avg_re:.2f}" if avg_re is not None else "Rephrases:—",
        ]
        if rag_vote is not None:
            expl_parts.append(f"RAG:{rag_score:.2f}")

        indicators_str = rules_explanation(text, UNSAFE_KEYWORDS)
        if feats.get("contradiction", 0.0) >= 0.5:
            expl_parts.append(
                f"Contradiction:{feats['contradiction']:.2f} (conflicting/negating instructions detected)"
            )
        explanation = ", ".join(expl_parts) + " | " + indicators_str

        # ---- Assemble final ----
        final = dict(base_primary)
        final.update(
            {
                "label": ens_label,
                "score": blended,
                "confidence": conf_cal,
                "intent_summary": intent_summary,
                "fallback_used": True,
                "explanation": explanation,
            }
        )

        # Recalibrate confidence by margin & agreement
        agree_final_rules = (final["label"] == r_label)
        margin = abs(final["score"] - 0.5)
        final["confidence"] = min(1.0, max(final.get("confidence", 0.5), 0.45 + 0.8 * margin))
        if agree_final_rules:
            final["confidence"] = min(1.0, final["confidence"] + 0.05)

        # Recommendation & latency
        final = enforce_consistency(final)
        print(f"Final Score:{final.get('score', 0.0)}")
        if final["label"] == 1 or final.get("score", 0.0) > 0.8:
            final["recommendation"] = "Block this prompt - suspected manipulation."
        elif final.get("score", 0.0) > 0.3:
            final["recommendation"] = "Proceed with caution."
        else:
            final["recommendation"] = "Proceed."
        final["latency_ms"] = int((time.time() - t0) * 1000)
        return final

    def classify(self,baseline: Dict[str, Any], text: str, use_fallback: bool = True) -> Dict[str, Any]:
        t0 = time.time()

        # --- Primary pass ---
        try:
            print("Run LLM once")
            b = predict_baseline(text)
            primary = self._classify_once(text)
            print(f"LLM Primary output:{primary}")
            primary["fallback_used"] = False
            if(primary["confidence"]<0.5):
                primary["fallback_used"] = True
                return self.Run_Fallback(t0, f"Primary Confidence score({primary["confidence"]}) less than 0.5; ", text, primary=None)
            elif(baseline["label"] != primary["label"]):
                primary["fallback_used"] = True
                return self.Run_Fallback(t0, f"Baseline label({baseline["label"]}) does not match with LLM model label output({primary["label"]}) ", text, primary=None)
        except Exception as e:
            print("LLM failed outright; switching to fallback.")
            return self.Run_Fallback(t0, f"Primary LLM parse error: {e}", text, primary=None)

        # Decide if we should fallback due to low confidence
        prim_conf = float(primary.get("confidence", primary.get("score", 0.0)))
        if self.use_fallback and prim_conf < 0.5:
            return self.Run_Fallback(
                t0,
                f"Primary confidence {prim_conf:.2f} < 0.50",
                text,
                primary=primary,
            )

        # --- No fallback needed: finalize primary cleanly ---
        final = dict(primary)
        # Ensure intent summary is present
        if not final.get("intent_summary"):
            final["intent_summary"] = self._summarize_intent(text)

        # Append rules explanation for transparency
        indicators_str = rules_explanation(text, UNSAFE_KEYWORDS)
        if final.get("explanation"):
            final["explanation"] = f"{final['explanation']} | {indicators_str}"
        else:
            final["explanation"] = indicators_str

        # Normalize recommendation & latency
        final = enforce_consistency(final)
        print(f"Final Score:{final.get('score', 0.0)}")
        if final["label"] == 1 or final.get("score", 0.0) > 0.8:
            final["recommendation"] = "Block this prompt - suspected manipulation."
        elif final.get("score", 0.0) > 0.3:
            final["recommendation"] = "Proceed with caution."
        else:
            final["recommendation"] = "Proceed."
        final["latency_ms"] = int((time.time() - t0) * 1000)
        return final