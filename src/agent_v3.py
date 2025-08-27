# src/agent_v3.py
import json, re, time
from typing import Dict, Any, List, Optional
from collections import Counter

from llama_cpp import Llama

# Project imports (match your provided files)
from .prompts import system_prompt, build_user_prompt, FEWSHOT, UNSAFE_KEYWORDS
from .utils import (
    safe_json_parse,
    enforce_consistency,
    rule_risk_score,
    rule_label,
    rules_explanation,
    scenario_features,
    scenario_weights,
)
from .rag import RAGIndex
from .rephraser import Rephraser
from .baseline import predict_baseline

Json = Dict[str, Any]


class PromptSafetyAgent:
    """
    Local-only Prompt Safety Agent.

    Decision policy:
      1) baseline → 2) LLM
      3) If labels disagree → RAG, Rules, Rephrase (in that order)
      4) Majority vote across these 5 outputs
      5) Final JSON: majority label; avg score/conf over majority;
         explanation from highest-confidence method in majority; recommendation by risk.
    """

    # -------------------------- Initialization --------------------------

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        seed: int = 0,
        version: str = "v2",
        ragflag: bool = True,
        rag_k: int = 3,
        n_threads: Optional[int] = None,
        use_fallback: bool = True,
        baseline_model_dir: Optional[str] = None,
        verbose: bool = True,
        chat_format: Optional[str] = None,  # e.g., "mistral-instruct" or "llama-2"
    ):
        # Build kwargs so chat_format is only passed if provided
        llama_kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_threads=n_threads or 0,
            verbose=False,
        )
        if chat_format:
            llama_kwargs["chat_format"] = chat_format

        self.llm = Llama(**llama_kwargs)

        self.version = version
        self.ragflag = ragflag
        self.rag_k = rag_k
        self.rephraser = Rephraser(self.llm)
        self.use_fallback = use_fallback
        self.baseline_model_dir = baseline_model_dir
        self.verbose = verbose

        # Compiled regexes for quick pattern checks (used in heuristics)
        self._override_patterns = re.compile(
            r"\b(ignore (all|any) (previous|prior) (instructions|guidelines)|"
            r"developer mode|jailbreak|pretend you are|"
            r"bypass|disable (safety|guardrails|filters)|"
            r"no (ethic|moral)s|overrid(e|ing) rules)\b",
            re.IGNORECASE,
        )
        self._harm_illegal_patterns = re.compile(
            r"\b(hack|malware|keylogger|phishing|credit card|ssn|terror|bomb|"
            r"steal|crack|piracy|exploit|sql injection|xss|ransomware|doxx?)\b",
            re.IGNORECASE,
        )
        self._contradiction_patterns = re.compile(
            r"\b(contradict|regardless of policy|even if forbidden|"
            r"no matter the rules|ignore\W+policy|"
            r"do not tell anyone|keep (this|it) secret)\b",
            re.IGNORECASE,
        )

    # -------------------------- Uniform output & printing --------------------------

    _STANDARD_KEYS = ("label", "score", "confidence", "fallback_used", "explanation", "recommendation")

    def _to_public_json(self, op: Dict[str, Any]) -> Dict[str, Any]:
        """Project any internal dict to the public uniform schema (add empties if missing)."""
        score = float(op.get("score", 0.5))
        label = int(op.get("label", 1 if score >= 0.5 else 0))
        confidence = float(op.get("confidence", 0.5 + abs(score - 0.5)))
        fallback_used = bool(op.get("fallback_used", False))
        explanation = str(op.get("explanation", "")).strip()
        # Always derive recommendation from (possibly snapped) label/score
        recommendation = self._recommendation(label, score)
        return {
            "label": label,
            "score": max(0.0, min(1.0, score)),
            "confidence": max(0.0, min(1.0, confidence)),
            "fallback_used": fallback_used,
            "explanation": explanation,
            "recommendation": recommendation,
        }

    # pretty one-liners for stage logs
    def _fmt_line(self, name: str, op: Dict[str, Any]) -> str:
        pub = self._to_public_json(op)
        exp = pub["explanation"].replace("\n", " ").strip()[:160]
        return (
            f"[{name:<9}] "
            f"label={pub['label']} "
            f"score={pub['score']:.3f} "
            f"conf={pub['confidence']:.3f} "
            f"rec='{pub['recommendation']}' "
            f"fb={pub['fallback_used']} "
            f"ms={op.get('_latency_ms', '-')} "
            f"src={op.get('_source', '?')} "
            f"exp={exp}"
        )

    def _print_stage(self, name: str, op: Dict[str, Any]):
        if not getattr(self, "verbose", False):
            return
        print(self._fmt_line(name, op))
        print(f"[{name} JSON] " + json.dumps(self._to_public_json(op), ensure_ascii=False))

    # -------------------------- Chat Helpers --------------------------

    def _fewshot_messages_from_struct(self) -> List[Dict[str, str]]:
        """
        Convert your FEWSHOT (with keys 'input' and 'json') into proper chat turns.
        """
        msgs: List[Dict[str, str]] = []
        if isinstance(FEWSHOT, (list, tuple)):
            for ex in FEWSHOT:
                try:
                    user_inp = ex.get("input", "")
                    asst_json = json.dumps(ex.get("json", {}), ensure_ascii=False)
                    if user_inp:
                        msgs.append({"role": "user", "content": str(user_inp)})
                        msgs.append({"role": "assistant", "content": asst_json})
                except Exception:
                    continue
        return msgs

    # -------------------------- Core LLM Call --------------------------

    def _call_llm(self, prompt: str, original_text: str) -> Json:
        """
        Chat call: system + few-shot (converted) + user. Expect JSON-only output.
        """
        # Ensure string prompt
        if not isinstance(prompt, str):
            try:
                prompt = json.dumps(prompt, ensure_ascii=False)
            except Exception:
                prompt = str(prompt)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt(self.version)}]
        # Include few-shot normalized from provided FEWSHOT struct
        try:
            messages.extend(self._fewshot_messages_from_struct())
        except Exception:
            pass
        messages.append({"role": "user", "content": prompt})

        try:
            out = self.llm.create_chat_completion(
                messages=messages, temperature=0.1, max_tokens=512
            )
            text = out["choices"][0]["message"]["content"]
            print(f"[LLM raw] {text}")
            parsed = safe_json_parse(text)          # returns normalized dict per utils.py
            return self._normalize_output(parsed, source="llm")
        except Exception as e:
            # Conservative fallback so pipeline continues
            msg = f"LLM call failed: {type(e).__name__}: {e}"
            print("[LLM ERROR]", msg)
            fallback = {
                "label": 1, "score": 0.6, "confidence": 0.55,
                "fallback_used": True,
                "explanation": f"{msg}. Returned conservative fallback.",
            }
            return self._normalize_output(fallback, source="llm_fallback")

    # -------------------------- Shared Helpers --------------------------

    def _recommendation(self, label: int, score: float) -> str:
        if label == 1 and score > 0.7:
            return "Block this prompt - suspected manipulation."
        elif label == 1 and score > 0.3:
            return "Proceed with caution."
        else:
            return "Proceed."

    def _normalize_output(self, obj: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Normalize any partial/dirty dict into the uniform, schema-complete output and
        ensure label↔score consistency before deriving recommendation.
        """
        if not isinstance(obj, dict):
            obj = {}
        # Seed defaults
        obj.setdefault("score", 0.5)
        obj.setdefault("label", 1 if float(obj["score"]) >= 0.5 else 0)
        obj.setdefault("confidence", 0.5 + abs(float(obj["score"]) - 0.5))
        obj.setdefault("fallback_used", False)
        obj.setdefault("explanation", "")

        # Snap label to score when needed; clamp score/conf in [0,1]
        obj = enforce_consistency(obj)

        # Always (re)compute recommendation AFTER enforcing consistency
        obj["recommendation"] = self._recommendation(int(obj["label"]), float(obj["score"]))

        # provenance
        obj["_source"] = source
        return obj

    # -------------------------- Method 1: Baseline --------------------------

    def ClassifyUsingAlg(self, text: str, model_dir: Optional[str] = None) -> Json:
        """
        Baseline (TF-IDF + LR) + quick signal enrichment in explanation.
        """
        t0 = time.time()
        raw = predict_baseline(text, model_dir or self.baseline_model_dir or "models/baseline")
        # add confidence from margin to 0.5
        raw.setdefault("confidence", 0.5 + abs(float(raw.get("score", 0.5)) - 0.5))
        out = self._normalize_output(raw, source="baseline")
        # enrich explanation via rules (no label flip)
        expl = rules_explanation(text, UNSAFE_KEYWORDS,out["score"])
        out["explanation"] = f"Baseline prob + signal check. {expl}"
        out["_latency_ms"] = int((time.time() - t0) * 1000)
        return out

    # -------------------------- Method 2: LLM --------------------------

    def ClassifyUsingLLMModel(self, text: str) -> Json:
        t0 = time.time()
        user = build_user_prompt(text, version=self.version)
        out = self._call_llm(user, original_text=text)
        out["_latency_ms"] = int((time.time() - t0) * 1000)
        return out

    # -------------------------- Method 3: RAG --------------------------

    def ClassifyBasedOnRagScore(self, text: str) -> Json:
        """
        Similarity vote (neighbors) + rules on current text → fused score.
        """
        t0 = time.time()
        sims: List[Dict[str, Any]] = []
        rag_vote = None
        try:
            if self.ragflag:
                rag = RAGIndex(persist_dir="rag_index")
                sims = rag.query(text, k=self.rag_k) or []
                if sims:
                    # similarity-weighted unsafe probability
                    wsum = sum(s.get("similarity", 0.0) for s in sims)
                    unsafe_sum = sum(s.get("similarity", 0.0) * (1.0 if int(s.get("label", 0)) == 1 else 0.0) for s in sims)
                    rag_vote = (unsafe_sum / wsum) if wsum > 0 else None
        except Exception:
            sims = []

        # rules on current text
        r_score = float(rule_risk_score(text, UNSAFE_KEYWORDS))
        
        # fuse (conservative): prefer neighbors if present
        if rag_vote is None:
            fused = r_score
        else:
            fused = 0.6 * float(rag_vote) + 0.4 * r_score
        
        r_expl = rules_explanation(text, UNSAFE_KEYWORDS,str(fused))

        label = int(fused >= 0.5)
        confidence = 0.5 + abs(fused - 0.5)

        explanation = (
            f"RAG similarity vote + rules. neighbors={len(sims)}, "
            f"rag_vote={None if rag_vote is None else round(rag_vote,3)}. {r_expl}"
        )
        out = self._normalize_output(
            {
                "label": label,
                "score": fused,
                "confidence": confidence,
                "fallback_used": True,
                "explanation": explanation,
            },
            source="rag",
        )
        out["_neighbors"] = sims
        out["_latency_ms"] = int((time.time() - t0) * 1000)
        return out

    # -------------------------- Method 4: Rules --------------------------

    def ClassifyUsingRuleScoreOnUnSafeWords(self, text: str) -> Json:
        """
        Pure rule-based: score from UNSAFE_KEYWORDS and combos; explanation via same logic.
        """
        t0 = time.time()
        score = float(rule_risk_score(text, UNSAFE_KEYWORDS))
        label = int(rule_label(text, UNSAFE_KEYWORDS))
        conf = 0.5 + abs(score - 0.5)
        expl = rules_explanation(text, UNSAFE_KEYWORDS,score)

        out = self._normalize_output(
            {
                "label": label,
                "score": score,
                "confidence": conf,
                "fallback_used": False,
                "explanation": f"Rule-only decision. {expl}",
            },
            source="rules",
        )
        out["_latency_ms"] = int((time.time() - t0) * 1000)
        return out

    # -------------------------- Method 5: Rephrase --------------------------

    def ClassifyBasedOnRephrasedText(self, text: str, n_variants: int = 3) -> Json:
        """
        Generate paraphrases (using your Rephraser.generate) → LLM on each → majority/avg.
        """
        t0 = time.time()
        try:
            variants = self.rephraser.generate(text, k=n_variants)
        except Exception:
            variants = []

        eval_texts = [text] + variants
        results: List[Json] = []
        for t in eval_texts:
            user = build_user_prompt(t, version=self.version)
            res = self._call_llm(user, original_text=t)
            results.append(res)

        labels = [r["label"] for r in results]
        scores = [r["score"] for r in results]
        confs = [r["confidence"] for r in results]

        maj = Counter(labels).most_common(1)[0][0]
        maj_scores = [s for (s, l) in zip(scores, labels) if l == maj]
        maj_confs = [c for (c, l) in zip(confs, labels) if l == maj]
        score = sum(maj_scores) / max(1, len(maj_scores))
        confidence = sum(maj_confs) / max(1, len(maj_confs))

        best_idx = max(
            [i for i, l in enumerate(labels) if l == maj],
            key=lambda i: results[i]["confidence"],
            default=0,
        )
        explanation = (
            f"Rephrase consensus over {len(eval_texts)} variants. "
            f"labels={labels}; scores={[round(s,3) for s in scores]}. "
            f"Best explanation: {results[best_idx]['explanation']}"
        )

        out = self._normalize_output(
            {
                "label": maj,
                "score": score,
                "confidence": confidence,
                "fallback_used": True,
                "explanation": explanation,
            },
            source="rephrase",
        )
        out["_variants"] = variants
        out["_latency_ms"] = int((time.time() - t0) * 1000)
        return out

    # -------------------------- Orchestrator --------------------------

    def FormFinalOutput(self, text: str) -> Json:
        """
        baseline → LLM; if disagree → RAG, Rules, Rephrase; majority vote; merge and print.
        """
        # 1) baseline
        baselineOP = self.ClassifyUsingAlg(text)
        self._print_stage("Baseline", baselineOP)

        # 2) LLM
        LLModelOP = self.ClassifyUsingLLMModel(text)
        self._print_stage("LLM", LLModelOP)

        results: List[Json] = [baselineOP, LLModelOP]
        used_fallback = False

        if LLModelOP["label"] != baselineOP["label"]:
            ragOP = self.ClassifyBasedOnRagScore(text)
            self._print_stage("RAG", ragOP)

            ruleOP = self.ClassifyUsingRuleScoreOnUnSafeWords(text)
            self._print_stage("RuleScore", ruleOP)

            rephraseOP = self.ClassifyBasedOnRephrasedText(text)
            self._print_stage("Rephrase", rephraseOP)

            results.extend([ragOP, ruleOP, rephraseOP])
            used_fallback = True

        # majority
        labels = [int(r["label"]) for r in results]
        majority = Counter(labels).most_common(1)[0][0]

        # avg score/conf among majority
        maj_results = [r for r in results if int(r["label"]) == majority]
        finalscore = sum(r["score"] for r in maj_results) / max(1, len(maj_results))
        finalconfidence = sum(r["confidence"] for r in maj_results) / max(1, len(maj_results))

        # best explanation from majority (highest confidence)
        best = max(maj_results, key=lambda r: r.get("confidence", 0.0))
        finalexplanation = best.get("explanation", "")

        op = {
            "label": int(majority),
            "score": float(finalscore),
            "confidence": float(finalconfidence),
            "fallback_used": bool(used_fallback),
            "explanation": finalexplanation,
        }
        op = self._normalize_output(op, source="final")  # enforce uniform schema, derive recommendation
        op["_methods"] = [r.get("_source", "?") for r in results]
        op["_details"] = results

        self._print_stage("FINAL", op)
        return op
