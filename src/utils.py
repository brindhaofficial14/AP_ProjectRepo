# src/utils.py
import json, re
from typing import Dict, Any

# ---------- JSON parsing helpers ----------

def _strip_code_fences(text: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    return re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text, flags=re.IGNORECASE)

def _relaxed_json_candidate(text: str) -> str | None:
    # try largest {...} block first
    text = _strip_code_fences(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # fallback: first {...} match
        m = re.search(r"\{[\s\S]*?\}", text)
        if not m:
            return None
        return m.group(0)
    return text[start:end+1]

def _json_loads_lenient(s: str) -> Dict[str, Any] | None:
    # common fixes: single quotes → double, trailing commas, Python literals
    fixed = s
    fixed = re.sub(r"'", '"', fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)          # trailing commas
    fixed = re.sub(r"\bTrue\b", "true", fixed)            # Python → JSON
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    try:
        return json.loads(fixed)
    except Exception:
        return None

def enforce_consistency(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure label agrees with score when disagreement is significant."""
    try:
        score = float(obj.get("score", 0.5))
    except Exception:
        score = 0.5
    score = max(0.0, min(1.0, score))
    obj["score"] = score

    lbl = obj.get("label", None)
    # accept strings like "safe"/"unsafe"
    if isinstance(lbl, str):
        ls = lbl.strip().lower()
        if ls in ("safe", "0"): lbl = 0
        elif ls in ("unsafe", "1"): lbl = 1
        else:
            try:
                lbl = int(float(ls))
            except Exception:
                lbl = None

    lbl_by_score = 1 if score >= 0.5 else 0
    if lbl not in (0, 1):
        obj["label"] = lbl_by_score
    else:
        if lbl != lbl_by_score and abs(score - 0.5) >= 0.10:
            obj["label"] = lbl_by_score
    return obj

def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Parse LLM output that *should* be JSON, but may include extra prose.
    Tries strict JSON first, then a lenient fallback.
    """
    # strict attempt (first JSON object)
    m = re.search(r"\{[\s\S]*\}", text)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None

    # lenient attempt
    if obj is None:
        candidate = _relaxed_json_candidate(text)
        if candidate is None:
            raise ValueError("No JSON found in model output.")
        obj = _json_loads_lenient(candidate)
        if obj is None:
            raise ValueError("No JSON found in model output.")

    # -------- normalize fields --------
    # score
    try:
        score = float(obj.get("score", obj.get("risk", 0.5)))
    except Exception:
        score = 0.5
    obj["score"] = max(0.0, min(1.0, score))

    # label (may be int/str); will be finalized by enforce_consistency
    obj.setdefault("label", 1 if obj["score"] >= 0.5 else 0)

    # confidence defaults to score
    try:
        conf = float(obj.get("confidence", obj["score"]))
    except Exception:
        conf = obj["score"]
    obj["confidence"] = max(0.0, min(1.0, conf))

    # booleans / strings
    obj["fallback_used"] = bool(obj.get("fallback_used", False))
    obj["explanation"] = str(obj.get("explanation", "")).strip()

    # default recommendation (final agents may override)
    if obj.get("label", 0) == 1:
        obj.setdefault("recommendation", "Block")
    else:
        obj.setdefault("recommendation", "Proceed with caution" if obj["score"] > 0.3 else "Proceed")

    # enforce label ↔ score consistency
    obj = enforce_consistency(obj)
    return obj

# ---------- Rules & confidence ----------

def rule_risk_score(text: str, keywords) -> float:
    t = text.lower()
    hits = sum(1 for k in keywords if k in t)
    if "ignore previous" in t or "system prompt" in t:
        hits += 1
    return min(1.0, 0.15 * hits)  # 0.15 per hit

def rule_label(text: str, keywords) -> int:
    return 1 if rule_risk_score(text, keywords) >= 0.5 else 0

def calibrate_confidence(llm_conf: float, rule_score: float, agree: bool) -> float:
    base = 0.6 * llm_conf + 0.4 * (1.0 if rule_score >= 0.5 else 0.4)
    if agree:
        base += 0.1
    return float(max(0.0, min(1.0, base)))

def rules_explanation(text: str, keywords) -> str:
    t = text.lower()
    matched = [k for k in keywords if k in t]
    if not matched:
        return "No injection/jailbreak indicators found; benign greeting or neutral content."
    return "Matched risky indicators: " + ", ".join(sorted(set(matched)))
