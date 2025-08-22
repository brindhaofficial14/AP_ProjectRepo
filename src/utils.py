# src/utils.py
import json, re
from typing import Dict, Any

def safe_json_parse(text: str) -> Dict[str, Any]:
    # Extract first JSON object in text
    match = re.search(r'\{.*\}', text, flags=re.S)
    if not match:
        raise ValueError("No JSON found in model output.")
    obj = json.loads(match.group(0))
    # normalize fields
    obj.setdefault("label", 1 if obj.get("score", 1.0) >= 0.5 else 0)
    obj.setdefault("confidence", min(1.0, max(0.0, obj.get("confidence", obj.get("score", 0.5)))))
    obj.setdefault("fallback_used", False)
    obj.setdefault("explanation", "")
    obj.setdefault("recommendation", "Proceed with caution" if obj["label"]==0 else "Block")
    return obj

def rule_risk_score(text: str, keywords) -> float:
    t = text.lower()
    hits = sum(1 for k in keywords if k in t)
    if "ignore previous" in t or "system prompt" in t: hits += 1
    return min(1.0, 0.15 * hits)  # 0.15 per hit

def rule_label(text: str, keywords) -> int:
    return 1 if rule_risk_score(text, keywords) >= 0.5 else 0

def calibrate_confidence(llm_conf: float, rule_score: float, agree: bool) -> float:
    base = 0.6 * llm_conf + 0.4 * (1.0 if rule_score >= 0.5 else 0.4)
    if agree: base += 0.1
    return float(max(0.0, min(1.0, base)))
