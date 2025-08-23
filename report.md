
---

# 📄 report.md 

```md
# Prompt Safety Agent — Analysis Report

## Approach
We combine (1) a baseline linear classifier (TF-IDF + LR) and (2) a local 7B chat model wrapped as a Prompt Safety Agent. The agent performs multi-step analysis (pattern extraction, intent analysis, anomaly checks, decision) and outputs structured JSON. Confidence is calibrated by ensembling LLM confidence with rule-based signals. Low-confidence cases trigger a fallback (rephrased query + rules).

## Key Decisions
- **Model**: Mistral-7B-Instruct (GGUF, Q4_K_M) for balanced latency/quality on CPU.
- **Prompts**: Iterated from v1 to v2 with tighter rubric and short explanations to reduce drift.
- **Fallback**: Rule keywords + second LLM pass + simple score averaging; majority vote on labels.
- **Confidence**: Separate from score; blends LLM self-estimate, rule score, and agreement bonus.

## Results (Test 20%)
- **Baseline**: P/R/F1 and error modes (typically misses paraphrased jailbreaks; strong on obvious keywords).
- **Agent**: Accuracy / Precision / Recall / F1; Confidence mean/std; Fallback rate; Avg latency.
- **Comparisons**: 20 curated cases where disagreement/low-confidence/errors reveal behavior; insights per case.

## Learnings
- Prompt structure (A–D rubric) improved consistency; few-shot helped coverage of variants (DAN, developer mode).
- Rules add reliability for classic override phrases.
- Weak spots: subtle social engineering without keywords; extremely long, benign-looking contexts with a single manipulative tail.
- Future: Add RAG of hard negatives; train a small calibrator for confidence; GPU acceleration.

## Optional RAG (if done)
- Embedding with sentence-transformers, retrieval from ChromaDB of nearest examples; small but measurable lift on paraphrased jailbreaks.
