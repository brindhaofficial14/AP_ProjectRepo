# Prompt Safety Agent — Analysis Report

> This document summarizes the approach, design choices, evaluation highlights, and ablation insights for a fully local, explainable prompt‑injection detector. It integrates a linear baseline, an on‑device 7B chat model, interpretable rules, and an optional RAG layer, with rich reporting artifacts for auditability.

---

# Project Structure

```text
prompt-safety-agent/
├─ src/
│  ├─ agent_v3.py           # LLM-powered classifier (JSON output, fallbacks, majority vote)
│  ├─ prompts.py            # System/user prompt templates (v1–v4) + few-shot + keyword lists
│  ├─ utils.py              # JSON parsing, rule scoring, explanations, confidence helpers
│  ├─ evaluate_v3.py        # Full evaluation + reports (HTML/Excel/CSV/JSONL)
│  ├─ eval_ablation_v3.py   # Ablation: compare prompt templates (e.g., v3 → v4) on subsets
│  ├─ rag.py                # (Optional) RAG index + retrieval (ChromaDB + sentence-transformers)
│  ├─ rephraser.py          # Prompt paraphrasing (LLM with regex fallback) for robustness
│  ├─ baseline.py           # TF-IDF + Logistic Regression baseline (train/predict)
│  └─ model_resolver.py     # Maps aliases (phi2/stablelm3b/llama2/mistral) → GGUF path
├─ safety_agent_chat.py     # Desktop UI for the Prompt Safety Agent (run full pipeline)
├─ llama2_chat.py           # Lightweight desktop chat UI for a local GGUF model
├─ data/                    # Datasets (train/test CSVs)
│  └─ README.md             # How to download/use the dataset
├─ models/
│  ├─ baseline/             # Saved baseline artifacts (e.g., tfidf.joblib)
│  └─ gguf/                 # Local GGUF model files (not tracked)
│  └─ README.md             # Model choices & filenames
├─ rag_index/               # On-disk vector store for RAG (created on demand)
├─ reports/                 # Time-stamped evaluation folders + ablation outputs
├─ notebooks/               # (Optional) EDA notebooks
├─ Dockerfile               # (Optional) container image for CLI/batch runs
├─ requirements.txt         # Python dependencies
├─ report.md                # Analysis & findings (project report)
└─ README.md                # Project overview, setup, usage


## 1) Approach

We combine four complementary components:

1) **Baseline** — TF‑IDF + Logistic Regression: a fast, calibrated prior that is strong on classic keyworded attacks.  
2) **LLM Agent** — a local 7B chat model (GGUF via `llama.cpp`) prompted to emit **schema‑locked JSON** with the fields: `label`, `score`, `confidence`, `fallback_used`, `explanation`, and `recommendation`.  
3) **Rules Layer** — interpretable scoring based on keyword hits, proximity‑boosted combinations (e.g., *bypass* near *policy/guardrails*), and multi‑word unsafe phrases; produces concise, evidence‑backed explanations.  
4) **Optional RAG** — a ChromaDB index over labeled examples using sentence‑transformer embeddings; returns a similarity‑weighted unsafe vote that can be fused with rules.

**Decision policy (orchestrator):** run **Baseline** and **LLM**; if they disagree or confidence is low, run **RAG**, **Rules**, and **Rephrase+LLM**, then take a **majority vote** across methods. The final JSON includes the majority label, averaged score/confidence over the majority, the best explanation from the highest‑confidence method, and a recommendation derived from the risk score.

**Confidence vs Score:** `score` is the estimated risk in [0,1]; `confidence` reflects evidence strength (not a duplicate of risk). Confidence calibration blends LLM self‑estimate, rules signal, and an agreement bonus.

---

## 2) Key Decisions

- **Model** — A CPU‑friendly **Mistral‑7B‑Instruct (Q4_K_M)** / **Llama‑2‑7B‑Chat (GGUF)** balance for latency/quality.  
- **Prompting (V2 → V3 → V4)** —  
  - *V2* enforces “think stepwise but output **JSON only**.”  
  - *V3* adds a hard **evidence requirement** (unsafe=1 only when quoting an explicit cue) plus a **benign‑scaffolding whitelist** (e.g., “You will be given…”, “paraphrase: …”), which reduces false positives on dataset‑style prompts.  
  - *V4* schema‑locks a one‑line JSON, clarifies **score vs confidence**, and guarantees valid JSON even when uncertain (`fallback_used: true`).  
- **Fallbacks** — When low‑confidence or disagreement occurs: RAG vote + rules pass + **rephrase & re‑classify**, then majority vote.  
- **Auditability** — Every evaluation produces **HTML/Excel/JSONL/CSV** artifacts, including “interesting cases,” confusion matrices, confidence/latency plots, and full stage logs.

---

## 3) System Architecture (brief)

- **`src/baseline.py`** — train/predict TF‑IDF + LR.  
- **`src/agent_v3.py` + `src/prompts.py`** — orchestrator + prompt templates (V1–V4) + few‑shots.  
- **`src/utils.py`** — JSON parsing/normalization, rule scoring, explanations, contradiction heuristics, scenario weights.  
- **`src/rag.py`** — SBERT + ChromaDB index, similarity‑weighted unsafe vote.  
- **`src/rephraser.py`** — paraphrase variants (LLM first, regex fallback).  
- **`src/evaluate_v3.py`** — end‑to‑end evaluation with reports.  
- **`src/eval_ablation_v3.py`** — V3 vs V4 ablation, especially on scaffolding subset.

---

## 4) Dataset

**Safe‑Guard Prompt Injection** (CSV with `text,label`) with ~70/30 safe/unsafe split. Unsafe classes include explicit override/jailbreak requests (e.g., “ignore previous instructions,” “developer mode,” “DAN”), system‑prompt exfiltration, and requests for secrets/credentials. This mix supports both keyword‑driven signals and paraphrase‑heavy variants that benefit from RAG/rephrase.

---

## 5) Evaluation & Metrics (held‑out ≈20%)

**Artifacts:** Each run writes to `reports/eval_YYYY‑MM‑DD_HH‑MM‑SS/`:
- **`report.html`** — tables + plots (confusion matrices; confidence/latency histograms; fallback pie).  
- **`report.xlsx`** — multi‑sheet workbook (config, metrics summary, per‑example, interesting cases, text info, stage logs).  
- **`per_example.csv` / `eval_outputs.jsonl`** — row‑level outputs (with/without stage details).  
- **`per_example_stream.xlsx`** — optional streaming workbook (if `openpyxl` is available).

**Highlights (from the project report and example slice):**
- The **fusion pipeline** achieves **very high final accuracy** on the held‑out slice; raw LLM accuracy is lower but improves substantially after fusion.  
- **V3** increases **calibration quality** (higher mean confidence) and **reduces false positives** on scaffolding‑style prompts due to the evidence requirement + whitelist.  
- **Fallback rate** rises slightly under V3 (the model chooses to “double‑check” more often), which, on CPU‑only runs, **increases latency**—primarily from RAG/rephrase branches.  
- “**Interesting cases**” reveal typical errors: subtle social‑engineering with minimal cues, or long benign‑looking contexts with a manipulative tail.

---

## 6) Ablation: Scaffolding Subset (V3 vs V4)

A focused ablation on dataset‑style scaffolding shows **fewer false positives** with **V3** compared to **V2**, and **V4** maintains strict JSON validity while preserving V3’s gains. The per‑row comparison tables (Excel/HTML) make flips explicit (FP→OK, FN→OK), aiding prompt and rule iteration.

---

## 7) Optional RAG Extension

- **Indexing** — Sentence‑transformer embeddings persisted in **ChromaDB** (telemetry disabled).  
- **Inference** — Retrieve top‑K neighbors; compute a **similarity‑weighted unsafe probability (`rag_vote`)**; fuse with the rules score (e.g., `0.6·RAG + 0.4·Rules`).  
- **Effect** — Helps where keyword cues are weak but semantics match past unsafe examples (paraphrased jailbreaks).

---

## 8) Error Modes & Learnings

- **Rules excel at**: explicit overrides (“ignore previous”), exfiltration (“reveal system prompt”), and persona toggles (“developer mode”, “DAN”).  
- **Weak spots**: subtle coercion/social‑engineering with few explicit terms; single‑sentence manipulative suffixes appended to long neutral contexts.  
- **Helpful tactics**: the A→D rubric, **schema‑locked JSON**, and **evidence‑quoting requirement** made decisions stricter and easier to audit; “interesting cases” tables guided prompt/rules refinements.

---

## 9) Reproducibility & Tooling

- **Repeatable reports**: HTML + Excel + JSONL/CSV per run, with time‑stamped folders.  
- **Desktop UIs**: a safety‑agent UI to run the full pipeline and a plain chat UI for quick model checks.  
- **Local‑only**: requires GGUF weights on disk; no external API calls during inference.

---

## 10) Limitations & Future Work

- **Latency**: fallback branches (RAG + rephrase) dominate on CPU; mitigations include neighbor/result caching, token caps, and GPU layers when available.  
- **Coverage**: rules require ongoing curation for new jailbreak phrasing; consider semi‑automatic mining from false‑negative logs.  
- **Calibration**: add a small learned **calibrator** for confidence, or temperature scaling on top of the fused score.  
- **RAG++**: tag neighbors by attack type and **learn fusion weights** per type; add hard‑negative mining.

---

## 11) Conclusion

This project delivers a **hybrid, fully local** prompt‑safety agent with **explainable** decisions and robust fallbacks. Across the evaluated slice, the **final, fused decision** achieves strong accuracy, while **V3/V4 templates** improve calibration and reduce scaffolding false positives versus earlier prompts. The reporting pipeline (HTML/Excel/JSONL) and desktop UI make outcomes **auditable**, **debuggable**, and **iterable**, providing a practical path to on‑device deployment with transparent safeguards.
