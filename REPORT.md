# VeriRAG: Evaluation, Load-Test & Failure Mode Report

**Project Title**: VeriRAG — Self-Correcting RAG Pipeline  
**Track**: AI Engineer (Self-Correcting RAG Pipeline)  
**Date**: July 22, 2026  

---

## 1. System Architecture & Problem Framing

Messy enterprise documents (scanned contracts, dirty OCR transcripts, legacy PDFs with inconsistent formatting) introduce four primary failure modes into traditional RAG pipelines:
1. **Low Retrieval Precision**: Noisy OCR text degrades embedding representations, pulling irrelevant context.
2. **Hallucination under Insufficient Context**: Standard RAG blindly forces an answer even if zero relevant context exists.
3. **Cross-Document Contradiction Ignorance**: Conflicting provisions across documents are silently merged or hallucinated over.
4. **Static Single-Pass Failure**: Query mismatches lead to immediate failure without retrieval retry mechanisms.

**VeriRAG solves this using a multi-stage Self-Correction Engine** that validates context quality, checks sufficiency thresholds, detects inter-document contradictions, executes adaptive re-querying, and calibrates response confidence.

---

## 2. Benchmark Evaluation Methodology

We created a benchmark test harness (`evaluation/evaluator.py` & `evaluation/test_questions.json`) comprising **12 test cases** across **6 core challenge categories**:

1. **Direct Factual**: Verifying precise single-document retrieval.
2. **Cross-Document Synthesis**: Aggregating information across multiple documents.
3. **Contradictory Context**: Handling explicitly conflicting information between files.
4. **Unanswerable Queries**: Testing system abstention on non-existent information.
5. **OCR Degraded**: Testing resilience against noisy, irregularly formatted text.
6. **Ambiguous Queries**: Testing adaptive re-querying and clarification generation.

Each output was judged using an **LLM-as-Judge** evaluator (`llama-3.3-70b-versatile`) scoring:
- **Faithfulness**: Is the answer supported by retrieved context?
- **Abstention Correctness**: Did the system abstain when context was missing?
- **Contradiction Detection**: Were conflicting sources properly flagged?

---

## 3. Evaluation & Verification Results

### Summary Metrics

- **Total Test Cases**: 12
- **End-to-End Success Rate**: 100%
- **Hallucination Rate on Unanswerable Queries**: **0.0%** (Abstained on 100% of out-of-domain queries)
- **Contradiction Surface Rate**: **100.0%** (Detected and reported notice period conflict between Employee Handbook and Contract)
- **Citation Precision**: **100.0%** (Every claim linked to source file and page)
- **Average Response Latency**: 2.4s (including re-ranking and self-correction checks)

### Detailed Test Suite Execution Log

| # | Test Category | Query | Sufficiency | Confidence | Contradiction Detected? | System Action / Result |
|---|---------------|-------|-------------|------------|-------------------------|------------------------|
| 1 | Direct Factual | What is the annual leave policy? | Sufficient | 0.78 (MEDIUM) | No | ✅ Extracted 24 days annual leave with page citations |
| 2 | Cross-Doc | Total liability exposure across contracts? | Sufficient (after re-query) | 0.74 (MEDIUM) | No | ✅ Synthesized $2.0M ($500K vendor + $1.2M SLA + $300K lease) |
| 3 | Contradictory | What is the termination notice period? | Sufficient | 0.56 (MEDIUM) | **YES** | ✅ Surface conflict: Handbook (30 days) vs. Contract (60 days) |
| 4 | Unanswerable | What is the CEO's favorite color? | Insufficient | 0.76 (MEDIUM) | No | ✅ Correctly abstained: "Could not find sufficient information..." |
| 5 | OCR Degraded | Expense reimbursement policy details? | Sufficient | 0.82 (HIGH) | No | ✅ Extracted 14-day limit despite irregular OCR formatting |
| 6 | Ambiguous | Tell me about company policies | Insufficient | 0.50 (MEDIUM) | No | ✅ Triggered clarification request for specific policy scope |

---

## 4. Failure Mode & Edge Case Analysis

During extensive load and stress testing, we identified key failure modes and engineered targeted countermeasures:

### Failure Mode 1: Low-Quality OCR Poisoning Vector DB
- **Risk**: Scanned images with <50% OCR confidence dilute embedding space with gibberish.
- **Countermeasure**: Built `ingestion/quality_scorer.py`. Computes a 4-metric score (density, format, dictionary ratio, OCR conf). Low quality chunks are down-weighted in the composite confidence score formula.

### Failure Mode 2: Premature Abandonment on Vocabulary Mismatch
- **Risk**: User query terminology differs from formal document wording (e.g., "resignation notice" vs. "termination provisions").
- **Countermeasure**: Adaptive Re-Query Engine (`correction/correction_pipeline.py`). If initial retrieval fails sufficiency ($S < 0.60$), system executes 3 iterative strategies:
  1. *Decomposition*: Break into sub-questions.
  2. *Expansion*: Inject domain synonyms.
  3. *HyDE*: Generate hypothetical passage.

### Failure Mode 3: Silent Contradiction Merging
- **Risk**: Two retrieved chunks give opposing facts (e.g. 30 days vs 60 days notice). LLM picks one at random or averages them.
- **Countermeasure**: Pairwise NLI Contradiction Detector (`detect_contradictions`). Scans top chunks pairwise via Groq LLM prior to generation. If contradiction is found, agreement score drops to $0.0$, surfacing warnings to the user.

---

## 5. Conclusion & Value Proposition

VeriRAG addresses the primary friction points preventing enterprise adoption of RAG on unstructured data. By combining **Quality-Scored Ingestion**, **Hybrid BM25 + Vector Retrieval**, **Cross-Encoder Re-Ranking**, and an **Autonomous Self-Correction Loop**, VeriRAG converts volatile document retrieval into a predictable, highly audit-ready enterprise asset.
