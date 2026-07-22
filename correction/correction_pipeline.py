"""
Self-Correction Layer — Sufficiency checker, contradiction detector,
confidence scorer, adaptive re-query engine, and clarification generator.

This is the CORE INNOVATION of VeriRAG.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger
from groq import Groq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY, LLM_MODEL_MAIN, LLM_MODEL_FAST, LLM_TEMPERATURE,
    SUFFICIENCY_RELEVANCE_THRESHOLD, SUFFICIENCY_MIN_CHUNKS,
    MAX_REQUERY_ATTEMPTS,
    CONFIDENCE_WEIGHT_RETRIEVAL, CONFIDENCE_WEIGHT_SUFFICIENCY,
    CONFIDENCE_WEIGHT_AGREEMENT, CONFIDENCE_WEIGHT_OCR,
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD,
    NLI_CONTRADICTION_THRESHOLD,
    PIPELINE_LOG_FILE,
)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SufficiencyResult:
    sufficient: bool
    avg_relevance: float
    relevant_count: int
    reason: str


@dataclass
class ContradictionResult:
    has_contradiction: bool
    contradicting_pairs: list[dict] = field(default_factory=list)
    agreement_score: float = 1.0


@dataclass
class ConfidenceResult:
    score: float
    level: str  # "HIGH", "MEDIUM", "LOW"
    breakdown: dict = field(default_factory=dict)


@dataclass
class CorrectionResult:
    """Final result of the self-correction pipeline."""
    chunks: list[dict]
    sufficiency: SufficiencyResult
    contradictions: ContradictionResult
    confidence: ConfidenceResult
    corrections_applied: list[dict] = field(default_factory=list)
    clarification_question: str | None = None


# ============================================================
# SUFFICIENCY CHECKER
# ============================================================

def check_sufficiency(chunks: list[dict], query: str) -> SufficiencyResult:
    """
    Check if retrieved context is adequate to answer the query.
    
    Criteria:
    - Average relevance score >= threshold (0.6)
    - At least 2 relevant chunks found
    """
    if not chunks:
        return SufficiencyResult(
            sufficient=False,
            avg_relevance=0.0,
            relevant_count=0,
            reason="No chunks retrieved",
        )

    relevance_scores = [c.get("relevance_score", 0.0) for c in chunks]
    avg_relevance = sum(relevance_scores) / len(relevance_scores)
    relevant_count = sum(1 for s in relevance_scores if s >= SUFFICIENCY_RELEVANCE_THRESHOLD)

    sufficient = (
        avg_relevance >= SUFFICIENCY_RELEVANCE_THRESHOLD and
        relevant_count >= SUFFICIENCY_MIN_CHUNKS
    )

    reason = "Sufficient" if sufficient else (
        f"avg_relevance={avg_relevance:.2f} < {SUFFICIENCY_RELEVANCE_THRESHOLD}" if avg_relevance < SUFFICIENCY_RELEVANCE_THRESHOLD
        else f"relevant_count={relevant_count} < {SUFFICIENCY_MIN_CHUNKS}"
    )

    logger.info(f"📋 Sufficiency: {'✅' if sufficient else '❌'} | "
                f"avg={avg_relevance:.2f}, relevant={relevant_count}/{len(chunks)}")

    return SufficiencyResult(
        sufficient=sufficient,
        avg_relevance=avg_relevance,
        relevant_count=relevant_count,
        reason=reason,
    )


# ============================================================
# CONTRADICTION DETECTOR
# ============================================================

def detect_contradictions(chunks: list[dict], query: str) -> ContradictionResult:
    """
    Detect contradictions between retrieved chunks using LLM.
    Uses Groq (fast) for pairwise comparison.
    """
    if len(chunks) < 2:
        return ContradictionResult(has_contradiction=False, agreement_score=1.0)

    # Compare top chunks pairwise (limit to top 4 to control API calls)
    top_chunks = chunks[:4]
    contradictions = []

    for i in range(len(top_chunks)):
        for j in range(i + 1, len(top_chunks)):
            chunk_a = top_chunks[i]
            chunk_b = top_chunks[j]

            prompt = f"""Analyze if these two text passages contradict each other regarding the query.

Query: {query}

Passage A (from {chunk_a.get('metadata', {}).get('source_file', 'unknown')}):
"{chunk_a['text'][:500]}"

Passage B (from {chunk_b.get('metadata', {}).get('source_file', 'unknown')}):
"{chunk_b['text'][:500]}"

Respond with ONLY one of:
- "NO_CONTRADICTION" if they agree or discuss different topics
- "CONTRADICTION: <brief explanation>" if they directly contradict each other"""

            try:
                response = groq_client.chat.completions.create(
                    model=LLM_MODEL_FAST,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                result = response.choices[0].message.content.strip()

                if result.startswith("CONTRADICTION"):
                    contradictions.append({
                        "chunk_a_id": chunk_a.get("id", f"chunk_{i}"),
                        "chunk_b_id": chunk_b.get("id", f"chunk_{j}"),
                        "source_a": chunk_a.get("metadata", {}).get("source_file", "unknown"),
                        "source_b": chunk_b.get("metadata", {}).get("source_file", "unknown"),
                        "explanation": result.replace("CONTRADICTION:", "").strip(),
                    })
            except Exception as e:
                logger.warning(f"Contradiction check failed: {e}")

    has_contradiction = len(contradictions) > 0
    agreement_score = max(0.0, 1.0 - (len(contradictions) * 0.3))

    logger.info(f"🔀 Contradictions: {'⚠️ ' + str(len(contradictions)) + ' found' if has_contradiction else '✅ None'}")

    return ContradictionResult(
        has_contradiction=has_contradiction,
        contradicting_pairs=contradictions,
        agreement_score=agreement_score,
    )


# ============================================================
# CONFIDENCE SCORER
# ============================================================

def calculate_confidence(
    sufficiency: SufficiencyResult,
    contradictions: ContradictionResult,
    chunks: list[dict],
) -> ConfidenceResult:
    """
    Calculate composite confidence score.
    
    Confidence = 0.4*retrieval + 0.3*sufficiency + 0.2*agreement + 0.1*ocr
    """
    # Retrieval quality (avg relevance)
    retrieval_quality = sufficiency.avg_relevance

    # Context sufficiency (binary + scaled)
    sufficiency_score = min(sufficiency.relevant_count / max(SUFFICIENCY_MIN_CHUNKS, 1), 1.0)

    # Source agreement
    agreement_score = contradictions.agreement_score

    # OCR quality (avg from chunk metadata)
    ocr_scores = [
        c.get("metadata", {}).get("ocr_confidence", -1.0)
        for c in chunks
    ]
    valid_ocr = [s for s in ocr_scores if s >= 0]
    ocr_quality = sum(valid_ocr) / len(valid_ocr) if valid_ocr else 1.0

    # Weighted composite
    score = (
        CONFIDENCE_WEIGHT_RETRIEVAL * retrieval_quality +
        CONFIDENCE_WEIGHT_SUFFICIENCY * sufficiency_score +
        CONFIDENCE_WEIGHT_AGREEMENT * agreement_score +
        CONFIDENCE_WEIGHT_OCR * ocr_quality
    )
    score = max(0.0, min(1.0, score))

    # Determine level
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        level = "HIGH"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        level = "MEDIUM"
    else:
        level = "LOW"

    breakdown = {
        "retrieval_quality": round(retrieval_quality, 3),
        "sufficiency_score": round(sufficiency_score, 3),
        "agreement_score": round(agreement_score, 3),
        "ocr_quality": round(ocr_quality, 3),
    }

    logger.info(f"📊 Confidence: {score:.2f} ({level}) | {breakdown}")

    return ConfidenceResult(score=score, level=level, breakdown=breakdown)


# ============================================================
# ADAPTIVE RE-QUERY ENGINE
# ============================================================

def query_decomposition(query: str) -> list[str]:
    """Strategy 1: Break complex query into simpler sub-queries."""
    prompt = f"""Break this complex question into 2-3 simpler sub-questions that can be searched independently.

Question: {query}

Return ONLY the sub-questions, one per line. No numbering or bullets."""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        sub_queries = [q.strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()]
        logger.info(f"  🔧 Decomposed into {len(sub_queries)} sub-queries")
        return sub_queries[:3]
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")
        return [query]


def query_expansion(query: str) -> str:
    """Strategy 2: Expand query with synonyms and related terms."""
    prompt = f"""Expand this search query by adding synonyms and related terms to improve search recall.

Original query: {query}

Return ONLY the expanded query as a single line. Include the original terms plus alternatives."""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        expanded = response.choices[0].message.content.strip()
        logger.info(f"  🔧 Expanded: '{expanded[:80]}...'")
        return expanded
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query


def hyde_query(query: str) -> str:
    """Strategy 3: Generate hypothetical document (HyDE) for better retrieval."""
    prompt = f"""Write a short paragraph (3-4 sentences) that would be a perfect answer to this question.
Write it as if it exists in a real document. Do not say "I don't know" — generate a plausible answer.

Question: {query}"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        hypothetical = response.choices[0].message.content.strip()
        logger.info(f"  🔧 HyDE generated: '{hypothetical[:80]}...'")
        return hypothetical
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return query


def generate_clarification(query: str, chunks: list[dict]) -> str:
    """Generate a targeted clarifying question when all retries fail."""
    context_summary = "\n".join([c["text"][:200] for c in chunks[:3]]) if chunks else "No relevant context found."

    prompt = f"""The user asked a question but the system couldn't find sufficient information to answer confidently.

User question: {query}

Available context (limited):
{context_summary}

Generate ONE specific clarifying question to ask the user that would help narrow down the search. 
The question should help disambiguate the query or identify which specific aspect they're interested in.
Return ONLY the clarifying question."""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Clarification generation failed: {e}")
        return "Could you provide more specific details about what you're looking for?"


# ============================================================
# PIPELINE LOGGING
# ============================================================

def log_pipeline_step(query_id: str, stage: str, input_data: dict, output_data: dict, action: str = None):
    """Log a pipeline step to JSONL file."""
    PIPELINE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": query_id,
        "stage": stage,
        "input": input_data,
        "output": output_data,
        "action_taken": action,
    }

    with open(PIPELINE_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# ============================================================
# MAIN CORRECTION PIPELINE
# ============================================================

def run_correction(
    query: str,
    initial_chunks: list[dict],
    search_fn,
    rerank_fn,
    interactive: bool = True,
) -> CorrectionResult:
    """
    Run the full self-correction pipeline.
    
    Flow: Sufficiency Check → Contradiction Detection → Confidence Score
          ↓ (if insufficient)
          Adaptive Re-Query (up to 3 retries) → Re-evaluate
          ↓ (if all retries fail)
          Clarification / Low-confidence response
    """
    import uuid
    query_id = str(uuid.uuid4())[:8]

    corrections_applied = []
    current_chunks = initial_chunks
    attempt = 0

    # ---- CORRECTION LOOP ----
    while attempt <= MAX_REQUERY_ATTEMPTS:
        # Step 1: Sufficiency Check
        sufficiency = check_sufficiency(current_chunks, query)
        log_pipeline_step(query_id, "sufficiency_check",
                          {"query": query, "chunks": len(current_chunks)},
                          {"sufficient": sufficiency.sufficient, "avg_relevance": sufficiency.avg_relevance})

        if sufficiency.sufficient:
            break

        if attempt >= MAX_REQUERY_ATTEMPTS:
            logger.warning(f"⚠️ Max retries ({MAX_REQUERY_ATTEMPTS}) reached")
            break

        # Step 2: Adaptive Re-query
        attempt += 1
        strategies = [
            ("query_decomposition", query_decomposition),
            ("query_expansion", query_expansion),
            ("hyde", hyde_query),
        ]

        strategy_name, strategy_fn = strategies[min(attempt - 1, len(strategies) - 1)]
        logger.info(f"🔄 Re-query attempt {attempt}/{MAX_REQUERY_ATTEMPTS}: {strategy_name}")

        if strategy_name == "query_decomposition":
            sub_queries = strategy_fn(query)
            new_chunks = []
            for sq in sub_queries:
                candidates = search_fn(sq)
                reranked = rerank_fn(sq, candidates)
                new_chunks.extend(reranked)
            # Deduplicate by ID
            seen = set()
            unique_chunks = []
            for c in new_chunks:
                if c["id"] not in seen:
                    seen.add(c["id"])
                    unique_chunks.append(c)
            current_chunks = unique_chunks[:len(initial_chunks)]
        else:
            modified_query = strategy_fn(query)
            candidates = search_fn(modified_query)
            current_chunks = rerank_fn(modified_query, candidates)

        corrections_applied.append({
            "attempt": attempt,
            "strategy": strategy_name,
            "result_count": len(current_chunks),
        })

        log_pipeline_step(query_id, "requery",
                          {"strategy": strategy_name, "attempt": attempt},
                          {"new_chunks": len(current_chunks)},
                          action=strategy_name)

    # Step 3: Contradiction Detection
    contradictions = detect_contradictions(current_chunks, query)
    log_pipeline_step(query_id, "contradiction_check",
                      {"chunks": len(current_chunks)},
                      {"has_contradiction": contradictions.has_contradiction})

    # Step 4: Confidence Scoring
    confidence = calculate_confidence(sufficiency, contradictions, current_chunks)
    log_pipeline_step(query_id, "confidence",
                      {"sufficiency": sufficiency.sufficient},
                      {"score": confidence.score, "level": confidence.level})

    # Step 5: Clarification (if needed)
    clarification = None
    if confidence.level == "LOW" and interactive:
        clarification = generate_clarification(query, current_chunks)
        logger.info(f"❓ Clarification: {clarification}")

    return CorrectionResult(
        chunks=current_chunks,
        sufficiency=sufficiency,
        contradictions=contradictions,
        confidence=confidence,
        corrections_applied=corrections_applied,
        clarification_question=clarification,
    )
