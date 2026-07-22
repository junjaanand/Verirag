"""
Response Generator — Grounded LLM generation with citations and confidence display.
"""

from dataclasses import dataclass, field
from loguru import logger
from groq import Groq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL_MAIN, LLM_TEMPERATURE, LLM_MAX_TOKENS

groq_client = Groq(api_key=GROQ_API_KEY)


@dataclass
class Citation:
    source: str
    page: int
    chunk_text: str
    relevance: float


@dataclass
class VeriRAGResponse:
    """Final structured response from VeriRAG."""
    answer: str
    confidence: float
    confidence_level: str
    citations: list[Citation] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    corrections_applied: list[dict] = field(default_factory=list)
    clarification_needed: str | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "confidence_level": self.confidence_level,
            "citations": [
                {"source": c.source, "page": c.page, "relevance": round(c.relevance, 3)}
                for c in self.citations
            ],
            "contradictions": self.contradictions,
            "corrections_applied": self.corrections_applied,
            "clarification_needed": self.clarification_needed,
        }


def build_citations(chunks: list[dict]) -> list[Citation]:
    """Build citations from retrieved chunks."""
    citations = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        citations.append(Citation(
            source=meta.get("source_file", "unknown"),
            page=meta.get("page_num", 0),
            chunk_text=chunk["text"][:200],
            relevance=chunk.get("relevance_score", 0.0),
        ))
    return citations


def generate_response(query: str, correction_result) -> VeriRAGResponse:
    """
    Generate a grounded response using LLM with retrieved context.
    """
    chunks = correction_result.chunks
    confidence = correction_result.confidence

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        source = meta.get("source_file", "unknown")
        page = meta.get("page_num", "?")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)

    # Adjust prompt based on confidence level
    if confidence.level == "HIGH":
        confidence_instruction = "You have high-quality context. Provide a confident, detailed answer with citations."
    elif confidence.level == "MEDIUM":
        confidence_instruction = (
            "Context quality is moderate. Provide an answer but note any uncertainties. "
            "Use phrases like 'Based on the available information...' where appropriate."
        )
    else:
        confidence_instruction = (
            "Context quality is LOW. Be very cautious. Clearly state that the available information "
            "is limited and your answer may be incomplete. Recommend the user verify this information."
        )

    # Add contradiction warning
    contradiction_note = ""
    if correction_result.contradictions.has_contradiction:
        pairs = correction_result.contradictions.contradicting_pairs
        contradiction_note = (
            "\n\nIMPORTANT: The sources contain CONTRADICTORY information. "
            "Present BOTH viewpoints and clearly indicate which source says what. "
            f"Contradictions found: {len(pairs)}"
        )

    system_prompt = f"""You are VeriRAG, a self-correcting RAG assistant. Your answers MUST be:
1. GROUNDED — Only use information from the provided context. Never make up facts.
2. CITED — Reference the source document and page for every claim using [Source: filename, Page: X] format.
3. HONEST — If the context doesn't contain enough information, say so explicitly.
4. CALIBRATED — {confidence_instruction}
{contradiction_note}

If the context does not contain the answer, respond with: "Based on the available documents, I could not find sufficient information to answer this question."
"""

    user_prompt = f"""Context:
{context}

---

Question: {query}

Provide a grounded answer with citations."""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_MAIN,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        answer = f"Error generating response: {e}"

    # Build final response
    citations = build_citations(chunks)

    result = VeriRAGResponse(
        answer=answer,
        confidence=confidence.score,
        confidence_level=confidence.level,
        citations=citations,
        contradictions=correction_result.contradictions.contradicting_pairs,
        corrections_applied=correction_result.corrections_applied,
        clarification_needed=correction_result.clarification_question,
    )

    # Log confidence badge
    badge = "🟢" if confidence.level == "HIGH" else "🟡" if confidence.level == "MEDIUM" else "🔴"
    logger.info(f"{badge} Response generated | confidence={confidence.score:.2f} ({confidence.level})")

    return result
