"""
Evaluation Harness — Test questions, hallucination scoring, and A/B comparison.
"""

import json
from dataclasses import dataclass
from loguru import logger
from groq import Groq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL_MAIN, EVAL_TEST_QUESTIONS_PATH

groq_client = Groq(api_key=GROQ_API_KEY)


@dataclass
class EvalResult:
    question: str
    category: str
    expected: str
    actual_answer: str
    confidence: float
    confidence_level: str
    hallucination_label: str  # "faithful", "partial", "hallucinated"
    correct: bool
    corrections_applied: int


def load_test_questions() -> list[dict]:
    """Load test questions from JSON file."""
    with open(EVAL_TEST_QUESTIONS_PATH, "r") as f:
        return json.load(f)


def score_hallucination(question: str, expected: str, actual: str, context_used: str) -> str:
    """
    Use LLM-as-judge to score hallucination.
    Returns: "faithful", "partial", or "hallucinated"
    """
    prompt = f"""You are an evaluation judge. Determine if the AI's answer is faithful to the source context.

Question: {question}

Expected Answer: {expected}

AI's Answer: {actual}

Source Context Used: {context_used[:1500]}

Classify the AI's answer as ONE of:
- "faithful" — The answer is fully supported by the source context and matches the expected answer
- "partial" — The answer is partially correct but contains some unsupported claims
- "hallucinated" — The answer contains fabricated information or contradicts the sources

If the expected answer is "UNANSWERABLE" and the AI correctly said it couldn't find the answer, classify as "faithful".
If the expected answer is "UNANSWERABLE" but the AI gave a specific answer anyway, classify as "hallucinated".

Respond with ONLY one word: faithful, partial, or hallucinated"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_MAIN,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip().lower()
        if label not in ["faithful", "partial", "hallucinated"]:
            label = "partial"
        return label
    except Exception as e:
        logger.error(f"Hallucination scoring failed: {e}")
        return "partial"


def run_evaluation(query_fn, test_questions: list[dict] = None) -> dict:
    """
    Run full evaluation harness.
    
    Args:
        query_fn: Function that takes a query string and returns VeriRAGResponse
        test_questions: List of test questions (loads from file if None)
    
    Returns:
        Evaluation results with metrics
    """
    if test_questions is None:
        test_questions = load_test_questions()

    results = []
    
    for i, tq in enumerate(test_questions):
        question = tq["question"]
        expected = tq["expected_answer"]
        category = tq["category"]

        logger.info(f"\n📝 Eval [{i+1}/{len(test_questions)}]: {question[:60]}...")

        # Run query through VeriRAG
        response = query_fn(question)

        # Get context text for scoring
        context_text = " | ".join([c.chunk_text for c in response.citations]) if response.citations else ""

        # Score hallucination
        label = score_hallucination(question, expected, response.answer, context_text)

        # Check correctness
        correct = label == "faithful"

        results.append(EvalResult(
            question=question,
            category=category,
            expected=expected,
            actual_answer=response.answer[:500],
            confidence=response.confidence,
            confidence_level=response.confidence_level,
            hallucination_label=label,
            correct=correct,
            corrections_applied=len(response.corrections_applied),
        ))

        logger.info(f"  Result: {label} | confidence={response.confidence:.2f}")

    # Calculate metrics
    total = len(results)
    faithful = sum(1 for r in results if r.hallucination_label == "faithful")
    partial = sum(1 for r in results if r.hallucination_label == "partial")
    hallucinated = sum(1 for r in results if r.hallucination_label == "hallucinated")

    metrics = {
        "total_questions": total,
        "faithful": faithful,
        "partial": partial,
        "hallucinated": hallucinated,
        "hallucination_rate": round((hallucinated + partial * 0.5) / max(total, 1) * 100, 1),
        "accuracy": round(faithful / max(total, 1) * 100, 1),
        "avg_confidence": round(sum(r.confidence for r in results) / max(total, 1), 3),
        "results": [
            {
                "question": r.question[:80],
                "category": r.category,
                "label": r.hallucination_label,
                "confidence": r.confidence,
                "correct": r.correct,
            }
            for r in results
        ],
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Hallucination Rate: {metrics['hallucination_rate']}%")
    logger.info(f"Accuracy: {metrics['accuracy']}%")
    logger.info(f"Avg Confidence: {metrics['avg_confidence']}")

    return metrics
