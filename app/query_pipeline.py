from app import config
from app.models import RagAnswer
from app.security.guard import validate_query, QueryValidationError
from app.retrieval.retriever import retrieve
from app.generation.generator import generate_answer


def answer_question(question: str, top_k: int = None, document_filter: str = None) -> RagAnswer:
    top_k = top_k or config.TOP_K
    try:
        question = validate_query(question, config.MAX_QUERY_LENGTH)
    except QueryValidationError as e:
        return RagAnswer(
            answer=config.INSUFFICIENT_EVIDENCE,
            citations=[], supporting_passages=[], evidence="", inference="",
            confidence=0.0, refusal_reason=str(e), latency_seconds=0.0,
            security_flags=["Query rejected at validation."],
        )

    passages = retrieve(question, top_k=top_k, document_filter=document_filter)
    # Drop passages that are too weak semantically to be trustworthy evidence
    passages = [p for p in passages if p.combined_score >= config.MIN_RETRIEVAL_SCORE] or passages[:0]
    return generate_answer(question, passages)
