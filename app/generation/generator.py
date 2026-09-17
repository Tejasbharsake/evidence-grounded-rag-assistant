import json
import time
import re
from typing import List

from app import config
from app.models import RagAnswer, Citation, RetrievedChunk
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.generation.llm_client import generate_raw, local_fallback_response
from app.security.guard import scan_chunks_for_injection


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _parse_llm_json(raw: str) -> dict:
    raw = _strip_code_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage the first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _validate_citations(citations: List[dict], valid_chunks: List[RetrievedChunk]) -> List[Citation]:
    """Never let a fabricated citation reach the user: only keep citations
    that reference chunk_ids we actually retrieved and sent to the model."""
    valid_ids = {c.chunk_id: c for c in valid_chunks}
    result = []
    for c in citations or []:
        cid = c.get("chunk_id")
        if cid in valid_ids:
            src = valid_ids[cid]
            result.append(Citation(document_name=src.document_name, page_number=src.page_number, chunk_id=cid))
    return result


def generate_answer(question: str, passages: List[RetrievedChunk]) -> RagAnswer:
    start = time.time()
    security_flags = []

    injection_hits = scan_chunks_for_injection(passages)
    if injection_hits:
        security_flags.append(
            f"Prompt-injection pattern detected in {len(injection_hits)} retrieved chunk(s); "
            "treated as inert document content, not instructions."
        )

    if not passages:
        latency = time.time() - start
        return RagAnswer(
            answer=config.INSUFFICIENT_EVIDENCE,
            citations=[],
            supporting_passages=[],
            evidence="",
            inference="",
            confidence=0.0,
            refusal_reason="No passages met the retrieval relevance threshold for this query.",
            latency_seconds=round(latency, 3),
            security_flags=security_flags,
        )

    user_prompt = build_user_prompt(question, passages)
    raw = generate_raw(SYSTEM_PROMPT, user_prompt)

    if raw is None:
        raw = local_fallback_response("LLM unavailable or GROQ_API_KEY not configured")
        security_flags.append("LLM fallback used (no API key or backend unreachable).")

    try:
        parsed = _parse_llm_json(raw)
    except Exception:
        parsed = json.loads(local_fallback_response("model returned invalid JSON"))
        security_flags.append("Model output failed schema validation; safe fallback returned.")

    citations = _validate_citations(parsed.get("citations", []), passages)
    answer_text = parsed.get("answer") or config.INSUFFICIENT_EVIDENCE

    # If the model claims an answer but produced zero valid citations,
    # downgrade to a refusal rather than showing an uncited claim.
    refusal_reason = parsed.get("refusal_reason")
    if answer_text.strip().lower() != config.INSUFFICIENT_EVIDENCE.lower() and not citations:
        security_flags.append("Answer had no valid citations after validation; downgraded to refusal.")
        answer_text = config.INSUFFICIENT_EVIDENCE
        refusal_reason = refusal_reason or "Model answer could not be validated against retrieved evidence."

    latency = time.time() - start
    return RagAnswer(
        answer=answer_text,
        citations=citations,
        supporting_passages=passages,
        evidence=parsed.get("evidence", ""),
        inference=parsed.get("inference", ""),
        confidence=float(parsed.get("confidence", 0.0) or 0.0),
        refusal_reason=refusal_reason,
        latency_seconds=round(latency, 3),
        security_flags=security_flags,
    )
