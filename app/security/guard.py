"""
Security module.

Two complementary layers, since neither alone is reliable:

1. Detection: flag retrieved chunks that *look* like they're trying to
   issue instructions, so the UI can show a visible warning. This is a
   defense-in-depth signal, not the primary defense.
2. Structural defense (the real protection): retrieved text is always
   passed to the LLM as clearly-delimited, labeled DATA inside the
   prompt, with an explicit system instruction to never treat document
   content as commands. This is enforced in app/generation/generator.py.
   Detection here cannot be bypassed by rewording, because the actual
   safety property comes from the prompt structure, not keyword matching.
"""
import re
from typing import List, Tuple

INJECTION_PATTERNS = [
    r"ignore (the )?(previous|prior|above|user'?s?) (instructions?|question)",
    r"reveal (the )?system prompt",
    r"you are now",
    r"disregard (all|any) (previous|prior) instructions",
    r"act as (if|though)",
    r"print your (instructions|system prompt|api key)",
    r"new instructions?:",
    r"override (your|the) (rules|instructions|guidelines)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def scan_text_for_injection(text: str) -> List[str]:
    """Returns a list of matched pattern descriptions (empty if clean)."""
    hits = []
    for pattern in _COMPILED:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def scan_chunks_for_injection(chunks) -> List[Tuple[str, List[str]]]:
    """chunks: list of RetrievedChunk. Returns [(chunk_id, matched_patterns)]
    for any chunk containing suspicious instruction-like text."""
    flagged = []
    for c in chunks:
        hits = scan_text_for_injection(c.text)
        if hits:
            flagged.append((c.chunk_id, hits))
    return flagged


class QueryValidationError(Exception):
    pass


def validate_query(query: str, max_length: int) -> str:
    if query is None or not query.strip():
        raise QueryValidationError("Query cannot be empty.")
    query = query.strip()
    if len(query) > max_length:
        raise QueryValidationError(
            f"Query is too long ({len(query)} chars). Maximum is {max_length} characters."
        )
    return query
