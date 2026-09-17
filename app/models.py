"""
Shared data schemas used across ingestion, retrieval, generation and the UI.
Using dataclasses keeps this dependency-free and easy to serialize to JSON.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str
    embedding: Optional[List[float]] = None

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str
    semantic_score: float
    lexical_score: float
    combined_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    document_name: str
    page_number: int
    chunk_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RagAnswer:
    answer: str
    citations: List[Citation]
    supporting_passages: List[RetrievedChunk]
    evidence: str
    inference: str
    confidence: float
    refusal_reason: Optional[str]
    latency_seconds: float
    security_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "supporting_passages": [p.to_dict() for p in self.supporting_passages],
            "evidence": self.evidence,
            "inference": self.inference,
            "confidence": self.confidence,
            "refusal_reason": self.refusal_reason,
            "latency_seconds": self.latency_seconds,
            "security_flags": self.security_flags,
        }
