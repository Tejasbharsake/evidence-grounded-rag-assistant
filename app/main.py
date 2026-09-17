"""
FastAPI backend. Thin layer over the ingestion/query pipelines so the
system can also be used programmatically or from a different frontend.
The Streamlit UI (app/ui/streamlit_app.py) calls these pipelines directly
in-process for simplicity and to guarantee true single-command startup;
this API is provided for completeness and external/API-based testing.

Run: uvicorn app.main:app --reload
"""
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app import config
from app.ingestion.pipeline import ingest_file, current_stats
from app.query_pipeline import answer_question

app = FastAPI(title="Evidence-Grounded AI Research Assistant")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    document_filter: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return current_stats()


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        suffix = Path(f.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(f.file, tmp)
            tmp_path = Path(tmp.name)
        # Preserve the original filename for citations/duplicate detection
        target = config.KNOWLEDGE_BASE_DIR / f.filename
        shutil.move(str(tmp_path), str(target))
        result = ingest_file(target)
        results.append(result.__dict__)
    return {"results": results}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    answer = answer_question(req.question, top_k=req.top_k, document_filter=req.document_filter)
    return answer.to_dict()
