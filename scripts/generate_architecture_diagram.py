"""
Generates docs/architecture_diagram.png using Graphviz.
Run: python scripts/generate_architecture_diagram.py
(Requires the `graphviz` Python package AND the system `dot` binary.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphviz import Digraph

g = Digraph("architecture", format="png")
g.attr(rankdir="TB", fontsize="11", fontname="Helvetica")
g.attr("node", fontname="Helvetica", fontsize="10")

# --- Ingestion flow ---
with g.subgraph(name="cluster_ingest") as c:
    c.attr(label="Document Ingestion", style="rounded", color="gray50")
    c.node("upload", "Upload\nPDF / TXT / MD", shape="box")
    c.node("validate_file", "File validation\n(type, size, empty)", shape="box")
    c.node("extract", "Text extraction\n(PyMuPDF)", shape="box")
    c.node("corrupt", "Corrupt/empty file\n-> error to UI", shape="box", style="dashed")
    c.node("chunk", "Chunking\n(sentence-aware,\nsize+overlap)", shape="box")
    c.node("embed_ingest", "Local embedding\n(MiniLM-L6-v2)", shape="box")
    c.node("dup", "Duplicate\ndetection (hash)", shape="diamond")
    c.node("faiss_write", "FAISS index", shape="cylinder")
    c.node("sqlite_write", "SQLite metadata\n(doc, page, chunk_id)", shape="cylinder")

    c.edge("upload", "validate_file")
    c.edge("validate_file", "corrupt", label="invalid", style="dashed")
    c.edge("validate_file", "extract", label="valid")
    c.edge("extract", "corrupt", label="extraction error", style="dashed")
    c.edge("extract", "dup")
    c.edge("dup", "chunk", label="new doc")
    c.edge("dup", "corrupt", label="duplicate", style="dashed")
    c.edge("chunk", "embed_ingest")
    c.edge("embed_ingest", "faiss_write")
    c.edge("embed_ingest", "sqlite_write")

# --- Query flow ---
with g.subgraph(name="cluster_query") as c:
    c.attr(label="Query Flow", style="rounded", color="gray50")
    c.node("user", "User", shape="oval")
    c.node("ui", "Streamlit UI", shape="box")
    c.node("api", "FastAPI\n(app/main.py)", shape="box")
    c.node("qvalid", "Query validation\n(length, empty)", shape="box")
    c.node("qreject", "Reject with\nclear error", shape="box", style="dashed")
    c.node("embed_query", "Embed query\n(MiniLM-L6-v2)", shape="box")
    c.node("semantic", "Semantic search\n(FAISS)", shape="box")
    c.node("lexical", "Lexical search\n(BM25)", shape="box")
    c.node("hybrid", "Hybrid scoring +\nduplicate-context removal", shape="box")
    c.node("guard", "Prompt-injection guard\n(pattern scan, flags only)", shape="box")
    c.node("prompt", "RAG prompt build\n(evidence = untrusted DATA)", shape="box")
    c.node("groq", "Groq LLM\n(free tier)", shape="box")
    c.node("retry", "Timeout / retry\n(backoff)", shape="diamond")
    c.node("fallback", "Local fallback:\n'Insufficient evidence'\n(never invents answer)", shape="box", style="dashed")
    c.node("outvalid", "Output validation\n(schema + citation check\nagainst retrieved chunk_ids)", shape="box")
    c.node("answer", "Cited answer +\nevidence/inference split", shape="box", style="bold")

    c.edge("user", "ui")
    c.edge("ui", "api", style="dotted", label="(or in-process call)")
    c.edge("api", "qvalid")
    c.edge("qvalid", "qreject", label="invalid", style="dashed")
    c.edge("qvalid", "embed_query", label="valid")
    c.edge("embed_query", "semantic")
    c.edge("embed_query", "lexical")
    c.edge("semantic", "hybrid")
    c.edge("lexical", "hybrid")
    c.edge("hybrid", "guard")
    c.edge("guard", "prompt")
    c.edge("prompt", "groq")
    c.edge("groq", "retry")
    c.edge("retry", "groq", label="retry on\ntimeout/failure")
    c.edge("retry", "fallback", label="exhausted /\nno API key", style="dashed")
    c.edge("groq", "outvalid")
    c.edge("fallback", "outvalid")
    c.edge("outvalid", "answer")
    c.edge("answer", "ui")

# cross-links from ingestion stores into query flow
g.edge("faiss_write", "semantic", style="dotted", color="gray40")
g.edge("sqlite_write", "hybrid", style="dotted", color="gray40", label="citation metadata")
g.edge("sqlite_write", "lexical", style="dotted", color="gray40")

# --- Evaluation pipeline ---
with g.subgraph(name="cluster_eval") as c:
    c.attr(label="Evaluation Pipeline", style="rounded", color="gray50")
    c.node("dataset", "evaluation_dataset.json\n(20 cases: answerable,\nunanswerable, contradictory,\nprompt-injection)", shape="box")
    c.node("run_eval", "run_evaluation.py\n(calls real pipeline)", shape="box")
    c.node("metrics", "Metrics:\nhit rate, citation correctness,\ngroundedness, refusal accuracy,\navg latency", shape="box")
    c.node("evallog", "Evaluation Log tab\n(color-coded)", shape="box")

    c.edge("dataset", "run_eval")
    c.edge("run_eval", "answer", style="dotted", label="exercises")
    c.edge("run_eval", "metrics")
    c.edge("metrics", "evallog")

g.render(filename="architecture_diagram", directory="docs", cleanup=True)
print("Wrote docs/architecture_diagram.png")
