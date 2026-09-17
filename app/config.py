"""
Central configuration for the RAG Research Assistant.
All values are read from environment variables (via .env) so that no
secrets or environment-specific values are hardcoded in source code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # characters
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))  # weight: semantic vs lexical

# --- Security / limits ---
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1500"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
SQLITE_PATH = DATA_DIR / "sqlite" / "metadata.db"
FAISS_INDEX_PATH = DATA_DIR / "sqlite" / "faiss.index"
EVAL_DATASET_PATH = DATA_DIR / "evaluation_dataset.json"
EVAL_RESULTS_PATH = DATA_DIR / "evaluation_results.json"

DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True, parents=True)
(DATA_DIR / "sqlite").mkdir(exist_ok=True, parents=True)

# Minimum similarity score (cosine, 0-1) below which we do not trust a chunk
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.15"))

INSUFFICIENT_EVIDENCE = "Insufficient evidence"
