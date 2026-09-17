"""
Downloads genuine, publicly available AI research papers from arXiv to
satisfy the assessment's requirement for real research documents
(minimum 5 documents, 20+ pages total). Requires internet access on the
machine this is run on (the development sandbox used to build this
project did not have arXiv in its network allowlist, which is why these
are fetched here rather than bundled directly).

Run: python scripts/download_knowledge_base.py
"""
import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config

PAPERS = {
    # (filename, arxiv PDF url, topic)
    "arxiv_attention_is_all_you_need.pdf": (
        "https://arxiv.org/pdf/1706.03762", "Transformer architecture"),
    "arxiv_retrieval_augmented_generation.pdf": (
        "https://arxiv.org/pdf/2005.11401", "Retrieval-Augmented Generation"),
    "arxiv_llm_evaluation_survey.pdf": (
        "https://arxiv.org/pdf/2307.03109", "LLM evaluation"),
    "arxiv_react_agents.pdf": (
        "https://arxiv.org/pdf/2210.03629", "Agentic AI / reasoning + acting"),
    "arxiv_survey_hallucination_llms.pdf": (
        "https://arxiv.org/pdf/2311.05232", "Hallucination in LLMs"),
}


def download_all():
    config.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for filename, (url, topic) in PAPERS.items():
        dest = config.KNOWLEDGE_BASE_DIR / filename
        try:
            print(f"Downloading {topic}: {url}")
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            ok.append(filename)
            print(f"  saved -> {dest}")
        except Exception as e:
            failed.append((filename, str(e)))
            print(f"  FAILED: {e}")

    print(f"\nDownloaded {len(ok)}/{len(PAPERS)} papers.")
    if failed:
        print("Failed downloads (check your internet connection or the URLs, which can move):")
        for f, err in failed:
            print(f"  - {f}: {err}")
        print("\nYou can still run scripts/create_sample_pdfs.py for a fully offline demo knowledge base.")


if __name__ == "__main__":
    download_all()
