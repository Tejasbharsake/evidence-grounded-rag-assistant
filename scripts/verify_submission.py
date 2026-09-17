"""Offline pre-submission verifier for Assessment B."""
from pathlib import Path
import json, sys
try:
    import fitz
except Exception:
    fitz = None

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "knowledge_base"
DATASET = ROOT / "data" / "evaluation_dataset.json"

def main():
    errors=[]; warnings=[]
    files=[p for p in KB.iterdir() if p.is_file()] if KB.exists() else []
    pdfs=[p for p in files if p.suffix.lower()==".pdf"]
    if len(pdfs)<5: errors.append(f"Need at least 5 PDFs; found {len(pdfs)}")
    pages=0
    if fitz:
        for p in pdfs:
            try: pages += len(fitz.open(p))
            except Exception as e: errors.append(f"Unreadable PDF {p.name}: {e}")
    else: warnings.append("PyMuPDF unavailable; page count not checked")
    if pages<20: errors.append(f"Need 20+ PDF pages; found {pages}")
    if not DATASET.exists(): errors.append("Missing evaluation_dataset.json")
    else:
        cases=json.loads(DATASET.read_text())
        counts={k:sum(c.get('category')==k for c in cases) for k in ('answerable','unanswerable','contradictory','prompt_injection')}
        for k,n in {'answerable':10,'unanswerable':5,'contradictory':3,'prompt_injection':2}.items():
            if counts[k]<n: errors.append(f"Need {n} {k} cases; found {counts[k]}")
    for bad in ('.env','.venv','.pytest_cache'):
        if (ROOT/bad).exists(): errors.append(f"Remove {bad} before submission")
    print(f"PDFs: {len(pdfs)} | pages: {pages} | dataset: {DATASET.exists()}")
    if warnings:
        print("WARNINGS:"); [print(' -',w) for w in warnings]
    if errors:
        print("FAIL:"); [print(' -',e) for e in errors]; return 1
    print("PASS: offline submission checks")
    return 0
if __name__=='__main__': sys.exit(main())
