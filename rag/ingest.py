"""
OISD Regulatory RAG — TF-IDF ingestion (sklearn only, no heavy deps)
Replaces sentence-transformers + faiss for Vercel compatibility.
"""

import json, joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS_PATH = Path(__file__).parent / "oisd_corpus.txt"
TFIDF_PATH  = Path(__file__).parent / "tfidf.pkl"
DOCSTORE_PATH = Path(__file__).parent / "docstore.json"

def ingest():
    text = CORPUS_PATH.read_text()
    chunks = [c.strip() for c in text.split("---CHUNK---") if c.strip()]
    print(f"Loaded {len(chunks)} chunks")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(chunks)

    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, TFIDF_PATH)

    docstore = {str(i): {"text": chunk, "id": i} for i, chunk in enumerate(chunks)}
    DOCSTORE_PATH.write_text(json.dumps(docstore, indent=2))

    print(f"TF-IDF index saved: {TFIDF_PATH} ({matrix.shape})")
    print(f"Docstore saved: {DOCSTORE_PATH}")

if __name__ == "__main__":
    ingest()
