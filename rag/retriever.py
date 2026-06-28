"""
RAG retriever: query -> top-k OISD chunks using TF-IDF + cosine similarity.
sklearn only — no sentence-transformers, no faiss, Vercel-compatible.
"""

import json, joblib
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent

class OISDRetriever:
    def __init__(self):
        self._loaded = False
        self.vectorizer = None
        self.matrix = None
        self.docstore = None

    def _lazy_load(self):
        if self._loaded:
            return
        tfidf_path = BASE / "tfidf.pkl"
        if not tfidf_path.exists():
            raise FileNotFoundError("TF-IDF index not found. Run: python rag/ingest.py")
        artifact = joblib.load(tfidf_path)
        self.vectorizer = artifact["vectorizer"]
        self.matrix = artifact["matrix"]
        self.docstore = json.loads((BASE / "docstore.json").read_text())
        self._loaded = True

    def query(self, question: str, k: int = 3) -> dict:
        self._lazy_load()
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_k = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k:
            if scores[idx] > 0:
                doc = self.docstore[str(idx)]
                results.append({
                    "text": doc["text"],
                    "relevance_score": round(float(scores[idx]), 3),
                    "chunk_id": int(idx),
                })

        answer = self._build_answer(results)
        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "source_count": len(results),
        }

    def _build_answer(self, results):
        if not results:
            return "No relevant regulatory guidance found."
        top = results[0]["text"]
        sentences = [s.strip() for s in top.split(".") if len(s.strip()) > 30][:3]
        return ". ".join(sentences) + "."
