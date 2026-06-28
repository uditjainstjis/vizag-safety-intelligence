"""
RAG retriever: query -> top-k OISD chunks
No LLM needed — retrieval + template = clear, cited answer
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class OISDRetriever:
    def __init__(self):
        self.model = None
        self.index = None
        self.docstore = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        index_path = Path("/Users/uditjain/Desktop/vizag_safety/rag/faiss.index")
        docstore_path = Path("/Users/uditjain/Desktop/vizag_safety/rag/docstore.json")
        if not index_path.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run: python rag/ingest.py"
            )
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(index_path))
        self.docstore = json.loads(docstore_path.read_text())
        self._loaded = True

    def query(self, question: str, k: int = 3) -> dict:
        """
        Returns top-k relevant OISD chunks for the question.

        Args:
            question: Natural language safety question
            k: Number of top chunks to retrieve (default 3)

        Returns:
            dict with keys: question, answer, sources, source_count
        """
        self._lazy_load()
        emb = self.model.encode([question], show_progress_bar=False)
        emb = np.array(emb, dtype="float32")
        faiss.normalize_L2(emb)
        scores, indices = self.index.search(emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc = self.docstore[str(idx)]
                results.append(
                    {
                        "text": doc["text"],
                        "relevance_score": round(float(score), 3),
                        "chunk_id": int(idx),
                    }
                )

        # Build a coherent answer from retrieved chunks
        answer = self._build_answer(question, results)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "source_count": len(results),
        }

    def _build_answer(self, question: str, results: list) -> str:
        """
        Synthesizes a concise cited answer from the top retrieved chunks.
        Extracts key regulatory statements from the most relevant chunk,
        then appends a brief note from the second-most-relevant chunk if present.
        """
        if not results:
            return "No relevant regulatory guidance found."

        # Extract source tag from top chunk (e.g., "[SOURCE: OISD-GS-1 §4.2.3 ...]")
        top_text = results[0]["text"]
        source_tag = ""
        if top_text.startswith("[SOURCE:"):
            end = top_text.find("]")
            if end != -1:
                source_tag = top_text[: end + 1]
                top_text = top_text[end + 1 :].strip()

        # Pull first 3 substantive sentences from the top chunk body
        sentences = [s.strip() for s in top_text.split(".") if len(s.strip()) > 40][:3]
        primary = ". ".join(sentences) + "."

        # Append a supplementary sentence from the second chunk if available and distinct
        supplement = ""
        if len(results) > 1:
            second_text = results[1]["text"]
            if second_text.startswith("[SOURCE:"):
                end = second_text.find("]")
                second_source = second_text[: end + 1] if end != -1 else ""
                second_body = second_text[end + 1 :].strip() if end != -1 else second_text
            else:
                second_source = ""
                second_body = second_text
            supp_sentences = [
                s.strip() for s in second_body.split(".") if len(s.strip()) > 40
            ][:1]
            if supp_sentences and second_source:
                supplement = f" Additionally, {second_source}: {supp_sentences[0]}."

        if source_tag:
            return f"{source_tag}: {primary}{supplement}"
        return f"{primary}{supplement}"


# Convenience function for direct import use
_retriever_singleton = None


def get_retriever() -> OISDRetriever:
    """Returns a lazily initialized singleton retriever instance."""
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = OISDRetriever()
    return _retriever_singleton


if __name__ == "__main__":
    # Quick smoke-test queries
    retriever = OISDRetriever()

    test_questions = [
        "What actions are required when H2S reaches 35 ppm?",
        "Can hot work continue if gas alarms activate?",
        "What are the gas testing requirements before confined space entry?",
        "What must be done during shift handover for gas readings?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = retriever.query(q, k=3)
        print(f"A: {result['answer']}")
        print(f"   Sources: {[r['chunk_id'] for r in result['sources']]} "
              f"(scores: {[r['relevance_score'] for r in result['sources']]})")
