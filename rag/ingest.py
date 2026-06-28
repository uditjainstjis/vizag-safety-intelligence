"""
Ingests oisd_corpus.txt -> builds FAISS index
Uses: sentence-transformers (all-MiniLM-L6-v2, free, ~80MB)
      faiss-cpu (free vector search)
"""

import json
import os
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CORPUS_PATH = Path("/Users/uditjain/Desktop/vizag_safety/rag/oisd_corpus.txt")
INDEX_PATH = Path("/Users/uditjain/Desktop/vizag_safety/rag/faiss.index")
DOCSTORE_PATH = Path("/Users/uditjain/Desktop/vizag_safety/rag/docstore.json")


def ingest():
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Parse corpus
    text = CORPUS_PATH.read_text()
    chunks = [c.strip() for c in text.split("---CHUNK---") if c.strip()]
    print(f"Loaded {len(chunks)} chunks")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (inner product after L2 normalization = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    # Save docstore
    docstore = {str(i): {"text": chunk, "id": i} for i, chunk in enumerate(chunks)}
    DOCSTORE_PATH.write_text(json.dumps(docstore, indent=2))

    print(f"FAISS index saved: {INDEX_PATH} ({index.ntotal} vectors)")
    print(f"Docstore saved: {DOCSTORE_PATH}")


if __name__ == "__main__":
    ingest()
