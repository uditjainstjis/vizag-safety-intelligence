"""
Vizag Safety Intelligence — FastAPI Backend
Stateless serverless-compatible version (Vercel).
Playback is client-side (vizag_timeline.json served as static).
Server handles: dashboard serve, RAG queries, zone API.
"""

import json, sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

app = FastAPI(title="Vizag Safety Intelligence", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ── Lazy singletons ───────────────────────────────────────────────────────
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from rag.retriever import OISDRetriever
        _retriever = OISDRetriever()
    return _retriever

# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

@app.get("/api/status")
async def status():
    return {"status": "ok", "engine": "TF-IDF RAG + IsolationForest", "version": "1.0.0"}

class QueryRequest(BaseModel):
    question: str

@app.post("/api/query")
async def rag_query(req: QueryRequest):
    """Query OISD regulatory corpus via TF-IDF retrieval."""
    try:
        retriever = get_retriever()
        return retriever.query(req.question)
    except FileNotFoundError:
        return {
            "question": req.question,
            "answer": "OISD knowledge base not initialised. Run: python rag/ingest.py",
            "sources": [],
            "source_count": 0,
        }
    except Exception as e:
        return {
            "question": req.question,
            "answer": f"Query error: {e}",
            "sources": [],
            "source_count": 0,
        }

@app.get("/api/plant_zones")
async def plant_zones(risk_score: float = 0.0):
    """Zone definitions for plant heatmap. risk_score passed from frontend."""
    zones = [
        {"id": "A", "name": "Control Room",       "x": 10, "y": 10, "w": 18, "h": 22, "risk": 0.02, "workers": 6,  "permits": 0},
        {"id": "B", "name": "Boiler House",        "x": 32, "y": 10, "w": 16, "h": 20, "risk": 0.08, "workers": 4,  "permits": 1},
        {"id": "C", "name": "Coke Oven Battery",   "x": 52, "y":  8, "w": 22, "h": 28, "risk": risk_score, "workers": 7, "permits": 2, "incident_zone": True},
        {"id": "D", "name": "Raw Materials",       "x": 10, "y": 38, "w": 20, "h": 18, "risk": 0.05, "workers": 3,  "permits": 0},
        {"id": "E", "name": "Blast Furnace",       "x": 34, "y": 35, "w": 18, "h": 22, "risk": 0.12, "workers": 5,  "permits": 1},
        {"id": "F", "name": "Perimeter/Utilities", "x": 78, "y": 10, "w": 14, "h": 55, "risk": 0.03, "workers": 2,  "permits": 0},
    ]
    return {"zones": zones, "active_alert_zone": "C" if risk_score > 0.7 else None}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(" VIZAG SAFETY INTELLIGENCE")
    print(" ET AI Hackathon 2026 — Industrial Safety PS#1")
    print("=" * 60)
    print(" Dashboard: http://localhost:8001")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
