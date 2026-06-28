"""
Vizag Safety Intelligence — FastAPI Backend
Serves dashboard, streams sensor data via WebSocket, provides risk API
"""

import json, asyncio, sys, os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, "/Users/uditjain/Desktop/vizag_safety")

app = FastAPI(title="Vizag Safety Intelligence", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="/Users/uditjain/Desktop/vizag_safety/static"), name="static")

# ── State ────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.playback_mode = "normal"  # "normal" | "vizag"
        self.playback_index = 0
        self.is_playing = False
        self.vizag_df = None
        self.normal_df = None
        self.permits = []
        self.shift_info = {}
        self.engine = None
        self.retriever = None
        self.connected_clients = []

state = AppState()

def load_data():
    """Load simulation data and initialize engines"""
    data_dir = Path("/Users/uditjain/Desktop/vizag_safety/data")

    try:
        state.vizag_df = pd.read_csv(data_dir / "vizag_timeline.csv")
        print(f"Loaded Vizag timeline: {len(state.vizag_df)} rows")
    except Exception as e:
        print(f"Warning: Could not load vizag_timeline.csv: {e}")
        state.vizag_df = _generate_fallback_vizag()

    try:
        state.normal_df = pd.read_csv(data_dir / "normal_ops.csv")
    except:
        state.normal_df = _generate_fallback_normal()

    try:
        with open(data_dir / "permits.json") as f:
            state.permits = json.load(f)
        with open(data_dir / "shifts.json") as f:
            state.shift_info = json.load(f)
    except:
        state.permits = []
        state.shift_info = {"handover_complete": False}

    # Load compound risk engine
    try:
        from engine.compound_risk import CompoundRiskEngine
        state.engine = CompoundRiskEngine()
        print("Compound risk engine loaded")
    except Exception as e:
        print(f"Warning: Engine load failed: {e}")

    # Load RAG retriever (lazy — loads on first query)
    try:
        from rag.retriever import OISDRetriever
        state.retriever = OISDRetriever()
        print("OISD retriever ready (lazy load)")
    except Exception as e:
        print(f"Warning: Retriever init failed: {e}")

def _generate_fallback_vizag():
    """Fallback vizag data if CSV not generated yet"""
    n = 120
    t = np.linspace(60, 0, n)
    np.random.seed(42)
    df = pd.DataFrame({
        "minutes_to_explosion": t,
        "timestamp": pd.date_range("2025-01-18 14:00", periods=n, freq="30s").astype(str),
        "zone": "C",
        "h2s_ppm": np.where(t > 47, np.random.normal(10, 1, n),
                   np.where(t > 30, np.interp(t, [30, 47], [43, 12]) + np.random.normal(0, 1, n),
                   np.interp(t, [0, 30], [180, 43]) + np.random.normal(0, 2, n))),
        "co_ppm": np.where(t > 40, np.random.normal(25, 3, n), np.interp(t, [0, 40], [240, 25]) + np.random.normal(0, 5, n)),
        "pressure_kpa": np.where(t > 45, np.random.normal(785, 5, n), np.interp(t, [0, 45], [920, 785]) + np.random.normal(0, 3, n)),
        "temperature_c": np.where(t > 35, np.random.normal(390, 3, n), np.interp(t, [0, 35], [445, 390]) + np.random.normal(0, 2, n)),
        "vibration_ms2": np.where(t > 25, np.random.normal(1.2, 0.2, n), np.interp(t, [0, 25], [9.5, 1.2]) + np.random.normal(0, 0.3, n)),
        "hot_work_permit_active": (t < 47).astype(int),
        "shift_changeover": ((t < 47) & (t > 35)).astype(int),
        "confined_space_entry": (t < 45).astype(int),
        "compound_alarm": (t < 47).astype(int),
        "single_sensor_alarm": (t < 30).astype(int),
    })
    df["h2s_ppm"] = df["h2s_ppm"].clip(5, 200)
    return df

def _generate_fallback_normal():
    np.random.seed(99)
    n = 300
    return pd.DataFrame({
        "h2s_ppm": np.random.normal(8, 1.5, n).clip(2, 15),
        "co_ppm": np.random.normal(20, 4, n).clip(5, 35),
        "pressure_kpa": np.random.normal(783, 6, n).clip(765, 800),
        "temperature_c": np.random.normal(388, 4, n).clip(378, 398),
        "vibration_ms2": np.random.normal(1.1, 0.25, n).clip(0.4, 2.0),
    })

@app.on_event("startup")
async def startup():
    load_data()

# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("/Users/uditjain/Desktop/vizag_safety/static/index.html")

@app.get("/api/status")
async def status():
    return {
        "mode": state.playback_mode,
        "index": state.playback_index,
        "playing": state.is_playing,
        "engine_ready": state.engine is not None,
        "rag_ready": state.retriever is not None,
    }

@app.post("/api/playback/{mode}")
async def set_playback(mode: str):
    """mode: 'normal' | 'vizag'"""
    state.playback_mode = mode
    state.playback_index = 0
    state.is_playing = True
    return {"status": "ok", "mode": mode}

@app.post("/api/stop")
async def stop_playback():
    state.is_playing = False
    state.playback_index = 0
    return {"status": "stopped"}

@app.get("/api/sensor_snapshot")
async def sensor_snapshot():
    """Get current sensor reading + risk assessment"""
    row = _get_current_row()
    assessment = _assess_row(row)
    return {"sensor": row, "assessment": assessment}

class QueryRequest(BaseModel):
    question: str

@app.post("/api/query")
async def rag_query(req: QueryRequest):
    """Query OISD regulatory corpus"""
    if state.retriever is None:
        return {"error": "RAG not initialized", "answer": "OISD retriever not loaded. Run setup.sh first."}
    try:
        result = state.retriever.query(req.question)
        return result
    except Exception as e:
        return {"error": str(e), "answer": "Query failed. Ensure rag/ingest.py has been run."}

@app.get("/api/plant_zones")
async def plant_zones():
    """Zone definitions for plant heatmap"""
    row = _get_current_row()
    assessment = _assess_row(row)
    risk_score = assessment.get("compound_risk_score", 0)

    zones = [
        {"id": "A", "name": "Control Room", "x": 10, "y": 10, "w": 18, "h": 22, "risk": 0.02, "workers": 6, "permits": 0},
        {"id": "B", "name": "Boiler House", "x": 32, "y": 10, "w": 16, "h": 20, "risk": 0.08, "workers": 4, "permits": 1},
        {"id": "C", "name": "Coke Oven Battery", "x": 52, "y": 8, "w": 22, "h": 28, "risk": risk_score, "workers": 7, "permits": 2, "incident_zone": True},
        {"id": "D", "name": "Raw Materials", "x": 10, "y": 38, "w": 20, "h": 18, "risk": 0.05, "workers": 3, "permits": 0},
        {"id": "E", "name": "Blast Furnace", "x": 34, "y": 35, "w": 18, "h": 22, "risk": 0.12, "workers": 5, "permits": 1},
        {"id": "F", "name": "Perimeter / Utilities", "x": 78, "y": 10, "w": 14, "h": 55, "risk": 0.03, "workers": 2, "permits": 0},
    ]
    return {"zones": zones, "active_alert_zone": "C" if risk_score > 0.7 else None}

@app.websocket("/ws/sensors")
async def sensor_stream(websocket: WebSocket):
    """Real-time sensor data stream — advances playback index on each tick"""
    await websocket.accept()
    state.connected_clients.append(websocket)
    try:
        while True:
            row = _get_current_row()
            assessment = _assess_row(row)

            payload = {
                "type": "sensor_update",
                "timestamp": row.get("timestamp", datetime.now().isoformat()),
                "minutes_to_explosion": row.get("minutes_to_explosion", None),
                "mode": state.playback_mode,
                "index": state.playback_index,
                "sensors": {
                    "h2s_ppm": round(float(row.get("h2s_ppm", 10)), 1),
                    "co_ppm": round(float(row.get("co_ppm", 20)), 1),
                    "pressure_kpa": round(float(row.get("pressure_kpa", 785)), 1),
                    "temperature_c": round(float(row.get("temperature_c", 390)), 1),
                    "vibration_ms2": round(float(row.get("vibration_ms2", 1.2)), 2),
                },
                "permits": {
                    "hot_work_active": int(row.get("hot_work_permit_active", 0)),
                    "shift_changeover": int(row.get("shift_changeover", 0)),
                    "confined_space_entry": int(row.get("confined_space_entry", 0)),
                },
                "assessment": assessment,
            }
            await websocket.send_json(payload)

            # Advance playback
            if state.is_playing:
                df = state.vizag_df if state.playback_mode == "vizag" else state.normal_df
                if df is not None:
                    state.playback_index = (state.playback_index + 1) % len(df)

            await asyncio.sleep(0.8)  # ~0.8s per tick = 1 min of incident in ~60 seconds
    except WebSocketDisconnect:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)
    except Exception:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)

def _get_current_row() -> dict:
    df = state.vizag_df if state.playback_mode == "vizag" else state.normal_df
    if df is None or len(df) == 0:
        return {}
    idx = min(state.playback_index, len(df) - 1)
    row = df.iloc[idx].to_dict()
    # Add permit/shift context for vizag mode
    if state.playback_mode == "vizag":
        row["hot_work_permit_active"] = 1 if row.get("minutes_to_explosion", 60) < 47 else 0
        row["shift_changeover"] = 1 if 35 < row.get("minutes_to_explosion", 60) < 47 else 0
        row["confined_space_entry"] = 1 if row.get("minutes_to_explosion", 60) < 45 else 0
    return row

def _assess_row(row: dict) -> dict:
    if state.engine is None:
        # Fallback assessment without engine
        h2s = row.get("h2s_ppm", 0)
        permits_active = row.get("hot_work_permit_active", 0) and row.get("confined_space_entry", 0)
        score = 0.0
        if h2s > 35 and permits_active:
            score = 0.82
        elif h2s > 50:
            score = 0.65
        elif h2s > 30:
            score = 0.30
        return {
            "compound_risk_score": round(score, 3),
            "risk_level": "CRITICAL" if score > 0.75 else "HIGH" if score > 0.5 else "ELEVATED" if score > 0.25 else "NORMAL",
            "would_traditional_alert": h2s > 50,
            "compound_detects_before_traditional": score > 0.75 and h2s < 50,
            "triggered_rules": [],
            "recommendation": "IMMEDIATE EVACUATION — compound risk pattern matches Vizag incident." if score > 0.75 else "Continue monitoring.",
            "oisd_citations": ["OISD-GS-1 §6.3.2"] if score > 0.75 else [],
        }

    permits = []
    if row.get("hot_work_permit_active", 0):
        permits = [p for p in state.permits if p.get("type") == "HOT_WORK"] or [{"type": "HOT_WORK", "zone": "C", "status": "ACTIVE"}]
    if row.get("confined_space_entry", 0):
        permits += [p for p in state.permits if p.get("type") == "CONFINED_SPACE_ENTRY"] or [{"type": "CONFINED_SPACE_ENTRY", "zone": "C", "status": "ACTIVE"}]

    shift = state.shift_info or {"handover_complete": False}
    if row.get("shift_changeover", 0):
        shift["handover_complete"] = False

    return state.engine.assess(row, permits, shift)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" VIZAG SAFETY INTELLIGENCE")
    print(" ET AI Hackathon 2026 — Industrial Safety PS#1")
    print("="*60)
    print(" Dashboard: http://localhost:8001")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
