# Vizag Safety Intelligence
## ET AI Hackathon 2026 · Problem Statement #1
### AI-Powered Industrial Safety Intelligence for Zero-Harm Operations

---

## The Problem
On January 18, 2025, 8 workers died at the Visakhapatnam Steel Plant when entrapped gases triggered a sudden explosion in the coke oven battery. The facility had functioning gas detectors, permit-to-work systems, and SCADA controls. Warning signals from gas pressure sensors existed. No intelligence layer connected those readings to operational decisions in time.

**The problem is not absence of technology. It is absence of intelligence.**

---

## What This System Does
This prototype demonstrates **Compound Risk Detection** — identifying dangerous combinations that no single sensor would flag alone.

| Traditional Monitoring | Vizag Safety Intelligence |
|----------------------|--------------------------|
| Monitors each sensor independently | Correlates sensors + permits + shift patterns |
| Alarms at H2S > 50ppm | Detects risk when H2S = 43ppm + hot work permit active |
| Would have alarmed at T-30 min | **Detects compound risk at T-47 min** |
| Only 30 min before explosion | **47 min before explosion — 17 min earlier** |

---

## Quick Start
```bash
git clone <repo>
cd vizag_safety
bash setup.sh
python app.py
# Open http://localhost:8000
```

No API keys. No cloud services. Runs fully local.

---

## Architecture
- **Compound Risk Engine**: IsolationForest (sklearn) + 5 compound rules
- **OISD RAG**: sentence-transformers + FAISS over 28 regulatory chunks
- **Sensor Simulation**: Physics-based Vizag incident timeline (30s resolution)
- **Dashboard**: WebSocket real-time updates, plant heatmap, side-by-side comparison

---

## Demo Flow
1. Page loads → Normal operations (all green)
2. Click "REPLAY VIZAG INCIDENT"
3. Watch: sensors rise → compound engine fires at T-47 (right panel goes CRITICAL)
4. Traditional monitoring (left panel): still shows NO ALARM
5. At T-30: traditional alarm finally fires — 17 minutes later
6. Ask the RAG: "What protocol applies for hot work near H2S elevated zones?"

---

## Regulatory Compliance
All alerts cite OISD (Oil Industry Safety Directorate), Factory Act 1948, and DGMS regulations by section number.

---

## Scalability
This engine is sensor-schema agnostic. Feed it any plant's sensor + permit data with a config file. Deployable to all 1,900+ notified hazardous factories in India.
