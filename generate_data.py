"""
Vizag Steel Plant Incident Simulation Data Generator
January 18, 2025 - Coke Oven Battery Explosion, Zone C
8 workers killed. Gas pressure warnings existed; no intelligence layer acted.

Generates:
  data/vizag_timeline.csv  — 120 rows (T-60 to T=0, 30s resolution)
  data/normal_ops.csv      — 300 rows (baseline safe operations)
"""

import os
import csv
import json
import random
import math
from datetime import datetime, timedelta

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

EXPLOSION_DT = datetime(2025, 1, 18, 15, 0, 0)   # T=0 moment
START_DT     = EXPLOSION_DT - timedelta(minutes=60)  # T=-60


def noisy(value, pct=0.05):
    """Add ±pct gaussian noise to a value."""
    sigma = abs(value) * pct
    return value + random.gauss(0, sigma)


def lerp(a, b, t):
    """Linear interpolation between a and b at fraction t in [0,1]."""
    return a + (b - a) * t


def fraction_in(lo_min, hi_min, cur_min):
    """
    cur_min = minutes remaining to explosion (60 → 0).
    lo_min / hi_min define the window (e.g. lo=40, hi=30 means T-40 to T-30).
    Returns fraction through that window (0.0 at entry, 1.0 at exit).
    Clamps outside window.
    """
    # cur_min counts DOWN: 60 at start, 0 at explosion.
    # Window from T-lo_min to T-hi_min  means cur_min goes from lo_min DOWN to hi_min.
    if cur_min >= lo_min:
        return 0.0
    if cur_min <= hi_min:
        return 1.0
    return (lo_min - cur_min) / (lo_min - hi_min)


def h2s_value(minutes_left):
    """H2S ppm profile across 60-minute window."""
    m = minutes_left
    if m >= 50:                     # T-60 to T-50
        return lerp(8, 12, fraction_in(60, 50, m))
    elif m >= 40:                   # T-50 to T-40
        return lerp(12, 25, fraction_in(50, 40, m))
    elif m >= 35:                   # T-40 to T-35
        return lerp(25, 38, fraction_in(40, 35, m))
    elif m >= 30:                   # T-35 to T-30
        return lerp(38, 43, fraction_in(35, 30, m))
    elif m >= 20:                   # T-30 to T-20
        return lerp(43, 52, fraction_in(30, 20, m))
    else:                           # T-20 to T=0
        return lerp(52, 180, fraction_in(20, 0, m))


def co_value(minutes_left):
    """CO ppm profile."""
    m = minutes_left
    if m >= 50:
        return lerp(15, 22, fraction_in(60, 50, m))
    elif m >= 40:
        return lerp(22, 55, fraction_in(50, 40, m))
    elif m >= 35:
        return lerp(55, 90, fraction_in(40, 35, m))
    elif m >= 30:
        return lerp(90, 130, fraction_in(35, 30, m))
    elif m >= 20:
        return lerp(130, 175, fraction_in(30, 20, m))
    else:
        return lerp(175, 280, fraction_in(20, 0, m))


def pressure_value(minutes_left):
    """Coke oven gas pressure kPa profile."""
    m = minutes_left
    if m >= 45:
        return lerp(780, 790, fraction_in(60, 45, m))
    elif m >= 30:
        return lerp(790, 820, fraction_in(45, 30, m))
    else:
        return lerp(820, 920, fraction_in(30, 0, m))


def temperature_value(minutes_left):
    """Zone temperature Celsius profile."""
    m = minutes_left
    if m >= 35:
        return lerp(385, 395, fraction_in(60, 35, m))
    else:
        return lerp(395, 445, fraction_in(35, 0, m))


def vibration_value(minutes_left):
    """Equipment vibration m/s² profile."""
    m = minutes_left
    if m >= 25:
        return lerp(0.5, 2.0, fraction_in(60, 25, m))
    else:
        return lerp(2.0, 9.5, fraction_in(25, 0, m))


def compound_risk(minutes_left, hot_work, shift_changeover, confined):
    """
    Compound risk score 0.0-1.0.
    Phase 1 (T-60 to T-47): background noise, 0.05-0.15.
    Phase 2 (T-47 to T=0): compound conditions detected — immediately CRITICAL (>0.80)
      at T-47min (the '47 minutes early' detection moment), rising to 1.0 at T=0.

    Note: risk does NOT drop when shift changeover ends (T-35) because hot work +
    confined space entry remain active — the compound hazard persists.
    """
    m = minutes_left

    # Phase 1: pre-compound, baseline noise (T-60 to T-47)
    if m > 47:
        t = (60.0 - m) / 13.0   # 0 at T-60, 1 approaching T-47
        return round(lerp(0.05, 0.15, t), 4)

    # Phase 2: compound risk active, crosses CRITICAL (>0.80) immediately at T-47.
    # Uses power curve so it accelerates toward 1.0 in the final minutes.
    t = (47.0 - m) / 47.0       # 0 at T-47, 1 at T=0
    score = lerp(0.82, 1.0, t ** 0.65)
    return round(min(score, 1.0), 4)


def generate_incident_timeline():
    rows = []
    for i in range(120):   # 120 steps × 30s = 60 min
        minutes_left = 60 - i * 0.5   # 60.0, 59.5, ... 0.5, 0.0
        dt = START_DT + timedelta(seconds=i * 30)

        h2s  = round(noisy(h2s_value(minutes_left)), 2)
        co   = round(noisy(co_value(minutes_left)), 2)
        pres = round(noisy(pressure_value(minutes_left), 0.03), 1)
        temp = round(noisy(temperature_value(minutes_left), 0.02), 1)
        vib  = round(noisy(vibration_value(minutes_left), 0.05), 3)

        # Operational flags
        # Hot work permit issued at T-47 min → minutes_left <= 47
        hot_work = 1 if minutes_left <= 47 else 0

        # Shift changeover T-47 to T-35
        shift_chg = 1 if (35 <= minutes_left <= 47) else 0

        # Confined space entry T-45 onward
        confined = 1 if minutes_left <= 45 else 0

        # Crew count: ramps to 7 by T-45, stays until explosion
        if minutes_left > 47:
            crew = 0
        elif minutes_left > 45:
            crew = random.randint(2, 4)
        elif minutes_left > 20:
            crew = random.randint(6, 8)
        else:
            crew = random.randint(7, 8)

        # Single sensor alarm fires only when h2s_ppm > 50 (T-30 min)
        single_alarm = 1 if h2s > 50 else 0

        # Compound alarm fires at T-47 min (17 min before single sensor)
        compound_alarm = 1 if (hot_work and shift_chg and minutes_left <= 47) else 0
        # Keep compound alarm on after shift changeover ends too (risk persists)
        if minutes_left < 35 and hot_work:
            compound_alarm = 1

        risk = compound_risk(minutes_left, hot_work, shift_chg, confined)

        rows.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "minutes_to_explosion": round(minutes_left, 1),
            "zone": "C",
            "h2s_ppm": h2s,
            "co_ppm": co,
            "pressure_kpa": pres,
            "temperature_c": temp,
            "vibration_ms2": vib,
            "hot_work_permit_active": hot_work,
            "shift_changeover": shift_chg,
            "confined_space_entry": confined,
            "maintenance_crew_count": crew,
            "compound_risk_score": risk,
            "single_sensor_alarm": single_alarm,
            "compound_alarm": compound_alarm,
        })

    return rows


def generate_normal_ops(n=300):
    """Generate 300 rows of normal safe operations across several zones."""
    zones = ["A", "B", "D", "E"]
    rows = []
    base_dt = datetime(2025, 1, 10, 6, 0, 0)

    for i in range(n):
        dt = base_dt + timedelta(minutes=i * 10)
        zone = random.choice(zones)

        h2s  = round(noisy(random.uniform(2, 10)), 2)
        co   = round(noisy(random.uniform(10, 50)), 2)
        pres = round(noisy(random.uniform(760, 790), 0.03), 1)
        temp = round(noisy(random.uniform(375, 390), 0.02), 1)
        vib  = round(noisy(random.uniform(0.3, 1.5), 0.05), 3)

        risk = round(random.uniform(0.02, 0.15), 4)

        rows.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "minutes_to_explosion": None,
            "zone": zone,
            "h2s_ppm": h2s,
            "co_ppm": co,
            "pressure_kpa": pres,
            "temperature_c": temp,
            "vibration_ms2": vib,
            "hot_work_permit_active": 0,
            "shift_changeover": 0,
            "confined_space_entry": 0,
            "maintenance_crew_count": 0,
            "compound_risk_score": risk,
            "single_sensor_alarm": 0,
            "compound_alarm": 0,
        })

    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_permits_json():
    permits = [
        {
            "permit_id": "PTW-2025-0847",
            "type": "HOT_WORK",
            "zone": "C",
            "description": "Repair of coke oven battery door seals - welding required",
            "issued_at": "2025-01-18 14:13:00",
            "crew_size": 4,
            "supervisor": "Rajan Mehta",
            "valid_until": "2025-01-18 17:00:00",
            "gas_clearance_taken": True,
            "gas_clearance_time": "2025-01-18 14:10:00",
            "status": "ACTIVE"
        },
        {
            "permit_id": "PTW-2025-0848",
            "type": "CONFINED_SPACE_ENTRY",
            "zone": "C",
            "description": "Inspection of underfiring system in coke oven battery compartment 7",
            "issued_at": "2025-01-18 14:15:00",
            "crew_size": 3,
            "supervisor": "Priya Nair",
            "valid_until": "2025-01-18 16:00:00",
            "gas_clearance_taken": True,
            "gas_clearance_time": "2025-01-18 14:12:00",
            "status": "ACTIVE"
        }
    ]
    with open(os.path.join(DATA_DIR, "permits.json"), "w") as f:
        json.dump(permits, f, indent=2)


def write_shifts_json():
    shifts = {
        "current_shift": "B",
        "shift_start": "2025-01-18 14:00:00",
        "shift_end": "2025-01-18 22:00:00",
        "previous_shift": "A",
        "handover_start": "2025-01-18 14:00:00",
        "handover_end": "2025-01-18 14:30:00",
        "handover_complete": False,
        "zone_C_handover_briefing_done": False,
        "note": (
            "Shift A did not verbally brief Shift B about elevated H2S readings "
            "in Zone C battery compartments 6-7"
        )
    }
    with open(os.path.join(DATA_DIR, "shifts.json"), "w") as f:
        json.dump(shifts, f, indent=2)


if __name__ == "__main__":
    print("Generating Vizag Steel Plant incident simulation data...")

    incident_rows = generate_incident_timeline()
    normal_rows   = generate_normal_ops(300)

    timeline_path   = os.path.join(DATA_DIR, "vizag_timeline.csv")
    normal_ops_path = os.path.join(DATA_DIR, "normal_ops.csv")

    write_csv(timeline_path, incident_rows)
    write_csv(normal_ops_path, normal_rows)

    write_permits_json()
    write_shifts_json()

    print(f"  vizag_timeline.csv : {len(incident_rows)} rows  →  {timeline_path}")
    print(f"  normal_ops.csv     : {len(normal_rows)} rows  →  {normal_ops_path}")
    print(f"  permits.json       →  {os.path.join(DATA_DIR, 'permits.json')}")
    print(f"  shifts.json        →  {os.path.join(DATA_DIR, 'shifts.json')}")

    # Spot-check key moments
    print("\nSpot-check key moments (incident timeline):")
    for row in incident_rows:
        m = row["minutes_to_explosion"]
        if m in (60.0, 47.0, 35.0, 30.0, 20.0, 0.5):
            print(
                f"  T-{m:5.1f}min | H2S={row['h2s_ppm']:6.1f}ppm "
                f"| CO={row['co_ppm']:6.1f}ppm "
                f"| P={row['pressure_kpa']:6.1f}kPa "
                f"| risk={row['compound_risk_score']:.3f} "
                f"| s_alarm={row['single_sensor_alarm']} "
                f"| c_alarm={row['compound_alarm']}"
            )

    print("\nDone.")
