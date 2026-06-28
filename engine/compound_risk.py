"""
Compound Risk Engine for Vizag Industrial Safety AI.

Combines:
  1. Traditional single-sensor threshold alarms
  2. IsolationForest ML anomaly detection
  3. Compound rule evaluation (the innovation layer)

into a unified risk score and structured alert payload.
"""

import datetime
import joblib
import numpy as np

from engine.rule_engine import RuleEngine, CompoundRule


class CompoundRiskEngine:
    """
    Main compound risk assessment engine.

    Wraps ML anomaly detection (IsolationForest) and deterministic compound
    rules into a single `assess()` call that returns a structured JSON-serialisable
    assessment dict.
    """

    # ------------------------------------------------------------------ #
    # Single-sensor thresholds — what traditional systems check alone     #
    # ------------------------------------------------------------------ #
    SINGLE_THRESHOLDS = {
        "h2s_ppm":       50.0,
        "co_ppm":        200.0,
        "pressure_kpa":  850.0,
        "temperature_c": 420.0,
        "vibration_ms2": 6.0,
    }

    # ------------------------------------------------------------------ #
    # ML feature order (must match training)                              #
    # ------------------------------------------------------------------ #
    FEATURE_KEYS = ["h2s_ppm", "co_ppm", "pressure_kpa", "temperature_c", "vibration_ms2"]

    def __init__(
        self,
        model_path: str = "/Users/uditjain/Desktop/vizag_safety/models/iso_forest.pkl",
    ):
        self.rule_engine   = RuleEngine()
        self.model         = None
        self.scaler        = None
        self.model_path    = model_path
        self.sensor_history: list = []   # rolling window of last 20 readings
        self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading                                                       #
    # ------------------------------------------------------------------ #
    def _load_model(self) -> None:
        """
        Load the IsolationForest artifact saved by models/train_anomaly.py.
        The artifact is a dict {"model": ..., "scaler": ..., "features": [...]}.
        Silently degrades if the file is absent — ML score returns 0.0.
        """
        try:
            artifact = joblib.load(self.model_path)
            if isinstance(artifact, dict):
                self.model  = artifact.get("model")
                self.scaler = artifact.get("scaler")
            else:
                # Legacy: plain model object
                self.model  = artifact
                self.scaler = None
        except Exception:
            self.model  = None
            self.scaler = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def assess(
        self,
        sensor_row: dict,
        permits:    list,
        shift_info: dict,
    ) -> dict:
        """
        Main assessment function.

        Parameters
        ----------
        sensor_row : dict
            Required keys: h2s_ppm, co_ppm, pressure_kpa, temperature_c,
            vibration_ms2.
            Optional keys: zone (str), reading_history (list[dict]).
        permits : list[dict]
            Each item: {"type": str, "zone": str, "status": str}
        shift_info : dict
            Keys: handover_complete (bool), shift_start_baseline (dict)

        Returns
        -------
        dict with keys:
            compound_risk_score          : float 0.0–1.0
            risk_level                   : "NORMAL"|"ELEVATED"|"HIGH"|"CRITICAL"
            triggered_rules              : list[dict]  (CompoundRule.__dict__)
            ml_anomaly_score             : float
            single_sensor_alarms         : list[dict]
            compound_alarms_count        : int
            primary_alert                : str or None
            oisd_citations               : list[str]
            recommendation               : str
            would_traditional_alert      : bool
            compound_detects_before_traditional : bool
            timestamp                    : str (ISO-8601)
        """
        # ---- 0. Maintain rolling history --------------------------------
        self.sensor_history.append(sensor_row)
        if len(self.sensor_history) > 20:
            self.sensor_history.pop(0)

        # Inject history into sensor_row so rule engine can see it
        if "reading_history" not in sensor_row:
            sensor_row = dict(sensor_row)
            sensor_row["reading_history"] = self.sensor_history[:-1]  # exclude current

        # ---- 1. Single-sensor threshold check ---------------------------
        single_alarms = self._check_single_sensors(sensor_row)

        # ---- 2. ML anomaly score ----------------------------------------
        ml_score = self._ml_anomaly(sensor_row)

        # ---- 3. Compound rule evaluation --------------------------------
        triggered_rules: list[CompoundRule] = self.rule_engine.evaluate(
            sensor_row, permits, shift_info
        )

        # ---- 4. Compound risk score -------------------------------------
        compound_score = self._calculate_compound_score(
            ml_score, triggered_rules, sensor_row
        )

        # ---- 5. Risk level ----------------------------------------------
        risk_level = self._score_to_level(compound_score)

        # ---- 6. Collate human-readable outputs --------------------------
        primary_alert: str | None = None
        recommendations: list[str] = []
        oisd_citations:  list[str] = []

        for rule in triggered_rules:
            if rule.oisd_ref:
                oisd_citations.append(rule.oisd_ref)
            if rule.recommendation:
                recommendations.append(rule.recommendation)
            if rule.severity == "CRITICAL" and primary_alert is None:
                primary_alert = (
                    f"COMPOUND RISK: {rule.name} — {rule.description}"
                )

        # Fall back to HIGH if no CRITICAL
        if primary_alert is None and triggered_rules:
            r = triggered_rules[0]
            primary_alert = f"COMPOUND RISK: {r.name} — {r.description}"

        return {
            "compound_risk_score": round(compound_score, 3),
            "risk_level": risk_level,
            "triggered_rules": [r.__dict__ for r in triggered_rules],
            "ml_anomaly_score": round(float(ml_score), 3),
            "single_sensor_alarms": single_alarms,
            "compound_alarms_count": len(triggered_rules),
            "primary_alert": primary_alert,
            "oisd_citations": list(set(oisd_citations)),
            "recommendation": (
                recommendations[0] if recommendations else "Continue monitoring."
            ),
            "would_traditional_alert": len(single_alarms) > 0,
            "compound_detects_before_traditional": (
                len(triggered_rules) > 0 and len(single_alarms) == 0
            ),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _check_single_sensors(self, row: dict) -> list:
        """Return a list of individual sensor alarms (threshold crossings)."""
        alarms = []
        for sensor, threshold in self.SINGLE_THRESHOLDS.items():
            value = row.get(sensor, 0)
            if value > threshold:
                alarms.append({
                    "sensor":    sensor,
                    "value":     value,
                    "threshold": threshold,
                    "excess_pct": round((value - threshold) / threshold * 100, 1),
                })
        return alarms

    def _ml_anomaly(self, row: dict) -> float:
        """
        Score the current reading with the IsolationForest.

        IsolationForest.score_samples() returns negative values; more negative
        means more anomalous.  We normalise to [0, 1] where 1 = most anomalous.

        Returns 0.0 if the model is not loaded.
        """
        if self.model is None:
            return 0.0

        features = np.array(
            [[row.get(k, 0) for k in self.FEATURE_KEYS]], dtype=float
        )

        try:
            if self.scaler is not None:
                features = self.scaler.transform(features)
            raw_score = self.model.score_samples(features)[0]
            # Typical range of score_samples: roughly -0.5 (anomaly) to +0.1 (normal)
            # Map: score <= -0.5 → 1.0 (anomalous), score >= 0.1 → 0.0 (normal)
            normalised = (raw_score - 0.1) / (-0.5 - 0.1)   # linear map
            normalised = float(np.clip(normalised, 0.0, 1.0))
        except Exception:
            normalised = 0.0

        return normalised

    def _calculate_compound_score(
        self,
        ml_score:        float,
        triggered_rules: list,
        row:             dict,
    ) -> float:
        """
        Weighted combination of:
          - ML anomaly score (30%)
          - Rule severity weights (cumulative, capped)
          - Sensor proximity to thresholds (bonus factor)
        """
        # ML contribution (30% weight)
        base = ml_score * 0.30

        # Rule severity contributions
        rule_score = 0.0
        for rule in triggered_rules:
            if rule.severity == "CRITICAL":
                rule_score += 0.45
            elif rule.severity == "HIGH":
                rule_score += 0.25
            elif rule.severity == "MEDIUM":
                rule_score += 0.10

        # Sensor proximity factor — how close each sensor is to its threshold
        # Adds up to ~0.4 max across 4 sensors
        proximity = 0.0
        for sensor, threshold in [
            ("h2s_ppm",       self.SINGLE_THRESHOLDS["h2s_ppm"]),
            ("co_ppm",        self.SINGLE_THRESHOLDS["co_ppm"]),
            ("pressure_kpa",  self.SINGLE_THRESHOLDS["pressure_kpa"]),
            ("temperature_c", self.SINGLE_THRESHOLDS["temperature_c"]),
        ]:
            val = row.get(sensor, 0)
            # Only contribute when ≥50% of threshold
            proximity += max(0.0, (val / threshold) - 0.5) * 0.10

        return min(1.0, base + rule_score + proximity)

    def _score_to_level(self, score: float) -> str:
        """Map 0–1 compound risk score to a categorical level."""
        if score >= 0.75:
            return "CRITICAL"
        if score >= 0.50:
            return "HIGH"
        if score >= 0.25:
            return "ELEVATED"
        return "NORMAL"
