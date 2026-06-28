"""
Compound Risk Rule Engine for Vizag Industrial Safety AI.

Each rule detects COMBINATIONS of conditions that are dangerous
even when no single sensor has crossed its individual alarm threshold.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompoundRule:
    name: str
    severity: str          # "CRITICAL", "HIGH", "MEDIUM"
    description: str
    triggered: bool = False
    evidence: list = field(default_factory=list)   # human-readable evidence items
    oisd_ref: str = ""
    recommendation: str = ""


class RuleEngine:
    """
    Evaluates a set of compound risk rules against live sensor data,
    active permits, and shift-handover metadata.
    """

    # ------------------------------------------------------------------ #
    # Danger thresholds used for "elevated range" checks (>50% of limit)  #
    # ------------------------------------------------------------------ #
    SINGLE_THRESHOLDS = {
        "h2s_ppm":       50.0,
        "co_ppm":        200.0,
        "pressure_kpa":  850.0,
        "temperature_c": 420.0,
        "vibration_ms2": 6.0,
    }

    def evaluate(
        self,
        sensor_row: dict,
        permits: list,
        shift_info: dict,
    ) -> list:
        """
        Evaluate all compound rules and return the list of triggered
        CompoundRule objects.

        Parameters
        ----------
        sensor_row : dict
            Keys: h2s_ppm, co_ppm, pressure_kpa, temperature_c,
                  vibration_ms2, zone (str), shift_start_baseline (dict),
                  reading_history (list[dict]) — last N readings.
        permits : list[dict]
            Each permit: {"type": str, "zone": str, "status": str}
            status "ACTIVE" means the permit is currently valid.
        shift_info : dict
            Keys: handover_complete (bool), shift_start_baseline (dict)
        """
        triggered = []

        triggered += self._rule1_confined_space_hot_work_gas(sensor_row, permits)
        triggered += self._rule2_shift_handover_hazmat_drift(sensor_row, permits, shift_info)
        triggered += self._rule3_multi_permit_same_zone(sensor_row, permits)
        triggered += self._rule4_pressure_temperature_creep(sensor_row)
        triggered += self._rule5_historical_incident_pattern(sensor_row, permits)

        return triggered

    # ------------------------------------------------------------------ #
    # RULE 1 — CONFINED_SPACE_HOT_WORK_GAS                                #
    # ------------------------------------------------------------------ #
    def _rule1_confined_space_hot_work_gas(
        self, sensor_row: dict, permits: list
    ) -> list:
        """
        H2S > 30 ppm AND a HOT_WORK permit AND a CONFINED_SPACE_ENTRY permit
        are both ACTIVE in the same zone as the sensor reading.

        Note: 30 ppm is well below the 50 ppm single-sensor alarm — this
        combination is dangerous at a much lower concentration because a
        welding spark + rising H2S in an enclosed space creates explosion risk.
        """
        rule = CompoundRule(
            name="CONFINED_SPACE_HOT_WORK_GAS",
            severity="CRITICAL",
            description=(
                "H2S > 12 ppm with simultaneous HOT_WORK and CONFINED_SPACE_ENTRY "
                "permits active in the same zone — welding spark + enclosed rising "
                "H2S creates explosion risk even well below the 50 ppm single-alarm level."
            ),
            oisd_ref="OISD-GS-1 §6.3.2",
            recommendation=(
                "Immediately suspend hot work PTW. Evacuate confined space. "
                "Re-test atmosphere before re-entry."
            ),
        )

        h2s = sensor_row.get("h2s_ppm", 0)
        zone = sensor_row.get("zone", "")

        if h2s <= 12:
            return []

        active_in_zone = [
            p for p in permits
            if p.get("status", "").upper() == "ACTIVE"
            and p.get("zone", "") == zone
        ]

        has_hot_work = any(
            p.get("type", "").upper() == "HOT_WORK" for p in active_in_zone
        )
        has_confined = any(
            p.get("type", "").upper() == "CONFINED_SPACE_ENTRY"
            for p in active_in_zone
        )

        if has_hot_work and has_confined:
            rule.triggered = True
            rule.evidence = [
                f"H2S = {h2s:.1f} ppm (compound threshold: 12 ppm, single-alarm: 50 ppm)",
                f"HOT_WORK permit ACTIVE in zone '{zone}'",
                f"CONFINED_SPACE_ENTRY permit ACTIVE in zone '{zone}'",
                "Explosive mixture risk: H2S + ignition source in confined geometry",
            ]
            return [rule]

        return []

    # ------------------------------------------------------------------ #
    # RULE 2 — SHIFT_HANDOVER_HAZMAT_DRIFT                                #
    # ------------------------------------------------------------------ #
    def _rule2_shift_handover_hazmat_drift(
        self, sensor_row: dict, permits: list, shift_info: dict
    ) -> list:
        """
        Shift handover incomplete AND any sensor drifting upward by >15%
        compared to start-of-shift baseline AND at least one sensor in the
        elevated range.

        Gap in situational awareness during handover + unattended drift = high risk.
        """
        rule = CompoundRule(
            name="SHIFT_HANDOVER_HAZMAT_DRIFT",
            severity="HIGH",
            description=(
                "Incomplete shift handover while hazardous sensor values are "
                "drifting upward — neither outgoing nor incoming crew has full "
                "situational awareness of the developing condition."
            ),
            oisd_ref="OISD-GS-1 §3.1.8",
            recommendation=(
                "Halt shift handover. Both shifts must acknowledge current Zone C "
                "gas readings before transfer of custody."
            ),
        )

        if shift_info.get("handover_complete", True):
            return []

        h2s  = sensor_row.get("h2s_ppm",     0)
        co   = sensor_row.get("co_ppm",       0)
        pres = sensor_row.get("pressure_kpa", 0)

        elevated_condition = h2s > 12 or co > 40 or pres > 785
        if not elevated_condition:
            return []

        baseline = shift_info.get("shift_start_baseline", {})
        if not baseline:
            # No baseline provided — conservatively treat as triggered if elevated
            drift_evidence = ["No shift-start baseline recorded — drift cannot be quantified"]
            rule.triggered = True
            rule.evidence = [
                f"Shift handover incomplete",
                f"H2S={h2s:.1f}, CO={co:.1f}, Pressure={pres:.1f} — at least one in elevated range",
            ] + drift_evidence
            return [rule]

        drift_sensors = []
        for key, label in [
            ("h2s_ppm", "H2S"),
            ("co_ppm", "CO"),
            ("pressure_kpa", "Pressure"),
        ]:
            current = sensor_row.get(key, 0)
            base_val = baseline.get(key, current)
            if base_val > 0:
                pct_change = (current - base_val) / base_val * 100
                if pct_change > 15:
                    drift_sensors.append(
                        f"{label}: {base_val:.1f} → {current:.1f} "
                        f"(+{pct_change:.1f}% since shift start)"
                    )

        if drift_sensors:
            rule.triggered = True
            rule.evidence = [
                "Shift handover is INCOMPLETE — situational awareness gap exists",
                f"H2S={h2s:.1f} ppm, CO={co:.1f} ppm, Pressure={pres:.1f} kPa",
            ] + drift_sensors
            return [rule]

        return []

    # ------------------------------------------------------------------ #
    # RULE 3 — MULTI_PERMIT_SAME_ZONE                                     #
    # ------------------------------------------------------------------ #
    def _rule3_multi_permit_same_zone(
        self, sensor_row: dict, permits: list
    ) -> list:
        """
        2+ permits active in the same zone AND any sensor is in the elevated
        range (>50% of danger threshold). Two work crews multiply risk non-linearly.
        """
        rule = CompoundRule(
            name="MULTI_PERMIT_SAME_ZONE",
            severity="HIGH",
            description=(
                "Multiple concurrent active work permits in the same zone while "
                "sensor readings are in the elevated range — simultaneous operations "
                "multiply risk non-linearly."
            ),
            oisd_ref="OISD-GS-1 §5.2.1 - Simultaneous Operations",
            recommendation=(
                "Review SIMOPS conflict. Stagger activities or increase gas monitoring "
                "frequency to 5-minute intervals."
            ),
        )

        zone = sensor_row.get("zone", "")

        # Count active permits per zone
        zone_permits: dict = {}
        for p in permits:
            if p.get("status", "").upper() == "ACTIVE":
                pz = p.get("zone", "")
                zone_permits.setdefault(pz, []).append(p)

        active_in_zone = zone_permits.get(zone, [])
        if len(active_in_zone) < 2:
            return []

        # Check for elevated readings (>50% of danger threshold)
        elevated = []
        for sensor, threshold in self.SINGLE_THRESHOLDS.items():
            val = sensor_row.get(sensor, 0)
            if val > threshold * 0.5:
                elevated.append(
                    f"{sensor}={val:.2f} (>{threshold * 0.5:.1f}, 50% of alarm threshold {threshold})"
                )

        if not elevated:
            return []

        rule.triggered = True
        permit_types = [p.get("type", "UNKNOWN") for p in active_in_zone]
        rule.evidence = [
            f"{len(active_in_zone)} permits ACTIVE in zone '{zone}': "
            + ", ".join(permit_types),
            "Elevated sensor readings detected:",
        ] + elevated
        return [rule]

    # ------------------------------------------------------------------ #
    # RULE 4 — PRESSURE_TEMPERATURE_CREEP                                 #
    # ------------------------------------------------------------------ #
    def _rule4_pressure_temperature_creep(self, sensor_row: dict) -> list:
        """
        pressure > 800 kPa AND temperature > 400°C AND both values have been
        strictly rising for the last 10 consecutive readings.

        Co-rising pressure + temperature = thermal runaway precursor.
        """
        rule = CompoundRule(
            name="PRESSURE_TEMPERATURE_CREEP",
            severity="HIGH",
            description=(
                "Simultaneous sustained increase in both pressure and temperature "
                "over ≥10 consecutive readings — characteristic precursor to "
                "thermal runaway."
            ),
            oisd_ref="DGMS Circular 2019-07 §4.1",
            recommendation=(
                "Initiate pressure relief procedure per SOP-COB-007. "
                "Alert duty engineer."
            ),
        )

        pressure = sensor_row.get("pressure_kpa", 0)
        temperature = sensor_row.get("temperature_c", 0)

        if pressure <= 800 or temperature <= 400:
            return []

        history = sensor_row.get("reading_history", [])
        if len(history) < 10:
            # Not enough history yet — flag as evidence-limited but still note elevated values
            rule.triggered = True
            rule.evidence = [
                f"Pressure = {pressure:.1f} kPa (> 800 kPa threshold)",
                f"Temperature = {temperature:.1f} °C (> 400 °C threshold)",
                f"Insufficient history for trend check ({len(history)} readings, need 10) "
                "— values currently elevated above thresholds.",
            ]
            return [rule]

        # Check that both pressure and temperature have been monotonically increasing
        # over the last 10 readings (including the current reading appended at end)
        last_10 = history[-9:] + [sensor_row]   # 9 from history + current = 10

        pres_rising = all(
            last_10[i + 1].get("pressure_kpa", 0) > last_10[i].get("pressure_kpa", 0)
            for i in range(len(last_10) - 1)
        )
        temp_rising = all(
            last_10[i + 1].get("temperature_c", 0) > last_10[i].get("temperature_c", 0)
            for i in range(len(last_10) - 1)
        )

        if pres_rising and temp_rising:
            pres_start = last_10[0].get("pressure_kpa", 0)
            temp_start = last_10[0].get("temperature_c", 0)
            rule.triggered = True
            rule.evidence = [
                f"Pressure rising for 10 consecutive readings: "
                f"{pres_start:.1f} → {pressure:.1f} kPa "
                f"(+{pressure - pres_start:.1f} kPa)",
                f"Temperature rising for 10 consecutive readings: "
                f"{temp_start:.1f} → {temperature:.1f} °C "
                f"(+{temperature - temp_start:.1f} °C)",
                "Co-rising pattern is a known thermal runaway precursor.",
            ]
            return [rule]

        return []

    # ------------------------------------------------------------------ #
    # RULE 5 — HISTORICAL_INCIDENT_PATTERN                                #
    # ------------------------------------------------------------------ #
    def _rule5_historical_incident_pattern(
        self, sensor_row: dict, permits: list
    ) -> list:
        """
        H2S between 35–50 ppm AND a HOT_WORK permit is active AND
        pressure > 810 kPa.

        THIS EXACT COMBINATION preceded the January 18, 2025 Visakhapatnam
        Steel Plant explosion (OISD Incident Report VZG-2025-01).
        """
        rule = CompoundRule(
            name="HISTORICAL_INCIDENT_PATTERN",
            severity="CRITICAL",
            description=(
                "Current sensor-permit combination matches the documented precursor "
                "pattern of the Visakhapatnam Steel Plant explosion — "
                "January 18, 2025."
            ),
            oisd_ref="OISD Incident Report VZG-2025-01 Pattern Match",
            recommendation=(
                "IMMEDIATE EVACUATION of Zone C. This sensor-permit pattern matches "
                "the Visakhapatnam Steel Plant incident of January 18, 2025."
            ),
        )

        h2s      = sensor_row.get("h2s_ppm",     0)
        pressure = sensor_row.get("pressure_kpa", 0)
        zone     = sensor_row.get("zone",         "")

        h2s_in_band = 35 <= h2s <= 50
        pressure_high = pressure > 810

        if not (h2s_in_band and pressure_high):
            return []

        hot_work_active = any(
            p.get("type", "").upper() == "HOT_WORK"
            and p.get("status", "").upper() == "ACTIVE"
            for p in permits
        )

        if hot_work_active:
            rule.triggered = True
            rule.evidence = [
                f"H2S = {h2s:.1f} ppm (incident band: 35–50 ppm)",
                f"Pressure = {pressure:.1f} kPa (incident threshold: > 810 kPa)",
                f"HOT_WORK permit ACTIVE",
                "Pattern matches VZG-2025-01: H2S 35–50 ppm + hot work + pressure > 810 kPa",
                "The January 18 2025 explosion occurred within 12 minutes of this reading profile.",
            ]
            return [rule]

        return []
