from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd


def build_report_records(
    normalized_df: pd.DataFrame,
    features_df: pd.DataFrame,
    gps_df: Optional[pd.DataFrame] = None,
    report_date: Optional[date] = None,
    lookback_days: int = 7,
    period_label: str = "7d",
) -> List[Dict[str, Any]]:
    """
    Résumé texte MVP par joueur sur les N derniers jours de wellness.
    """
    target = report_date or date.today()
    records: List[Dict[str, Any]] = []

    if normalized_df.empty:
        return records

    df = normalized_df.copy()
    df["metric_date"] = pd.to_datetime(df["metric_date"]).dt.date

    last_roll = {}
    if not features_df.empty:
        roll = features_df[features_df["metric_key"] == "fatigue_rolling_7"].copy()
        if not roll.empty:
            roll["feature_date"] = pd.to_datetime(roll["feature_date"]).dt.date
            roll_latest = roll.sort_values("feature_date").groupby("player_id").tail(1)
            for row in roll_latest.itertuples(index=False):
                last_roll[int(row.player_id)] = float(row.metric_value)

    for pid, grp_all in df.groupby("player_id"):
        grp = grp_all.sort_values("metric_date").tail(lookback_days)
        player_target = grp["metric_date"].max() if not grp.empty else target
        fatigue_m = float(grp["fatigue"].mean())
        stress_m = float(grp["stress"].mean())
        roll = last_roll.get(int(pid))
        gps_mean = None
        hsr_mean = None
        if gps_df is not None and not gps_df.empty:
            gdf = gps_df.copy()
            gdf["metric_date"] = pd.to_datetime(gdf["metric_date"]).dt.date
            ggrp = gdf[gdf["player_id"] == pid].sort_values("metric_date").tail(lookback_days)
            if not ggrp.empty:
                gps_mean = float(pd.to_numeric(ggrp["player_load_total"], errors="coerce").mean())
                hsr_mean = float(pd.to_numeric(ggrp["hsr_distance"], errors="coerce").mean())
        parts = [
            f"Période {period_label} jusqu'au {player_target}: fatigue moyenne {fatigue_m:.2f}/10, stress moyen {stress_m:.2f}/10."
        ]
        if gps_mean is not None:
            parts.append(f"Charge GPS moyenne (player load): {gps_mean:.2f}.")
        if hsr_mean is not None:
            parts.append(f"Distance HSR moyenne: {hsr_mean:.2f}.")
        if roll is not None:
            parts.append(f"Moyenne mobile fatigue (7 j) la plus récente: {roll:.2f}.")
        if fatigue_m >= 6.5:
            parts.append("Signaux de fatigue élevée: surveiller charge et récupération.")
        summary = " ".join(parts)
        summary_json = {
            "fatigue_mean_7d": fatigue_m,
            "stress_mean_7d": stress_m,
            "player_load_mean_7d": gps_mean,
            "hsr_distance_mean_7d": hsr_mean,
            "fatigue_rolling_7_latest": roll,
            "days_with_data": int(len(grp)),
        }
        records.append(
            {
                "player_id": int(pid),
                "report_date": player_target,
                "period_label": period_label,
                "summary_text": summary,
                "summary_json": summary_json,
            }
        )

    return records
