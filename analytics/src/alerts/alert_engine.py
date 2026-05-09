from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd


def build_alert_records(
    normalized_df: pd.DataFrame,
    features_df: pd.DataFrame,
    gps_df: Optional[pd.DataFrame] = None,
    alert_date: Optional[date] = None,
) -> List[dict]:
    """
    Règles MVP : seuils simples sur fatigue/stress journaliers et tendance (rolling 7).
    """
    target_date = alert_date or date.today()
    records: List[Dict[str, Any]] = []

    if normalized_df.empty:
        return records

    latest = normalized_df.sort_values("metric_date").groupby("player_id", as_index=False).tail(1)
    for row in latest.itertuples(index=False):
        pid = int(row.player_id)
        d = row.metric_date
        fatigue = float(row.fatigue) if row.fatigue is not None else None
        stress = float(row.stress) if row.stress is not None else None

        if fatigue is not None and fatigue >= 7.0:
            records.append(
                {
                    "player_id": pid,
                    "alert_date": target_date,
                    "alert_type": "elevated_fatigue_daily",
                    "severity": "high" if fatigue >= 8.0 else "moderate",
                    "evidence": {
                        "metric_date": str(d),
                        "fatigue": fatigue,
                        "rule": "fatigue >= 7",
                    },
                }
            )
        if (
            fatigue is not None
            and stress is not None
            and fatigue >= 6.0
            and stress >= 6.0
        ):
            records.append(
                {
                    "player_id": pid,
                    "alert_date": target_date,
                    "alert_type": "fatigue_stress_combo",
                    "severity": "moderate",
                    "evidence": {
                        "metric_date": str(d),
                        "fatigue": fatigue,
                        "stress": stress,
                        "rule": "fatigue >= 6 AND stress >= 6",
                    },
                }
            )

    if not features_df.empty:
        roll_f = features_df[features_df["metric_key"] == "fatigue_rolling_7"].copy()
        if not roll_f.empty:
            roll_latest = roll_f.sort_values("feature_date").groupby("player_id", as_index=False).tail(1)
            for row in roll_latest.itertuples(index=False):
                val = float(row.metric_value)
                if val >= 6.5:
                    records.append(
                        {
                            "player_id": int(row.player_id),
                            "alert_date": target_date,
                            "alert_type": "elevated_fatigue_trend_7d",
                            "severity": "moderate" if val < 7.5 else "high",
                            "evidence": {
                                "feature_date": str(row.feature_date),
                                "fatigue_rolling_7": val,
                                "rule": "rolling mean fatigue (7j) >= 6.5",
                            },
                        }
                    )

    if gps_df is not None and not gps_df.empty and not normalized_df.empty:
        gps_latest = gps_df.sort_values("metric_date").groupby("player_id", as_index=False).tail(1)
        wellness_latest = (
            normalized_df.sort_values("metric_date")
            .groupby("player_id", as_index=False)
            .tail(1)[["player_id", "metric_date", "fatigue"]]
        )
        joined = gps_latest.merge(wellness_latest, on="player_id", how="inner", suffixes=("_gps", "_wellness"))
        for row in joined.itertuples(index=False):
            gps_load = float(row.player_load_total) if row.player_load_total is not None else None
            fatigue = float(row.fatigue) if row.fatigue is not None else None
            if gps_load is not None and fatigue is not None and gps_load >= 800 and fatigue >= 6.0:
                records.append(
                    {
                        "player_id": int(row.player_id),
                        "alert_date": target_date,
                        "alert_type": "high_gps_load_with_fatigue",
                        "severity": "high" if fatigue >= 7.0 else "moderate",
                        "evidence": {
                            "gps_date": str(row.metric_date_gps),
                            "wellness_date": str(row.metric_date_wellness),
                            "player_load_total": gps_load,
                            "fatigue": fatigue,
                            "rule": "player_load_total >= 800 AND fatigue >= 6",
                        },
                    }
                )

    return records
