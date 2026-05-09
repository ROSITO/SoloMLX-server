import pandas as pd


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=["player_id", "feature_date", "metric_key", "metric_value"])


def _build_wellness_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_features()
    out = df.sort_values(["player_id", "metric_date"]).copy()
    out["fatigue_rolling_7"] = out.groupby("player_id")["fatigue"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    out["stress_rolling_7"] = out.groupby("player_id")["stress"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    melted = out.melt(
        id_vars=["player_id", "metric_date"],
        value_vars=["fatigue", "stress", "fatigue_rolling_7", "stress_rolling_7"],
        var_name="metric_key",
        value_name="metric_value",
    )
    return melted.rename(columns={"metric_date": "feature_date"}).dropna(subset=["metric_value"])


def _build_gps_features(gps_df: pd.DataFrame) -> pd.DataFrame:
    if gps_df is None or gps_df.empty:
        return _empty_features()
    out = gps_df.sort_values(["player_id", "metric_date"]).copy()
    for col in ["player_load_total", "hsr_distance", "acceleration_load"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["player_load_total_rolling_7"] = out.groupby("player_id")["player_load_total"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    out["hsr_distance_rolling_7"] = out.groupby("player_id")["hsr_distance"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    out["acceleration_load_rolling_7"] = out.groupby("player_id")["acceleration_load"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    melted = out.melt(
        id_vars=["player_id", "metric_date"],
        value_vars=[
            "player_load_total",
            "hsr_distance",
            "acceleration_load",
            "player_load_total_rolling_7",
            "hsr_distance_rolling_7",
            "acceleration_load_rolling_7",
        ],
        var_name="metric_key",
        value_name="metric_value",
    )
    return melted.rename(columns={"metric_date": "feature_date"}).dropna(subset=["metric_value"])


def build_daily_features(wellness_df: pd.DataFrame, gps_df: pd.DataFrame = None) -> pd.DataFrame:
    wellness_features = _build_wellness_features(wellness_df)
    gps_features = _build_gps_features(gps_df)
    if wellness_features.empty and gps_features.empty:
        return _empty_features()
    return pd.concat([wellness_features, gps_features], ignore_index=True)
