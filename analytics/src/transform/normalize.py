import pandas as pd


def normalize_wellness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["metric_date"] = pd.to_datetime(out["metric_date"]).dt.date
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    for col in ("fatigue", "stress"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["player_id", "metric_date"])
    out = out.sort_values(["player_id", "metric_date"])
    out["fatigue"] = out.groupby("player_id")["fatigue"].ffill()
    out["stress"] = out.groupby("player_id")["stress"].ffill()
    out = out.dropna(subset=["fatigue", "stress"], how="any")
    out["player_id"] = out["player_id"].astype(int)
    return out
