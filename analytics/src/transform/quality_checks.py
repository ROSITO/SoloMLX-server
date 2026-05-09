import pandas as pd


def validate_input(df: pd.DataFrame) -> None:
    required = {"player_id", "metric_date", "fatigue", "stress"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
