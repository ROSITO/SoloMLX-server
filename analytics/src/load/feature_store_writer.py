from analytics.config.settings import get_settings
from analytics.src.common.db import get_connection


def write_daily_features(df, source: str = "monitoring"):
    if df.empty:
        return 0

    s = get_settings()
    conn = get_connection()
    inserted = 0
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO analytics_features_daily
                (player_id, feature_date, metric_key, metric_value, source, calc_version)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                metric_value = VALUES(metric_value),
                source = VALUES(source),
                calc_version = VALUES(calc_version),
                updated_at = CURRENT_TIMESTAMP
        """
        for row in df.itertuples(index=False):
            cursor.execute(
                query,
                (
                    int(row.player_id),
                    row.feature_date,
                    str(row.metric_key),
                    float(row.metric_value),
                    source,
                    s.calc_version,
                ),
            )
            inserted += 1
        conn.commit()
        return inserted
    finally:
        conn.close()
