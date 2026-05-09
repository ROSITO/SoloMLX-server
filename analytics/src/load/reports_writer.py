from __future__ import annotations

import json
from typing import Any, Dict, List

from analytics.config.settings import get_settings
from analytics.src.common.db import get_connection


def upsert_reports(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0

    s = get_settings()
    conn = get_connection()
    n = 0
    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO analytics_reports
                (player_id, report_date, period_label, summary_text, summary_json, calc_version)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                summary_text = VALUES(summary_text),
                summary_json = VALUES(summary_json),
                calc_version = VALUES(calc_version),
                updated_at = CURRENT_TIMESTAMP
        """
        for r in records:
            cur.execute(
                sql,
                (
                    int(r["player_id"]),
                    r["report_date"],
                    str(r["period_label"]),
                    str(r["summary_text"]),
                    json.dumps(r.get("summary_json") or {}, ensure_ascii=False),
                    s.calc_version,
                ),
            )
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()
