from __future__ import annotations

import json
from typing import Any, Dict, List

from analytics.src.common.db import get_connection


def upsert_alerts(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0

    conn = get_connection()
    n = 0
    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO analytics_alerts
                (player_id, alert_date, alert_type, severity, evidence_json, status)
            VALUES (%s, %s, %s, %s, %s, 'open')
            ON DUPLICATE KEY UPDATE
                severity = VALUES(severity),
                evidence_json = VALUES(evidence_json),
                status = 'open',
                updated_at = CURRENT_TIMESTAMP
        """
        for r in records:
            cur.execute(
                sql,
                (
                    int(r["player_id"]),
                    r["alert_date"],
                    str(r["alert_type"]),
                    str(r["severity"]),
                    json.dumps(r.get("evidence") or {}, ensure_ascii=False),
                ),
            )
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()
