from __future__ import annotations

from typing import Dict, List

from analytics.src.common.db import get_connection


def build_player_documents(days: int = 90) -> List[Dict]:
    """
    Build one compact RAG document per player from analytics features + latest report.
    """
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT DISTINCT player_id
            FROM analytics_features_daily
            WHERE feature_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
            ORDER BY player_id
            """,
            (days,),
        )
        player_ids = [row["player_id"] for row in cur.fetchall()]
        names_by_id: Dict[int, str] = {}
        if player_ids:
            placeholders = ",".join(["%s"] * len(player_ids))
            cur.execute(
                f"""
                SELECT id_sportif AS sid, prenom_sportif, nom_sportif
                FROM sportif
                WHERE id_sportif IN ({placeholders})
                """,
                tuple(player_ids),
            )
            for row in cur.fetchall():
                sid = int(row["sid"])
                pre = (row.get("prenom_sportif") or "").strip()
                nom = (row.get("nom_sportif") or "").strip()
                label = f"{pre} {nom}".strip() or f"Joueur {sid}"
                names_by_id[sid] = label

        docs: List[Dict] = []

        metric_keys = (
            "fatigue",
            "stress",
            "fatigue_rolling_7",
            "player_load_total",
            "player_load_total_rolling_7",
            "hsr_distance",
            "hsr_distance_rolling_7",
        )

        for player_id in player_ids:
            cur.execute(
                """
                SELECT metric_key, AVG(metric_value) AS avg_value, MAX(feature_date) AS last_date
                FROM analytics_features_daily
                WHERE player_id = %s
                  AND feature_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                  AND metric_key IN (%s, %s, %s, %s, %s, %s, %s)
                GROUP BY metric_key
                ORDER BY metric_key
                """,
                (player_id, days, *metric_keys),
            )
            rows = cur.fetchall()
            metrics = {r["metric_key"]: float(r["avg_value"]) for r in rows if r["avg_value"] is not None}
            last_date = max((r["last_date"] for r in rows if r["last_date"] is not None), default=None)

            cur.execute(
                """
                SELECT summary_text, report_date
                FROM analytics_reports
                WHERE player_id = %s
                ORDER BY report_date DESC, id DESC
                LIMIT 1
                """,
                (player_id,),
            )
            report = cur.fetchone()
            summary = report["summary_text"] if report else "Aucun rapport généré."
            display = names_by_id.get(int(player_id), f"Joueur {player_id}")

            text = (
                f"Joueur: {display} (player_id={player_id}). "
                f"Période: {days} jours. "
                f"Dernière date métrique: {last_date}. "
                f"fatigue={metrics.get('fatigue')} stress={metrics.get('stress')} "
                f"fatigue_rolling_7={metrics.get('fatigue_rolling_7')} "
                f"player_load_total={metrics.get('player_load_total')} "
                f"player_load_total_rolling_7={metrics.get('player_load_total_rolling_7')} "
                f"hsr_distance={metrics.get('hsr_distance')} "
                f"hsr_distance_rolling_7={metrics.get('hsr_distance_rolling_7')}. "
                f"Rapport: {summary}"
            )
            docs.append(
                {
                    "doc_id": f"player:{player_id}",
                    "player_id": int(player_id),
                    "player_display": display,
                    "days": int(days),
                    "text": text,
                    "metrics": metrics,
                    "report_summary": summary,
                }
            )
        return docs
    finally:
        conn.close()
