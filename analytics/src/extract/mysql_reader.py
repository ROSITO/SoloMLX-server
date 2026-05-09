import pandas as pd
import re

from analytics.config.settings import get_settings
from analytics.src.common.db import get_connection


def read_wellness_last_days(days: int = 30) -> pd.DataFrame:
    """
    Wellness fatigue/stress depuis sport_est_critere + critere (groupe wellness),
    aligné sur data_access_functions.getWellnessData.
    """
    s = get_settings()
    team_clause = ""
    params: list = [days]
    if s.critere_equipe:
        team_clause = " AND c.equipe = %s"
        params.append(int(s.critere_equipe))

    query = f"""
        SELECT
            sec.id_sportif AS player_id,
            DATE(sec.`update`) AS metric_date,
            MAX(CASE WHEN c.nom_critere = 'fatigue' OR sec.id_critrere = 'fatigue' THEN sec.valeur_critere END) AS fatigue,
            MAX(CASE WHEN c.nom_critere = 'stress' OR sec.id_critrere = 'stress' THEN sec.valeur_critere END) AS stress
        FROM sport_est_critere sec
        LEFT JOIN critere c ON sec.id_critrere = c.id_critere
        WHERE (
                (c.groupe = 'wellness' AND c.nom_critere IN ('fatigue', 'stress'))
                OR sec.id_critrere IN ('fatigue', 'stress')
              )
          AND sec.`update` >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
          {team_clause}
        GROUP BY sec.id_sportif, DATE(sec.`update`)
        ORDER BY player_id, metric_date
    """
    conn = get_connection()
    try:
        return pd.read_sql(query, conn, params=params)
    finally:
        conn.close()


def read_gps_last_days(days: int = 30) -> pd.DataFrame:
    """
    GPS depuis la table équipe (ex: GPS_18), reliée au joueur via
    UPPER(prenom + nom), comme dans les APIs historiques.
    """
    s = get_settings()
    if not re.match(r"^[A-Za-z0-9_]+$", s.gps_table):
        raise ValueError(f"Invalid GPS table name: {s.gps_table}")

    query = f"""
        SELECT
            sp.id_sportif AS player_id,
            DATE(g.date) AS metric_date,
            g.total_player_load AS player_load_total,
            g.hsr_15_8_20_km_h AS hsr_distance,
            g.total_acceleration_load AS acceleration_load
        FROM {s.gps_table} g
        INNER JOIN sportif sp
            ON UPPER(CONCAT(sp.prenom_sportif, ' ', sp.nom_sportif)) = g.player_name
        WHERE g.date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
          AND sp.id_equipe = %s
        ORDER BY player_id, metric_date
    """
    conn = get_connection()
    try:
        return pd.read_sql(query, conn, params=[days, s.analytics_equipe_id])
    finally:
        conn.close()
