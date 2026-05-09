import mysql.connector

from analytics.config.settings import get_settings


def get_connection():
    s = get_settings()
    return mysql.connector.connect(
        host=s.db_host,
        port=s.db_port,
        user=s.db_user,
        password=s.db_pass,
        database=s.db_name,
    )
