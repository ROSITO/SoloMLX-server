from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os

from dotenv import load_dotenv


def _load_env_files() -> None:
    root = Path(__file__).resolve().parents[2]
    for rel in (".env", "deploy-docker/.env.docker"):
        p = root / rel
        if p.is_file():
            load_dotenv(p, override=False)


_load_env_files()


@dataclass
class Settings:
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(
        default_factory=lambda: int(os.getenv("DB_PORT", os.getenv("MYSQL_PORT", "3306")))
    )
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "root"))
    db_pass: str = field(default_factory=lambda: os.getenv("DB_PASS", ""))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "xlencesoncapp"))
    calc_version: str = field(default_factory=lambda: os.getenv("ANALYTICS_CALC_VERSION", "v1"))
    critere_equipe: Optional[str] = field(
        default_factory=lambda: os.getenv("ANALYTICS_CRITERE_EQUIPE") or None
    )
    analytics_equipe_id: int = field(default_factory=lambda: int(os.getenv("ANALYTICS_EQUIPE_ID", "18")))
    gps_table: str = field(default_factory=lambda: os.getenv("ANALYTICS_GPS_TABLE", "GPS_18"))


def get_settings() -> Settings:
    return Settings()
