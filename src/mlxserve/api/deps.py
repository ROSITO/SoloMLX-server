import secrets

from fastapi import Header, HTTPException

from mlxserve.config import settings
from mlxserve.memory.guardian import MemoryGuardian
from mlxserve.runtime.engine import InferenceEngine

engine = InferenceEngine(backend_mode=settings.runtime_backend)
guardian = MemoryGuardian(
    soft_limit_gb=settings.max_memory_gb,
    hard_limit_gb=settings.hard_memory_gb,
    idle_unload_minutes=settings.idle_unload_minutes,
)


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
