import secrets

from fastapi import Header, HTTPException

from mlxserve.config import settings
from mlxserve.memory.guardian import MemoryGuardian
from mlxserve.models.manager import ModelManager
from mlxserve.observability import MetricsStore
from mlxserve.runtime.engine import InferenceEngine

engine = InferenceEngine(backend_mode=settings.runtime_backend)
guardian = MemoryGuardian(
    soft_limit_gb=settings.max_memory_gb,
    hard_limit_gb=settings.hard_memory_gb,
    idle_unload_minutes=settings.idle_unload_minutes,
)
model_manager = ModelManager()
metrics = MetricsStore()


def _jwt_hs256_accepted(token: str) -> bool:
    if not settings.jwt_hs256_secret:
        return False
    try:
        import jwt
    except ImportError:
        return False
    try:
        if settings.jwt_audience:
            jwt.decode(
                token,
                settings.jwt_hs256_secret,
                algorithms=["HS256"],
                audience=settings.jwt_audience,
            )
        else:
            jwt.decode(token, settings.jwt_hs256_secret, algorithms=["HS256"])
        return True
    except Exception:
        return False


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_key and not settings.jwt_hs256_secret:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.removeprefix("Bearer ").strip()
    if _jwt_hs256_accepted(token):
        return
    if settings.api_key and secrets.compare_digest(token, settings.api_key):
        return
    raise HTTPException(status_code=401, detail="Invalid API key")
