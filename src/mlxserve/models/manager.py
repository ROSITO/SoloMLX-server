import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LocalModel:
    id: str
    source: str
    local_path: str
    pulled_at: str
    size_bytes: int
    quantization: str


class ModelManager:
    def __init__(self, base_dir: str | None = None) -> None:
        root = Path(base_dir) if base_dir else (Path.home() / ".mlxserve" / "models")
        self.base_dir = root
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "registry.json"
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({"models": []}, indent=2), encoding="utf-8")

    def _load_registry(self) -> dict:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save_registry(self, payload: dict) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _model_alias(model_id: str) -> str:
        return model_id.split("/")[-1].lower().replace(" ", "-")

    @staticmethod
    def _guess_quantization(model_id: str) -> str:
        low = model_id.lower()
        if "4bit" in low or "q4" in low:
            return "4bit"
        if "8bit" in low or "q8" in low:
            return "8bit"
        if "6bit" in low or "q6" in low:
            return "6bit"
        return "unknown"

    @staticmethod
    def _dir_size_bytes(path: str) -> int:
        root = Path(path)
        total = 0
        for p in root.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    def list_local(self) -> list[LocalModel]:
        payload = self._load_registry()
        return [LocalModel(**m) for m in payload.get("models", [])]

    def pull(self, model_id: str) -> LocalModel:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(repo_id=model_id)
        now = datetime.now(timezone.utc).isoformat()
        alias = self._model_alias(model_id)
        model = LocalModel(
            id=alias,
            source=model_id,
            local_path=local_path,
            pulled_at=now,
            size_bytes=self._dir_size_bytes(local_path),
            quantization=self._guess_quantization(model_id),
        )

        payload = self._load_registry()
        models = [m for m in payload.get("models", []) if m.get("source") != model_id]
        models.append(model.__dict__)
        payload["models"] = models
        self._save_registry(payload)
        return model

    def remove(self, model_alias_or_source: str) -> bool:
        payload = self._load_registry()
        models = payload.get("models", [])
        target = None
        keep = []
        for m in models:
            if m.get("id") == model_alias_or_source or m.get("source") == model_alias_or_source:
                target = m
            else:
                keep.append(m)
        if target is None:
            return False

        payload["models"] = keep
        self._save_registry(payload)

        # Best effort cache cleanup for this model.
        try:
            from huggingface_hub import scan_cache_dir

            info = scan_cache_dir()
            repo = next((r for r in info.repos if r.repo_id == target["source"]), None)
            if repo:
                strategy = info.delete_revisions(*[rev.commit_hash for rev in repo.revisions])
                strategy.execute()
        except Exception:
            pass

        local_path = Path(target.get("local_path", ""))
        if local_path.exists() and str(local_path).startswith(str(self.base_dir)):
            try:
                shutil.rmtree(local_path, ignore_errors=True)
            except Exception:
                pass
        return True
