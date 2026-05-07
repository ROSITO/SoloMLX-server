from mlxserve.models.manager import ModelManager


def test_model_manager_registry_roundtrip(tmp_path):
    manager = ModelManager(base_dir=str(tmp_path))
    assert manager.list_local() == []

    payload = manager._load_registry()
    payload["models"].append(
        {
            "id": "demo",
            "source": "mlx-community/demo",
            "local_path": "/tmp/demo",
            "pulled_at": "2026-01-01T00:00:00+00:00",
            "size_bytes": 123,
            "quantization": "4bit",
        }
    )
    manager._save_registry(payload)
    items = manager.list_local()
    assert len(items) == 1
    assert items[0].id == "demo"


def test_model_manager_remove_not_found(tmp_path):
    manager = ModelManager(base_dir=str(tmp_path))
    assert manager.remove("missing") is False
