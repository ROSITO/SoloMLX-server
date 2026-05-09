from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from mlxserve.api.app import app
from mlxserve.api.deps import engine
from mlxserve.config import settings


@pytest.fixture(autouse=True)
def restore_preload():
    prev = settings.preload_default_model
    yield
    settings.preload_default_model = prev


def test_lifespan_preloads_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "preload_default_model", True)
    mock = AsyncMock()
    monkeypatch.setattr(engine, "ensure_model", mock)
    with TestClient(app):
        pass
    mock.assert_awaited_once_with(settings.default_model)


def test_lifespan_skips_preload_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "preload_default_model", False)
    mock = AsyncMock()
    monkeypatch.setattr(engine, "ensure_model", mock)
    with TestClient(app):
        pass
    mock.assert_not_awaited()
