"""Tests for FastAPI inference app."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def api_client(
    monkeypatch: pytest.MonkeyPatch,
    manifest_path: Path,
    label_map_path: Path,
    checkpoint_path: Path,
) -> TestClient:
    monkeypatch.setenv("PLANT_DISEASE_MANIFEST", str(manifest_path))
    monkeypatch.setenv("PLANT_DISEASE_LABEL_MAP", str(label_map_path))
    monkeypatch.setenv("PLANT_DISEASE_CHECKPOINT", str(checkpoint_path))
    import api as api_module

    importlib.reload(api_module)
    with TestClient(api_module.app) as client:
        yield client


def test_health_ok(api_client: TestClient) -> None:
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_predict_jpeg(api_client: TestClient, tiny_image: Path) -> None:
    with tiny_image.open("rb") as f:
        r = api_client.post(
            "/predict", files={"file": ("x.jpg", f, "image/jpeg")}, params={"topk": 2}
        )
    assert r.status_code == 200
    data = r.json()
    assert data["top1"] is not None
    assert len(data["top_k"]) <= 2
