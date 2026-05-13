from __future__ import annotations

from typing import Any
from unittest.mock import patch, AsyncMock

import pytest

from replicate.tools import _dispatch
from replicate import (
    cancel_prediction,
    get_model,
    get_prediction,
    list_collections,
    list_models,
    list_predictions,
    list_versions,
    run_prediction,
)
from replicate.types import (
    CancelResult,
    ListCollectionsResult,
    ListModelsResult,
    ListPredictionsResult,
    ListVersionsResult,
    ModelDetailResult,
    PredictionResult,
)


@pytest.fixture(autouse=True)
def _fake_dispatch():
    """All tests mock _dispatch to avoid real API calls."""

    async def _noop(path, *, method=None, body=None):
        return {}, "not mocked"

    with patch("replicate.tools._dispatch", side_effect=_noop):
        yield


# --- list_models ---


@pytest.mark.asyncio
async def test_list_models_success():
    async def fake_dispatch(path, **kw):
        assert "/models" in path
        return {
            "results": [
                {"owner": "a", "name": "model-1", "description": "desc1"},
                {"owner": "b", "name": "model-2"},
            ],
            "next": "https://api.replicate.com/v1/models?cursor=abc123",
        }, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await list_models()

    assert isinstance(result, ListModelsResult)
    assert result.success
    assert len(result.models) == 2
    assert result.models[0].owner == "a"
    assert result.next_cursor == "abc123"


@pytest.mark.asyncio
async def test_list_models_error():
    async def fake_dispatch(path, **kw):
        return {}, "unauthorized"

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await list_models()

    assert not result.success
    assert result.error == "unauthorized"


@pytest.mark.asyncio
async def test_list_models_with_cursor_and_limit():
    captured: dict[str, Any] = {}

    async def fake_dispatch(path, **kw):
        captured["path"] = path
        return {"results": [], "next": None}, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await list_models(cursor="xyz", limit=10)

    assert result.success
    assert "cursor=xyz" in captured["path"]
    assert "limit=10" in captured["path"]


# --- get_model ---


@pytest.mark.asyncio
async def test_get_model_success():
    async def fake_dispatch(path, **kw):
        assert path == "/models/stability-ai/sdxl"
        return {
            "owner": "stability-ai",
            "name": "sdxl",
            "description": "SDXL model",
            "visibility": "public",
            "latest_version": {"id": "v1"},
        }, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await get_model(owner="stability-ai", name="sdxl")

    assert isinstance(result, ModelDetailResult)
    assert result.success
    assert result.model is not None
    assert result.model.owner == "stability-ai"
    assert result.model.latest_version_id == "v1"


# --- list_versions ---


@pytest.mark.asyncio
async def test_list_versions_success():
    async def fake_dispatch(path, **kw):
        return {
            "results": [
                {"id": "v1", "created_at": "2024-01-01"},
                {"id": "v2", "created_at": "2024-02-01"},
            ],
            "next": None,
        }, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await list_versions(owner="a", name="b")

    assert isinstance(result, ListVersionsResult)
    assert result.success
    assert len(result.versions) == 2
    assert result.versions[0].id == "v1"


# --- run_prediction ---


@pytest.mark.asyncio
async def test_run_prediction_no_wait():
    async def fake_dispatch(path, **kw):
        return {
            "id": "pred-1",
            "status": "starting",
            "model": "a/b",
            "input": {"prompt": "hello"},
        }, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await run_prediction(version="v123", input={"prompt": "hello"}, wait=False)

    assert isinstance(result, PredictionResult)
    assert result.success
    assert result.prediction is not None
    assert result.prediction.status == "starting"


@pytest.mark.asyncio
async def test_run_prediction_with_model_owner():
    captured: dict[str, Any] = {}

    async def fake_dispatch(path, **kw):
        captured["path"] = path
        captured["body"] = kw.get("body")
        return {"id": "pred-2", "status": "starting"}, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await run_prediction(model_owner="a", model_name="b", input={"x": 1}, wait=False)

    assert result.success
    assert captured["path"] == "/models/a/b/predictions"
    assert captured["body"]["input"] == {"x": 1}


@pytest.mark.asyncio
async def test_run_prediction_no_version_or_model():
    result = await run_prediction()

    assert not result.success
    assert "Must provide" in result.error


@pytest.mark.asyncio
async def test_run_prediction_polls_to_completion():
    call_count = 0

    async def fake_dispatch(path, **kw):
        nonlocal call_count
        call_count += 1
        if "predictions" in path and path.startswith("/predictions") and path.endswith("/predictions") is False and "cancel" not in path:
            pass
        if call_count == 1:
            return {"id": "pred-3", "status": "starting"}, None
        if call_count == 2:
            return {"id": "pred-3", "status": "processing"}, None
        return {"id": "pred-3", "status": "succeeded", "output": "result_url"}, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        with patch("replicate.tools.asyncio.sleep", new_callable=AsyncMock):
            result = await run_prediction(version="v1", wait=True)

    assert result.success
    assert result.prediction is not None
    assert result.prediction.status == "succeeded"
    assert result.prediction.output == "result_url"


@pytest.mark.asyncio
async def test_run_prediction_poll_timeout():
    async def fake_dispatch(path, **kw):
        return {"id": "pred-4", "status": "processing"}, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        with patch("replicate.tools.asyncio.sleep", new_callable=AsyncMock):
            result = await run_prediction(version="v1", wait=True, max_attempts=2)

    assert result.success
    assert "timed out" in result.error


# --- get_prediction ---


@pytest.mark.asyncio
async def test_get_prediction():
    async def fake_dispatch(path, **kw):
        assert path == "/predictions/abc-123"
        return {"id": "abc-123", "status": "succeeded", "output": ["url1"]}, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await get_prediction(id="abc-123")

    assert result.success
    assert result.prediction is not None
    assert result.prediction.status == "succeeded"


# --- list_predictions ---


@pytest.mark.asyncio
async def test_list_predictions():
    async def fake_dispatch(path, **kw):
        return {
            "results": [
                {"id": "p1", "status": "succeeded"},
                {"id": "p2", "status": "failed", "error": "oom"},
            ],
            "next": None,
        }, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await list_predictions()

    assert isinstance(result, ListPredictionsResult)
    assert result.success
    assert len(result.predictions) == 2
    assert result.predictions[1].error == "oom"


# --- cancel_prediction ---


@pytest.mark.asyncio
async def test_cancel_prediction():
    async def fake_dispatch(path, **kw):
        assert "/cancel" in path
        return {}, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await cancel_prediction(id="pred-1")

    assert isinstance(result, CancelResult)
    assert result.success


# --- list_collections ---


@pytest.mark.asyncio
async def test_list_collections():
    async def fake_dispatch(path, **kw):
        return {
            "results": [
                {"slug": "sdxl", "name": "SDXL Models", "description": "desc"},
                {"slug": "flux", "name": "Flux Models"},
            ],
            "next": None,
        }, None

    with patch("replicate.tools._dispatch", side_effect=fake_dispatch):
        result = await list_collections()

    assert isinstance(result, ListCollectionsResult)
    assert result.success
    assert len(result.collections) == 2
    assert result.collections[0].slug == "sdxl"
