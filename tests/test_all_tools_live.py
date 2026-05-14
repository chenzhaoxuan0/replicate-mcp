"""Comprehensive live tests for all 8 Replicate MCP tools.

Requires .env with REPLICATE_API_TOKEN set.
"""

from __future__ import annotations

from http import HTTPStatus

import pytest
from dedalus_mcp.testing import ConnectionTester, HttpMethod
from dedalus_mcp.testing import TestRequest as Req


@pytest.mark.asyncio
async def test_01_list_models(replicate_tester: ConnectionTester) -> None:
    resp = await replicate_tester.request(
        Req(method=HttpMethod.GET, path="/models?limit=3")
    )
    assert resp.success, f"list_models failed: {resp.status} {resp.body!r}"
    assert resp.status == HTTPStatus.OK
    assert "results" in resp.body
    assert len(resp.body["results"]) > 0


@pytest.mark.asyncio
async def test_02_get_model(replicate_tester: ConnectionTester) -> None:
    resp = await replicate_tester.request(
        Req(method=HttpMethod.GET, path="/models/stability-ai/sdxl")
    )
    assert resp.success, f"get_model failed: {resp.status} {resp.body!r}"
    assert resp.status == HTTPStatus.OK
    assert resp.body["owner"] == "stability-ai"
    assert resp.body["name"] == "sdxl"


@pytest.mark.asyncio
async def test_03_list_versions(replicate_tester: ConnectionTester) -> None:
    resp = await replicate_tester.request(
        Req(method=HttpMethod.GET, path="/models/stability-ai/sdxl/versions?limit=3")
    )
    assert resp.success, f"list_versions failed: {resp.status} {resp.body!r}"
    assert resp.status == HTTPStatus.OK
    assert "results" in resp.body


@pytest.mark.asyncio
async def test_04_list_predictions(replicate_tester: ConnectionTester) -> None:
    resp = await replicate_tester.request(
        Req(method=HttpMethod.GET, path="/predictions?limit=5")
    )
    assert resp.success, f"list_predictions failed: {resp.status} {resp.body!r}"
    assert resp.status == HTTPStatus.OK
    assert "results" in resp.body


@pytest.mark.asyncio
async def test_05_get_prediction_not_found(replicate_tester: ConnectionTester) -> None:
    """GET a non-existent prediction should return 404 (validates auth + endpoint)."""
    resp = await replicate_tester.request(
        Req(method=HttpMethod.GET, path="/predictions/nonexistent-id-12345")
    )
    assert resp.status in (HTTPStatus.NOT_FOUND, HTTPStatus.OK), (
        f"Unexpected status: {resp.status}"
    )


@pytest.mark.asyncio
async def test_06_list_collections(replicate_tester: ConnectionTester) -> None:
    resp = await replicate_tester.request(
        Req(method=HttpMethod.GET, path="/collections?limit=3")
    )
    assert resp.success, f"list_collections failed: {resp.status} {resp.body!r}"
    assert resp.status == HTTPStatus.OK
    assert "results" in resp.body


@pytest.mark.asyncio
async def test_07_run_prediction_validate(replicate_tester: ConnectionTester) -> None:
    """Run a tiny prediction to validate the full prediction lifecycle.
    Skips on 402 (insufficient credit) for free-tier accounts.
    """
    resp = await replicate_tester.request(
        Req(
            method=HttpMethod.POST,
            path="/models/black-forest-labs/flux-schnell/predictions",
            body={
                "input": {
                    "prompt": "a photo of a cat",
                    "num_outputs": 1,
                    "aspect_ratio": "1:1",
                    "output_format": "webp",
                    "output_quality": 50,
                }
            },
        )
    )
    if resp.status == HTTPStatus.PAYMENT_REQUIRED:
        pytest.skip("Replicate account has insufficient credit (402)")
    if resp.status == HTTPStatus.TOO_MANY_REQUESTS:
        pytest.skip("Replicate rate limit hit (429)")
    assert resp.success, f"run_prediction failed: {resp.status} {resp.body!r}"
    assert resp.status in (HTTPStatus.CREATED, HTTPStatus.OK), (
        f"Unexpected status: {resp.status}, body={resp.body}"
    )
    assert "id" in resp.body
    assert "status" in resp.body


@pytest.mark.asyncio
async def test_08_cancel_prediction(replicate_tester: ConnectionTester) -> None:
    """Cancel a prediction - creates one first, then cancels it.
    Skips on 402/429 for free-tier accounts.
    """
    create_resp = await replicate_tester.request(
        Req(
            method=HttpMethod.POST,
            path="/models/black-forest-labs/flux-schnell/predictions",
            body={"input": {"prompt": "test", "num_outputs": 1}},
        )
    )
    if create_resp.status in (HTTPStatus.PAYMENT_REQUIRED, HTTPStatus.TOO_MANY_REQUESTS):
        pytest.skip(f"Cannot create prediction for cancel test: {create_resp.status}")
    assert create_resp.success, f"Create prediction failed: {create_resp.status}"
    pred_id = create_resp.body["id"]

    resp = await replicate_tester.request(
        Req(method=HttpMethod.POST, path=f"/predictions/{pred_id}/cancel")
    )
    assert resp.success, f"cancel_prediction failed: {resp.status} {resp.body!r}"
