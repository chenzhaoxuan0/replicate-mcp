from __future__ import annotations

from http import HTTPStatus

import pytest
from dedalus_mcp.testing import ConnectionTester, HttpMethod
from dedalus_mcp.testing import TestRequest as Req


@pytest.mark.asyncio
async def test_list_models_live(replicate_tester: ConnectionTester) -> None:
    """GET /models should return a paginated list."""
    resp = await replicate_tester.request(
        Req(method=HttpMethod.GET, path="/models?limit=3")
    )
    assert resp.success, f"List models failed: status={resp.status} body={resp.body!r}"
    assert resp.status == HTTPStatus.OK
    assert resp.body is not None
    assert "results" in resp.body


@pytest.mark.asyncio
async def test_list_collections_live(replicate_tester: ConnectionTester) -> None:
    """GET /collections should return collections."""
    resp = await replicate_tester.request(
        Req(method=HttpMethod.GET, path="/collections?limit=3")
    )
    assert resp.success, f"List collections failed: status={resp.status} body={resp.body!r}"
    assert resp.status == HTTPStatus.OK
    assert resp.body is not None
    assert "results" in resp.body
