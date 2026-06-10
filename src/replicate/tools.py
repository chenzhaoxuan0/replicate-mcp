from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlparse, parse_qs

from dedalus_mcp import HttpMethod, HttpRequest, get_context, tool
from dedalus_mcp.auth import Connection, SecretKeys
from dedalus_mcp.types import ToolAnnotations

from replicate.types import (
    CancelResult,
    CollectionInfo,
    ListCollectionsResult,
    ListModelsResult,
    ListPredictionsResult,
    ListVersionsResult,
    ModelDetailResult,
    ModelInfo,
    PredictionInfo,
    PredictionResult,
    VersionInfo,
)

_CONN_NAME = os.getenv("MCP_SERVER_SLUG", "Zx/replicate-mcp").replace("/", "-")

replicate_conn = Connection(
    name=_CONN_NAME,
    secrets=SecretKeys(token="REPLICATE_API_TOKEN"),
    base_url="https://api.replicate.com/v1",
    auth_header_format="Bearer {api_key}",
)

_POLL_INTERVAL = 1.0
_POLL_MAX_ATTEMPTS = 120


async def _dispatch(path: str, *, method: HttpMethod = HttpMethod.GET, body: dict | None = None) -> tuple[dict, str | None]:
    ctx = get_context()
    resp = await ctx.dispatch(
        _CONN_NAME,
        HttpRequest(method=method, path=path, body=body),
    )
    if resp.success and resp.response is not None:
        raw = resp.response.body
        if resp.response.status >= 400:
            return {}, f"Replicate API error ({resp.response.status}): {raw}"
        return raw if isinstance(raw, dict) else {}, None
    return {}, resp.error.message if resp.error else "Replicate request failed"


def _extract_cursor(next_url: str | None) -> str:
    if not next_url:
        return ""
    parsed = urlparse(next_url)
    params = parse_qs(parsed.query)
    cursors = params.get("cursor", [])
    return cursors[0] if cursors else ""


# --- Tools ---


@tool(
    description="List available ML models on Replicate",
    tags=["replicate", "models", "list"],
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_models(cursor: str = "", limit: int = 25) -> ListModelsResult:
    params: list[str] = []
    if cursor:
        params.append(f"cursor={cursor}")
    if limit:
        params.append(f"limit={limit}")
    path = "/models"
    if params:
        path += "?" + "&".join(params)

    body, err = await _dispatch(path)
    if err:
        return ListModelsResult(success=False, error=err)

    results = body.get("results", [])
    models = [ModelInfo.from_api(m) for m in results]
    next_cursor = _extract_cursor(body.get("next"))
    return ListModelsResult(success=True, models=models, next_cursor=next_cursor)


@tool(
    description="Get details for a specific Replicate model",
    tags=["replicate", "models", "read"],
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_model(owner: str, name: str) -> ModelDetailResult:
    body, err = await _dispatch(f"/models/{owner}/{name}")
    if err:
        return ModelDetailResult(success=False, error=err)
    return ModelDetailResult(success=True, model=ModelInfo.from_api(body))


@tool(
    description="List versions of a Replicate model",
    tags=["replicate", "models", "versions", "list"],
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_versions(owner: str, name: str, cursor: str = "", limit: int = 25) -> ListVersionsResult:
    params: list[str] = []
    if cursor:
        params.append(f"cursor={cursor}")
    if limit:
        params.append(f"limit={limit}")
    path = f"/models/{owner}/{name}/versions"
    if params:
        path += "?" + "&".join(params)

    body, err = await _dispatch(path)
    if err:
        return ListVersionsResult(success=False, error=err)

    results = body.get("results", [])
    versions = [VersionInfo.from_api(v) for v in results]
    next_cursor = _extract_cursor(body.get("next"))
    return ListVersionsResult(success=True, versions=versions, next_cursor=next_cursor)


@tool(
    description="Run a prediction on a Replicate model. Optionally waits for completion.",
    tags=["replicate", "predictions", "run"],
    annotations=ToolAnnotations(readOnlyHint=False),
)
async def run_prediction(
    version: str = "",
    model_owner: str = "",
    model_name: str = "",
    input: dict | None = None,
    wait: bool = True,
    poll_interval: float = 1.0,
    max_attempts: int = 120,
) -> PredictionResult:
    if not input:
        input = {}

    if model_owner and model_name:
        path = f"/models/{model_owner}/{model_name}/predictions"
        req_body: dict[str, Any] = {"input": input}
    elif version:
        path = "/predictions"
        req_body = {"version": version, "input": input}
    else:
        return PredictionResult(success=False, error="Must provide either version or model_owner+model_name")

    body, err = await _dispatch(path, method=HttpMethod.POST, body=req_body)
    if err:
        return PredictionResult(success=False, error=err)

    prediction = PredictionInfo.from_api(body)

    if not wait:
        return PredictionResult(success=True, prediction=prediction)

    terminal = {"succeeded", "failed", "canceled"}
    interval = max(0.5, min(poll_interval, 10.0))
    attempts = 0
    max_poll = max(1, min(max_attempts, 600))

    while prediction.status not in terminal:
        attempts += 1
        if attempts > max_poll:
            return PredictionResult(
                success=True,
                prediction=prediction,
                error=f"Polling timed out after {max_poll} attempts (status={prediction.status})",
            )
        await asyncio.sleep(interval)
        body, err = await _dispatch(f"/predictions/{prediction.id}")
        if err:
            return PredictionResult(success=True, prediction=prediction, error=err)
        prediction = PredictionInfo.from_api(body)

    return PredictionResult(success=True, prediction=prediction)


@tool(
    description="Get the status and output of a prediction",
    tags=["replicate", "predictions", "read"],
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_prediction(id: str) -> PredictionResult:
    body, err = await _dispatch(f"/predictions/{id}")
    if err:
        return PredictionResult(success=False, error=err)
    return PredictionResult(success=True, prediction=PredictionInfo.from_api(body))


@tool(
    description="List recent predictions",
    tags=["replicate", "predictions", "list"],
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_predictions(cursor: str = "", limit: int = 25) -> ListPredictionsResult:
    params: list[str] = []
    if cursor:
        params.append(f"cursor={cursor}")
    if limit:
        params.append(f"limit={limit}")
    path = "/predictions"
    if params:
        path += "?" + "&".join(params)

    body, err = await _dispatch(path)
    if err:
        return ListPredictionsResult(success=False, error=err)

    results = body.get("results", [])
    predictions = [PredictionInfo.from_api(p) for p in results]
    next_cursor = _extract_cursor(body.get("next"))
    return ListPredictionsResult(success=True, predictions=predictions, next_cursor=next_cursor)


@tool(
    description="Cancel a running prediction",
    tags=["replicate", "predictions", "cancel"],
    annotations=ToolAnnotations(readOnlyHint=False),
)
async def cancel_prediction(id: str) -> CancelResult:
    _, err = await _dispatch(f"/predictions/{id}/cancel", method=HttpMethod.POST)
    if err:
        return CancelResult(success=False, error=err)
    return CancelResult(success=True)


@tool(
    description="List featured model collections on Replicate",
    tags=["replicate", "collections", "list"],
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_collections(cursor: str = "", limit: int = 25) -> ListCollectionsResult:
    params: list[str] = []
    if cursor:
        params.append(f"cursor={cursor}")
    if limit:
        params.append(f"limit={limit}")
    path = "/collections"
    if params:
        path += "?" + "&".join(params)

    body, err = await _dispatch(path)
    if err:
        return ListCollectionsResult(success=False, error=err)

    results = body.get("results", [])
    collections = [CollectionInfo.from_api(c) for c in results]
    next_cursor = _extract_cursor(body.get("next"))
    return ListCollectionsResult(success=True, collections=collections, next_cursor=next_cursor)
