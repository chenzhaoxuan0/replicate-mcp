"""End-to-end integration test that exercises tool functions against real APIs.

Uses ConnectionTester for HTTP + patches _dispatch to route through it.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dedalus_mcp import HttpMethod
from dedalus_mcp.testing import ConnectionTester
from dedalus_mcp.testing import TestRequest as Req

from replicate import replicate_conn


async def make_real_dispatch(tester: ConnectionTester):
    async def _real_dispatch(path, *, method=HttpMethod.GET, body=None):
        resp = await tester.request(Req(method=method, path=path, body=body))
        if resp.success:
            return resp.body, None
        return {}, f"HTTP {resp.status}"

    return _real_dispatch


async def main():
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        print("REPLICATE_API_TOKEN not set")
        return False

    from replicate import (
        list_models,
        get_model,
        list_versions,
        run_prediction,
        get_prediction,
        list_predictions,
        cancel_prediction,
        list_collections,
    )

    tester = ConnectionTester.from_env(replicate_conn)
    real_dispatch = await make_real_dispatch(tester)

    print("=" * 60)
    print("Replicate MCP — End-to-End Integration Test")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    with patch("replicate.tools._dispatch", side_effect=real_dispatch):

        # 1. list_models
        print("\n--- list_models ---")
        result = await list_models(limit=3)
        if result.success and len(result.models) > 0:
            print(f"  PASS — found {len(result.models)} models")
            print(f"  First: {result.models[0].owner}/{result.models[0].name}")
            passed += 1
        else:
            print(f"  FAIL — {result.error}")
            failed += 1

        # 2. get_model
        print("\n--- get_model ---")
        result = await get_model(owner="stability-ai", name="sdxl")
        if result.success and result.model:
            print(f"  PASS — {result.model.owner}/{result.model.name}")
            print(f"  Latest version: {result.model.latest_version_id}")
            passed += 1
        else:
            print(f"  FAIL — {result.error}")
            failed += 1

        # 3. list_versions
        print("\n--- list_versions ---")
        result = await list_versions(owner="stability-ai", name="sdxl", limit=3)
        if result.success:
            print(f"  PASS — found {len(result.versions)} versions")
            passed += 1
        else:
            print(f"  FAIL — {result.error}")
            failed += 1

        # 4. list_predictions
        print("\n--- list_predictions ---")
        result = await list_predictions(limit=5)
        if result.success:
            print(f"  PASS — found {len(result.predictions)} predictions")
            passed += 1
        else:
            print(f"  FAIL — {result.error}")
            failed += 1

        # 5. run_prediction (may skip on free tier)
        print("\n--- run_prediction ---")
        result = await run_prediction(
            model_owner="black-forest-labs",
            model_name="flux-schnell",
            input={"prompt": "a photo of a cat", "num_outputs": 1},
            wait=False,
        )
        if result.success and result.prediction:
            print(f"  PASS — prediction {result.prediction.id}, status={result.prediction.status}")
            passed += 1
            pred_id = result.prediction.id

            # 6. get_prediction
            print("\n--- get_prediction ---")
            result2 = await get_prediction(id=pred_id)
            if result2.success and result2.prediction:
                print(f"  PASS — status={result2.prediction.status}")
                passed += 1
            else:
                print(f"  FAIL — {result2.error}")
                failed += 1

            # 7. cancel_prediction
            print("\n--- cancel_prediction ---")
            result3 = await cancel_prediction(id=pred_id)
            if result3.success:
                print(f"  PASS — cancelled")
                passed += 1
            else:
                print(f"  FAIL — {result3.error}")
                failed += 1
        elif "402" in str(result.error) or "429" in str(result.error):
            print(f"  SKIP — {result.error[:80]}")
            skipped += 3
        else:
            print(f"  FAIL — {result.error}")
            failed += 3

        # 8. list_collections
        print("\n--- list_collections ---")
        result = await list_collections(limit=3)
        if result.success:
            print(f"  PASS — found {len(result.collections)} collections")
            if result.collections:
                print(f"  First: {result.collections[0].slug} — {result.collections[0].name}")
            passed += 1
        else:
            print(f"  FAIL — {result.error}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
