from __future__ import annotations

import os

import pytest
from dedalus_mcp.testing import ConnectionTester
from dotenv import load_dotenv

from replicate import replicate_conn


@pytest.fixture(scope="session")
def replicate_tester() -> ConnectionTester:
    load_dotenv()
    if not os.getenv("REPLICATE_API_TOKEN"):
        pytest.skip("REPLICATE_API_TOKEN not set; skipping live connection tests")
    return ConnectionTester.from_env(replicate_conn)
