from __future__ import annotations

import os

from dedalus_mcp import MCPServer
from dedalus_mcp.server import TransportSecuritySettings

from replicate import (
    cancel_prediction,
    get_model,
    get_prediction,
    list_collections,
    list_models,
    list_predictions,
    list_versions,
    replicate_conn,
    run_prediction,
)


def create_server() -> MCPServer:
    as_url = os.getenv("DEDALUS_AS_URL", "https://as.dedaluslabs.ai")
    return MCPServer(
        name="replicate-mcp",
        connections=[replicate_conn],
        http_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        streamable_http_stateless=True,
        authorization_server=as_url,
    )


async def main() -> None:
    server = create_server()
    server.collect(
        list_models,
        get_model,
        list_versions,
        run_prediction,
        get_prediction,
        list_predictions,
        cancel_prediction,
        list_collections,
    )
    await server.serve(8080)
