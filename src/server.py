from __future__ import annotations

from dedalus_mcp import MCPServer

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

server = MCPServer(
    name="replicate-mcp",
    connections=[replicate_conn],
    tools=[
        list_models,
        get_model,
        list_versions,
        run_prediction,
        get_prediction,
        list_predictions,
        cancel_prediction,
        list_collections,
    ],
)


async def main() -> None:
    await server.serve(8080)
