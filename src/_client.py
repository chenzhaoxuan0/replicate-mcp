import asyncio

from dotenv import load_dotenv
from dedalus_mcp.client import MCPClient

load_dotenv()


async def main() -> None:
    async with MCPClient("http://localhost:8080/mcp") as client:
        tools = await client.list_tools()
        print("Available tools:")
        for t in tools:
            print(f"  - {t.name}: {t.description[:80]}...")
        print()

        print("--- list_models ---")
        result = await client.call_tool("list_models", {"limit": 3})
        print(result)
        print()

        print("--- get_model ---")
        result = await client.call_tool("get_model", {"owner": "stability-ai", "name": "sdxl"})
        print(result)
        print()

        print("--- list_versions ---")
        result = await client.call_tool("list_versions", {"owner": "stability-ai", "name": "sdxl", "limit": 3})
        print(result)
        print()

        print("--- list_collections ---")
        result = await client.call_tool("list_collections", {"limit": 3})
        print(result)
        print()

        print("--- list_predictions ---")
        result = await client.call_tool("list_predictions", {"limit": 3})
        print(result)
        print()

        print("--- run_prediction ---")
        result = await client.call_tool("run_prediction", {
            "model_owner": "stability-ai",
            "model_name": "sdxl",
            "input": {"prompt": "a beautiful sunset"},
            "wait": True,
        })
        print(result)
        print()

        prediction_id = ""
        if hasattr(result, 'data') and result.data:
            pred = result.data[0] if isinstance(result.data, list) else result.data
            prediction_id = pred.get("id", "") if isinstance(pred, dict) else ""

        if prediction_id:
            print("--- get_prediction ---")
            result = await client.call_tool("get_prediction", {"id": prediction_id})
            print(result)
            print()

            print("--- cancel_prediction ---")
            result = await client.call_tool("cancel_prediction", {"id": prediction_id})
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
