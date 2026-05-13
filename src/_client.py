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

        print("--- list_collections ---")
        result = await client.call_tool("list_collections", {"limit": 3})
        print(result)
        print()

        print("--- get_model ---")
        result = await client.call_tool("get_model", {"owner": "stability-ai", "name": "sdxl"})
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
