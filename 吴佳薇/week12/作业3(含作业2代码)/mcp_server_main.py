import asyncio
from fastmcp import FastMCP, Client

from news import mcp as news_mcp
from saying import mcp as saying_mcp
from tool import mcp as tool_mcp
from sentiment import mcp as sentiment_mcp

mcp = FastMCP(
    name="MCP-Server"
)

async def setup():
    await mcp.import_server(news_mcp, prefix="")
    await mcp.import_server(saying_mcp, prefix="")
    await mcp.import_server(tool_mcp, prefix="")
    await mcp.import_server(sentiment_mcp,prefix="")

async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        Available_tools=[t.name for t in tools]
        print("Available tools:", [t.name for t in tools])
        return Available_tools


if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=5000)
