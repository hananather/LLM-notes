"""Single-server MCP client for research_server.py."""

import asyncio

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


class MCP_ChatBot:
    def __init__(self):
        self.session: ClientSession = None
        self.anthropic = Anthropic()
        self.available_tools: list[dict] = []

    async def process_query(self, query):
        messages = [{"role": "user", "content": query}]
        response = self.anthropic.messages.create(
            max_tokens=2024,
            model="claude-sonnet-4-6",
            tools=self.available_tools,
            messages=messages,
        )
        process_query = True
        while process_query:
            assistant_content = []
            for content in response.content:
                if content.type == "text":
                    print(content.text)
                    assistant_content.append(content)
                    if len(response.content) == 1:
                        process_query = False
                elif content.type == "tool_use":
                    assistant_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_content})
                    print(f"Calling tool {content.name} with args {content.input}")
                    result = await self.session.call_tool(
                        content.name, arguments=content.input
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": result.content,
                                }
                            ],
                        }
                    )
                    response = self.anthropic.messages.create(
                        max_tokens=2024,
                        model="claude-sonnet-4-6",
                        tools=self.available_tools,
                        messages=messages,
                    )
                    if (
                        len(response.content) == 1
                        and response.content[0].type == "text"
                    ):
                        print(response.content[0].text)
                        process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop."""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                await self.process_query(query)
                print("\n")
            except Exception as error:
                print(f"\nError: {error}")

    async def connect_to_server_and_run(self):
        server_params = StdioServerParameters(
            command="uv",
            args=[
                "run",
                "--with",
                "mcp>=1.2,<2",
                "--with",
                "arxiv",
                "research_server.py",
            ],
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                response = await session.list_tools()
                print(
                    "\nConnected to server with tools:",
                    [tool.name for tool in response.tools],
                )
                self.available_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in response.tools
                ]
                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
