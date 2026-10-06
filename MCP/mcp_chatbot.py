"""Chatbot that connects to every server in server_config.json."""

import asyncio
import json
from contextlib import AsyncExitStack

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


class MCP_ChatBot:
    def __init__(self):
        self.sessions: list[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.available_tools: list[dict] = []
        self.tool_to_session: dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            print(
                f"\nConnected to {server_name} with tools:",
                [tool.name for tool in response.tools],
            )
            for tool in response.tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                )
        except Exception as error:
            print(f"Failed to connect to {server_name}: {error}")

    async def connect_to_servers(self) -> None:
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            for server_name, server_config in data.get("mcpServers", {}).items():
                await self.connect_to_server(server_name, server_config)
        except Exception as error:
            print(f"Error loading server configuration: {error}")
            raise

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
                    session = self.tool_to_session[content.name]
                    result = await session.call_tool(
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

    async def cleanup(self):
        """Close every client and session in reverse order."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
