# MCP

Notes on the Model Context Protocol. A server lists tools, resources, and prompts. A client calls them over stdio. **Stdio** means the server runs as a child process.

Read the notebooks in this order:

- [`mcp.ipynb`](./mcp.ipynb) calls the arXiv tools in the notebook process.
- [`mcp-server.ipynb`](./mcp-server.ipynb) wraps those tools in a FastMCP server.
- [`mcp-client.ipynb`](./mcp-client.ipynb) connects one chatbot to that server.
- [`reference-servers.ipynb`](./reference-servers.ipynb) connects the chatbot to several servers.
- [`prompts-and-resources.ipynb`](./prompts-and-resources.ipynb) adds read-only resources and a prompt template.
- [`subprocess.ipynb`](./subprocess.ipynb) shows how a parent process starts a child process.

Files next to the notebooks:

- [`research_server.py`](./research_server.py) is the research server.
- [`mcp_client.py`](./mcp_client.py) is the single-server client.
- [`mcp_chatbot.py`](./mcp_chatbot.py) reads [`server_config.json`](./server_config.json).
- [`papers/`](./papers) holds saved arXiv records.
- [`mcp-subprocess-server/`](./mcp-subprocess-server) is a separate server. Its only tool runs a subprocess.

Run the multi-server chatbot from this folder. Put `ANTHROPIC_API_KEY` in a local `.env` file. That file stays untracked.

```bash
uv run --with "mcp>=1.2,<2" --with arxiv --with anthropic --with python-dotenv mcp_chatbot.py
```
