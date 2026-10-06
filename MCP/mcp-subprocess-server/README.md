# MCP subprocess server

A small FastMCP server. Its one tool runs `echo` through `subprocess`.

From this folder:

```bash
uv run server.py
```

`test_script.py` is a sample child process. It writes to stdout and stderr.
