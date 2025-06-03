import subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test")

@mcp.tool()
def test_subprocess():
    """Run a subprocess"""
    result = subprocess.run(["echo", "Hello from subprocess!"], capture_output=True, text=True)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

if __name__ == "__main__":
    mcp.run()