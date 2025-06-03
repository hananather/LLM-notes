import sys
import time

print("🚀 MCP Subprocess Test Running!")
print("This is stdout output")
time.sleep(1)
print("Process ID:", sys.executable)
sys.stderr.write("⚠️  This is stderr output\n")
print("✅ Subprocess completed successfully")