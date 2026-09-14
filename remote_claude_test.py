import subprocess
import json
import sys

HOST = "CNS2026330003@CNS202633000332.roboscience.xyz"

prompt = """
You are a remote planner test.

Return ONLY valid JSON with exactly this structure:

{
  "ok": true,
  "planner": "claude",
  "message": "remote planning works",
  "action": {
    "name": "test_move",
    "x": 123,
    "y": 456
  }
}
""".strip()

cmd = [
    "ssh",
    HOST,
    'bash -lc "claude -p --output-format json"',
]

print(f"[local] calling remote planner: {HOST}")

result = subprocess.run(
    cmd,
    input=prompt,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=120,
)

print(f"[local] ssh return code: {result.returncode}")

if result.stderr.strip():
    print("\n[remote stderr]")
    print(result.stderr)

if result.returncode != 0:
    print("\nRemote Claude call failed.")
    sys.exit(result.returncode)

print("\n[remote raw output]")
print(result.stdout)

# Claude --output-format json normally wraps the actual response,
# so first make sure at least the outer response is valid JSON.
try:
    outer = json.loads(result.stdout)
except json.JSONDecodeError as exc:
    print("\nERROR: remote output is not valid JSON:")
    print(exc)
    sys.exit(1)

print("\n[local] outer JSON parsed successfully")
print(json.dumps(outer, indent=2, ensure_ascii=False))

print("\n=== REMOTE PLANNER TEST PASSED ===")
