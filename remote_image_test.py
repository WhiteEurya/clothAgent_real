import json
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

HOST = "CNS2026330003@CNS202633000332.roboscience.xyz"

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} IMAGE")
    sys.exit(2)

image = Path(sys.argv[1]).resolve()
if not image.is_file():
    raise SystemExit(f"Image not found: {image}")

job_id = uuid.uuid4().hex[:10]
remote_dir = f"/tmp/cloth_agent_remote_test_{job_id}"
remote_image = f"{remote_dir}/{image.name}"

print(f"[1] create remote workspace: {remote_dir}")
subprocess.run(
    ["ssh", HOST, f"mkdir -p {shlex.quote(remote_dir)}"],
    check=True,
)

print(f"[2] upload: {image.name}")
subprocess.run(
    ["scp", str(image), f"{HOST}:{remote_image}"],
    check=True,
)

prompt = f"""
This is a remote vision test.

Use the Read tool to inspect this image:

{remote_image}

Return ONLY a JSON object with this exact structure:

{{
  "ok": true,
  "image_read": true,
  "description": "brief description of what is visible"
}}
""".strip()

remote_cmd = (
    f"cd {shlex.quote(remote_dir)} && "
    "claude -p "
    "--output-format json "
    "--permission-mode dontAsk "
    "--allowedTools Read "
    "--tools Read "
    "--no-session-persistence "
    f"--add-dir {shlex.quote(remote_dir)}"
)

print("[3] call remote Claude")

completed = subprocess.run(
    ["ssh", HOST, "bash", "-lc", shlex.quote(remote_cmd)],
    input=prompt,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=120,
)

print(f"[4] return code: {completed.returncode}")

if completed.stderr.strip():
    print("\n[stderr]")
    print(completed.stderr)

if completed.returncode != 0:
    raise SystemExit("Remote Claude failed")

outer = json.loads(completed.stdout)

if outer.get("is_error"):
    raise SystemExit(f"Claude error: {outer}")

result_text = outer.get("result")
if not isinstance(result_text, str):
    raise SystemExit("Claude response has no string result")

proposal = json.loads(result_text)

print("\n=== RESULT ===")
print(json.dumps(proposal, indent=2, ensure_ascii=False))

print("\n[5] cleanup")
subprocess.run(
    ["ssh", HOST, f"rm -rf {shlex.quote(remote_dir)}"],
    check=False,
)

print("\nREMOTE IMAGE TEST PASSED")
