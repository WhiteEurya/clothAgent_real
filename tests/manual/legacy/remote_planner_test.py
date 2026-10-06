import hashlib
import json
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

HOST = "company-planner"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_tempfile(path: Path) -> str:
    result = subprocess.run(
        [
            "curl", "-fsS",
            "-X", "POST",
            "https://tempfile.org/api/upload/local",
            "-F", f"files=@{path}",
            "-F", "expiryHours=1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    payload = json.loads(result.stdout)

    if not payload.get("success"):
        raise RuntimeError(f"upload failed: {payload}")

    file_info = payload["files"][0]
    return f"https://tempfile.org/{file_info['id']}/download"


def run_ssh(command: str, *, input_text=None, capture=False):
    kwargs = {
        "text": True,
        "check": True,
    }

    if input_text is not None:
        kwargs["input"] = input_text

    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    return subprocess.run(
        ["ssh", HOST, command],
        **kwargs,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} IMAGE")

    image = Path(sys.argv[1]).resolve()

    if not image.is_file():
        raise SystemExit(f"Image not found: {image}")

    job = uuid.uuid4().hex[:12]
    remote_dir = f"/tmp/cloth_remote_{job}"
    remote_image = f"{remote_dir}/{image.name}"

    local_hash = sha256(image)

    print(f"IMAGE: {image}")
    print(f"SHA256: {local_hash}")

    print("\n[1/5] Uploading original image via HTTPS...")
    t0 = time.time()
    url = upload_tempfile(image)
    print(f"upload completed in {time.time() - t0:.2f}s")
    print(f"url: {url}")

    print("\n[2/5] Company computer downloading image...")
    t0 = time.time()

    cmd = (
        f"mkdir -p {shlex.quote(remote_dir)} && "
        f"curl -fsSL {shlex.quote(url)} "
        f"-o {shlex.quote(remote_image)}"
    )

    run_ssh(cmd)

    print(f"remote download completed in {time.time() - t0:.2f}s")

    print("\n[3/5] Verifying SHA256...")

    result = run_ssh(
        f"sha256sum {shlex.quote(remote_image)}",
        capture=True,
    )

    remote_hash = result.stdout.split()[0]

    print("LOCAL :", local_hash)
    print("REMOTE:", remote_hash)

    if local_hash != remote_hash:
        raise RuntimeError("SHA256 mismatch")

    print("exact original image confirmed")

    print("\n[4/5] Calling company Claude...")
    t0 = time.time()

    prompt = f"""
You are performing a remote robot-vision planning connectivity test.

Use the Read tool to inspect this exact image:

{remote_image}

Return ONLY valid JSON with this structure:

{{
  "ok": true,
  "image_read": true,
  "observation": "brief but specific description",
  "suggested_grasp_region": "brief description of a visually plausible grasp region",
  "confidence": 0.0
}}

The confidence must be between 0 and 1.
Do not perform any robot action.
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

    completed = subprocess.run(
        ["ssh", HOST, remote_cmd],
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=120,
    )

    print(f"Claude completed in {time.time() - t0:.2f}s")

    outer = json.loads(completed.stdout)

    if outer.get("is_error"):
        raise RuntimeError(json.dumps(outer, indent=2))

    result_text = outer.get("result", "")

    print("\n=== CLAUDE RAW RESULT ===")
    print(repr(result_text))

    if not isinstance(result_text, str) or not result_text.strip():
        raise RuntimeError(
            "Claude returned an empty/non-string result:\n"
            + json.dumps(outer, indent=2, ensure_ascii=False)
        )

    cleaned = result_text.strip()

    # Claude may occasionally wrap JSON in Markdown fences.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Fallback: extract the outermost JSON object if Claude added prose.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]

    try:
        proposal = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Claude result was not valid JSON: {exc}\n"
            f"RAW RESULT:\n{result_text}"
        ) from exc

    print("\n=== CLAUDE RESULT ===")
    print(json.dumps(proposal, indent=2, ensure_ascii=False))

    print("\n[5/5] Cleaning remote workspace...")
    run_ssh(f"rm -rf {shlex.quote(remote_dir)}")

    print("\nREMOTE PLANNER PIPELINE PASSED")


if __name__ == "__main__":
    main()
