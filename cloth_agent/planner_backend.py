"""Claude planner execution backends.

The remote backend deliberately transports RGB PNGs over HTTPS.  SSH carries
only the small orchestration command and Claude's structured response.
"""

from __future__ import annotations

import json
import hashlib
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


def parse_claude_json(stdout: str) -> dict[str, Any]:
    """Decode the CLI envelope and a fenced/prose-wrapped planner object."""
    try:
        outer = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PlannerBackendError("Claude returned an invalid JSON envelope") from exc
    if not isinstance(outer, dict) or outer.get("is_error") is True:
        raise PlannerBackendError("Claude returned an error envelope")
    if str(outer.get("subtype", "success")).startswith("error"):
        raise PlannerBackendError("Claude did not complete successfully")
    value = outer.get("structured_output", outer.get("structuredOutput"))
    if value is None:
        value = outer.get("result", outer)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= start:
            try:
                result = json.loads(text[start:end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
    raise PlannerBackendError("Claude result did not contain one valid JSON object")


class PlannerBackendError(RuntimeError):
    """A planner backend failed; callers must not execute an action."""


@dataclass(frozen=True)
class BackendResult:
    stdout: str
    stderr: str
    returncode: int
    command: tuple[str, ...]


class LocalClaudeBackend:
    """Run Claude on the local machine (the historical behaviour)."""

    def __init__(self, binary: str = "claude", timeout_s: int = 900):
        self.binary = binary
        self.timeout_s = int(timeout_s)

    def invoke(self, *, prompt: str, command: list[str], cwd: Path) -> BackendResult:
        try:
            completed = subprocess.run(
                command, cwd=cwd, text=True, capture_output=True,
                timeout=self.timeout_s, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PlannerBackendError(f"local Claude invocation failed: {exc}") from exc
        if completed.returncode != 0:
            raise PlannerBackendError(
                f"Claude exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return BackendResult(completed.stdout, completed.stderr, completed.returncode, tuple(command))


class RemoteClaudeBackend:
    """Relay RGB PNGs to a company Claude host and return its JSON envelope.

    ``ssh_host`` must resolve through the user's SSH config (normally
    ``company-planner``).  Every call gets a unique remote directory and the
    directory is removed in a shell ``trap`` even when curl or Claude fails.
    """

    def __init__(
        self,
        ssh_host: str = "company-planner",
        upload_url: str = "https://tempfile.org/api/upload/local",
        download_base_url: str = "https://tempfile.org",
        timeout_s: int = 900,
        ssh_binary: str = "ssh",
        curl_binary: str = "curl",
    ):
        self.ssh_host = ssh_host
        self.upload_url = upload_url.rstrip("/")
        self.download_base_url = download_base_url.rstrip("/")
        self.timeout_s = int(timeout_s)
        self.ssh_binary = ssh_binary
        self.curl_binary = curl_binary
        if self.timeout_s <= 0 or not ssh_host or ssh_host.startswith("-"):
            raise ValueError("positive timeout and a valid SSH host are required")

    def _upload(self, image: Path) -> str:
        command = [
            self.curl_binary, "-fsS", "-X", "POST", self.upload_url,
            "-F", f"files=@{image}", "-F", "expiryHours=1",
        ]
        try:
            completed = subprocess.run(command, text=True, capture_output=True,
                                       timeout=self.timeout_s, check=False, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PlannerBackendError(f"HTTPS image upload failed: {exc}") from exc
        if completed.returncode != 0:
            raise PlannerBackendError(
                f"HTTPS image upload exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            payload: Any = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PlannerBackendError("HTTPS upload returned invalid JSON") from exc
        if isinstance(payload, dict) and payload.get("success") is False:
            raise PlannerBackendError("HTTPS relay rejected the upload")
        # tempfile.org has returned several equivalent shapes over time:
        # ``{"id": ...}``, ``{"files": [{"id": ...}]}``, and a direct URL.
        # Accept only an ID belonging to the uploaded file; never guess from
        # arbitrary response text.
        def find_file_id(value: Any) -> str | None:
            if isinstance(value, dict):
                for key in ("id", "fileId", "file_id"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                for key in ("file", "files", "data", "result"):
                    found = find_file_id(value.get(key))
                    if found:
                        return found
            elif isinstance(value, list):
                for item in value:
                    found = find_file_id(item)
                    if found:
                        return found
            return None

        direct_url = None
        if isinstance(payload, dict):
            for key in ("downloadUrl", "download_url", "url"):
                candidate = payload.get(key)
                if (isinstance(candidate, str) and urlparse(candidate).scheme == "https"
                        and urlparse(candidate).netloc == urlparse(self.download_base_url).netloc):
                    direct_url = candidate
                    break
        if direct_url:
            return direct_url
        file_id = find_file_id(payload)
        if not file_id:
            preview = completed.stdout.strip().replace("\n", " ")[:500]
            raise PlannerBackendError(
                "HTTPS upload response did not contain a file ID "
                f"(response={preview!r})"
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise PlannerBackendError("HTTPS upload returned an invalid file ID")
        return f"{self.download_base_url}/{file_id}/download"

    def invoke(self, *, prompt: str, image_paths: Iterable[Path],
               schema: dict[str, Any], system_prompt: str,
               timeout_s: int | None = None) -> BackendResult:
        call_timeout = self.timeout_s if timeout_s is None else int(timeout_s)
        if call_timeout <= 0:
            raise PlannerBackendError("remote timeout must be positive")
        images = [Path(path).resolve() for path in image_paths]
        if not images:
            raise PlannerBackendError("remote planner requires at least one RGB image")
        if any(path.suffix.lower() != ".png" or not path.is_file() for path in images):
            raise PlannerBackendError("remote planner accepts existing PNG images only")
        for path in images:
            with path.open("rb") as stream:
                if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise PlannerBackendError("remote planner image is not a PNG")
        # Uploading is intentionally sequential: each URL is short-lived and the
        # relay service has a small request quota.
        urls = [self._upload(path) for path in images]
        job = f"/tmp/cloth_remote_{uuid.uuid4().hex}"
        quoted_job = shlex.quote(job)
        downloads = " && ".join(
            f"curl -fsSL --connect-timeout 20 --max-time 120 {shlex.quote(url)} "
            f"-o {quoted_job}/image_{i}.png && "
            f"printf '%s  %s\\n' {hashlib.sha256(images[i].read_bytes()).hexdigest()} "
            f"{quoted_job}/image_{i}.png | sha256sum -c - >&2"
            for i, url in enumerate(urls)
        )
        # Claude receives the prompt through stdin.  This avoids putting a large
        # prompt or image paths into the SSH command line.
        remote = (
            f"set -eu; trap 'rm -rf {quoted_job}' EXIT; "
            f"mkdir -p {quoted_job}; {downloads} || exit $?; "
            f"timeout {call_timeout}s claude -p --output-format json --permission-mode dontAsk "
            f"--allowedTools Read --tools Read --no-session-persistence "
            f"--add-dir {quoted_job} --json-schema {shlex.quote(json.dumps(schema, separators=(',', ':')))} "
            f"--system-prompt {shlex.quote(system_prompt)}"
        )
        # Tell remote Claude where the downloaded files are without exposing any
        # local paths or depth/XYZ artifacts.
        remote_prompt = (
            f"{prompt}\n\nRGB files available to inspect:\n" +
            "\n".join(f"- {job}/image_{i}.png" for i in range(len(images))) +
            "\nReturn only the requested JSON object."
        )
        ssh = [self.ssh_binary, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", self.ssh_host]
        command = [*ssh, remote]
        try:
            completed = subprocess.run(
                command, input=remote_prompt, text=True, capture_output=True,
                timeout=call_timeout, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PlannerBackendError(f"remote Claude SSH invocation failed: {exc}") from exc
        finally:
            # A local SSH timeout may prevent the remote shell's EXIT trap.
            # The independent best-effort cleanup is bounded and job-specific.
            try:
                subprocess.run([*ssh, f"rm -rf -- {quoted_job}"], input="", text=True,
                               capture_output=True, timeout=25, check=False, shell=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if completed.returncode != 0:
            raise PlannerBackendError(
                f"remote Claude exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        parse_claude_json(completed.stdout)
        return BackendResult(completed.stdout, completed.stderr, completed.returncode, tuple(command))
