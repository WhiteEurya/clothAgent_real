"""Claude planner execution backends.

The remote backend deliberately transports RGB PNGs over HTTPS.  SSH carries
only the small orchestration command and Claude's structured response.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
                if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
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
        return f"{self.download_base_url}/{file_id}/download"

    def invoke(self, *, prompt: str, image_paths: Iterable[Path],
               schema: dict[str, Any], system_prompt: str) -> BackendResult:
        images = [Path(path).resolve() for path in image_paths]
        if not images:
            raise PlannerBackendError("remote planner requires at least one RGB image")
        if any(path.suffix.lower() != ".png" or not path.is_file() for path in images):
            raise PlannerBackendError("remote planner accepts existing PNG images only")
        # Uploading is intentionally sequential: each URL is short-lived and the
        # relay service has a small request quota.
        urls = [self._upload(path) for path in images]
        job = f"/tmp/cloth_remote_{uuid.uuid4().hex}"
        quoted_job = shlex.quote(job)
        downloads = " && ".join(
            f"{shlex.quote(self.curl_binary)} -fsSL {shlex.quote(url)} -o {quoted_job}/image_{i}.png"
            for i, url in enumerate(urls)
        )
        # Claude receives the prompt through stdin.  This avoids putting a large
        # prompt or image paths into the SSH command line.
        remote = (
            f"set -eu; trap 'rm -rf {quoted_job}' EXIT; "
            f"mkdir -p {quoted_job}; {downloads}; "
            f"claude -p --output-format json --permission-mode dontAsk "
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
        command = [self.ssh_binary, self.ssh_host, remote]
        try:
            completed = subprocess.run(
                command, input=remote_prompt, text=True, capture_output=True,
                timeout=self.timeout_s, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PlannerBackendError(f"remote Claude SSH invocation failed: {exc}") from exc
        if completed.returncode != 0:
            raise PlannerBackendError(
                f"remote Claude exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return BackendResult(completed.stdout, completed.stderr, completed.returncode, tuple(command))
