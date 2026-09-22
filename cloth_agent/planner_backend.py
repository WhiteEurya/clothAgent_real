"""Claude planner execution backends.

The remote backend deliberately transports RGB PNGs over HTTPS.  SSH carries
only the small orchestration command and Claude's structured response.
"""

from __future__ import annotations

import json
import base64
import hashlib
import os
import re
import queue
import shlex
import signal
import subprocess
import threading
import traceback
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .claude_stream import ClaudeStreamProgress


def claude_result_envelope(stdout: str) -> dict[str, Any]:
    """Accept legacy JSON or the terminal result of verbose stream-json.

    An assistant/tool message is never a substitute for a terminal result.
    """
    try:
        outer = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        results = []
        for line in (stdout or '').splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlannerBackendError('Claude returned an invalid JSON stream') from exc
            if isinstance(item, dict) and item.get('type') == 'result':
                results.append(item)
        if len(results) != 1:
            raise PlannerBackendError('Claude stream did not contain exactly one final result')
        outer = results[0]
    if not isinstance(outer, dict) or outer.get('type') in {'assistant', 'user', 'system', 'stream_event'}:
        raise PlannerBackendError('Claude returned no final result envelope')
    return outer


def parse_claude_json(stdout: str) -> dict[str, Any]:
    """Decode the CLI envelope and a fenced/prose-wrapped planner object."""
    outer = claude_result_envelope(stdout)
    if not isinstance(outer, dict) or outer.get("is_error") is True:
        detail = outer.get("result") if isinstance(outer, dict) else None
        if detail:
            detail = str(detail).replace("\n", " ")[:1200]
            raise PlannerBackendError(f"Claude returned an error envelope: {detail}")
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
    timings: dict[str, float] = field(default_factory=dict)
    image_tool_events: tuple[dict[str, Any], ...] = ()
    image_sources: tuple[dict[str, Any], ...] = ()


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
        image_tools: bool = True,
        max_turns: int = 16,
    ):
        self.ssh_host = ssh_host
        self.upload_url = upload_url.rstrip("/")
        self.download_base_url = download_base_url.rstrip("/")
        self.timeout_s = int(timeout_s)
        self.ssh_binary = ssh_binary
        self.curl_binary = curl_binary
        self.image_tools = bool(image_tools)
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError('max_turns must be a positive integer')
        self.max_turns = max_turns
        self._upload_cache = {}
        self._transfer_deadline = None
        self._overall_deadline = None
        self.progress_callback = None
        self.last_timings: dict[str, float] = {}
        self.last_image_tool_events: list[dict[str, Any]] = []
        self._debug_session = None
        self._seen_events = set()
        self._seen_timings = set()
        if self.timeout_s <= 0 or not ssh_host or ssh_host.startswith("-"):
            raise ValueError("positive timeout and a valid SSH host are required")

    def _remaining_timeout(self, limit):
        if self._overall_deadline is None:
            return limit
        remaining = self._overall_deadline - time.monotonic()
        if remaining <= 0:
            raise PlannerBackendError('remote Claude overall deadline exceeded')
        return min(limit, remaining)

    def _upload_timeout(self):
        remaining = (self._transfer_deadline - time.monotonic()
                     if self._transfer_deadline is not None else 180.)
        if remaining <= 0:
            raise PlannerBackendError("image transfer batch deadline exceeded")
        return min(self._remaining_timeout(30), remaining)

    def _upload(self, image: Path) -> str:
        command = [
            self.curl_binary, "-fsS", "--http1.1", "--connect-timeout", "10",
            "--max-time", "30", "--speed-limit", "1024", "--speed-time", "15",
            "-X", "POST", self.upload_url,
            "-F", f"files=@{image}", "-F", "expiryHours=1",
        ]
        for attempt in range(1, 4):
            try:
                completed = subprocess.run(command, text=True, capture_output=True,
                                           timeout=self._upload_timeout(),
                                           check=False, shell=False)
            except subprocess.TimeoutExpired as exc:
                failure = PlannerBackendError("HTTPS image upload timed out")
                retryable = True
            except OSError as exc:
                raise PlannerBackendError(f"HTTPS image upload failed: {exc}") from exc
            else:
                if completed.returncode == 0:
                    break
                failure = PlannerBackendError(
                    f"HTTPS image upload exited with {completed.returncode}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
                retryable = completed.returncode in {5, 6, 7, 18, 28, 35, 52, 55, 56, 92}
            if not retryable or attempt == 3:
                raise failure
            delay = 2.0 * attempt
            if self._remaining_timeout(delay + .1) <= delay:
                raise failure
            self._progress('image_upload', 'retry', image_name=image.name,
                           attempt=attempt, next_attempt=attempt + 1, max_attempts=3,
                           retry_delay_s=delay, reason=str(failure))
            time.sleep(delay)
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
               timeout_s: int | None = None, debug_dir: Path | None = None,
               image_edit_limit: int | None = None, max_turns: int | None = None,
               overall_timeout_s: float | None = None,
               orientation_correction: bool = False) -> BackendResult:
        if overall_timeout_s is not None and (type(overall_timeout_s) not in (int, float)
                or not 0 < overall_timeout_s < float('inf')):
            raise ValueError('overall_timeout_s must be finite and positive')
        self._overall_deadline = None if overall_timeout_s is None else time.monotonic() + overall_timeout_s
        if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
            raise ValueError('max_turns must be a positive integer')
        self._call_max_turns = self.max_turns if max_turns is None else max_turns
        if image_edit_limit is not None and (type(image_edit_limit) is not int or not 0 <= image_edit_limit <= 24):
            raise ValueError('image_edit_limit must be an integer in [0, 24]')
        self._image_edit_limit = image_edit_limit
        if orientation_correction and (not self.image_tools or image_edit_limit is None):
            raise ValueError('orientation correction requires audited image tools and a finite edit budget')
        self._orientation_correction = bool(orientation_correction)
        self.last_timings = {}
        self.last_image_tool_events = []
        self._seen_events = set()
        self._seen_timings = set()
        self._debug_session = None
        image_paths = list(image_paths)
        if debug_dir is not None:
            from .claude_image_debug import ImageDebugSession
            self._debug_session = ImageDebugSession(debug_dir, image_paths,
                {"prompt": prompt, "system_prompt": system_prompt, "schema": schema,
                 "image_edit_limit": image_edit_limit,
                 "max_turns": self._call_max_turns,
                 "overall_timeout_s": overall_timeout_s,
                 "orientation_correction": self._orientation_correction,
                 "image_paths": [str(p) for p in image_paths]})
        started = time.monotonic()
        try:
            result = self._invoke(prompt=prompt, image_paths=image_paths, schema=schema,
                                  system_prompt=system_prompt, timeout_s=timeout_s)
        except BaseException as exc:
            self.last_timings["total_s"] = time.monotonic() - started
            exc.timings = dict(self.last_timings)
            exc.image_tool_events = tuple(self.last_image_tool_events)
            self._progress("call", "failed", self.last_timings["total_s"])
            if self._debug_session:
                self._debug_session.append_stream("exception.log", traceback.format_exc())
                self._debug_session.finish("INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                                           f"{type(exc).__name__}: {exc}")
            raise
        self.last_timings["total_s"] = time.monotonic() - started
        self._progress("call", "completed", self.last_timings["total_s"])
        if self._debug_session:
            self._debug_session.finish("COMPLETED")
        return replace(result, timings=dict(self.last_timings),
                       image_tool_events=tuple(self.last_image_tool_events),
                       image_sources=tuple(self._debug_session.state['views']) if self._debug_session else ())

    def _progress(self, stage, event, duration_s=None, **details):
        if self._debug_session:
            self._debug_session.progress(stage, event, duration_s, details)
        if self.progress_callback is not None:
            self.progress_callback(stage, event, duration_s, **details)

    def _finish_phase(self, stage, started):
        duration = time.monotonic() - started
        self.last_timings[f"{stage}_s"] = duration
        self._progress(stage, "finished", duration)

    def _remote_timings(self, stderr):
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        for stage, ns in re.findall(r"^__CLOTH_TIMING__ (download_\d+|hash_\d+|claude) (\d+)$", stderr or "", re.M):
            if (stage, ns) in self._seen_timings:
                continue
            self._seen_timings.add((stage, ns))
            value = int(ns) / 1e9
            self.last_timings[f"remote_{stage}_s"] = value
            self._progress(f"remote_{stage}", "measured", value)
        for raw in re.findall(r"^__CLOTH_IMAGE_TOOL__ (.+)$", stderr or "", re.M):
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                identity = event.get("event_id") or json.dumps(event, sort_keys=True)
                if identity in self._seen_events:
                    continue
                self._seen_events.add(identity)
                self.last_image_tool_events.append(event)
                if self._debug_session:
                    self._debug_session.consume(event)
                self._progress("image_tool", event.get("status", "unknown"),
                               event.get("duration_s"), tool=event.get("tool"),
                               edit_budget=event.get('edit_budget'))
                if event.get('kind') == 'orientation_guard':
                    self._progress('orientation_correction', event['status'],
                                   classification=event.get('classification'),
                                   audit_facts=event.get('audit_facts'))

    def _image_tool_setup(self, job, count):
        """Stage the small tool implementation over SSH, never additional RGB."""
        from .image_tools_mcp import INSTRUCTIONS, TOOL_NAMES

        source = Path(__file__).with_name("image_tools_mcp.py").read_bytes()
        encoded = base64.b64encode(source).decode("ascii")
        bootstrap = ("import base64,pathlib; "
                     f"pathlib.Path({job + '/image_tools.py'!r}).write_bytes(base64.b64decode({encoded!r}))")
        quoted_job = shlex.quote(job)
        limit = getattr(self, '_image_edit_limit', None)
        budget_flag = f' --edit-limit {limit}' if limit is not None else ''
        correction_flag = ' --orientation-correction' if getattr(self, '_orientation_correction', False) else ''
        setup = (
            'cloth_image_python=${CLOTH_REMOTE_IMAGE_PYTHON:-python3}; '
            f'"$cloth_image_python" -c {shlex.quote(bootstrap)}; '
            f'"$cloth_image_python" {quoted_job}/image_tools.py --prepare '
            f'--job {quoted_job} --image-count {count}{budget_flag}{correction_flag}; '
        )
        allowed = TOOL_NAMES if limit != 0 else tuple(t for t in TOOL_NAMES
            if not t.endswith(('__crop_image', '__rotate_image', '__resize_image')))
        flags = (f"--allowedTools {shlex.quote(','.join(('Read', *allowed)))} --tools Read "
                 f"--mcp-config {quoted_job}/image_tools.mcp.json --strict-mcp-config "
                 f"--settings {quoted_job}/image_tools.settings.json "
                 "--disable-slash-commands ")
        prompt = (f"\n\nImage inspection tool list: {job}/tool_list.json\n" + INSTRUCTIONS)
        if limit is not None:
            prompt += (f' This call has at most {limit} edit attempts total (rotation/crop/resize combined). '
                       'Each result reports edit_budget. At zero, use existing views to finish, or report insufficient evidence under the requested schema. '
                       'Do not repeatedly request denied edits. Invalid edit attempts also consume budget.')
        return setup, flags, prompt

    def _run_streaming(self, command, prompt, timeout_s):
        """Drain SSH pipes continuously so image audit reaches Viser mid-call."""
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1,
                                   start_new_session=True)
        inbox = queue.Queue()
        output = {"stdout": [], "stderr": []}

        def reader(name, stream):
            try:
                for line in stream:
                    inbox.put((name, line))
            finally:
                inbox.put((name, None))

        def writer():
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

        threads = [threading.Thread(target=reader, args=(name, getattr(process, name)), daemon=True)
                   for name in output]
        threads.append(threading.Thread(target=writer, daemon=True))
        for thread in threads:
            thread.start()
        started = time.monotonic()
        stream_progress = ClaudeStreamProgress(self._progress)
        closed = set()

        def receive(name, line):
            if line is None:
                closed.add(name)
                return
            output[name].append(line)
            self._debug_session.append_stream(name + ".log", line)
            if name == "stderr":
                self._remote_timings(line)
            else:
                self._debug_session.consume_claude_line(line, on_event=stream_progress.consume)

        try:
            while len(closed) < 2 or process.poll() is None:
                if time.monotonic() - started >= timeout_s:
                    raise subprocess.TimeoutExpired(command, timeout_s,
                        output=''.join(output['stdout']), stderr=''.join(output['stderr']))
                try:
                    receive(*inbox.get(timeout=.1))
                except queue.Empty:
                    pass
                stream_progress.tick()
                self._debug_session.flush()
        finally:
            # Stop inherited pipe owners too. Killing only the shell can leave
            # Claude/audit children alive and stream.close() blocked forever.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            for thread in threads:
                thread.join(timeout=1)
            while not inbox.empty():
                receive(*inbox.get_nowait())
            stream_progress.tick(force=True)
            self._debug_session.flush(force=True)
            for stream in (process.stdout, process.stderr):
                stream.close()
        return subprocess.CompletedProcess(command, process.returncode,
                                           "".join(output["stdout"]), "".join(output["stderr"]))

    def _invoke(self, *, prompt, image_paths, schema, system_prompt, timeout_s):
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
        self._transfer_deadline = time.monotonic() + 180.0
        urls = []
        for i, path in enumerate(images):
            started = time.monotonic()
            self._progress(f"upload_{i}", "started", image_name=path.name,
                           bytes=path.stat().st_size, image_count=len(images))
            try:
                self._upload_timeout()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                cached = self._upload_cache.get(digest)
                if cached and time.monotonic() < cached[1]:
                    urls.append(cached[0])
                    self._progress("image_upload", "cache_hit", image_name=path.name, sha256=digest)
                else:
                    url = self._upload(path)
                    self._upload_cache[digest] = (url, time.monotonic() + 2400.)
                    urls.append(url)
            finally:
                self._finish_phase(f"upload_{i}", started)
        transfer_remaining = max(1, int(self._transfer_deadline - time.monotonic()))
        call_timeout = self._remaining_timeout(call_timeout)
        job = f"/tmp/cloth_remote_{uuid.uuid4().hex}"
        quoted_job = shlex.quote(job)
        setup, tool_flags, tool_prompt = (self._image_tool_setup(job, len(images))
            if self.image_tools else ("", "--allowedTools Read --tools Read --strict-mcp-config ", ""))
        # Retry each GET at most twice, overwriting partial output on retry.
        # Use a shell loop for compatibility with older remote curl versions.
        # Keep hash verification before any model invocation.
        downloads = " && ".join(
            f"cloth_stage=download_{i} && cloth_begin=$(date +%s%N) && "
            f"( cloth_cache=\"$HOME/.cache/cloth-agent-images/{hashlib.sha256(images[i].read_bytes()).hexdigest()}.png\"; "
            f"if [ -f \"$cloth_cache\" ] && printf '%s  %s\\n' {hashlib.sha256(images[i].read_bytes()).hexdigest()} \"$cloth_cache\" | sha256sum -c - >/dev/null 2>&1; then "
            f"cp -- \"$cloth_cache\" {quoted_job}/image_{i}.png; "
            f"else cloth_attempt=1; while :; do "
            f"if curl -fsSL --http1.1 --connect-timeout 10 --max-time 30 --speed-limit 1024 --speed-time 15 {shlex.quote(url)} "
            f"-o {quoted_job}/image_{i}.png; then break; else cloth_rc=$?; fi; "
            'if [ "$cloth_attempt" -ge 3 ]; then exit "$cloth_rc"; fi; '
            'case "$cloth_rc" in 5|6|7|18|28|35|52|55|56|92) ;; *) exit "$cloth_rc" ;; esac; '
            f"printf 'download_{i}: retry after curl exit %s (attempt %s/3)\\n' "
            '\"$cloth_rc\" \"$cloth_attempt\" >&2; '
             'sleep 2; cloth_attempt=$((cloth_attempt + 1)); done; '
            f"printf '%s  %s\\n' {hashlib.sha256(images[i].read_bytes()).hexdigest()} {quoted_job}/image_{i}.png | sha256sum -c - >&2 || exit 1; "
            f"mkdir -p \"$HOME/.cache/cloth-agent-images\"; cp {quoted_job}/image_{i}.png \"$cloth_cache.$$.tmp\" && mv \"$cloth_cache.$$.tmp\" \"$cloth_cache\"; fi ) && cloth_done && "
            f"cloth_stage=hash_{i} && cloth_begin=$(date +%s%N) && "
            f"printf '%s  %s\\n' {hashlib.sha256(images[i].read_bytes()).hexdigest()} "
            f"{quoted_job}/image_{i}.png | sha256sum -c - >&2 && cloth_done"
            for i, url in enumerate(urls)
        )
        download_script = (
            "set -eu; cloth_begin=0; cloth_stage=init; "
            "cloth_done() { cloth_end=$(date +%s%N); "
            "printf '__CLOTH_TIMING__ %s %s\\n' \"$cloth_stage\" \"$((cloth_end-cloth_begin))\" >&2; "
            "cloth_begin=0; }; " + downloads)
        # Claude receives the prompt through stdin.  This avoids putting a large
        # prompt or image paths into the SSH command line.
        remote = (
            "set -eu; cloth_begin=0; cloth_stage=init; cloth_audit_pid=; "
            "cloth_done() { if [ \"$cloth_begin\" != 0 ]; then "
            "cloth_end=$(date +%s%N); "
            "printf '__CLOTH_TIMING__ %s %s\\n' \"$cloth_stage\" \"$((cloth_end-cloth_begin))\" >&2; "
            "cloth_begin=0; fi; }; "
            f"trap 'cloth_rc=$?; "
            'if [ -n "$cloth_audit_pid" ]; then kill "$cloth_audit_pid" 2>/dev/null || true; '
            'wait "$cloth_audit_pid" 2>/dev/null || true; fi; '
            # Child runtimes can leave the inherited SSH pipe nonblocking.
            # Restore blocking writes after stopping the forwarder, before
            # flushing the complete audit. Do not mask audit delivery errors.
            '\"${cloth_image_python:-python3}\" -c \"import os; os.set_blocking(1, True); os.set_blocking(2, True)\"; '
            'cloth_done; '
            f"if [ -f {quoted_job}/image_tool_calls.jsonl ]; then "
            f"sed \"s/^/__CLOTH_IMAGE_TOOL__ /\" {quoted_job}/image_tool_calls.jsonl >&2; fi; "
            'printf "__CLOTH_IMAGE_TOOL__ {\\\"tool\\\":\\\"audit_finished\\\",\\\"status\\\":\\\"ok\\\"}\\n" >&2; '
            f"rm -rf {quoted_job}; exit $cloth_rc' EXIT; "
            f"mkdir -p {quoted_job}; timeout {transfer_remaining}s sh -c {shlex.quote(download_script)} || exit $?; "
            f"{setup}cd {quoted_job}; "
            + (f'"$cloth_image_python" {quoted_job}/image_tools.py --audit-forward --job {quoted_job} '
               f'--image-count {len(images)} < /dev/null & cloth_audit_pid=$!; ' if self.image_tools else "") +
            "cloth_stage=claude; cloth_begin=$(date +%s%N); "
            f"timeout {call_timeout}s claude -p --output-format stream-json --verbose --include-partial-messages --permission-mode dontAsk "
            f"{tool_flags}--no-session-persistence --max-turns {self._call_max_turns} "
            f"--add-dir {quoted_job} --json-schema {shlex.quote(json.dumps(schema, separators=(',', ':')))} "
            f"--system-prompt {shlex.quote(system_prompt)}"
        )
        # Tell remote Claude where the downloaded files are without exposing any
        # local paths or depth/XYZ artifacts.
        remote_prompt = (
            f"{prompt}\n\nRGB files available to inspect:\n" +
            "\n".join(f"- {job}/image_{i}.png" for i in range(len(images))) +
            tool_prompt +
            f"\nThis call allows at most {self._call_max_turns} model turns. Reuse saved image history and finish promptly. "
            "If evidence is insufficient, report it using the requested schema; never invent an action. " +
            "\nReturn only the requested JSON object."
        )
        ssh = [self.ssh_binary, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", self.ssh_host]
        command = [*ssh, remote]
        if self._debug_session:
            self._debug_session.request.update(remote_prompt=remote_prompt, command=command)
            self._debug_session.write("request.json", self._debug_session.request)
            self._debug_session.save_prompt(remote_prompt)
        started = time.monotonic()
        self._progress("ssh_download_and_claude", "started", image_count=len(images))
        completed = None
        try:
            completed = (self._run_streaming(command, remote_prompt, call_timeout) if self._debug_session else subprocess.run(
                command, input=remote_prompt, text=True, capture_output=True,
                timeout=call_timeout, check=False, shell=False,
            ))
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._remote_timings(getattr(exc, "stderr", ""))
            detail = (f"timed out after {call_timeout}s" if isinstance(exc, subprocess.TimeoutExpired)
                      else str(exc))
            raise PlannerBackendError(f"remote Claude SSH invocation failed: {detail}") from exc
        finally:
            self._finish_phase("ssh_download_and_claude", started)
            if completed is not None:
                self._remote_timings(completed.stderr)
                # Preserve the complete CLI streams in the run artifact.  The
                # live fold log intentionally truncates long stderr messages,
                # which previously hid the actual schema/CLI rejection.
                if self._debug_session is not None:
                    (self._debug_session.directory / 'claude_stdout.txt').write_text(completed.stdout or '', encoding='utf-8')
                    (self._debug_session.directory / 'claude_stderr.txt').write_text(completed.stderr or '', encoding='utf-8')
            # A local SSH timeout may prevent the remote shell's EXIT trap.
            # The independent best-effort cleanup is bounded and job-specific.
            cleanup_started = time.monotonic()
            self._progress("cleanup", "started")
            try:
                cleanup = subprocess.run([*ssh, f"rm -rf -- {quoted_job}"], input="", text=True,
                               capture_output=True, timeout=25, check=False, shell=False)
                if cleanup.returncode:
                    self._progress("cleanup", "failed", returncode=cleanup.returncode)
            except (OSError, subprocess.TimeoutExpired):
                self._progress("cleanup", "failed")
            finally:
                self._finish_phase("cleanup", cleanup_started)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            try:
                envelope = claude_result_envelope(completed.stdout)
                if isinstance(envelope, dict):
                    detail = json.dumps({key: envelope[key] for key in
                        ('subtype', 'stop_reason', 'terminal_reason', 'num_turns', 'result', 'errors')
                        if key in envelope}, ensure_ascii=False) + '\n' + detail
            except (ValueError, TypeError, PlannerBackendError):
                pass
            raise PlannerBackendError(
                f"remote Claude exited with {completed.returncode}: "
                f"{detail}"
            )
        parse_started = time.monotonic()
        try:
            parse_claude_json(completed.stdout)
        finally:
            self._finish_phase("json_parse", parse_started)
        # Preserve the raw stream in stdout.log; existing planner consumers
        # continue receiving the same final JSON envelope as before.
        return BackendResult(json.dumps(claude_result_envelope(completed.stdout)),
                             completed.stderr, completed.returncode, tuple(command))
