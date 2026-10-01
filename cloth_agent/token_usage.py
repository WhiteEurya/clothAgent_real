"""Actual Claude CLI usage, recorded once per process attempt before validation."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
MODEL_FIELDS = dict(zip(TOKEN_FIELDS, (
    "inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens",
)))
LOG = logging.getLogger(__name__)


def _envelope(stdout):
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    try:
        value = json.loads(stdout)
    except (TypeError, ValueError):
        results = []
        for line in (stdout or "").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("type") == "result":
                results.append(item)
        return results[0] if len(results) == 1 else {}
    return value if isinstance(value, dict) and value.get("type") in (None, "result") else {}


def _number(value, *, integer=False):
    if type(value) not in (int, float) or value < 0:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if integer:
        return int(value) if int(value) == value else None
    return value


def _usage(raw, fields, cost):
    raw = raw if isinstance(raw, dict) else {}
    result = {key: _number(raw.get(source), integer=True) for key, source in fields.items()}
    result["total_tokens"] = (sum(result.values()) if all(v is not None for v in result.values()) else None)
    result["total_cost_usd"] = _number(cost)
    return result


def parse_usage(stdout):
    """Use terminal totals only; assistant events and per-model totals overlap."""
    envelope = _envelope(stdout)
    models = {
        name: _usage(details, MODEL_FIELDS, details.get("costUSD"))
        for name, details in (envelope.get("modelUsage") or {}).items()
        if isinstance(details, dict)
    } if isinstance(envelope.get("modelUsage"), dict) else {}
    usage = _usage(envelope.get("usage"), dict(zip(TOKEN_FIELDS, TOKEN_FIELDS)), envelope.get("total_cost_usd"))
    # Some CLI versions report only per-model counts. Never add both sources.
    for key in (*TOKEN_FIELDS, "total_cost_usd"):
        if usage[key] is None and models and all(model[key] is not None for model in models.values()):
            usage[key] = sum(model[key] for model in models.values())
    usage["total_tokens"] = (sum(usage[key] for key in TOKEN_FIELDS)
                             if all(usage[key] is not None for key in TOKEN_FIELDS) else None)
    return {**usage, "models": models, "session_id": envelope.get("session_id"),
            "reported": any(usage[key] is not None for key in TOKEN_FIELDS)}


def usage_directory(path):
    """Find the run root from a workspace or a nested debug artifact directory."""
    root = Path(path).resolve()
    for candidate in (root, *root.parents):
        if (candidate / "run_metadata.json").is_file():
            return candidate / "results"
        if candidate.name == "workspace":
            return candidate.parent / "results"
    return root / "results"


def record_usage(run_dir, *, stage, backend, stdout="", returncode=None, status=None):
    if run_dir is None:
        return
    usage = parse_usage(stdout)
    envelope = _envelope(stdout)
    record = {
        "schema_version": 1, "call_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "stage": stage, "backend": backend, "returncode": returncode,
        "status": status or ("failed" if returncode or envelope.get("is_error")
                              or str(envelope.get("subtype", "")).startswith("error") else "completed"),
        **usage,
    }
    try:
        directory = usage_directory(run_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "token_usage.jsonl").open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)
    except OSError:
        # Accounting must not change robot control flow or mask a model error.
        LOG.warning("Could not save Claude token usage", exc_info=True)


def tracked_call(runner, *args, usage_run_dir, usage_stage, usage_backend="local", **kwargs):
    """Preserve the runner's result/exception and count rejected outputs too."""
    try:
        result = runner(*args, **kwargs)
    except BaseException as exc:
        record_usage(usage_run_dir, stage=usage_stage, backend=usage_backend,
                     stdout=getattr(exc, "stdout", "") or "",
                     status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed")
        raise
    record_usage(usage_run_dir, stage=usage_stage, backend=usage_backend,
                 stdout=result.stdout, returncode=result.returncode)
    return result


def _totals(records):
    fields = (*TOKEN_FIELDS, "total_tokens", "total_cost_usd")
    return {
        "calls": len(records),
        "calls_with_usage": sum(bool(r.get("reported")) for r in records),
        "calls_without_usage": sum(not r.get("reported") for r in records),
        **{key: sum(r[key] for r in records if r.get(key) is not None) for key in fields},
        "missing_fields": {key: sum(r.get(key) is None for r in records) for key in fields},
    }


def summarize_usage(run_dir):
    path = usage_directory(run_dir) / "token_usage.jsonl"
    records = []
    invalid = 0
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            for line in stream:
                try:
                    value = json.loads(line)
                    if (not isinstance(value, dict) or value.get("schema_version") != 1
                            or not isinstance(value.get("stage"), str)
                            or not isinstance(value.get("models"), dict)):
                        raise ValueError("invalid usage record")
                    records.append(value)
                except ValueError:
                    invalid += 1
    stages = {}
    models = {}
    for record in records:
        stages.setdefault(record["stage"], []).append(record)
        for name, usage in record["models"].items():
            models.setdefault(name, []).append({**usage, "reported": any(usage[k] is not None for k in TOKEN_FIELDS)})
    return {"schema_version": 1, "source": str(path), "invalid_records": invalid,
            "totals": _totals(records),
            "by_stage": {key: _totals(items) for key, items in sorted(stages.items())},
            "by_model": {key: _totals(items) for key, items in sorted(models.items())}}


def format_summary(summary):
    totals = summary["totals"]
    lines = [f"Claude token usage: {totals['calls']} calls; {totals['calls_without_usage']} without reported usage"]
    for key in (*TOKEN_FIELDS, "total_tokens", "total_cost_usd"):
        missing = totals["missing_fields"][key]
        suffix = f" (unknown for {missing} calls; known subtotal only)" if missing else ""
        lines.append(f"  {key}: {totals[key]:,}{suffix}")
    for stage, values in summary["by_stage"].items():
        lines.append(f"  [{stage}] calls={values['calls']}, total_tokens={values['total_tokens']:,}, unknown_totals={values['missing_fields']['total_tokens']}")
    if summary["invalid_records"]:
        lines.append(f"  Unreadable records: {summary['invalid_records']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    if not Path(args.run_dir).is_dir():
        parser.error("--run-dir must be an existing directory")
    summary = summarize_usage(args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.as_json else format_summary(summary))
    return 0
