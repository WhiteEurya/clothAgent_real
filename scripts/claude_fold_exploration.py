#!/usr/bin/env python3
"""CLI wrapper for the video-backed five-step folding exploration pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.fold_exploration_pipeline import main


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Keep terminal Ctrl-C quiet and return the conventional shell status.
        # The pipeline/session cleanup has already run in their finally blocks.
        print("[fold-debug] operator interrupt received; stopping", file=sys.stderr, flush=True)
        try:
            from scripts.upload_fold_debug import main as upload_debug
            print("[fold-debug] packaging and uploading lightweight diagnostics...", file=sys.stderr, flush=True)
            upload_debug(["--project-root", str(PROJECT_ROOT)])
        except Exception as exc:
            print(f"[fold-debug] diagnostic upload failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(130)
