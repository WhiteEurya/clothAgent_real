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
        raise SystemExit(130)
