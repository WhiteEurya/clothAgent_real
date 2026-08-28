#!/usr/bin/env python3
"""Run the standalone Claude garment-folding experiment."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.neat_fold_pipeline import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
