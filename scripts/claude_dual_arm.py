#!/usr/bin/env python3
"""Independent dual-arm runtime; see docs/dual_arm_runtime.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.dual_arm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
