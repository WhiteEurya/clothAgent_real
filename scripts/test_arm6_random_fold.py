#!/usr/bin/env python3
"""One-shot Arm6 random-fold test, independent of collection and graph replay.

Physical test:
  python scripts/test_arm6_random_fold.py --execute
Vision-only test (no robot connection):
  python scripts/test_arm6_random_fold.py --image path/to/rgb.png

Defaults: central grasp radius <= 80 mm, lift 250–300 mm above the sampled surface,
random yaw ±45 degrees, drop within 40 mm of home XY, one retry if too similar,
TCP grasp offset -15 mm. Evidence is saved to a new timestamped reports folder.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.random_fold_reset import main

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('测试已中止。已产生的记录保留在输出目录。', flush=True)
        raise SystemExit(130)
