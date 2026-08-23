#!/usr/bin/env python3
"""Explicitly approve an externally reviewed system-skill patch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.skill_lifecycle import SkillStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("patch", type=Path, help="Candidate patch file under data/skills/patches")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    patch_path = args.patch.expanduser()
    if not patch_path.is_absolute():
        patch_path = project_root / patch_path
    store = SkillStore(project_root / "data" / "skills")
    activated = store.approve_patch(
        patch_path,
        reviewer=args.reviewer,
        note=args.note,
    )
    print(
        f"approved {activated.name} v{activated.version} "
        f"from {activated.source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
