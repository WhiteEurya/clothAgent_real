#!/usr/bin/env python3
"""Restore a crumpled garment to the initial demonstration, with iteration memory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.codex_loop import main

if __name__ == '__main__':
    raise SystemExit(main(['--arm7-only', '--unfold', '--remember-iterations', *sys.argv[1:]]))
