#!/usr/bin/env python3
"""Give a task directly to the locally installed Codex (GPT-6 Astra, high)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    # Keep explicit inspection/offline modes available; default is the full loop.
    if '--single' in sys.argv:
        sys.argv.remove('--single')
        from cloth_agent.codex_astra_pipeline import main
    elif any(flag in sys.argv for flag in ('--read-only', '--dry-run', '--scene', '--state')):
        from cloth_agent.codex_astra_pipeline import main
    else:
        from cloth_agent.codex_loop import main
    raise SystemExit(main())
