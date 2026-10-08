#!/usr/bin/env python3
"""Return BOTH arms to saved joint Home, then open both grippers.

Running this script executes real Home automatically using the saved config.
Use --simulate for offline execution or --preflight-only for read-only planning.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.dual_arm.homing import gripper_home
from cloth_agent.dual_arm.geometry import DualArmError


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--simulate', action='store_true')
    mode.add_argument('--real', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--confirm-real', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if args.simulate and args.confirm_real:
        parser.error('--simulate cannot be combined with --confirm-real')
    try:
        result = gripper_home(args.config, output=args.output, simulated=args.simulate,
                              preflight_only=args.preflight_only)
    except DualArmError as exc:
        print(f'Home failed: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0 if result['status'] in {'COMPLETED', 'PREFLIGHT_ONLY'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
