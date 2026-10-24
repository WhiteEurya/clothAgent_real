"""Standalone Arm6 manual trajectory recording with confirmation on START."""
import sys
from .gamepad_collect import main as collect_main


def main(argv=None):
    return collect_main(['--manual-only', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
