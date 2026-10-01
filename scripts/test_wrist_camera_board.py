#!/usr/bin/env python3
"""Interactive, read-only wrist-camera test using the measured AprilTag board.

Reuses the diagnostic capture/analysis implementation; never commands motion.
"""
from pathlib import Path
import argparse
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag-edge-mm', type=float, default=None,
                        help='Measured black tag edge; live default 33.5 mm. Offline inherits saved scale.')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--session', type=Path, help='Re-analyze saved session without hardware')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.tag_edge_mm is not None and (not math.isfinite(args.tag_edge_mm) or args.tag_edge_mm <= 0):
        parser.error('--tag-edge-mm must be finite and positive')
    if not 1 <= args.repeats <= 20:
        parser.error('--repeats must be 1..20')
    from scripts.diagnose_apriltag_mapping import main as diagnose
    forwarded = ['--repeats', str(args.repeats)]
    if args.session:
        forwarded += ['--session', str(args.session)]
    edge = args.tag_edge_mm if args.tag_edge_mm is not None else (None if args.session else 33.5)
    if edge is not None:
        forwarded += ['--board-scale', str(edge / 48.0)]
    if args.no_preview:
        forwarded += ['--no-preview']
    if args.output_dir:
        forwarded += ['--output-dir', str(args.output_dir)]
    elif not args.session:
        forwarded += ['--output-dir', str(PROJECT_ROOT / 'results' / 'wrist_camera_board')]
    if not args.session:
        print('标定板保持固定、平整。请自行移动机械臂，停稳后再采集。', flush=True)
        print('预览窗口：空格/Enter 采集；Q/Esc 退出。终端：Enter 采集，q + Enter 退出。', flush=True)
        print(f'每次采集 {args.repeats} 组；建议至少 5 个不同观察姿态，完整看见标定板。', flush=True)
        print('脚本只读相机和机器人状态，不移动机器人，不修改 TCP 或外参。', flush=True)
    return diagnose(forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
