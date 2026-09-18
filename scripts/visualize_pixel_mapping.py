#!/usr/bin/env python3
"""Export saved RGB-D as RGB grid + base XY; never connect to hardware."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.pixel_mapping_visualization import PixelMappingView
from scripts.check_camera_calibration import load_capture, deproject, to_base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perception', type=Path, required=True, help='Saved result.json or its directory')
    parser.add_argument('--camera', default='A')
    parser.add_argument('--pixel', nargs=2, type=int, action='append', default=[], metavar=('U', 'V'))
    parser.add_argument('--grid-mm', type=float, default=50)
    parser.add_argument('--output', type=Path, required=True, help='Output PNG file')
    args = parser.parse_args(argv)
    capture = load_capture(args.perception, args.camera)
    xyz = capture['xyz']
    if xyz is None:
        yy, xx = np.indices(capture['depth'].shape)
        xyz = to_base(capture, deproject(capture['k'], np.stack((xx, yy), axis=-1), capture['depth']))
        xyz[..., 2] += capture['offset']
    valid = np.isfinite(capture['depth']) & (capture['depth'] > 0) & np.isfinite(xyz).all(axis=2)
    view = PixelMappingView(np.asarray(capture['rgb']), xyz, valid, args.grid_mm)
    result = view.render(args.pixel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    for index, (u, v) in enumerate(args.pixel, start=1):
        print(f'P{index}: raw pixel ({u}, {v}) -> computed XYZ {xyz[v, u].tolist()} mm')
    print(f'Saved {args.output}; this shows the computed mapping, not measured physical error.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
