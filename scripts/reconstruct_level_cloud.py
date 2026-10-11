#!/usr/bin/env python3
"""Offline rigid tilt correction of a saved world cloud; no robot I/O."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from audit_support_depth import audit, fit_plane
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.depth_level import rotate_cloud


def save_ply(path, xyz, rgb):
    valid = np.isfinite(xyz).all(axis=-1)
    points = xyz[valid]
    vertices = np.empty(len(points), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                            ('red','u1'),('green','u1'),('blue','u1')])
    for i, key in enumerate(('x','y','z')):
        vertices[key] = points[:,i]
    for i, key in enumerate(('red','green','blue')):
        vertices[key] = rgb[valid][:,i]
    header = ('ply\nformat binary_little_endian 1.0\ncomment coordinates in millimeters\n'
              f'element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        vertices.tofile(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observation', type=Path)
    parser.add_argument('--pivot-mm', type=float, nargs=3, required=True)
    parser.add_argument('--angle-deg', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite([*args.pivot_mm,args.angle_deg]).all():
        parser.error('pivot and angle must be finite')
    args.output.mkdir(parents=True, exist_ok=False)
    xyz = np.load(args.observation/'world_xyz_mm.npy', allow_pickle=False)
    rgb = np.asarray(Image.open(args.observation/'rgb.png').convert('RGB'))
    corrected, rotation = rotate_cloud(xyz, args.pivot_mm, args.angle_deg)
    np.save(args.output/'corrected_world_xyz_mm.npy', corrected.astype(np.float32))
    save_ply(args.output/'corrected_cloud.ply', corrected, rgb)
    before, strips = audit(args.observation)
    report = dict(source=str(args.observation.resolve()), physical_execution=False,
                  angle_world_y_deg=args.angle_deg, pivot_world_mm=args.pivot_mm,
                  rotation=rotation.tolist(), translation_mm=(np.asarray(args.pivot_mm)-rotation@args.pivot_mm).tolist(),
                  method='Rigid point-cloud rotation, preserving clothing relief; original calibration unchanged. Strip fits are Z versus X, not full surface normals.',
                  surfaces={})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15,4), layout='constrained')
    for ax, (name,(points,coeff)) in zip(axes,strips.items()):
        leveled,_ = rotate_cloud(points,args.pivot_mm,args.angle_deg)
        fit,residual,keep = fit_plane(leveled)
        report['surfaces'][name] = dict(before=before['surfaces'][name],
            after_slope_x_deg=float(np.degrees(np.arctan(fit[0]))),
            after_z_difference_x700_minus_x400_mm=float(fit[0]*300),
            after_fit_z_ax_by_c=fit.tolist(), after_residual_std_mm=float(residual[keep].std()))
        ax.scatter(points[:,0],points[:,2],s=1,alpha=.15,label='Original')
        ax.scatter(leveled[:,0],leveled[:,2],s=1,alpha=.15,label='Corrected')
        xs=np.linspace(400,800,100)
        ax.plot(xs,coeff[0]*xs+coeff[2],color='C0')
        ax.plot(xs,fit[0]*xs+fit[2],color='C1')
        ax.set(title=name,xlabel='World X (mm)',ylabel='World Z (mm)')
        ax.legend()
    fig.suptitle(f'{args.angle_deg:g} degree world-Y rotation about arm7 reference (offline)')
    fig.savefig(args.output/'comparison.png',dpi=150)
    plt.close(fig)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
