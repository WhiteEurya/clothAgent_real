#!/usr/bin/env python3
"""Offline support-height diagnostics; no calibration corrections or robot I/O.

Masks are image-space strips checked against the current observation layout.
Dark support and bright left edge are separate surfaces, not surveyed ground truth.
"""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image


def fit_plane(points):
    A=np.c_[points[:,0],np.ones(len(points))]
    keep=np.ones(len(points),bool)
    for _ in range(8):
        coeff=np.linalg.lstsq(A[keep],points[keep,2],rcond=None)[0]
        residual=points[:,2]-A@coeff
        keep=np.abs(residual-np.median(residual))<3
    return np.array([coeff[0],0.,coeff[1]]),residual,keep


def audit(folder):
    xyz=np.load(folder/'world_xyz_mm.npy',allow_pickle=False)
    rgb=np.asarray(Image.open(folder/'rgb.png'))
    if xyz.shape!=(720,1280,3):raise ValueError('ROIs require the inspected 1280x720 observation layout')
    y,x=np.indices(xyz.shape[:2]);finite=np.isfinite(xyz).all(2)
    valid=finite&(y>=100)&(y<=600)
    masks={
        'dark_support_left_strip':valid&(x>=250)&(x<=300)&(rgb.max(2)<85),
        'dark_support_right_strip':valid&(x>=935)&(x<=995)&(rgb.max(2)<85),
        'bright_left_edge':valid&(x>=205)&(x<=215)&(rgb.min(2)>140),
    }
    out={'observation':str(folder),'surfaces':{}}
    plotted={}
    for name,mask in masks.items():
        points=xyz[mask][::4]
        if len(points)<100:raise ValueError(f'insufficient points in {name}')
        coeff,residual,keep=fit_plane(points)
        bins=[]
        for lo,hi in [(400,450),(475,525),(575,625),(675,725),(750,800)]:
            z=points[(points[:,0]>=lo)&(points[:,0]<hi),2]
            if len(z):bins.append(dict(x_range_mm=[lo,hi],count=len(z),median_z_mm=float(np.median(z)),p10_z_mm=float(np.percentile(z,10)),p90_z_mm=float(np.percentile(z,90))))
        out['surfaces'][name]=dict(count=len(points),plane_z_ax_by_c=coeff.tolist(),
            slope_along_world_x_deg=float(np.degrees(np.arctan(np.linalg.norm(coeff[:2])))),
            inlier_fraction=float(keep.mean()),inlier_residual_std_mm=float(residual[keep].std()),
            fitted_z_difference_x700_minus_x400_mm=float(coeff[0]*300),bands=bins)
        plotted[name]=(points,coeff)
    # Same local Y, to isolate the slope along the arm6 -> arm7 direction.
    return out,plotted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('results/codex_astra'))
    parser.add_argument('--output',type=Path,default=Path('results/support_depth_audit'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[];latest=None;excluded=[]
    for run in sorted(args.runs.glob('loop_*')):
        for folder in sorted(run.glob('round_*/before')):
            if not (folder/'world_xyz_mm.npy').exists():continue
            try:result,latest=audit(folder)
            except ValueError as exc:
                excluded.append(dict(observation=str(folder),reason=str(exc)));continue
            rows.append(result)
    report=dict(physical_execution=False,excluded=excluded,method='Fixed inspected strips, 4-pixel decimation, iteratively trimmed Z-vs-X line fit within each narrow image strip (3 mm residual); not a full surface-normal estimate. Surfaces not independently surveyed.',observations=rows)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    if latest:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for ax,(name,(p,c)) in zip(axes,latest.items()):
            ax.scatter(p[:,0],p[:,2],s=2,alpha=.12)
            xs=np.linspace(400,800,100)
            ax.plot(xs,c[0]*xs+c[1]*np.median(p[:,1])+c[2],color='red')
            ax.set(title=name,xlabel='World X (mm): arm6 side -> arm7 side',ylabel='Measured world Z (mm)',ylim=(10,50))
        fig.suptitle(rows[-1]['observation']);fig.savefig(args.output/'height_profiles.png',dpi=150);plt.close(fig)
    for row in rows:
        print(row['observation'])
        for name,s in row['surfaces'].items():
            print(name,'slope_x_deg',round(s['slope_along_world_x_deg'],2),'delta700_400',round(s['fitted_z_difference_x700_minus_x400_mm'],2),'bands',[(b['x_range_mm'],round(b['median_z_mm'],1)) for b in s['bands']])


if __name__=='__main__':main()
