#!/usr/bin/env python3
"""Offline table audit for the inspected Camera A/B perception layouts.

RGB masks select visible dark support, not cloth or metal rails. Never modifies
calibration or robot state. Camera Z is reconstructed from saved world XYZ and
extrinsics, not an independently saved raw depth frame. Hidden table is unknown.
"""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def analyze(folder, camera):
    xyz=np.load(folder/'world_xyz_mm.npy',allow_pickle=False)
    rgb=np.array(Image.open(folder/'rgb.png'))
    meta=json.loads((folder/'observation.json').read_text())
    if xyz.shape!=(720,1280,3):raise ValueError('Mask requires inspected 1280x720 layout')
    expected={'A':'317222073552','B':'233622079809'}
    if str(meta['camera_serial'])!=expected[camera]:raise ValueError('Camera/mask mismatch')
    y,x=np.indices(xyz.shape[:2])
    roi=[285,25,1125,695] if camera=='B' else [250,30,1030,620]
    x0,y0,x1,y1=roi
    mask=(x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)&(rgb.max(2)<100)
    if camera=='B':mask[(y<110)&(((x>470)&(x<620))|(x>1040))]=False
    mask=binary_erosion(mask,iterations=8)
    valid=mask&np.isfinite(xyz).all(2)
    points=xyz[valid][::4]
    if len(points)<1000:raise ValueError('Insufficient exposed support')
    design=np.c_[points[:,:2],np.ones(len(points))]
    keep=np.ones(len(points),bool)
    for _ in range(8):
        coeff=np.linalg.lstsq(design[keep],points[keep,2],rcond=None)[0]
        residual=points[:,2]-design@coeff
        keep=np.abs(residual-np.median(residual))<3.
    plane=xyz[:,:,:2]@coeff[:2]+coeff[2]
    transform=np.array(meta['world_from_camera_mm'])
    camera_z=np.einsum('ijk,k->ij',xyz-transform[:3,3],transform[:3,2])
    grid=[]
    for row in range(5):
        for col in range(6):
            xa,xb=np.linspace(x0,x1,7)[col:col+2]
            ya,yb=np.linspace(y0,y1,6)[row:row+2]
            local=valid&(x>=xa)&(x<xb)&(y>=ya)&(y<yb)
            count=int(local.sum())
            grid.append(dict(row=row,col=col,pixel_center=[float((xa+xb)/2),float((ya+yb)/2)],
                count=count,median_z_mm=float(np.median(xyz[local,2])) if count>=150 else None))
    medians=[cell['median_z_mm'] for cell in grid if cell['median_z_mm'] is not None]
    report=dict(observation=str(folder),captured_at=meta['captured_at'],camera=camera,
        method='Inspected image ROI, RGB max<100, 8-pixel erosion; robust z=ax+by+c fit, 3 mm trimming. All-point residuals also reported; no calibration change.',
        limitations='Only exposed dark support. Cloth, rails, occlusion excluded. No independent surveyed height or flatness reference.',
        roi=roi,valid_pixels=int(valid.sum()),selected_pixels=int(mask.sum()),
        plane_z_ax_by_c=coeff.tolist(),tilt_deg=float(np.degrees(np.arctan(np.linalg.norm(coeff[:2])))),
        tilt_x_deg=float(np.degrees(np.arctan(coeff[0]))),tilt_y_deg=float(np.degrees(np.arctan(coeff[1]))),
        world_z_p05_p50_p95_mm=np.percentile(points[:,2],[5,50,95]).tolist(),
        camera_z_p05_p50_p95_mm=np.percentile(camera_z[valid],[5,50,95]).tolist(),
        plane_residual_all_p05_p50_p95_mm=np.percentile(residual,[5,50,95]).tolist(),
        plane_residual_all_std_mm=float(residual.std()),plane_inlier_fraction=float(keep.mean()),
        grid_median_min_max_mm=[min(medians),max(medians)],grid=grid)
    return report,(rgb,xyz,valid,camera_z,plane),points


def render(report,arrays,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb,xyz,valid,camera_z,plane=arrays
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    axes[0,0].imshow(rgb)
    axes[0,0].contour(valid,levels=[.5],colors='cyan',linewidths=.35)
    axes[0,0].set_title('Audited exposed support (cyan boundary)')
    for ax,values,title,label,limits in [
        (axes[0,1],xyz[:,:,2],'World Z; labels = regional median','World Z (mm)',(20,55)),
        (axes[1,0],xyz[:,:,2]-plane,'Residual from fitted plane','Residual (mm)',(-8,8)),
        (axes[1,1],camera_z,'Reconstructed camera-axis depth','Camera Z (mm)',(None,None)),
    ]:
        image=ax.imshow(np.where(valid,values,np.nan),cmap='turbo',vmin=limits[0],vmax=limits[1])
        ax.set_title(title);fig.colorbar(image,ax=ax,label=label)
    for cell in report['grid']:
        value=cell['median_z_mm']
        if value is not None:
            axes[0,1].text(*cell['pixel_center'],f'{value:.1f}',ha='center',va='center',
                fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    for ax in axes.flat:ax.set_xlim(0,1280);ax.set_ylim(720,0)
    fig.suptitle(f"Camera {report['camera']} | {report['captured_at']} | fitted tilt {report['tilt_deg']:.2f} deg")
    fig.savefig(path,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',choices=['A','B'],required=True)
    parser.add_argument('--observations',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for i,folder in enumerate(args.observations):
        report,arrays,points=analyze(folder,args.camera);reports.append(report)
        print(folder,'tilt_deg',round(report['tilt_deg'],3),'Z5/50/95',np.round(report['world_z_p05_p50_p95_mm'],2).tolist())
        if i==len(args.observations)-1:
            render(report,arrays,args.output/'table_depth.png')
            Image.fromarray((arrays[2]*255).astype(np.uint8)).save(args.output/'audited_mask.png')
            np.save(args.output/'support_points_mm.npy',points)
    (args.output/'report.json').write_text(json.dumps(dict(physical_motion=False,observations=reports),indent=2)+'\n')


if __name__=='__main__':main()
