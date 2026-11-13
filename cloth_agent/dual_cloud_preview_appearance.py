"""Render saved camera-only preview captures with the normal planning renderer."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import numpy as np
from PIL import Image
from .dual_cloud_observation import scale_world
from .dual_cloud_render import render_appearance
from .depth_stability import table_depth_mask


def render_saved_preview(folder, fusion_config, roi, output=None, *, scene="garment"):
    if scene not in ("garment", "calibration"):
        raise ValueError(f"unknown preview scene: {scene}")
    folder=Path(folder);output=Path(output) if output else folder/'appearance'
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((folder/'report.json').read_text())
    calibration=json.loads((folder/'calibration_used.json').read_text())
    config={**fusion_config,'calibration':calibration}
    for key in ('zed_serial','realsense_serial'):
        if calibration[key]!=fusion_config['calibration'][key]:
            raise ValueError('preview and fusion camera identity mismatch')
    transforms=[np.array(calibration['world_from_zed_mm']),
        np.array(calibration['world_from_realsense_color_mm'])@np.array(report['realsense_depth_to_color_mm'])]
    raw=[np.load(folder/'zed_world_xyz_mm.npy'),np.load(folder/'realsense_world_xyz_mm.npy')]
    raw[1]=raw[1].reshape(*np.load(folder/'realsense_native_depth_m.npy').shape,3)
    cloud=[scale_world(a,t,config['depth_scale'][n]) for a,t,n in zip(raw,transforms,['zed','realsense'])]
    # The original RGBs remain intact; these files only supply renderer input.
    rsdir=output/'realsense';rsdir.mkdir();shutil.copyfile(folder/'realsense_rgb.png',rsdir/'color_rgb.png')
    observations=[SimpleNamespace(xyz=cloud[0],image=folder/'zed_rgb.png',directory=folder,
        meta={'world_from_camera_mm':transforms[0].tolist(),'intrinsics':report['zed_K']}),
        SimpleNamespace(xyz=cloud[1],directory=rsdir,meta={'world_from_camera_mm':transforms[1].tolist()})]
    mask=table_depth_mask(cloud[0].shape[:2],roi['table_polygon_px']);T=transforms[0];K=np.array(report['zed_K']);valid=[]
    for xyz in cloud:
        keep=np.isfinite(xyz).all(2)&(xyz[:,:,2]>=roi.get('display_min_world_z_mm',-20))&(xyz[:,:,2]<=400)
        y,x=np.nonzero(keep);cam=(xyz[y,x]-T[:3,3])@T[:3,:3];front=cam[:,2]>0
        uv=np.zeros((len(cam),2),int);uv[front]=np.rint((cam[front]@K.T)[:,:2]/cam[front,2:]).astype(int)
        inside=front&(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
        ids=np.flatnonzero(inside);inside[ids]&=mask[uv[ids,1],uv[ids,0]]
        keep[:]=False;keep[y[inside],x[inside]]=True;valid.append(keep)
    result=render_appearance(observations,valid,np.array(report['topdown_center_mm']),report['topdown_span_mm'],config,
        garment_profile=roi.get('appearance_garment_profile',roi.get('segmentation_profile')) if scene=='garment' else None,
        seam_mode=roi.get('appearance_seam'))
    Image.fromarray(result['rgb']).save(output/'fused_rgb.png')
    for name in ('source_camera','texture_inferred','display_z_mm','coverage'):
        np.save(output/(name+'.npy'),result[name])
    metadata={**result['report'],'input':str(folder),'depth_scale':config['depth_scale'],
        'scene':scene,'role':'appearance_preview_not_execution_depth','renderer':'cloth_agent.dual_cloud_render.render_appearance',
        'calibration_source':report.get('calibration_source'),'robot_calls':0}
    (output/'report.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return output/'fused_rgb.png'


def main():
    import argparse
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fusion-config',type=Path,default=root/'config/fusion.unfold_dual.json')
    parser.add_argument('--view-config',type=Path,default=root/'config/topdown.unfold_zed.json')
    parser.add_argument('--scene',choices=['garment','calibration'],default='garment',
        help='Calibration scenes omit garment silhouette ownership overrides; measured depth is unchanged.')
    args=parser.parse_args()
    print(render_saved_preview(args.input,json.loads(args.fusion_config.read_text()),
          json.loads(args.view_config.read_text()),args.output,scene=args.scene))


if __name__=='__main__':main()
