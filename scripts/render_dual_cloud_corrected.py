"""Apply saved experimental scales to static captures; no board detection or hardware."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.pixel_mapping_visualization import PixelMappingView
from view_dual_cloud import read_cloud
from reconstruct_level_cloud import save_ply


def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--parameters',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 meta=json.loads((a.capture/'report.json').read_text());params=json.loads(a.parameters.read_text());cal=json.loads(Path(meta['calibration']).read_text())
 if 'ULTRA' not in meta['depth_mode'] or meta.get('zed_resolution','HD720')!='HD720' or meta.get('realsense_depth_size_wh')!=[1280,720]:raise ValueError('Capture differs from tested depth scale settings')
 a.output.mkdir(parents=True,exist_ok=False);clouds=[];colors=[]
 for name,key in [('zed','world_from_zed_mm'),('realsense','world_from_realsense_color_mm')]:
  xyz,rgb=read_cloud(a.capture/f'{name}.ply');t=np.array(cal[key])
  if name=='realsense':t=t@np.array(meta['realsense_depth_to_color_mm'])
  scale=float(params['cameras'][name]['scale'])
  if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid scale')
  native=(xyz-t[:3,3])@t[:3,:3];corrected=(native*scale)@t[:3,:3].T+t[:3,3]
  clouds.append(corrected);colors.append(rgb)
 clouds.append(np.concatenate(clouds));colors.append(np.concatenate(colors));views=[]
 for name,x,c in zip(['zed','realsense','fused'],clouds,colors):
  save_ply(a.output/f'{name}.ply',x,c);views.append(PixelMappingView(c[None],x[None],np.ones((1,len(x)),bool),show_xy_grid=False,max_points=None))
 common=views[-1];canvas=Image.new('RGB',(1920,680),'#181e27');draw=ImageDraw.Draw(canvas)
 for i,(name,v) in enumerate(zip(['zed','realsense','fused'],views)):
  v.center=common.center;v.span=common.span;v._xy_image();canvas.paste(v.measured_image,(i*640,40));draw.text((i*640+10,10),name.upper()+' | fixed depth scale | no hole filling',fill='white');v.measured_image.save(a.output/f'{name}_topdown.png');np.save(a.output/f'{name}_topdown_xyz_mm.npy',v.projected_xyz)
 canvas.save(a.output/'comparison.png')
 masks=[np.isfinite(v.projected_xyz).all(2) for v in views];both=masks[0]&masks[1];delta=(views[1].projected_xyz[:,:,2]-views[0].projected_xyz[:,:,2])[both]
 report={'capture':str(a.capture),'parameters':str(a.parameters),'refitted':False,'robot_connections':0,'artifact_role':'EXPERIMENTAL_STATIC_OVERLAY','occupied_cells':{n:int(m.sum()) for n,m in zip(['zed','realsense','fused'],masks)},'cells_added_to_zed':int((masks[1]&~masks[0]).sum()),'overlap_median_absolute_height_difference_mm':float(np.median(np.abs(delta))) if len(delta) else None,'limitation':'Whole ROI metrics are not garment-only coverage. Highest-Z projection can hide disagreement; scale validated on board, not clothing.'}
 (a.output/'parameters_used.json').write_text(json.dumps(params,indent=2),encoding='utf-8');(a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
