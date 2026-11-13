"""Rebuild saved dual-wrist capture through the RGB-first planning view; no hardware."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.dual_cloud_render import texture_surface
from cloth_agent.dual_arm.observation import save_observation,sha
from cloth_agent.experiments.unfold_dual_topdown import reconstruct

def main():
 root=Path('reports/dual_wrist_tabletop_20261110');out=root/'rgb_first_color';out.mkdir(exist_ok=True);m=json.loads((root/'capture.json').read_text());intr=json.loads((root/'depth_intrinsics.json').read_text());corr=np.array(json.loads(Path('reports/dual_wrist_fixed_board_20261110/fixed_validation.json').read_text())['right_world_correction_mm']);observations=[]
 for key in ('left','right'):
  meta=m['cameras'][key];td=np.array(meta['world_from_depth_mm']);tc=np.array(meta['world_from_color_mm'])
  if key=='right':td=corr@td;tc=corr@tc
  di=intr[meta['serial']];k=meta['color_intrinsics'];K=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]]);KD=np.array([[di['fx'],0,di['cx']],[0,di['fy'],di['cy']],[0,0,1.]])
  d=np.load(root/f'{key}_depth.npy')*meta['depth_unit_m']*1000;y,x=np.indices(d.shape);cam=np.stack([(x-di['cx'])/di['fx']*d,(y-di['cy'])/di['fy']*d,d],-1);xyz=cam@td[:3,:3].T+td[:3,3];xyz[d<=0]=np.nan;rgb=np.asarray(Image.open(root/f'{key}_rgb.png').convert('RGB'));texture,known,_=texture_surface(xyz,rgb,tc,K);xyz[~known]=np.nan
  obs=save_observation(out/key,texture,xyz,SimpleNamespace(raw={'calibration_id':'dual_wrist_fixed_offline_candidate','camera':{}}),m['before'],extra={'camera_serial':meta['serial'],'intrinsics':KD.tolist(),'world_from_camera_mm':td.tolist(),'color_intrinsics':K.tolist(),'world_from_color_mm':tc.tolist(),'execution_allowed':False,'capture_source':str(root/'capture.json')});Image.fromarray(rgb).save(obs.directory/'color_rgb.png');observations.append(obs)
 before,secondary=observations;before.meta['dual_cloud']={'secondary_directory':'../right','secondary_observation_sha256':sha(secondary.directory/'observation.json'),'config':{}};(before.directory/'observation.json').write_text(json.dumps(before.meta,indent=2));roi=json.loads(Path('config/topdown.dual_wrist_rgb.json').read_text());view=reconstruct(before,out/'view',roi)
 report={'image':str(view.image),'appearance_pixels':int((~view.unknown_mask).sum()),'measured_pixels':int((~view.depth_unknown_mask).sum()),'visible_without_depth':int((~view.unknown_mask&view.depth_unknown_mask).sum()),'execution_allowed':False}
 (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
