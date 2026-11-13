"""Fit depth scale on one board placement; evaluate on another without refitting."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.pixel_mapping_visualization import PixelMappingView
from PIL import Image,ImageDraw

def load(folder,name):
 meta=json.loads((folder/'report.json').read_text());cal=json.loads(Path(meta['calibration']).read_text())
 k=np.array(meta['zed_K'] if name=='zed' else cal['realsense_K']);wcolor=np.array(cal['world_from_zed_mm' if name=='zed' else 'world_from_realsense_color_mm']);dc=np.eye(4) if name=='zed' else np.array(meta['realsense_depth_to_color_mm']);wd=wcolor@dc
 image=cv2.imread(str(folder/f'{name}_rgb.png'));board=cv2.aruco.CharucoBoard((7,5),40.,30.,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000));corners,ids,_,_=cv2.aruco.CharucoDetector(board).detectBoard(image)
 if ids is None or len(ids)<8:raise ValueError('Too few corners')
 obj=board.getChessboardCorners()[ids.ravel()].astype(float);ok,rv,tv=cv2.solvePnP(obj,corners,k,None)
 if not ok:raise ValueError('PnP failed')
 predicted=cv2.projectPoints(obj,rv,tv,k,None)[0].reshape(-1,2)
 if np.sqrt(np.mean(np.sum((predicted-corners.reshape(-1,2))**2,axis=1)))>1.5:raise ValueError('Poor board pose')
 cb=np.eye(4);cb[:3,:3]=cv2.Rodrigues(rv)[0];cb[:3,3]=tv.ravel();db=np.linalg.inv(dc)@cb
 world=np.load(folder/f'{name}_world_xyz_mm.npy').reshape(-1,3);native=(world-wd[:3,3])@wd[:3,:3];local=(native-db[:3,3])@db[:3,:3]
 valid=np.isfinite(local).all(1)&(local[:,0]>20)&(local[:,0]<260)&(local[:,1]>20)&(local[:,1]<180)&(np.abs(local[:,2])<100)
 return native[valid],db,wd,meta

def fit_depth_scale(points, board_pose):
 normal=board_pose[:3,2];distance=normal@board_pose[:3,3];denominator=points@normal
 if not np.isfinite(points).all() or np.any(np.abs(denominator)<1e-9):raise ValueError('Invalid plane/depth sample')
 scale=float(np.median(distance/denominator))
 if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid scale')
 return scale

def stats(values):return dict(median=float(np.median(values)),median_absolute=float(np.median(np.abs(values))),p95_absolute=float(np.percentile(np.abs(values),95)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--parameters',type=Path);p.add_argument('--fit',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 if a.fit.resolve()==a.validation.resolve():raise ValueError('Fit and validation must be separate')
 report={'status':'OFFLINE_LOCAL_DEPTH_SCALE_EXPERIMENT','fit':str(a.fit),'validation':str(a.validation),'method':'Scale native camera XYZ about native optical origin; preserve extrinsics. Fit median ray/board-plane depth ratio at placement A only.','limitations':['Board pose from RGB and user-provided dimensions, not independently surveyed truth.','Only two local board placements; not valid for all ranges or materials.','Not enabled in execution or camera capture.'],'cameras':{}};scaled=[];colors=[];original=[]
 frozen=None if a.parameters is None else json.loads(a.parameters.read_text())
 if frozen is not None:
  report['parameters_source']=str(a.parameters);report['refitted']=False
  report['method']='Apply previously saved native-depth scales unchanged; no parameter fitting.'
  report['limitations'][1]='Local board checks only; not validated for all ranges or materials.'
 for name in ['zed','realsense']:
  if frozen is None:
   fitted,db,_,_=load(a.fit,name);s=fit_depth_scale(fitted,db);fit_range=np.percentile(fitted[:,2],[5,50,95]).tolist()
  else:
   s=float(frozen['cameras'][name]['scale']);fit_range=frozen['cameras'][name]['fit_optical_z_mm_p05_p50_p95']
   if not np.isfinite(s) or s<=0:raise ValueError('Invalid saved scale')
  test,tb,wd,meta=load(a.validation,name)
  before=(test-tb[:3,3])@tb[:3,2];after=(test*s-tb[:3,3])@tb[:3,2]
  report['cameras'][name]={'scale':s,'fit_optical_z_mm_p05_p50_p95':fit_range,'held_out_before_mm':stats(before),'held_out_after_mm':stats(after)}
  with (a.validation/f'{name}.ply').open('rb') as f:
   while f.readline().strip()!=b'end_header':pass
   raw=np.fromfile(f,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
  world=np.stack([raw[k] for k in 'xyz'],1);rgb=np.stack([raw[k] for k in 'rgb'],1);native=(world-wd[:3,3])@wd[:3,:3];world_new=(native*s)@wd[:3,:3].T+wd[:3,3];scaled.append(world_new);colors.append(rgb);original.append(world)
 views=[]
 for x,c in zip(scaled+[np.concatenate(scaled)],colors+[np.concatenate(colors)]):views.append(PixelMappingView(c[None],x[None],np.ones((1,len(x)),bool),show_xy_grid=False,max_points=None))
 common=views[-1];canvas=Image.new('RGB',(1920,680),'#181e27');draw=ImageDraw.Draw(canvas)
 for i,(name,v) in enumerate(zip(['zed','realsense','fused'],views)):
  v.center=common.center;v.span=common.span;v._xy_image();canvas.paste(v.measured_image,(i*640,40));draw.text((i*640+10,10),name.upper()+' | experimental depth scale, fitted elsewhere',fill='white');np.save(a.output/f'{name}_topdown_xyz_mm.npy',v.projected_xyz)
 overlap=np.isfinite(views[0].projected_xyz).all(2)&np.isfinite(views[1].projected_xyz).all(2);delta=(views[1].projected_xyz[:,:,2]-views[0].projected_xyz[:,:,2])[overlap];report['held_out_overlap_height_difference_mm']=stats(delta)
 canvas.save(a.output/'comparison.png')
 base=np.concatenate(original);col=np.concatenate(colors);before=PixelMappingView(col[None],base[None],np.ones((1,len(base)),bool),show_xy_grid=False,max_points=None)
 before.center=common.center;before.span=common.span;before._xy_image()
 pair=Image.new('RGB',(1280,680),'#181e27');pair.paste(before.measured_image,(0,40));pair.paste(common.measured_image,(640,40));label=ImageDraw.Draw(pair);label.text((10,10),'ORIGINAL measured overlay',fill='white');label.text((650,10),'FIXED depth scale | held-out placement | no hole filling',fill='white');pair.save(a.output/'before_after.png')
 (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
