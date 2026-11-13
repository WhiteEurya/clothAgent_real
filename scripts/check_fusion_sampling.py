from pathlib import Path
import argparse,json,numpy as np,cv2
from PIL import Image,ImageDraw
parser=argparse.ArgumentParser();parser.add_argument('capture',type=Path);args=parser.parse_args();p=args.capture;out=p/'sampling_check';out.mkdir(exist_ok=True)
meta=json.loads((p/'report.json').read_text());cal=json.loads(Path(meta['calibration']).read_text())
def ply(name):
 with (p/f'{name}.ply').open('rb') as f:
  while f.readline().strip()!=b'end_header':pass
  a=np.fromfile(f,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
 return np.stack([a[k] for k in 'xyz'],1),np.stack([a[k] for k in 'rgb'],1)
report={};clouds={n:ply(n) for n in ['zed','realsense','fused']}
for name,key in [('zed','world_from_zed_mm'),('realsense','world_from_realsense_color_mm')]:
 x,_=clouds[name];T=np.array(cal[key]);opt=(x-T[:3,3])@T[:3,:3];dist=np.linalg.norm(opt,axis=1)
 report[name]={'camera_world_position_mm':T[:3,3].tolist(),'range_distance_mm_p10_p50_p90':np.percentile(dist,[10,50,90]).tolist(),'optical_depth_mm_p10_p50_p90':np.percentile(opt[:,2],[10,50,90]).tolist(),'view_angle_from_world_vertical_deg':float(np.degrees(np.arccos(abs(T[2,2]))))}
rw=np.load(p/'realsense_world_xyz_mm.npy').reshape(*np.load(p/'realsense_native_depth_m.npy').shape,3)
for axis in [0,1]:
 dd=np.linalg.norm(np.diff(rw[:,:,:2],axis=axis),axis=2);good=np.isfinite(dd)&(dd>0)&(dd<20)
 report['realsense'][f'neighbor_xy_spacing_axis{axis}_mm_p10_p50_p90']=np.percentile(dd[good],[10,50,90]).tolist()
center=np.array(meta['topdown_center_mm']);span=meta['topdown_span_mm'];low=center-span/2
steps=[span/520,4.,6.];canvas=Image.new('RGB',(1560,1620),'#181e27');draw=ImageDraw.Draw(canvas)
for row,(name,(x,c)) in enumerate(clouds.items()):
 report.setdefault(name,{})['raster_comparison']=[]
 for col,step in enumerate(steps):
  n=int(np.ceil(span/step))+1;ij=np.floor((x[:,:2]-low)/step).astype(int);ij[:,1]=n-1-ij[:,1];order=np.argsort(x[:,2]);flat=ij[order,1]*n+ij[order,0];_,rev=np.unique(flat[::-1],return_index=True);ids=order[len(flat)-1-rev]
  im=np.full((n,n,3),[24,30,39],np.uint8);im[ij[ids,1],ij[ids,0]]=c[ids]
  occ=np.zeros((n,n),np.uint8);occ[ij[:,1],ij[:,0]]=1
  hull=np.zeros_like(occ);cv2.fillConvexPoly(hull,cv2.convexHull(ij.astype(np.int32)),1)
  # Central hull excludes edge geometry, but can include true occlusions.
  hull=cv2.erode(hull,np.ones((5,5),np.uint8))>0
  missing=float((occ[hull]==0).mean());report[name]['raster_comparison'].append({'mm_per_cell':step,'unoccupied_fraction_inside_eroded_convex_hull':missing})
  pic=Image.fromarray(im).resize((520,520),Image.Resampling.NEAREST);canvas.paste(pic,(col*520,row*540+20));draw.text((col*520+10,row*540+4),f'{name} | {step:.2f} mm/cell | hull gaps {missing:.1%}',fill='white')
canvas.save(out/'grid_comparison.png');(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
