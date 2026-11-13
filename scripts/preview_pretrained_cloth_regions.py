"""Offline transfer diagnostic for author's IROS2020 depth region weights; no ROS."""
import sys,json,time,argparse
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
parser=argparse.ArgumentParser()
parser.add_argument('--source-dir',default='reports/segmentation_model_comparison_20261108/cloth_regions/source')
parser.add_argument('--weights',default='/tmp/cloth-region-network/weights.pt')
args=parser.parse_args()
sys.path.insert(0,args.source_dir)
from unet import unet
root=Path('reports/segmentation_model_comparison_20261108/cloth_regions');root.mkdir(exist_ok=True)
model=unet(n_classes=3,in_channels=1)
model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
model=model.cuda().eval();torch.set_num_threads(4)
reports=[]
for name,path in [('white_zed','reports/cloth_background_trial_20261108/zed_depth_m.npy'),('empty_zed','reports/empty_table_background_20261108/zed_depth_m.npy'),('white_rs','reports/cloth_background_trial_20261108/realsense_native_depth_m.npy')]:
 depth=np.load(path).astype('float32');depth[(depth<=0)|~np.isfinite(depth)]=np.nan
 # Fixed table-view crop, no mask/point chosen from outcome. Both input grids 1280x720.
 crop=[0,230,1190,720] if name.endswith('zed') else [0,80,1280,720]
 x0,y0,x1,y1=crop;d=depth[y0:y1:2,x0:x1:2].copy()
 d[~np.isfinite(d)]=np.nanmax(d);lo=float(d.min());hi=float(d.max());d=(d-lo)/(hi-lo)
 inp=torch.from_numpy(d)[None,None].cuda();t=time.perf_counter()
 with torch.inference_mode():pred=model(inp).sigmoid()[0].permute(1,2,0).cpu().numpy()
 torch.cuda.synchronize();elapsed=time.perf_counter()-t
 np.save(root/(name+'_probabilities.npy'),pred)
 canvas=Image.new('RGB',(1024,280),'white');draw=ImageDraw.Draw(canvas)
 for i,(label,arr) in enumerate([('depth input',d),('corners',pred[:,:,0]),('outer edges',pred[:,:,1]),('inner edges',pred[:,:,2])]):
  heat=cv2.applyColorMap(np.clip(arr*255,0,255).astype('uint8'),cv2.COLORMAP_VIRIDIS)[:,:,::-1]
  canvas.paste(Image.fromarray(heat).resize((256,240)),(256*i,30));draw.text((256*i+5,8),label,fill='black')
 canvas.save(root/(name+'.png'))
 reports.append(dict(name=name,input=path,crop_xyxy=crop,stride=2,normalization='author minmax; missing=max depth',seconds=elapsed,probability_p99=np.percentile(pred,99,axis=(0,1)).tolist(),pixels_over_half=(pred>.5).sum(axis=(0,1)).tolist(),note='region probabilities, NOT whole-garment mask; unadapted cross-camera transfer'))
 print(reports[-1],flush=True)
(root/'results.json').write_text(json.dumps(reports,indent=2))
