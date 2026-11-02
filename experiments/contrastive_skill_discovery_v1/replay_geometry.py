from run import *
import cv2,numpy as np
# Reused exactly from dataset.py mask(), applied at the original 640x360 analysis resolution.
def mask(im):
 rgb=im[:,:,::-1].astype(np.int16);m=((rgb[:,:,0]>rgb[:,:,1]+10)&(rgb[:,:,2]>rgb[:,:,1]+15)&(rgb[:,:,0]>70)).astype(np.uint8)*255
 m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8));n,labels,stats,_=cv2.connectedComponentsWithStats(m)
 if n<2:return m*0
 m=(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.uint8)*255
 contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(m,contours,-1,255,cv2.FILLED)
 return m

def main():
 for state in ['E29','E30']:
  src=DATA/f'cases/{state}/before.png';im=cv2.imread(str(src));h,w=im.shape[:2];m=cv2.resize(mask(cv2.resize(im,(640,360))),(w,h),interpolation=cv2.INTER_NEAREST)
  folder=OUT/'planner_replay/static_masks';folder.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(folder/f'{state}_mask.png'),m)
  overlay=im.copy();overlay[m>0]=(.7*overlay[m>0]+.3*np.array([40,220,40])).astype(np.uint8);cv2.imwrite(str(folder/f'{state}_mask_check.jpg'),overlay)
  din=cv2.distanceTransform(m,cv2.DIST_L2,5);dout=cv2.distanceTransform(255-m,cv2.DIST_L2,5)
  for f in (OUT/'planner_replay').glob(f'{state}_*/parsed.json'):
   action=read(f);p=f.parent/'geometry.json';d=read(p)
   if not d['finite_2d_in_bounds']:continue
   x,y=[int(round(z)) for z in action['Pa']];x=min(w-1,x);y=min(h-1,y);bx,by=[int(round(z)) for z in action['Pb']];bx=min(w-1,bx);by=min(h-1,by)
   d.update({'on_cloth':'HEURISTIC_COLOR_MASK_ONLY','Pa_in_heuristic_mask':bool(m[y,x]),'Pa_signed_distance_to_mask_boundary_px':float(din[y,x] if m[y,x] else -dout[y,x]),'Pb_in_heuristic_mask':bool(m[by,bx]),'mask_area_fraction':float(np.mean(m>0)),'mask_area_sanity':bool(.02<np.mean(m>0)<.7),'mask_source':'Reused prior dataset.py color segmentation; before-only, 640x360 then nearest-neighbor upsample','graspability':'NOT_CERTIFIED: 2D color segmentation cannot confirm layer count, depth, fabric acquisition or contact safety'})
   save(p,d)
  tiles=[]
  for rep in [1,2]:
   for g in ['Base','A','B','C']:
    f=OUT/f'planner_replay/{state}_{g}_r{rep}/action_overlay.jpg'
    if not f.exists():continue
    tile=Image.new('RGB',(640,390),'white');tile.paste(Image.open(f).resize((640,360)),(0,30));ImageDraw.Draw(tile).text((10,8),f'{state} | {g} | repeat {rep}',fill='black');tiles.append(tile)
  if len(tiles)==8:
   canvas=Image.new('RGB',(2560,780),'white')
   for i,tile in enumerate(tiles):canvas.paste(tile,((i%4)*640,(i//4)*390))
   canvas.save(OUT/f'planner_replay/{state}_comparison.jpg',quality=92)
 save(OUT/'planner_replay/static_masks/provenance.json',{'reused_source':'experiments/contrastive_skill_dataset_v1/dataset.py:27-33','method':'color threshold, closing, largest component, external contour fill','not_model_input':True,'physical_validation':False})
if __name__=='__main__':main()
