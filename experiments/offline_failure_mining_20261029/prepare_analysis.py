from index import *
import cv2,math
from PIL import ImageDraw
from cloth_agent.cloth_harness.cv_harness.segmentation import segment

def normalized(s):
 rgb=np.asarray(Image.open(ROOT/s['before']).convert('RGB'));mask,_=segment(rgb);y,x=np.where(mask)
 if len(x)==0:return None
 # Uniform scaling and centering: shape retrieval only, not material correspondence.
 crop=mask[y.min():y.max()+1,x.min():x.max()+1].astype('uint8');h,w=crop.shape;scale=120/max(h,w)
 small=cv2.resize(crop,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST);canvas=np.zeros((128,128),np.uint8);h,w=small.shape;canvas[(128-h)//2:(128-h)//2+h,(128-w)//2:(128-w)//2+w]=small
 return canvas.astype(bool)
def panel(seq,path):
 sheet=Image.new('RGB',(1280,len(seq)*410),'white');d=ImageDraw.Draw(sheet)
 for i,s in enumerate(seq):
  for k,side in enumerate(['before','after']):
   if not s[side]:continue
   im=Image.open(ROOT/s[side]).convert('RGB');im.thumbnail((630,370));sheet.paste(im,(k*640,i*410+35))
  ev=s['evaluation'];delta=ev['coverage']['delta_ratio'] if ev else None
  d.text((6,i*410+6),f"{s['step']} BEFORE (left) / AFTER (right) | Pa {s['action']['grasp_px']} -> Pb {s['action']['target_px']} | delta={None if delta is None else round(delta,4)}",fill='black')
 sheet.save(path)
def clean(s):
 e=s['evaluation']
 return {'id':s['id'],'step':s['step'],'action':s['action'],'execution':{k:s['execution'].get(k) for k in ['status','physical_execution','completed_commands','acquisition']},'fixed_evaluation':e,'image_size':list(Image.open(ROOT/s['before']).size),'real_before_after':bool(s['valid_pair'])}
def main():
 data=read(OUT/'index.json');assets=OUT/'analysis_images';assets.mkdir(exist_ok=True);packet={'constraints':'Single-view RGB; grasp -> lift -> pull -> release. Existing host checks workspace, collision, depth and allowed motion length; all proposed actions need existing approval. No robot execution or policy modification in this task. Completed commands do not prove grasp.','selection':'All six runs with first 3-action no-positive-area-progress window at <=2% per action; includes two absolute-area plateaus. Not all missing DONE runs are failures.','groups':[],'controls':[],'caveats':['Area is an uncalibrated segmentation proxy, not flatness. UNKNOWN registration is not no motion.','Adjacent states and separate runs are not controlled resets. Image resemblance cannot prove identical hidden layers.','No old skill, planner rationale, previous RSI hypotheses or code supplied. Choose the research problem yourself.']};mapping={};images=[]
 candidates=[r for r in data['runs'] if r['screen']['0.02']['windows']]
 for j,r in enumerate(candidates):
  w=r['screen']['0.02']['first_nonprogress'];n=next(i for i,s in enumerate(r['steps']) if s['id']==w['ids'][0]);seq=r['steps'][max(0,n-1):n+4];name=f'R{j+1}';image=assets/(name+'.jpg');panel(seq,image);images.append(str(image))
  packet['groups'].append({'anonymous_run':name,'image_index':len(images),'row_order':[s['step'] for s in seq],'termination':r['termination'],'first_nonprogress':w,'first_plateau':r['screen']['0.02']['first_plateau'],'steps':[clean(s) for s in seq]});mapping[name]={'run':r['path'],'ids':[s['id'] for s in seq]}
 progress=[(r,s) for r in data['runs'] for s in r['steps'] if s['evaluation'] and s['evaluation']['coverage']['delta_ratio'] is not None and s['evaluation']['coverage']['delta_ratio']>.05]
 masks={s['id']:normalized(s) for r,s in progress};matches=[]
 for j,r in enumerate(candidates):
  w=r['screen']['0.02']['first_plateau']
  if not w:continue
  q=next(s for s in r['steps'] if s['id']==w['ids'][0]);m=normalized(q);rank=[]
  qm=read(ROOT/q['before'].replace('rgb.png','observation.json'))
  for rr,s in progress:
   if rr['path']==r['path']:continue
   sm=read(ROOT/s['before'].replace('rgb.png','observation.json'))
   if (qm.get('camera_serial'),qm.get('calibration_id'))!=(sm.get('camera_serial'),sm.get('calibration_id')):continue
   n=masks[s['id']];score=float((m&n).sum()/max(1,(m|n).sum()));rank.append((score,rr,s))
  rank.sort(key=lambda t:t[0],reverse=True)
  for score,rr,s in rank[:5]:matches.append({'query_run':f'R{j+1}','query_id':q['id'],'candidate_run':rr['path'],'candidate_id':s['id'],'normalized_silhouette_iou':score,'delta':s['evaluation']['coverage']['delta_ratio']})
  if rank:
   score,rr,s=rank[0];name=f'C{len(packet["controls"])+1}';image=assets/(name+'.jpg');panel([s],image);images.append(str(image));packet['controls'].append({'id':name,'image_index':len(images),'retrieved_for':f'R{j+1}','normalized_silhouette_iou':score,'matching_limit':'uniform scale/translation-normalized outline only; internal folds, depth and material identities unmatched; not causal control','steps':[clean(s)]});mapping[name]={'run':rr['path'],'ids':[s['id']]}
 save(OUT/'analysis_packet.json',packet);save(OUT/'analysis_images.json',images);save(OUT/'case_mapping.json',mapping);save(OUT/'control_search.json',{'method':'all valid >5% area-increase actions; same camera/calibration; top 5 normalized silhouette IoU per plateau query, different run','searched_progress_actions':len(progress),'matches':matches})
 print('images',len(images),'progress actions searched',len(progress))
if __name__=='__main__':main()
