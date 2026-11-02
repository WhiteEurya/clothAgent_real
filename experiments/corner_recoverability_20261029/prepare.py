from pathlib import Path
import sys,json,hashlib
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from cloth_agent.cloth_harness.runtime.dataset import load_observation
from cloth_agent.cloth_harness.cv_harness.outcome_observer import same_view
from cloth_agent.cloth_harness.cv_harness.segmentation import segment
OUT=ROOT/'reports/corner_recoverability_20261029'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 d=read(ROOT/'reports/offline_failure_mining_20261029/index.json');runs=d['runs'];choices=[]
 done=sorted([r for r in runs if r['physical_command_steps'] and r['done_recorded']],key=lambda r:r['steps'][0]['timestamp'])
 choices += [('D1',done[0],'last'),('D2',done[-1],'last')]
 plateau=sorted([r for r in runs if r['screen']['0.02']['first_plateau']],key=lambda r:r['steps'][0]['timestamp'])
 choices += [(f'P{i+1}',r,'plateau') for i,r in enumerate(plateau)]
 interrupted=sorted([r for r in runs if r['human_interrupted'] and r['valid_pairs']>=3],key=lambda r:r['steps'][0]['timestamp'])
 choices += [('I1',interrupted[0],'first')]
 manifest=[];audit=[]
 for name,r,mode in choices:
  steps=r['steps'];valid=[s for s in steps if s['valid_pair']]
  if mode=='last':seq=valid[-3:]
  elif mode=='first':seq=valid[:3]
  else:
   ids=r['screen']['0.02']['first_plateau']['ids'];seq=[s for s in steps if s['id'] in ids]
  nums=[int(s['step'].split('_')[-1]) for s in seq];assert nums==list(range(nums[0],nums[0]+len(nums)))
  assert all(s['evaluation']['quality']['same_view'] for s in seq)
  bridges=[]
  for left,right in zip(seq,seq[1:]):
   a=load_observation((ROOT/left['after']).parent);b=load_observation((ROOT/right['before']).parent);same,reason=same_view(a,b)
   ma,_=segment(a.rgb);mb,_=segment(b.rgb);iou=float((ma&mb).sum()/max(1,(ma|mb).sum()));inter=ma&mb
   mae=float(np.abs(a.rgb.astype(float)-b.rgb)[inter].mean());gap=b.timestamp-a.timestamp
   ok=same and 0<gap<600 and iou>.97 and mae<12
   bridge={'after':left['after'],'next_before':right['before'],'same_view':same,'mask_iou':iou,'rgb_mae_on_overlap':mae,'gap_s':gap,'passes_screen':ok,'limitation':'near-identical RGB is not proof of identical hidden state; both bridge images retained for review'};bridges.append(bridge)
  if not all(b['passes_screen'] for b in bridges):audit.append({'id':name,'run':r['path'],'rejected_bridges':bridges});continue
  states=[]
  for i,s in enumerate(seq):states.append({'id':f'{name}_{i}','image':s['before'],'sha256':s['before_sha256'],'round':s['step'],'role':'before'})
  states.append({'id':f'{name}_{len(seq)}','image':seq[-1]['after'],'sha256':seq[-1]['after_sha256'],'round':seq[-1]['step'],'role':'after'})
  manifest.append({'id':name,'source_run':r['path'],'selection':mode,'termination':r['termination'],'states':states,'actions':[{k:s[k] for k in ['id','source','step','before','after','action','execution']} for s in seq],'bridges':bridges,'labels_status':'MODEL_PROVISIONAL_PENDING_HUMAN_REVIEW'})
  sheet=Image.new('RGB',(1280,380*len(states)),'white');draw=ImageDraw.Draw(sheet)
  for i,s in enumerate(states):
   im=Image.open(ROOT/s['image']).convert('RGB');im.thumbnail((640,360));sheet.paste(im,(0,i*380+20));draw.text((650,i*380+45),s['id']+' '+s['round']+' '+s['role'],fill='black')
  sheet.save(OUT/(name+'_sheet.jpg'))
 save(OUT/'manifest.json',manifest);save(OUT/'selection_audit.json',{'rule':'earliest and latest DONE runs: last up to3 physical actions; both area plateaus first3 actions; earliest interrupted with >=3 valid pairs first3. Selected before corner labeling, no search for positive swaps.','rejected':audit,'source_index_sha256':hashlib.sha256((ROOT/'reports/offline_failure_mining_20261029/index.json').read_bytes()).hexdigest()})
 print([(r['id'],len(r['states']),r['source_run']) for r in manifest]);print('rejected',audit)
if __name__=='__main__':main()
