"""Audit, source-aware splits, weak CV diagnostics and fixed sample selection."""
from common import *
import cv2,collections,itertools

FLAG_KEYS=['hem_sleeve_pull_rule','visible_sleeve_hem_enabled','ggo_manual_geometry','ggo_auto_anchors']
def inventory():
 old=read(ROOT/'experiments/failure_driven_rsi_pilot/inventory.json')
 protocol={'at':utc(),'scope':'Only observed-effect extraction and comparability dataset. No Skill Discovery, Prompt Evolution, Planner calls or online actions.','baseline_inventory_sha256':sha(ROOT/'experiments/failure_driven_rsi_pilot/inventory.json'),'target_count':40,'core_split_targets':{'train':20,'development':8,'test':8},'reference_target':4,'source_priority':'Core: no explicit target-rule flag enabled AND before earliest archived true flag; absence is UNKNOWN, not expert-free proof. Reference: explicit flag true. Never read prompt semantic content.','split':'Run-group-disjoint, source-family-stratified deterministic allocation before model labels; retain full episode paths and chronology.','selection':'Within split, distribute across groups; farthest-point diversity on pre-mask/shape plus weak post rigid-residual diagnostics, not named skills or human action labels. No outcome quota forcing.','effects':['unfolding_gain','whole_garment_transport','new_folding','exposure_gain'],'effect_values':'YES / NO_CLEAR_EVIDENCE / UNKNOWN with image-based evidence. These are categorical observation indicators, not calibrated scalar rewards.','review':'Every selected case: 2 fresh independent GPT sessions, different batch neighbors/order; reviewer R1 raw pre/post only, R2 adds factual action and chronological process montage. No reviewer sees other output or CV numeric diagnostics. Same model, so agreement is not independent ground truth.','conservative_labels':'Accept a dimension only when both reviewers agree YES or NO_CLEAR_EVIDENCE, confidence>=0.65 and both endpoint comparability=GOOD. Disagreement -> UNKNOWN. No-clear-evidence is not proof of absence.','groups':'A: clear unfolding and no clear new folding. B: transport YES and no clear unfolding. C: unfolding YES and new folding YES. Other/no-clear/unknown retained.','pairing':'Before-only geometric ranking, same split/camera/calibration/source family/lift and other numeric motion parameters. Exact prompt-hash matches prioritized. Contrastive pairs selected after labels, retaining all candidates; pre-only visual audit. No cross-split pair is training material.','budget':{'planned_effect_requests':20,'pair_audit_max':2,'max_total_attempts':24,'max_concurrent_requests':2},'future_affordance_gain':'NOT_ASSESSED_THIS_STAGE. Preserve adjacent rounds and full episode links for separate sequence analysis; no immediate-effect negative label implies no future value.','limitations':'Historical robots may be guided by human prompts. Mask area/rigid residual are sampling/diagnostic signals, never a success label. No causal counterfactual claims.'}
 save(OUT/'protocol.json',protocol)
 protected={str(p.relative_to(ROOT)):sha(p) for top in ['cloth_agent','config'] for p in (ROOT/top).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.toml','.txt','.md']};save(OUT/'protected_before.json',protected)
 rows=[];sources={}
 for index,x in enumerate(old):
  if not x['eligible']:continue
  r=ROOT/x['round'];ep=r.parent;group=ep.parent.name;pr=read(ep/'protocol.json') if (ep/'protocol.json').exists() else {};before=read(r/'before/observation.json');after=read(r/'after/observation.json');ex=read(r/'execution.json');g=read(r/'geometry.json');pts=read(r/'points.json');eprel=str(ep.relative_to(ROOT))
  if eprel not in sources:
   sources[eprel]={'episode':eprel,'group':group,'family':group.split('_2026')[0].replace('_ready',''),'prompt_sha256':sha(ep/'prompt.txt') if (ep/'prompt.txt').exists() else None,'prompt_semantics_read':False,'protocol_sha256':sha(ep/'protocol.json') if (ep/'protocol.json').exists() else None,'flags':{k:pr.get(k,'NOT_RECORDED') for k in FLAG_KEYS},'explicit_target_rule_enabled':any(pr.get(k) is True for k in FLAG_KEYS),'expert_free':False,'expert_free_note':'Not established; flag absence does not remove possible expert base-prompt influence.','model':pr.get('model'),'human_review_required':pr.get('human_review_required'),'camera_serial':before.get('camera_serial'),'calibration_id':before.get('calibration_id')}
  src=sources[eprel];native=read(r/'native_plan.json') if (r/'native_plan.json').exists() else [];speeds=sorted(set(n['data']['cartesian_speed_mm_s'] for n in native if 'cartesian_speed_mm_s' in n.get('data',{})));release=[{k:v for k,v in n['data'].items() if k in ['opening_delta_pulse','settle_s']} for n in native if any(k in n.get('data',{}) for k in ['opening_delta_pulse','settle_s'])]
  rows.append({'raw_id':f'R{index:04d}','round':x['round'],'episode':eprel,'group':group,'family':src['family'],'before_at':before['captured_at'],'after_at':after['captured_at'],'status':ex['status'],'physical_execution':ex.get('physical_execution'),'real':before.get('synthetic') is False and after.get('synthetic') is False,'Pa':pts[0]['pixel_xy'],'Pb':pts[1]['pixel_xy'],'geometry':g,'speeds_mm_s':speeds,'release':release,'camera_serial':before.get('camera_serial'),'calibration_id':before.get('calibration_id'),'recorded_camera_transform_equal':before.get('world_from_camera_mm')==after.get('world_from_camera_mm'),'prompt_sha256':src['prompt_sha256'],'explicit_target_rule_enabled':src['explicit_target_rule_enabled'],'process_frames':x['process_frames'],'basic_integrity':all((r/n).exists() for n in ['before/rgb.png','after/rgb.png','points.json','geometry.json','states.jsonl','motion.py'])})
 earliest=min(r['before_at'] for r in rows if r['explicit_target_rule_enabled'])
 for r in rows:r['core_source_eligible']=not r['explicit_target_rule_enabled'] and r['before_at']<earliest
 save(OUT/'source_audit.json',{'earliest_archived_explicit_target_rule':earliest,'not_claimed_actual_introduction_time':True,'sources':list(sources.values()),'prior_inventory_counts':{'rounds':len(old),'complete':len(rows)},'newer_records_not_silently_added':True})
 save(OUT/'raw_inventory.json',rows)
 cache=read(Path.home()/'.codex/models_cache.json');m=next(m for m in cache['models'] if m['slug']=='gpt-6-astra');save(OUT/'local_model_catalog.json',{'models':[m]})
 freeze('audit',[OUT/'protocol.json',OUT/'source_audit.json',OUT/'raw_inventory.json'])
 print('AUDIT',len(rows),'CORE',sum(r['core_source_eligible'] for r in rows),flush=True)

def mask(im):
 rgb=im[:,:,::-1].astype(np.int16);m=((rgb[:,:,0]>rgb[:,:,1]+10)&(rgb[:,:,2]>rgb[:,:,1]+15)&(rgb[:,:,0]>70)).astype(np.uint8)*255
 m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8));n,labels,stats,_=cv2.connectedComponentsWithStats(m)
 if n<2:return m*0
 m=(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.uint8)*255
 contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(m,contours,-1,255,cv2.FILLED)
 return m
def centroid(m):
 y,x=np.where(m>0);return np.array([x.mean(),y.mean()]) if len(x) else np.array([0.,0.])
def iou(a,b):return float(np.logical_and(a>0,b>0).sum()/max(1,np.logical_or(a>0,b>0).sum()))
def normalized(m,size=48):
 x,y,w,h=cv2.boundingRect(m);side=max(w,h)+12;out=np.zeros((side,side),np.uint8);crop=m[y:y+h,x:x+w];ox=(side-w)//2;oy=(side-h)//2;out[oy:oy+h,ox:ox+w]=crop;return cv2.resize(out,(size,size),interpolation=cv2.INTER_AREA)/255.
def rigid_align(a,b):
 ca,cb=centroid(a),centroid(b);best=(-1,None,None)
 for angle in range(-30,31,3):
  M=cv2.getRotationMatrix2D(tuple(cb),angle,1.);M[:,2]+=ca-cb;v=cv2.warpAffine(b,M,(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST);score=iou(a,v)
  if score>best[0]:best=(score,angle,v)
 return best
def diagnostics(r):
 folder=OUT/'cv'/r['raw_id'];folder.mkdir(parents=True,exist_ok=True)
 p=ROOT/r['round'];a=cv2.imread(str(p/'before/rgb.png'));b=cv2.imread(str(p/'after/rgb.png'));a=cv2.resize(a,(640,360));b=cv2.resize(b,(640,360));ma,mb=mask(a),mask(b)
 valid=ma.sum()>10000 and mb.sum()>10000
 # Camera check uses background feature correspondences, never garment optical flow.
 bg=255-cv2.dilate(cv2.bitwise_or(ma,mb),np.ones((25,25),np.uint8));orb=cv2.ORB_create(nfeatures=1500);ka,da=orb.detectAndCompute(a,bg);kb,db=orb.detectAndCompute(b,bg);reg={'status':'UNKNOWN','reason':'insufficient stable background features'};aligned_b=b;aligned_m=mb
 if da is not None and db is not None and len(da)>10 and len(db)>10:
  matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(db,da,k=2);good=[x[0] for x in matches if len(x)==2 and x[0].distance<.75*x[1].distance]
  if len(good)>=12:
   src=np.float32([kb[m.queryIdx].pt for m in good]);dst=np.float32([ka[m.trainIdx].pt for m in good]);M,inl=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.5)
   if M is not None:
    n=int(inl.sum());scale=float(np.linalg.norm(M[0,:2]));rot=float(np.degrees(np.arctan2(M[1,0],M[0,0])));err=float(np.median(np.linalg.norm(cv2.transform(src[None],M)[0][inl.ravel()>0]-dst[inl.ravel()>0],axis=1)));ok=n>=8 and .95<scale<1.05 and abs(rot)<5 and err<2.5
    reg={'status':'PASS' if ok else 'UNKNOWN','inliers':n,'matches':len(good),'scale':scale,'rotation_deg':rot,'median_error_px_at_half_resolution':err,'after_to_before_affine':M.tolist()}
    if ok:aligned_b=cv2.warpAffine(b,M,(640,360));aligned_m=cv2.warpAffine(mb,M,(640,360),flags=cv2.INTER_NEAREST)
 score,angle,rigid=rigid_align(ma,aligned_m);ca,cb=centroid(ma),centroid(aligned_m);contours,_=cv2.findContours(ma,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);c=max(contours,key=cv2.contourArea);area=max(1,cv2.contourArea(c));hull=max(1,cv2.contourArea(cv2.convexHull(c)));x,y,w,h=cv2.boundingRect(c)
 d={'id':r['raw_id'],'mask_quality_heuristic':bool(valid and .02<np.mean(ma>0)<.7 and .02<np.mean(mb>0)<.7),'background_registration':reg,'raw_area_ratio':float((mb>0).sum()/max(1,(ma>0).sum())),'centroid_shift_px':float(np.linalg.norm(cb-ca)*2),'rigid_mask_iou':score,'rigid_fit_rotation_deg':angle,'residual_after_rigid_fit':1-score,'pre_area_fraction':float(np.mean(ma>0)),'pre_solidity':area/hull,'pre_aspect':w/max(1,h),'pre_bbox':[int(x*2),int(y*2),int(w*2),int(h*2)],'not_a_success_metric':True}
 overlay=a.copy();overlay[ma>0]=(.65*a[ma>0]+.35*np.array([40,210,40])).astype(np.uint8);cv2.drawContours(overlay,[c],-1,(0,255,255),2);cv2.imwrite(str(folder/'before_mask.png'),ma);cv2.imwrite(str(folder/'after_mask.png'),mb);cv2.imwrite(str(folder/'mask_check.jpg'),overlay);np.save(folder/'pre_normalized.npy',normalized(ma));save(folder/'metrics.json',d)
 return d

def compute_cv():
 rows=read(OUT/'raw_inventory.json');out=[]
 for j,r in enumerate(rows):
  f=OUT/'cv'/r['raw_id']/'metrics.json'
  try:d=read(f) if f.exists() else diagnostics(r)
  except Exception as e:d={'id':r['raw_id'],'error':str(e),'mask_quality_heuristic':False}
  out.append(d)
  if j%50==0:print('CV',j,'/',len(rows),flush=True)
 save(OUT/'cv_diagnostics.json',out);freeze('cv',[OUT/'cv_diagnostics.json']);print('CV_DONE',len(out),flush=True)

def allocate_groups(rows):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['group']].append(r)
 gs=sorted(groups,key=key);best=None
 # Reserve three groups for Development, three for Test; capacity and family overlap,
 # not outcome labels, determine feasible split. Capacity cap keeps sources diverse.
 for dev in itertools.combinations(gs,3):
  if sum(min(6,len(groups[g])) for g in dev)<8:continue
  remaining=[g for g in gs if g not in dev]
  for test in itertools.combinations(remaining,3):
   train=[g for g in remaining if g not in test]
   if sum(min(6,len(groups[g])) for g in test)<8 or sum(min(6,len(groups[g])) for g in train)<20:continue
   parts=[train,list(dev),list(test)];coverage=sum(len({groups[g][0]['family'] for g in part}) for part in parts)
   score=(-coverage,key('|'.join(dev)+'::'+'|'.join(test)))
   if best is None or score<best[0]:best=(score,parts)
 if best is None:raise ValueError('Insufficient group-disjoint capacities; do not reduce blindly')
 return {g:s for s,part in zip(['train','development','test'],best[1]) for g in part}

def select():
 verify_freezes();rows=read(OUT/'raw_inventory.json');cv={x['id']:x for x in read(OUT/'cv_diagnostics.json')};core=[r for r in rows if r['core_source_eligible'] and cv[r['raw_id']]['mask_quality_heuristic'] and r['basic_integrity'] and r['real']];mapping=allocate_groups(core)
 save(OUT/'split_groups.json',mapping);freeze('split',[OUT/'split_groups.json'])
 chosen=[]
 for split,n in [('train',20),('development',8),('test',8)]:
  pool=[r for r in core if mapping[r['group']]==split];selected=[];counts=collections.Counter()
  def features(r):
   d=cv[r['raw_id']];return np.r_[np.load(OUT/'cv'/r['raw_id']/'pre_normalized.npy').ravel()/15,np.array([d['pre_area_fraction']*4,d['pre_solidity'],d['pre_aspect']/2,d['residual_after_rigid_fit'],min(1,d['centroid_shift_px']/200),np.clip(math.log(d['raw_area_ratio']),-1,1)])]
  ff={r['raw_id']:features(r) for r in pool}
  while len(selected)<n:
   available=[r for r in pool if r not in selected and counts[r['group']]<6]
   minimum=min(counts[r['group']] for r in available);available=[r for r in available if counts[r['group']]==minimum]
   pick=max(available,key=lambda r:(min(np.linalg.norm(ff[r['raw_id']]-ff[z['raw_id']]) for z in selected) if selected else 0,key(r['round'])))
   selected.append(pick);counts[pick['group']]+=1
  chosen += [{**r,'split':split,'selection_basis':'Group-balanced diversity over raw pre-shape and weak rigid-change diagnostics; no visual-model labels available.'} for r in selected]
 refs=[r for r in rows if r['explicit_target_rule_enabled'] and cv[r['raw_id']]['mask_quality_heuristic'] and r['real']];rg=sorted({r['group'] for r in refs},key=key)[:4]
 for g in rg:
  pool=sorted([r for r in refs if r['group']==g],key=lambda r:key(r['round']));chosen.append({**pool[0],'split':'reference','selection_basis':'Hash-fixed explicit-rule reference, excluded from core learning/evaluation.'})
 for i,r in enumerate(chosen):r['id']=f'E{i+1:02d}'
 save(OUT/'selection_manifest.json',{'at':utc(),'cases':chosen,'outcome_labels_seen':False,'fixed_sample_size':len(chosen),'selection_no_replacement_after_labels':True});freeze('selection',[OUT/'selection_manifest.json'])
 for r in chosen:packet(r)
 print('SELECTED',collections.Counter(r['split'] for r in chosen),flush=True)

def packet(r):
 folder=OUT/'cases'/r['id'];folder.mkdir(parents=True,exist_ok=True);source=ROOT/r['round']
 for n in ['before','after']:shutil.copyfile(source/n/'rgb.png',folder/f'{n}.png')
 facts={k:r[k] for k in ['id','Pa','Pb','geometry','speeds_mm_s','release','before_at','after_at','status','physical_execution','recorded_camera_transform_equal']};facts['intention']='UNKNOWN; historical planner rationale withheld';frames=[];counts={}
 for phase,n in [('execution_images',8),('return_perception/execution_images',2)]:
  f=source/phase/'frames.jsonl';rr=[json.loads(l) for l in f.read_text().splitlines()] if f.exists() else [];counts[phase]=len(rr)
  if rr:
   for i in sorted({round(j*(len(rr)-1)/(n-1)) for j in range(n)}):frames.append({**rr[i],'source':str((source/phase/rr[i]['file']).relative_to(ROOT)),'phase':phase})
 frames.sort(key=lambda x:x['captured_unix_s']);facts['available_process_frames']=counts;facts['sampled_process_times']=[x['captured_unix_s'] for x in frames];facts['process_warning']='Moving wrist camera; image motion is not necessarily cloth motion. Sampling does not identify exact grasp/lift/release instants.'
 if frames:
  im=Image.new('RGB',(1280,math.ceil(len(frames)/4)*202),'#111');d=ImageDraw.Draw(im)
  for j,fr in enumerate(frames):
   x,y=(j%4)*320,(j//4)*202;im.paste(Image.open(ROOT/fr['source']).convert('RGB').resize((320,180)),(x,y+22));d.text((x+3,y+4),f'{j+1} '+datetime.fromtimestamp(fr['captured_unix_s'],timezone.utc).strftime('%H:%M:%S.%f')[:-3],fill='white')
  im.save(folder/'process.jpg',quality=94)
 im=Image.open(folder/'before.png').convert('RGB');d=ImageDraw.Draw(im);d.line([tuple(r['Pa']),tuple(r['Pb'])],fill='yellow',width=4)
 for p,t,col in [(r['Pa'],'Pa','red'),(r['Pb'],'Pb','cyan')]:x,y=p;d.ellipse((x-6,y-6,x+6,y+6),outline=col,width=3);d.text((x+8,y),t,fill=col)
 im.save(folder/'actual_action.png');save(folder/'facts.json',facts)
 ep=source.parent;rounds=sorted(ep.glob('round_*'));idx=rounds.index(source);next_r=rounds[idx+1] if idx+1<len(rounds) else None
 timeline=[{'round':str(q.relative_to(ROOT)),'execution_status':read(q/'execution.json').get('status') if (q/'execution.json').exists() else 'UNKNOWN','before_at':read(q/'before/observation.json').get('captured_at') if (q/'before/observation.json').exists() else None} for q in rounds]
 save(folder/'sequence_context.json',{'episode':r['episode'],'ordered_rounds':timeline,'current_index':idx,'next_round':str(next_r.relative_to(ROOT)) if next_r else None,'future_affordance_gain':'NOT_ASSESSED','note':'Chronology retained. No counterfactual necessity or downstream credit inferred.'})
 names=['before/rgb.png','after/rgb.png','before/observation.json','after/observation.json','points.json','geometry.json','execution.json','native_plan.json','states.jsonl','motion.py','execution_images/frames.jsonl','return_perception/execution_images/frames.jsonl']
 save(folder/'provenance.json',{'round':r['round'],'files':{n:sha(source/n) for n in names if (source/n).exists()},'frames':[{**fr,'sha256':sha(ROOT/fr['source'])} for fr in frames]})
 freeze('packet_'+r['id'],[p for p in folder.iterdir() if p.is_file()])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','cv','select']);a=p.parse_args();{'inventory':inventory,'cv':compute_cv,'select':select}[a.stage]()
