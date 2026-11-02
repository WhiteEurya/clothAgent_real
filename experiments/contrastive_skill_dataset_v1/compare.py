"""Candidate comparisons from pre-state geometry and recorded conditions only."""
from common import *
from dataset import rigid_align
from transport import call
import cv2,itertools,collections

def angular_distance(a,b):
 return float(max(abs((x-y+180)%360-180) for x,y in zip(a,b)))

def rank_pairs():
 cases=read(OUT/'selection_manifest.json')['cases'];out=[]
 for a,b in itertools.combinations(cases,2):
  if a['split']!=b['split'] or a['split']=='reference':continue
  ma=cv2.imread(str(OUT/'cv'/a['raw_id']/'before_mask.png'),0);mb=cv2.imread(str(OUT/'cv'/b['raw_id']/'before_mask.png'),0)
  sim,rot,_=rigid_align(ma,mb);ratio=float((ma>0).sum()/max(1,(mb>0).sum()));pa=a['geometry']['primitive'];pb=b['geometry']['primitive'];ga=a['geometry']['rpy_world_deg'];gb=b['geometry']['rpy_world_deg'];ori=angular_distance(ga,gb)
  conditions={'same_camera':a['camera_serial']==b['camera_serial'],'same_calibration':a['calibration_id']==b['calibration_id'],'same_source_family':a['family']==b['family'],'same_prompt_hash':a['prompt_sha256']==b['prompt_sha256'] and a['prompt_sha256'] is not None,'same_lift':pa['lift_height_mm']==pb['lift_height_mm'],'same_primitive_parameters':pa==pb,'same_speed':a['speeds_mm_s']==b['speeds_mm_s'],'speed_recorded':bool(a['speeds_mm_s']) and bool(b['speeds_mm_s']),'same_release':a['release']==b['release'],'release_recorded':bool(a['release']) and bool(b['release']),'orientation_within_10deg':ori<=10,'different_episode':a['episode']!=b['episode'],'different_run_group':a['group']!=b['group']}
  exact=all(conditions.values());shape=sim>=.7 and .65<=ratio<=1.54
  relaxed=all(conditions[k] for k in ['same_camera','same_calibration','same_lift','same_primitive_parameters','same_speed','same_release','orientation_within_10deg']) and sim>=.55 and .5<=ratio<=2
  tier='STRICT_PRE_CANDIDATE' if exact and shape else 'PARTIAL_PRE_CANDIDATE' if relaxed else 'WEAK_OR_CONFOUNDED'
  v1=np.asarray(a['Pb'])-a['Pa'];v2=np.asarray(b['Pb'])-b['Pa'];den=np.linalg.norm(v1)*np.linalg.norm(v2);angle=float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1)))) if den else None
  out.append({'pair_id':a['id']+'_'+b['id'],'ids':[a['id'],b['id']],'split':a['split'],'pre_similarity_iou_after_rigid_alignment':sim,'pre_area_ratio':ratio,'alignment_rotation_deg':rot,'orientation_parameter_difference_deg':ori,'conditions':conditions,'pre_tier':tier,'action_direction_difference_image_deg':angle,'action_distance_mm':[a['geometry']['distance_mm'],b['geometry']['distance_mm']],'not_a_causal_match':True})
 save(OUT/'pre_pair_candidates.json',{'at':utc(),'uses_outcome_labels':False,'cases':out});freeze('pre_pairs',[OUT/'pre_pair_candidates.json']);print('PRE_PAIRS',collections.Counter(p['pre_tier'] for p in out),flush=True)

PAIR_SCHEMA=obj({'pairs':{'type':'array','items':obj({'pair_id':S,'before_structure_comparable':enum('YES','NO','UNKNOWN'),'confidence':NUM,'shared_visible_structure':SS,'important_initial_differences':SS,'unobserved_confounders':SS,'recommended_use':enum('MATCHED_OBSERVATIONAL_COMPARISON','QUALITATIVE_CONTRAST_ONLY','DO_NOT_COMPARE')})}})

def prepare_review():
 verify_freezes();es={e['id']:e for e in read(OUT/'effects_verified.json')['cases']};pairs=read(OUT/'pre_pair_candidates.json')['cases'];eligible=[]
 for p in pairs:
  a,b=[es[c] for c in p['ids']];va=a['z']['unfolding_gain']['value'];vb=b['z']['unfolding_gain']['value'];gn=a['contrast_group']!=b['contrast_group'];different={va,vb}=={'YES','NO_CLEAR_EVIDENCE'}
  q={**p,'effect_groups':[a['contrast_group'],b['contrast_group']],'unfolding_contrast_observed':different,'different_effect_group':gn,'has_clear_mixed_vs_unfolding':set([a['contrast_group'],b['contrast_group']])=={'A_LOCAL_UNFOLDING','C_MIXED'}}
  if q['pre_tier']!='WEAK_OR_CONFOUNDED' and (different or q['has_clear_mixed_vs_unfolding']):eligible.append(q)
 selected=[]
 for split in ['train','development','test']:
  ranked=sorted([p for p in eligible if p['split']==split],key=lambda p:(p['pre_tier']=='STRICT_PRE_CANDIDATE',p['conditions']['different_run_group'],p['conditions']['same_prompt_hash'],p['pre_similarity_iou_after_rigid_alignment']),reverse=True)
  selected+=ranked[:(2 if split=='train' else 1)]
 # The pre-only coverage audit found strict pairs split across partitions.
 # Audit their comparability separately; never expose them as training pairs.
 cases=read(OUT/'selection_manifest.json')['cases'];byround={c['round']:c for c in cases};cross=[]
 for item in read(OUT/'core_pool_pair_coverage.json')['near_pre_pairs']:
  if not item['both_selected'] or item['same_split']:continue
  a,b=[byround[r] for r in item['rounds']];ea,eb=[es[c['id']] for c in [a,b]]
  if {ea['z']['unfolding_gain']['value'],eb['z']['unfolding_gain']['value']}!={'YES','NO_CLEAR_EVIDENCE'}:continue
  conditions={k:True for k in ['same_camera','same_calibration','same_source_family','same_prompt_hash','same_lift','same_primitive_parameters','same_speed','speed_recorded','same_release','release_recorded','orientation_within_10deg','different_episode','different_run_group']}
  cross.append({'pair_id':a['id']+'_'+b['id'],'ids':[a['id'],b['id']],'split':'CROSS_SPLIT_AUDIT_ONLY','original_splits':[a['split'],b['split']],'for_learning':False,'pre_similarity_iou_after_rigid_alignment':item['pre_iou'],'conditions':conditions,'pre_tier':'STRICT_PRE_CANDIDATE','effect_groups':[ea['contrast_group'],eb['contrast_group']],'unfolding_contrast_observed':True,'not_a_causal_match':True})
 selected+=sorted(cross,key=lambda p:p['pre_similarity_iou_after_rigid_alignment'],reverse=True)[:4]
 save(OUT/'contrastive_pair_selection.json',{'at':utc(),'all_eligible':eligible,'selected_for_pre_only_review':selected,'selection_uses_effects':True,'reviewer_sees_effects':False,'not_for_estimating_population_effect_frequency':True});freeze('pair_review_selection',[OUT/'contrastive_pair_selection.json'])
 print('PAIR_REVIEW_READY',len(selected),'of',len(eligible),flush=True)

def review():
 if not (OUT/'contrastive_pair_selection.json').exists():prepare_review()
 selected=read(OUT/'contrastive_pair_selection.json')['selected_for_pre_only_review'];results=[]
 for split in ['train','development','test']:
  part=[p for p in selected if p['split']==split]
  if not part:continue
  # At most two batches overall; combine dev+test only if needed but keep separate IDs.
 # Do not mix Test labels into training: this judge sees only pre-state and conditions.
 batches=[selected[:4],selected[4:8]]
 for i,part in enumerate(batches):
  if not part:continue
  prompt='''你是独立初态可比较性审查者。仅比较每对操作前RGB，不看动作后图、效果标签、候选Skill或历史Planner理由，也不推荐动作。任务是判断两件当前布形是否足够相似，可以用于历史操作效果的观察性对照。
不能仅因外轮廓相似就判可比：请比较内部折叠/覆盖、可见布边及展开难度。把已知相机及原语参数作为条件背景，未知抓层、摩擦、复位差异等保留。YES只表示具有有限观察可比性，不是随机受控初态或因果证明。明显差异判NO，无法辨认判UNKNOWN。每项简短具体。
''';images=[]
  for p in part:
   prompt+='配对ID '+p['pair_id']+'，附件顺序'+','.join(p['ids'])+'的Before。已知条件：'+json.dumps(p['conditions'])+'。\n';images += [OUT/'cases'/c/'before.png' for c in p['ids']]
  d=call(f'clean_pair_review_{i+1:02d}',prompt,images,PAIR_SCHEMA);assert sorted(x['pair_id'] for x in d['pairs'])==sorted(x['pair_id'] for x in part);results+=d['pairs']
 final=[];lookup={r['pair_id']:r for r in results}
 for p in selected:
  r=lookup.get(p['pair_id']);strict=p['pre_tier']=='STRICT_PRE_CANDIDATE';yes=r and r['before_structure_comparable']=='YES' and r['confidence']>=.65
  quality='STRONG_OBSERVATIONAL_PAIR' if strict and yes else 'QUALITATIVE_PAIR' if yes and r['recommended_use']!='DO_NOT_COMPARE' else 'NOT_VERIFIED_COMPARABLE'
  final.append({**p,'visual_review':r,'final_quality':quality,'residual_confounding':True})
 save(OUT/'comparability.json',{'at':utc(),'pairs':final,'scope':'Within-split contrasts plus explicitly segregated cross-split audit-only pairs. Cross-split pairs must never be learning material; even strong means observational, not causal.','strong_count':sum(x['final_quality']=='STRONG_OBSERVATIONAL_PAIR' for x in final),'strong_within_split':sum(x['final_quality']=='STRONG_OBSERVATIONAL_PAIR' and x['split']!='CROSS_SPLIT_AUDIT_ONLY' for x in final)});freeze('comparability',[OUT/'comparability.json'])
 print('PAIR_DONE',collections.Counter(x['final_quality'] for x in final),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['rank','review']);a=p.parse_args();rank_pairs() if a.stage=='rank' else review()
