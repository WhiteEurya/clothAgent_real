"""Three-call maximum Hindsight Goal Inversion; offline, isolated API analysis."""
from pathlib import Path
import json,hashlib,sys,shutil,html,math
from PIL import Image,ImageDraw
from transport import call,read,save,sha
REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'reports/intention_outcome_skill_discovery_v0_20261030'
OUT=REPO/'reports/hindsight_goal_inversion_20261030'
S={'type':'string'}
def en(*v):return {'type':'string','enum':list(v)}
def arr(x):return {'type':'array','items':x}
def obj(x):return {'type':'object','properties':x,'required':list(x),'additionalProperties':False}
POINT={'type':['array','null'],'items':{'type':'integer'},'minItems':2,'maxItems':2}
IND=obj({'original_intended_region':S,'original_expected_change':S,'observed_target_related_change':S,
 'intention_outcome_mismatch':S,'corner_goal':en('Reveal(LS)','Reveal(RS)','Reveal(LH)','Reveal(RH)'),
 'evidence_status':en('PROVISIONAL','UNKNOWN'),'corner_evidence':S,'alternative_non_reveal_explanation':S,
 'worth_testing':{'type':'boolean'},'inverted_goal':S,'how_to_use_action_effect_relation':S,
 'hypotheses':arr(obj({'id':S,'explanation':S,'observable_applicability':S,'testable_action_principle':S,'predicted_effect':S,'falsifier':S,'unobserved_assumptions':S})),
 'needed_information':arr(S),'limitations':S})
SEL=obj({'reviews':arr(obj({'id':S,'target_state':en('VISIBLE','SUSPECTED_OCCLUDED','OCCLUDED','OUT_OF_VIEW','UNKNOWN'),
 'semantic_identity_evidence':S,'relevant_visible_clues':S,'applicable_hypothesis_ids':arr(S),'test_value':en('RELATED','WEAK','NONE'),'limitations':S})),
 'selected_id':{'type':['string','null']},'selection_reason':S,'if_none_missing_information':arr(S)})
ACT=obj({'status':en('EXPERIMENTAL_ACTION','NO_GROUNDED_ACTION'),'goal':en('Reveal(LS)','Reveal(RS)','Reveal(LH)','Reveal(RH)'),
 'hypothesis_id':S,'target_identity_evidence':S,'current_target_state':S,'grasp_region':S,
 'grasp_px':POINT,'target_px':POINT,'grasp_angle_deg':{'type':['integer','null'],'minimum':0,'maximum':180},
 'motion_description':S,'selection_reason':S,'expected_semantic_endpoint_change':S,'observable_prediction':S,
 'falsifying_observations':arr(S),'execution_failure_confound':S,'assumptions':arr(S),
 'missing_information':arr(obj({'category':en('TARGET_IDENTITY','GRASP_REGION','DIRECTION','OTHER'),'detail':S})),
 'limitations':S})

def prepare():
 OUT.mkdir(exist_ok=True)
 if (OUT/'protocol.json').exists():
  p=read(OUT/'protocol.json');assert sha(__file__)==p['code_sha256'];return p
 selected=read(BASE/'selected_case.json')
 source=next(x for x in read(BASE/'experience_index.json') if x['id']==selected['id'])
 previous=read(BASE/'skill_hypotheses.json')
 raw={'original_intention':source['raw_intention'],'actual_action':source['action'],'execution_record':source['execution'],
      'observed_corner_event':selected['observed'],'independent_intention_comparison':selected['comparison'],
      'previous_agent_physical_hypotheses':previous['hypotheses'],'previous_evidence_limitations':previous['limitations']}
 save(OUT/'source_experience.json',raw)
 images=[]
 for k in ['before','after']:
  src=Path(source[k]);dest=OUT/f'experience_{k}.png';shutil.copyfile(src,dest)
  images.append({'source':str(src),'sha256':sha(src),'copy':str(dest)})
 split=read(REPO/'reports/hindsight_fast_test_20261029/split.json')
 prior=split['validation']+read(REPO/'reports/hindsight_corner_continuation_20261030/input_manifest.json')['validation_before_only']
 prior+=read(BASE/'validation_inputs_before_only.json')+read(BASE/'interface_repair/validation_inputs_before_only.json')
 prior+=[read(REPO/'reports/hindsight_experience_ab_20261030/protocol.json')['current']]
 used={x['before_sha256'] for x in prior}
 training=read(BASE/'experience_index.json');used|={x[k+'_sha256'] for x in training for k in ['before','after']}
 pool=[]
 for run in split['heldout_complete_runs']:
  # Enumerate RGB paths only. Never load validation action logs or After images.
  for p in sorted((REPO/run).glob('**/round_*/before/rgb.png')):
   h=sha(p)
   if h not in used:pool.append({'run':run,'before':str(p),'before_sha256':h})
 unique={r['before_sha256']:r for r in pool}
 seed='goal-inversion-minimal-before-only-20261030'
 candidates=sorted(unique.values(),key=lambda r:hashlib.sha256((seed+r['before_sha256']).encode()).hexdigest())[:6]
 assert candidates
 for i,r in enumerate(candidates):
  r['id']=f'N{i+1:02d}';r['size']=list(Image.open(r['before']).size)
  assert r['run'] not in {x['run'] for x in training}
  shutil.copyfile(r['before'],OUT/(r['id']+'.png'))
 p={'version':'goal-inversion-minimal-v1','code_sha256':sha(__file__),'source_id':selected['id'],'locked_goal':selected['goal'],
    'source_images':images,'source_context_sha256':sha(OUT/'source_experience.json'),'candidates':candidates,
    'pool_count':len(unique),'selection_seed':seed,'candidate_limit':6,
    'selection_policy':'Before-only pool from pre-heldout complete runs; excludes earlier test/AB images; hashed order then agent relevance selection without actions or After.',
    'source_metadata_files':{str(p):sha(p) for p in [BASE/'selected_case.json',BASE/'experience_index.json',BASE/'skill_hypotheses.json']},
    'model':'gpt-6-astra','effort':'high','profile':'api','max_calls':3,'schemas':{'inversion':IND,'selection':SEL,'action':ACT},
    'forbidden_context':'No E353 analysis, human skills, old guiding prompts, coordinator proposals, prior new-image decisions, generated candidate skill or follow-up instructions.',
    'scope':'Only provisional hypothesis and one new-state action prediction. No general skill generation, policy editing, robot connection or execution.'}
 save(OUT/'protocol.json',p);save(OUT/'new_states_before_only.json',candidates)
 return p

def render(p):
 esc=lambda x:html.escape(str(x)); dump=lambda x:'<pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
 get=lambda n:read(OUT/n) if (OUT/n).exists() else None
 summary=get('summary.json');inv=get('goal_inversion.json');sel=get('state_selection.json');action=get('candidate_prediction.json')
 body=['<h1>Hindsight Goal Inversion 最小离线实验</h1>',dump(summary),
 '<p>模型只使用自动选中案例的原始意图、实际动作、Before/After、已保存观察和物理假设。没有传入协调者的方法、其他人工技能或旧引导Prompt。新状态只提供Before；报告中的动作未经物理验证。</p>',
 '<h2>原始经验与目标倒推</h2><div class="pair"><img src="experience_before.png"><img src="experience_after.png"></div>',dump(get('source_experience.json')),dump(inv),
 '<h2>新状态选择：仅Before</h2>',dump(sel)]
 for r in p['candidates']:
  body+=['<h3>'+r['id']+'</h3><a href="'+r['id']+'.png"><img src="'+r['id']+'.png"></a>']
 if action:
  row=next(x for x in p['candidates'] if x['id']==sel['selected_id'])
  im=Image.open(OUT/(row['id']+'.png')).convert('RGB');d=ImageDraw.Draw(im)
  if action['status']=='EXPERIMENTAL_ACTION':
   a=action['grasp_px'];b=action['target_px'];d.line([tuple(a),tuple(b)],fill='yellow',width=5)
   t=math.atan2(b[1]-a[1],b[0]-a[0]);d.polygon([tuple(b)]+[(b[0]-18*math.cos(t+s),b[1]-18*math.sin(t+s)) for s in [-.4,.4]],fill='yellow')
   for pt,col,label in [(a,'lime','Pa'),(b,'red','Pb')]:d.ellipse((pt[0]-7,pt[1]-7,pt[0]+7,pt[1]+7),fill=col);d.text((pt[0]+10,pt[1]),label,fill=col)
  im.save(OUT/'candidate_overlay.png')
  body+=['<h2>最终候选动作预测</h2><a href="candidate_overlay.png"><img src="candidate_overlay.png"></a>',dump(action)]
 body+=['<h2>复现与证据边界</h2><p>模型输出不是材料身份真值。选择了相关图不等于独立泛化评价。没有标定/深度或Host运动检查；图像范围检查不能证明物理可执行。主协调会话有历史上下文，发现模型为新独立API会话且禁止工具访问。</p>',
         '<p>'+ ' · '.join(f'<a href="{x}">{x}</a>' for x in ['protocol.json','goal_inversion.json','state_selection.json','candidate_prediction.json','summary.json','audit.json'] if (OUT/x).exists())+'</p>']
 (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Goal Inversion</title><style>body{max-width:1280px;margin:30px auto;font:17px/1.6 sans-serif}img{max-width:100%}.pair{display:flex}.pair img{width:50%}pre{white-space:pre-wrap;background:#edf2f7;padding:15px}</style>'+''.join(body))

def main():
 p=prepare()
 if '--prepare-only' in sys.argv:print(json.dumps({'pool':p['pool_count'],'candidates':[(x['id'],x['before']) for x in p['candidates']]},ensure_ascii=False));return
 status='UNKNOWN';results=[]
 def request(name,instruction,payload,images,schema):
  result=call(OUT,name,instruction,payload,images,schema);results.append(result)
  if result['status']!='COMPLETE':raise RuntimeError(result.get('error'))
  return result['output']
 inv=request('inversion',
   '开展最小Hindsight Goal Inversion。附件依次为原始经验Before/After。只使用所附原始意图、执行记录、观察和先前独立Agent的物理假设。先还原原来想改变哪里、实际观察到哪个目标相关额外变化；评估是否支持暂定Corner Reveal。目标限LS/RS袖口边缘中点、LH/RH下摆侧缝交角，左右按穿着者。不能把普通展开、边缘清晰或面积增加升级为角点恢复。locked_goal是原发现目标，不得替换。若证据不足保留UNKNOWN。若值得研究，反问“如果原本就想实现此效果，会如何利用已经观察到的动作—效果关系”，自主给出可检验操作假设和新RGB中的适用线索。先前假设可质疑，不是答案或物理真值。不得假定固定抓取侧/方向，不复用坐标。只提出实验预测，不宣称已经学会技能。中文。',
   {'locked_goal':p['locked_goal'],'experience':read(OUT/'source_experience.json')},[OUT/'experience_before.png',OUT/'experience_after.png'],IND)
 assert inv['corner_goal']==p['locked_goal'];save(OUT/'goal_inversion.json',inv)
 if inv['worth_testing']:
  sel=request('selection',
    '基于冻结的目标倒推分析，逐张查看本次新状态Before RGB。所有图片都未参与该假设归纳，不提供其历史动作或After。逐张说明目标身份/可见性、相关线索和局限，自主选择一个最值得验证当前假设的状态。仅考虑Corner Reveal，不以普通展开替代目标。不需要确认隐藏精确坐标，不因希望得到动作而捏造适用条件；若均无依据可selected_id=null。只选择状态，不生成动作或读取文件。中文。',
    {'frozen_inversion':inv,'image_order':[{'id':r['id'],'size':r['size']} for r in p['candidates']]},[OUT/(r['id']+'.png') for r in p['candidates']],SEL)
  assert {x['id'] for x in sel['reviews']}=={x['id'] for x in p['candidates']};assert len(sel['reviews'])==len(p['candidates'])
  save(OUT/'state_selection.json',sel)
  if sel['selected_id'] is not None:
   row=next(x for x in p['candidates'] if x['id']==sel['selected_id'])
   act=request('prediction',
     '附件只有选中的新状态Before。根据冻结的Hindsight Goal Inversion与状态选择，为同一语义端点自主提出最多一个具体可证伪实验动作。单臂基本接口是抓Pa、牵拉到Pb、释放；可说明希望的抬升方式，夹爪图像角度0..180或null保持默认。Pa/Pb使用所附原图整数像素，不能沿用历史坐标。Pb是牵拉终点，不是隐藏端点位置。给出选择理由、预计端点变化、可观察预测和反证，区分抓取失败与机制反证。没有深度/标定，不能称Host已通过。若无法产生有依据动作，明确缺少目标识别、可操作区域、方向或其他信息，Pa/Pb填null。允许条件性实验动作，不要求通用规则；不能更换目标或借用未提供的人为抓取规律。中文。',
     {'frozen_inversion':inv,'selected_state_review':next(x for x in sel['reviews'] if x['id']==row['id']),
      'selection_reason':sel['selection_reason'],'current_size':row['size'],'locked_goal':p['locked_goal']},[OUT/(row['id']+'.png')],ACT)
   assert act['goal']==p['locked_goal'];issues=[]
   for k in ['grasp_px','target_px']:
    pt=act[k]
    if act['status']=='EXPERIMENTAL_ACTION' and (pt is None or not (0<=pt[0]<row['size'][0] and 0<=pt[1]<row['size'][1])):issues.append(k+' outside image or missing')
    if act['status']=='NO_GROUNDED_ACTION' and pt is not None:issues.append(k+' must be null')
   save(OUT/'candidate_prediction.json',act);save(OUT/'static_checks.json',{'pixel_contract_issues':issues,'host_safety':'NOT_TESTED','physical_execution':False})
   status=act['status'] if not issues else 'INVALID_ACTION_CONTRACT'
  else:status='NO_RELATED_NEW_STATE'
 else:status='NO_SUPPORTED_INVERSION'
 save(OUT/'summary.json',{'status':status,'historical_evidence':inv['evidence_status'],'goal':p['locked_goal'],'model_calls':len(results),
  'latency_s':[r['latency_s'] for r in results],'total_request_s':sum(r['latency_s'] for r in results),
  'usage':{k:sum((r.get('usage') or {}).get(k,0) for r in results) for k in ['input_tokens','output_tokens','reasoning_output_tokens']},
  'physical_actions':0,'claim':'Specific falsifiable prediction only if status EXPERIMENTAL_ACTION; no physical validation or generalization claim.'})
 render(p);print(json.dumps(read(OUT/'summary.json'),ensure_ascii=False))

if __name__=='__main__':main()
