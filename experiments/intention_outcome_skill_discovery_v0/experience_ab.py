"""Frozen one-state, three-repeat offline experience-context comparison."""
from pathlib import Path
import sys,json,random,hashlib,math,statistics,shutil,html,itertools
from PIL import Image,ImageDraw
from transport import call,save,read,sha
from contracts import obj,en,S,B,POINT
REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'reports/intention_outcome_skill_discovery_v0_20261030'
OUT=REPO/'reports/hindsight_experience_ab_20261030'
SEED='experience-ab-one-state-20261030-v1'
PROMPT='''你是离线衣物操作规划者。目标是显露穿着者左袖口开口边缘中点LS，不是普通轮廓尖角，也不要求同时展平整件衣服。第一张附件始终是当前RGB。根据当前图自主判断目标状态并决定下一步。若有额外经验，后续图片仅是历史经验，不能当作当前状态，不能照搬历史坐标或把历史结果当作当前已知结果。历史模型的解释也不是物理真值。
基本动作接口：单臂抓取一个可见位置Pa，牵拉至Pb并释放；可选择图像夹爪角0..180度或保持默认null。Pa、Pb用当前原图整数像素。motion_description可说明希望采用的运动方式，但本轮不编译或执行机器人动作。没有深度/标定，不能声称Host可行性已通过。
若存在可检验依据可以输出实验性ACTION，明确预测和反证。目标已可见或证据不足时允许ABSTAIN，不为输出动作而捏造材料身份。ABSTAIN时Pa/Pb/角度均null。输出中文，不调用工具。'''
nullable_point={'type':['array','null'],'items':{'type':'integer'},'minItems':2,'maxItems':2}
SCHEMA=obj({'decision':en('ACTION','ABSTAIN'),'goal':en('Reveal(LS)'),
 'target_state':en('VISIBLE','SUSPECTED_OCCLUDED','OCCLUDED','OUT_OF_VIEW','UNKNOWN'),
 'target_identity_basis':S,'visual_evidence':S,'grasp_region':S,'grasp_px':nullable_point,'target_px':nullable_point,
 'grasp_angle_deg':{'type':['integer','null'],'minimum':0,'maximum':180},
 'motion_description':S,'reason':S,'expected_effect':S,'falsifier':S,'uncertainty':S,'abstain_reason':S})

def prepare():
 OUT.mkdir(exist_ok=True)
 if (OUT/'protocol.json').exists():
  p=read(OUT/'protocol.json');assert p['code_sha256']==sha(__file__);return p
 split=read(REPO/'reports/hindsight_fast_test_20261029/split.json')
 index=read(REPO/'reports/hindsight_fast_test_20261029/experience_index.json')
 prior=split['validation']+read(REPO/'reports/hindsight_corner_continuation_20261030/input_manifest.json')['validation_before_only']
 prior+=read(BASE/'validation_inputs_before_only.json')+read(BASE/'interface_repair/validation_inputs_before_only.json')
 used={r['before_sha256'] for r in prior}
 train=read(BASE/'experience_index.json');used|={r[k+'_sha256'] for r in train for k in ['before','after']}
 pool=[{k:r[k] for k in ['run','before','before_sha256']} for r in index if r['run'] in split['heldout_complete_runs'] and r['before_sha256'] not in used]
 assert pool
 chosen=min(pool,key=lambda x:hashlib.sha256((SEED+x['before_sha256']).encode()).hexdigest())
 assert chosen['run'] not in {r['run'] for r in train}
 p=REPO/chosen['before'];assert sha(p)==chosen['before_sha256']
 shutil.copyfile(p,OUT/'current.png')
 selected=read(BASE/'selected_case.json');source=next(r for r in train if r['id']==selected['id'])
 experience={'original_pre_action_intention':source['raw_intention'],'executed_action':source['action'],
             'execution_record':source['execution'],'agent_induction':read(BASE/'calls/induction/attempt_0/output.json'),
             'evidence_limit':'PROVISIONAL. Sleeve visibility improved; same-material self-occlusion transition and grasp acquisition not confirmed. Not a successful skill demonstration.'}
 historical=[]
 for k in ['before','after']:
  src=Path(source[k]);dst=OUT/f'historical_{k}.png';shutil.copyfile(src,dst)
  historical.append({'source':str(src),'sha256':sha(src),'copy':str(dst)})
 save(OUT/'experience_context.json',experience)
 jobs=[]
 for rep in range(3):
  block=[{'condition':c,'repeat':rep+1} for c in ['A','B']]
  random.Random(SEED+str(rep)).shuffle(block);jobs+=block
 protocol={'seed':SEED,'code_sha256':sha(__file__),'model':'gpt-6-astra','effort':'high','profile':'api',
  'calls':6,'concurrency':1,'repeats_per_condition':3,'conditions':{'A':'task + current RGB','B':'same + raw experience pair/action/intention + original agent induction'},
  'question_scope':'Does access to this bundled hindsight context change decisions? Not isolation of raw experience versus inferred rules.',
  'selection':'Smallest seeded SHA256 among unused Before images in previously heldout complete runs; frozen before viewing current image or outputs.',
  'eligible_pool':pool,'current':chosen,'current_size':list(Image.open(p).size),'historical_images':historical,
  'experience_sha256':sha(OUT/'experience_context.json'),'common_prompt':PROMPT,'schema':SCHEMA,'order':jobs,
  'no_current_after_or_historical_action_for_current':True,'no_coordinator_method_in_context':True,'robot_execution':False,
  'evaluation':'All outputs, decisions, nullable coordinates, model-reported regions (not true labels), within-condition and cross-condition Pa/Pb distances, request latency/tokens. No physical success scoring or significance test with one state.'}
 save(OUT/'protocol.json',protocol);save(OUT/'schema.json',SCHEMA)
 return protocol

def dist(a,b):return math.dist(a,b)
def render(results,p):
 def esc(x):return html.escape(str(x))
 def stat(xs):
  if not xs:return None
  xs=sorted(xs);q=(len(xs)-1)*.95;i=int(q);return {'n':len(xs),'mean':statistics.mean(xs),'median':statistics.median(xs),'p95':xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(q-i)}
 metrics={}
 for c in ['A','B']:
  rr=[x for x in results if x['condition']==c];acts=[x['output'] for x in rr if x.get('output',{}).get('decision')=='ACTION']
  metrics[c]={'calls':len(rr),'decisions':{d:sum(x.get('output',{}).get('decision')==d for x in rr) for d in ['ACTION','ABSTAIN']},
   'latency_s':stat([x['latency_s'] for x in rr]),'within_pa_distance_px':stat([dist(x['grasp_px'],y['grasp_px']) for x,y in itertools.combinations(acts,2)]),
   'within_pb_distance_px':stat([dist(x['target_px'],y['target_px']) for x,y in itertools.combinations(acts,2)])}
  metrics[c]['usage']={k:sum((x.get('usage') or {}).get(k,0) for x in rr) for k in ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']}
 aa=[x['output'] for x in results if x['condition']=='A' and x.get('output',{}).get('decision')=='ACTION']
 bb=[x['output'] for x in results if x['condition']=='B' and x.get('output',{}).get('decision')=='ACTION']
 metrics['cross_pa_distance_px']=stat([dist(x['grasp_px'],y['grasp_px']) for x in aa for y in bb])
 metrics['cross_pb_distance_px']=stat([dist(x['target_px'],y['target_px']) for x in aa for y in bb])
 metrics['limitations']='One fixed state, three requests each. No independent semantic labels, no physical execution. Correlated pairwise distances are descriptive, not independent trials. B bundles experience and agent inference, not context-length controlled.'
 save(OUT/'metrics.json',metrics)
 body=['<h1>Hindsight经验是否改变决策？</h1><p>一张固定新图，A无经验 / B原始经验与Agent自己的假设。各3次独立codex -p api请求。不使用协调者提出的方法，不执行机器人。</p>',
 '<p>当前图来自独立保留run的未使用Before，按固定种子选择。没有根据结果挑图。它是否适合Reveal尚无独立真值。</p>',
 '<a href="current.png"><img src="current.png"></a>', '<h2>固定协议与描述性指标</h2><pre>'+esc(json.dumps(metrics,ensure_ascii=False,indent=2))+'</pre>',
 '<p>坐标差异不是好坏；部位名称是模型自述，不是真值。B含原始经验及归纳假设，因此不能把效应单独归因于物理经验学习，也没有等Token对照。</p>']
 for c in ['A','B']:
  body.append('<h2>'+c+'：'+esc(p['conditions'][c])+'</h2>')
  for x in results:
   if x['condition']!=c:continue
   o=x.get('output',{});im=Image.open(OUT/'current.png').convert('RGB');d=ImageDraw.Draw(im)
   if o.get('decision')=='ACTION':
    a=o['grasp_px'];b=o['target_px'];d.line([tuple(a),tuple(b)],fill='yellow',width=5)
    ang=math.atan2(b[1]-a[1],b[0]-a[0]);tip=[tuple(b)]+[(b[0]-18*math.cos(ang+s),b[1]-18*math.sin(ang+s)) for s in [-.45,.45]];d.polygon(tip,fill='yellow')
    for pt,color,label in [(a,'lime','Pa'),(b,'red','Pb')]:d.ellipse((pt[0]-7,pt[1]-7,pt[0]+7,pt[1]+7),fill=color);d.text((pt[0]+9,pt[1]),label,fill=color)
   img=f'{c}{x["repeat"]}.png';im.save(OUT/img)
   body+=['<h3>'+c+str(x['repeat'])+' '+esc(o.get('decision',x['status']))+'</h3><a href="'+img+'"><img src="'+img+'"></a><pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>']
 body.append('<p><a href="protocol.json">协议</a> · <a href="experience_context.json">B组实际经验输入</a> · <a href="results.json">全部原始输出</a> · <a href="metrics.json">指标</a></p>')
 (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Hindsight经验 A/B</title><style>body{max-width:1200px;margin:30px auto;font:17px/1.6 sans-serif}img{max-width:100%}pre{white-space:pre-wrap;background:#eef2f6;padding:16px}</style>'+''.join(body))

def main():
 p=prepare()
 if '--prepare-only' in sys.argv:print(json.dumps(p['current']));return
 results=[]
 for j in p['order']:
  c=j['condition'];payload={'task':'Reveal(LS)','current_image_index':1,'current_size':p['current_size']}
  images=[OUT/'current.png']
  if c=='B':
   payload['historical_experience']=read(OUT/'experience_context.json');payload['historical_image_order']=['image 2: before','image 3: after']
   images += [OUT/'historical_before.png',OUT/'historical_after.png']
  name=f'{c}{j["repeat"]}'
  r=call(OUT,name,PROMPT,payload,images,SCHEMA)
  checks=[];o=r.get('output')
  if o:
   for k in ['grasp_px','target_px']:
    pt=o[k]
    if o['decision']=='ABSTAIN' and pt is not None:checks.append(k+' should be null')
    if o['decision']=='ACTION' and (pt is None or not (0<=pt[0]<p['current_size'][0] and 0<=pt[1]<p['current_size'][1])):checks.append(k+' invalid image coordinate')
  results.append({**j,**r,'static_pixel_issues':checks,'host_safety':'NOT_TESTED','physical_success':'NOT_TESTED'})
  save(OUT/'results.json',results);print(json.dumps({'completed':len(results),'total':6,'name':name,'status':r['status'],'decision':o.get('decision') if o else None}),flush=True)
  if r['status']!='COMPLETE':break
 render(results,p)
 print(str(OUT/'review.html'))

if __name__=='__main__':main()
