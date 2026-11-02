import copy,hashlib,json,random,shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.cloth_harness.runtime.gpt_policy import PROMPT
from cloth_agent.restricted_context_one_shot import SCHEMA

PARTS={'LS':'左袖口边缘中点','RS':'右袖口边缘中点','LH':'左下摆真实侧缝交角','RH':'右下摆真实侧缝交角'}
# Left/right defined anatomically: front-facing garment's left is viewer's right.
CONSTRAINT='''本实验只允许抓四个真实语义端部附近的可见布料：衣服自身左/右袖口边缘中点、左/右下摆侧缝与下摆边相交的真实角点。左右按穿着者左右定义，不能按旋转后的图像左右重新命名。不要把折叠形成的突起、衣身内部折角或下摆中点当成下摆角。Pa可在对应端部向同一可见布面内侧微调以夹牢。不可猜隐藏部位；没有可靠可见端部时ABSTAIN，不因不可见就DONE。'''

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def phash(path):
    rgb=np.asarray(Image.open(path).convert('RGB'));gray=cv2.cvtColor(rgb[:,int(rgb.shape[1]*.18):int(rgb.shape[1]*.84)],cv2.COLOR_RGB2GRAY)
    dct=cv2.dct(cv2.resize(gray,(32,32)).astype('float32'))[:8,:8].ravel()[1:]
    return dct>np.median(dct)

def prepare(root,count=24):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);(root/'images').mkdir();(root/'frozen').mkdir()
    rng=random.Random(20261025);episodes=[]
    for pattern in ['ggo_rollout_2026102*','reset_radius_sweep_2026102*','automatic_top5_batch_2026102*']:
        for run in sorted(Path('reports').glob(pattern)):
            for ep in sorted(run.glob('episode_*')):
                images=sorted(ep.glob('round_*/before/rgb.png')) if ep.is_dir() else []
                if images:episodes.append((ep,images))
    rng.shuffle(episodes);selected=[];seen=set();hashes=[];exclusions=[]
    for ep,images in episodes:
        rng.shuffle(images)
        for path in images:
            meta=path.parent/'observation.json'
            if not meta.exists():continue
            m=read(meta);digest=sha(path)
            if m.get('synthetic') is not False or m.get('sha256',{}).get('rgb.png')!=digest:continue
            bits=phash(path)
            if digest in seen or any(np.count_nonzero(bits!=prior)<8 for prior in hashes):continue
            key=f'S{len(selected)+1:02d}';shutil.copyfile(path,root/'images'/f'{key}.png');selected.append({'id':key,'source_rgb':str(path.resolve()),'episode':str(ep.resolve()),'rgb_sha256':digest,'size':list(Image.open(path).size),'phash_bits':''.join('1' if b else '0' for b in bits)})
            seen.add(digest);hashes.append(bits);break
        else:exclusions.append(str(ep))
        if len(selected)>=count:break
    goal=Path('data/reference/deployment_example/ggo_goal.png');shutil.copyfile(goal,root/'goal.png')
    save(root/'cases.json',selected);save(root/'annotations.json',{});save(root/'annotation_drafts.json',{});save(root/'blind_labels.json',{})
    save(root/'frozen/policy.json',{'prompt':PROMPT,'schema':SCHEMA,'model':io.MODEL,'effort':io.EFFORT,'constraint':CONSTRAINT,'goal_sha256':sha(root/'goal.png')})
    save(root/'protocol.json',{'title':'Semantic Corner Grounding Oracle Experiment','status':'WAITING_HUMAN_ENDPOINT_VERIFICATION','cases':len(selected),'conditions':['A0','A1','A2'],'repeats':1,'expected_requests':len(selected)*3,'seed':20261025,'selection':'one current RGB per distinct Episode; seeded episode/frame order; SHA exact dedup + central-crop 63bit pHash distance >=8','episodes_available':len(episodes),'excluded_no_eligible_image':exclusions,
        'independence_limit':'Different episodes reduce temporal correlation; same garment/camera and possibly related resets, not guaranteed statistically independent.',
        'endpoint_definition':PARTS,'left_right_convention':'wearer anatomical left/right, invariant to image rotation; when front upright, wearer left is image right',
        'A0':'frozen existing fast one-shot without history/feedback; current and common goal RGB','A1':'A0 plus four-endpoint restriction only','A2':'A1 plus human-confirmed visible coordinates as text only; no action arrows in inputs',
        'primary_endpoint_radius_px':20,'sensitivity_radii_px':[10,20,30],'radius_limit':'coordinate proximity diagnostic, not physical grip success or material identity proof',
        'semantic_evaluation':'condition-blinded independent manual adjudication of proposed Pa; no model score as truth','false_corner_rule':'off-endpoint alone is not false corner; must independently identify a fold-created pseudo-corner',
        'rates':'report all-response and ACTION-only denominators, uncertainty/ABSTAIN/DONE/API failures separately','no_visible_candidates':'retain case; A2 must abstain unless independently DONE; no oracle imputation',
        'task_success_inference':False,'robot_calls':False,'training':False,'production_modified':False})
    sheet=Image.new('RGB',(1200,((len(selected)+3)//4)*195),'#f0f0f0');draw=ImageDraw.Draw(sheet)
    for i,c in enumerate(selected):
        im=Image.open(root/'images'/f"{c['id']}.png");im.thumbnail((300,170));x=(i%4)*300;y=(i//4)*195;sheet.paste(im,(x,y+22));draw.text((x+5,y+5),c['id'],fill='black')
    sheet.save(root/'contact_sheet.jpg');print(json.dumps({'root':str(root),'cases':len(selected),'unique_episodes':len({c['episode'] for c in selected})}))


def validate_annotation(value,size):
    if not value.get('confirmed_by'):raise ValueError('annotator confirmation required')
    if set(value.get('points',{}))!=set(PARTS):raise ValueError('four endpoint fields required')
    for key,p in value['points'].items():
        state=p.get('visibility');xy=p.get('px')
        if state not in ('VISIBLE','HIDDEN','UNCERTAIN'):raise ValueError('explicit visibility required for '+key)
        if state!='VISIBLE' and xy is not None:raise ValueError('hidden or uncertain endpoints cannot have coordinates')
        if state=='VISIBLE' and (not isinstance(xy,list) or len(xy)!=2 or any(type(v) is not int for v in xy) or not 0<=xy[0]<size[0] or not 0<=xy[1]<size[1]):raise ValueError('invalid visible endpoint coordinates')


def build_request(root,case,condition,annotation=None):
    root=Path(root);policy=read(root/'frozen/policy.json');body=io.payload(root/'images'/f"{case['id']}.png",root/'goal.png',case['size'])
    body['model']=policy['model'];body['reasoning']['effort']=policy['effort'];body['instructions']=policy['prompt'];body['text']['format'].update(name='semantic_corner_action',schema=policy['schema'])
    if condition in ('A1','A2'):body['instructions']+='\n'+policy['constraint']
    if condition=='A2':
        validate_annotation(annotation,case['size']);visible={k:{'part':PARTS[k],'px':p['px']} for k,p in annotation['points'].items() if p['visibility']=='VISIBLE'}
        body['input'][0]['content'].append({'type':'input_text','text':'人工已核验的当前图真实可见端部候选（像素；左右按穿着者定义）：'+json.dumps(visible,ensure_ascii=False)+'。只能选其中可见端部附近的同一布面；允许稍向内调整以夹牢。不生成其他语义端部，不推测缺失端部。空列表不代表已展开；无可行候选时ABSTAIN。'})
    elif condition not in ('A0','A1'):raise ValueError('unknown condition')
    return body


def endpoint_match(plan,annotation,radius=20):
    if plan['decision']!='ACTION':return {'status':plan['decision'],'part':None,'distance_px':None}
    import math
    candidates=[(math.dist(plan['grasp_px'],p['px']),k) for k,p in annotation['points'].items() if p['visibility']=='VISIBLE']
    if not candidates:return {'status':'NO_VERIFIED_VISIBLE_ENDPOINT','part':None,'distance_px':None}
    distance,part=min(candidates)
    return {'status':'WITHIN_RADIUS' if distance<=radius else 'OUTSIDE_RADIUS_NOT_A_FALSE_CORNER_LABEL','part':part if distance<=radius else None,'nearest_part':part,'distance_px':distance}
