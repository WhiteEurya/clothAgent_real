"""Local action retrieval + one GPT call for applicability and current anchors.

No graph node matching and no robot commands in this module.
"""
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .discovery_data import save
from .discovery_schema import obj, S, N, array, enum
from .graph_minimal_grounding import SCHEMA as ANCHORS, validate as validate_anchors
from .graph_policy import Requests, checked_file
from .local_action_memory import load_library
from .retrieval_matcher import worker_python, ROOT

SCHEMA=obj(decision=enum('ACT','DONE','STOP'),edge_id=S,confidence=N,
    grasp_px=ANCHORS['properties']['grasp_px'],target_px=ANCHORS['properties']['target_px'],
    local_evidence=array(S),uncertainties=array(S),reason=S,expected_change=S,
    previous_effect=enum('INITIAL','PROGRESS','NO_CHANGE','WORSE','UNCERTAIN'),completion_evidence=array(S))

PROMPT='''目标是把衣物展开，允许普通小褶皱。你选择的是局部可复用动作，不是 graph 节点或整件衣物状态。
current 是当前完整照片；candidate 图每张包含示范 BEFORE（红点抓取、青点目标）、真实 AFTER、示范抓取局部、
检索出的当前候选局部。候选框坐标及全部输出坐标均对应 ORIGINAL CURRENT 图，绝不是拼图坐标。
DINO 相似分仅用于召回，不代表动作成功或适用性。候选局部可能只是同色布料、印花或桌面误匹配。
必须查看可见布边、翻折边界、覆盖方向、可接近的布层和移动空间，而不是只看颜色、纹理、袖子名称或轮廓。
先检查当前是否还有值得展开的覆盖层。衣身主体、两袖和下摆已展开且无遮挡时 DONE，说明全图完成证据。
不要求全局形状和示范一致，也不要求完全确定隐藏层数；但可见的局部抓取关系与动作目的必须有依据。
从给定候选中选一个符合展开目标的已有动作，并把示范抓取点与目标点迁移到当前图。可以微调检索框内的抓点，
不要把检索框中心自动当抓点；抓点必须在选中 query_box_xyxy 内，否则该候选不支持本次局部对应。
目标点可在框外，需依据当前整图找到与示范作用相同的接收/释放位置，不能照搬像素差或凭空设计另一种动作。
向内移动可能是为了露出下面的袖子；不要把“内移”一概当作折叠并排除。反之，局部像不等于这次内移有益。
参考 AFTER 是观察到的变化，不是成功保证。结合它和示范限制，说明该动作为什么有望减少遮挡或打开折层。
若有 previous_before，用它和 current 检查上一步效果，避免刚拉开又拉回、重复没有变化的动作。
ACT 必须提供所选 edge_id、局部对应证据、清楚可见的抓点和目标点、预期可验证变化。只输出一次动作。
不确定性如实说明、confidence 如实给出。没有合适候选则 STOP；不得因为必须继续而伪造对应关系。
DONE/STOP 的 edge_id 为空，两个点为 null。只返回 schema JSON。'''


def make_requests(output, model='gpt-6-astra', timeout=180., max_calls=8):
    from .discovery_model import CodexMultimodal
    return Requests(Path(output)/'model',CodexMultimodal(model,timeout),max_calls=max_calls,no_cache=True,
        common='Independent Arm6 local morphology action experiment. Goal: unfold. No state IDs, graph transitions or expected next state.',
        schemas={'local_action':SCHEMA},prompts={'local_action':PROMPT},protocol='arm6-local-action-policy-v1')


def validate_proposal(value, ranking, size):
    if value['decision']!='ACT':
        if value['edge_id'] or value['grasp_px'] is not None or value['target_px'] is not None:
            raise ValueError('DONE/STOP must not contain an action')
        if value['decision']=='DONE' and not value['completion_evidence']:
            raise ValueError('DONE requires current-image completion evidence')
        return
    rows=[r for r in ranking if r['edge_id']==value['edge_id']]
    if len(rows)!=1:raise ValueError('selected action is not in local shortlist')
    if not value['local_evidence'] or not value['expected_change'].strip():
        raise ValueError('missing local correspondence or expected effect')
    validate_anchors({k:value[k] for k in ('grasp_px','target_px','confidence')},size)
    if value['grasp_px'] is None or value['target_px'] is None:raise ValueError('ACT requires two points')
    x,y=value['grasp_px'];a,b,c,d=rows[0]['query_box_xyxy']
    if not a<=x<c or not b<=y<d:raise ValueError('grasp outside retrieved local region')
    if value['grasp_px']==value['target_px']:raise ValueError('grasp and target coincide')


def candidate_board(image, action, ranked, destination):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    with Image.open(checked_file(action['before'])) as im:before=im.convert('RGB')
    with Image.open(checked_file(action['after'])) as im:after=im.convert('RGB')
    with Image.open(image) as im:current=im.convert('RGB')
    crop=before.crop(ranked['reference_crop']['box_xyxy'])
    draw=ImageDraw.Draw(before)
    start,target=action['reference_start_px'],action['reference_target_px']
    draw.line([tuple(start),tuple(target)],fill='#ffd166',width=3)
    for pt,color,label in [(start,'#ff5252','START'),(target,'#42e5ff','TARGET')]:
        x,y=pt;draw.ellipse((x-9,y-9,x+9,y+9),outline=color,width=4)
        draw.text((x+10,y),label,font=font,fill=color,stroke_width=1,stroke_fill='black')
    board=Image.new('RGB',(1280,880),'#101827');draw=ImageDraw.Draw(board)
    entries=[('DEMO BEFORE: '+action['edge_id'],before),('DEMO AFTER (observed)',after),
             ('DEMO grasp neighborhood (unrotated)',crop),('CURRENT candidate neighborhood',current.crop(ranked['query_box_xyxy']))]
    for i,(label,im) in enumerate(entries):
        x=(i%2)*640;y=(i//2)*440
        draw.text((x+12,y+8),label,font=font,fill='white')
        shown=ImageOps.contain(im,(620,390));board.paste(shown,(x+(640-shown.width)//2,y+42+(390-shown.height)//2))
    board.save(destination)


def preview(image, proposal, ranking, destination):
    with Image.open(image) as im:im=im.convert('RGB')
    draw=ImageDraw.Draw(im)
    if proposal['decision']=='ACT':
        row=next(r for r in ranking if r['edge_id']==proposal['edge_id'])
        draw.rectangle(row['query_box_xyxy'],outline='yellow',width=3)
        for key,color in [('grasp_px','red'),('target_px','cyan')]:
            x,y=proposal[key];draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=4)
        draw.line([tuple(proposal['grasp_px']),tuple(proposal['target_px'])],fill='yellow',width=2)
    im.save(destination)


def retrieve(library, image, folder, *, top_k=6, python=None, exclude_edges=()):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    command=[worker_python(python),'-m','cloth_agent.experiments.local_action_memory','search',
             '--library',str(Path(library).resolve()),'--image',str(Path(image).resolve()),
             '--output',str((folder/'retrieval.json').resolve()),'--top-k',str(top_k),
             '--exclude-edges',','.join(exclude_edges)]
    started=time.perf_counter()
    with (folder/'retrieval.stdout.log').open('w') as out,(folder/'retrieval.stderr.log').open('w') as err:
        subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=True,timeout=180.)
    result=json.loads((folder/'retrieval.json').read_text())
    result['timing']['subprocess_wall_s']=time.perf_counter()-started
    save(folder/'retrieval.json',result)
    return result


def decide(library, image, retrieval, requests, folder, *, previous=None, history=()):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);lib=load_library(library)
    rank=retrieval['ranking'];images=[('current',Path(image))];candidates=[]
    for row in rank:
        action=lib['actions'][row['edge_id']];path=folder/(row['edge_id']+'_case.png')
        candidate_board(image,action,row,path);images.append((row['edge_id'],path))
        candidates.append(dict(edge_id=row['edge_id'],query_box_xyxy=row['query_box_xyxy'],
            similarity=row['similarity'],action_family=action['action_family'],
            action_template=action['action_template'],observed_outcome=action['observed_outcome'],
            source_evidence_limitations=action['source_evidence_limitations'],
            reference_start_px=action['reference_start_px'],reference_target_px=action['reference_target_px']))
    if previous:images.append(('previous_before',Path(previous['before'])))
    with Image.open(image) as im:size=im.size
    payload=dict(goal='unfold',current_image_size=list(size),candidates=candidates,
        previous_action=previous['proposal'] if previous else None,recent_history=list(history)[-3:])
    started=time.perf_counter()
    value=requests.ask('local_action',payload,images,validator=lambda v:validate_proposal(v,rank,size))
    save(folder/'proposal.json',value)
    save(folder/'decision_timing.json',dict(gpt_action_selection_and_grounding_s=time.perf_counter()-started))
    preview(image,value,rank,folder/'proposal_preview.png')
    return value
