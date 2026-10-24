"""Offline cloth boundary/layer relation matching; no DINO or robot connection.

Build: extract a visible local relation graph around each demonstrated grasp.
Query: parse current visible structures independently, then match typed subgraphs.
All graph facts are model hypotheses with pixel evidence, not measured topology.
"""
import argparse
import json
import math
import time
from pathlib import Path

import jsonschema
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .discovery_data import save, file_hash
from .discovery_schema import obj, S, N, array, enum
from .graph_policy import Requests, checked_file, evidence_file
from .local_action_memory import load_library

PROTOCOL='arm6-local-visible-structure-v1'
POINT={'type':'array','items':{'type':'number','minimum':0},'minItems':2,'maxItems':2}
ENTITY=obj(id=S,kind=enum('surface','free_edge','fold_line','occlusion_edge','opening_rim','table_region'),
           points=array(POINT),evidence=S,confidence=N)
RELATION=obj(source=S,target=S,kind=enum('bounds','folds_at','covers','border_touches'),
             evidence=S,confidence=N)
GRAPH=obj(entities=array(ENTITY),relations=array(RELATION),uncertainties=array(S))
DEMO=obj(before_structure=GRAPH,grasp_entity_id=S,moving_surface_id=S,
    operation=enum('open_fold','uncover_layer','spread_gather','reposition','unclear'),
    effect=enum('coverage_reduced','boundary_exposed','no_clear_change','worse','unclear'),
    effect_evidence=array(S),action_explanation=S,limitations=array(S))
SCENE=obj(structure=GRAPH,completion=enum('unfolded','not_unfolded','uncertain'),completion_evidence=array(S))
GROUND=obj(decision=enum('PROPOSE','STOP'),candidate_id=S,
    grasp_px={'anyOf':[POINT,{'type':'null'}]},target_px={'anyOf':[POINT,{'type':'null'}]},
    confidence=N,correspondence_evidence=array(S),expected_change=S,uncertainties=array(S),reason=S)

VOCAB='''可见结构图的词表：
entity.kind: surface=一片连续可见布面区域（polygon至少3点）；free_edge=能辨认真实自由布边（polyline至少2点）；
fold_line=可见回折连接线，不能把普通皱纹阴影标成折线；occlusion_edge=上层遮住下层的可见边界；
opening_rim=可见袖口/领口等开口边缘；table_region=明确露出的桌面区域（polygon至少3点）。
边界points为原图像素折线，区域points为原图像素多边形。尽量标整条相关边界而非随意框。
relations有方向且只有四种：
bounds: boundary -> surface，该边界限定这片布面的可见边缘；
folds_at: surface -> fold_line，该片布面在该线处折回（不是仅邻近）；
covers: upper_surface -> lower_surface，两片可见布面的T形终止/遮挡边界支持上覆关系；
border_touches: boundary -> surface/table_region，边界另一侧邻接的可见布面或桌面。
每条关系写实际可见依据，不要从任务、动作或常识倒推。无法确认的隐藏连通性/层数写uncertainties，不能补成事实。
不要把缝线或印花边界当自由边；不要用左右袖/领口等名称代替可见关系。
不能因某片区域很黑就认定桌面，需结合实际表面。confidence是你对此可见标注的判断，不是校准概率。'''

DEMO_PROMPT='''从 BEFORE 中提取与已记录抓取点相关的最小局部结构图，再用 AFTER 解释动作效果。
只围绕实际被抓边界、所连布面及直接相关的折线/被覆盖布面/外侧桌面建图。
before_structure必须来自BEFORE图，AFTER只用于effect，不能用后图推断原来看不到的层。
grasp_entity_id必须是真实抓取点附近的可见边界；moving_surface_id是该边界所限定的可见布面。
若看不清抓取依赖的边界，可把两个ID留空，说明限制，不强行建可复用模板。
范围控制在2至6个实体，1至6个关系。不要收录与这个动作无关的领口或整件衣物轮廓。
结合历史start/target和前后图描述该操作是在打开折层、移开遮挡还是简单重新摆放。
向内移可以为了露出下面布片；不能按移动向内/向外直接判定目的。动作执行不等于达到效果。
effect_evidence只写前后可见变化；证据不足选unclear。不能把示范被保存当作成功证明。
'''+VOCAB

SCENE_PROMPT='''独立分析CURRENT中可见的局部布层结构，不规划动作，不猜历史示范，不输出graph节点类别。
标出适合检查是否可以展开的布边/翻折区域及彼此可见关系。提取最多18个实体、最多24条关系。
优先保留构成完整局部关系的边界和相连布面；共享的同一布面使用同一ID，分离的边界不要合成一个实体。
不是画固定窗口，不是比较颜色/纹理；实体必须对应具体的自由边、折回线、遮挡边界或布面。
边界交叠难辨时明确不确定，不能仅因表面形似某个字母而发明连接关系。
衣身主体、袖部和下摆均外露且没有需要打开的大片回折时可判unfolded，允许普通皱纹。
completion和局部结构分开记录。只输出原图坐标和可见依据。
'''+VOCAB

GROUND_PROMPT='''目的：展开衣物。已通过可见结构子图匹配得到候选。禁止重新做全图状态分类或按图片外观另选动作。
每个candidate包含示范结构、当前结构的实体对应关系和该示范的真实前后图。先核对这些标注是否真的受图像支持。
拓扑吻合不是适用性保证：检查自由边是否真实、哪片布覆盖哪里、折层方向和当前释放空间。
只能选给定candidate_id，抓点必须靠近它映射的grasp实体且落在相应布面上。
参考示范动作相对于自由边/折线/上覆布面的作用，把它迁移到当前这套结构；不要复制像素差，不能把外翻变成内折。
输出CURRENT原图抓点和目标点，并逐项说明哪些实体/关系对应、为何动作会有助展开。
如果结构标注错误、关系关键处不确定、或示范效果不足以支持此次操作，就STOP并说明。
不需要整件衣服或衣服部位相同。不要为了延续动作而猜隐藏层。PROPOSE仅是待审核建议，不能证明抓取成功。
STOP时candidate_id为空，点位为null。'''


def requests_at(output,model,timeout,max_calls):
    from .discovery_model import CodexMultimodal
    return Requests(Path(output)/'model',CodexMultimodal(model,timeout),max_calls=max_calls,no_cache=True,
        schemas={'demo_structure':DEMO,'scene_structure':SCENE,'structure_transfer':GROUND},
        prompts={'demo_structure':DEMO_PROMPT,'scene_structure':SCENE_PROMPT,'structure_transfer':GROUND_PROMPT},
        common='Visible cloth structure experiment. Return schema JSON. Treat all inferred boundaries and relations as inspectable hypotheses.',
        protocol=PROTOCOL)


def validate_graph(graph,size,max_entities=18):
    jsonschema.validate(graph,GRAPH)
    entities=graph['entities'];ids=[e['id'] for e in entities]
    if len(set(ids))!=len(ids) or any(not i for i in ids) or len(ids)>max_entities:
        raise ValueError('invalid or duplicate structure entity IDs')
    lookup={e['id']:e for e in entities};seen=set()
    for e in entities:
        minimum=3 if e['kind'] in ('surface','table_region') else 2
        if len(e['points'])<minimum or not e['evidence'].strip():raise ValueError('entity needs geometry and visible evidence')
        xy=np.asarray(e['points'],float)
        if not np.isfinite(xy).all() or np.any(xy<0) or np.any(xy[:,0]>=size[0]) or np.any(xy[:,1]>=size[1]):
            raise ValueError('structure coordinates outside original image')
        if len(np.unique(xy,axis=0))<minimum:raise ValueError('degenerate structure geometry')
    for r in graph['relations']:
        s,t,k=r['source'],r['target'],r['kind']
        if s not in lookup or t not in lookup or s==t or (s,t,k) in seen:raise ValueError('invalid/duplicate relation')
        seen.add((s,t,k))
        a,b=lookup[s]['kind'],lookup[t]['kind']
        boundary=('free_edge','fold_line','occlusion_edge','opening_rim')
        valid={'bounds':a in boundary and b=='surface',
               'folds_at':a=='surface' and b=='fold_line',
               'covers':a==b=='surface',
               'border_touches':a in boundary and b in ('surface','table_region')}[k]
        if not valid or not r['evidence'].strip():raise ValueError('relation violates typed vocabulary')


def point_edge_distance(point,points):
    p=np.asarray(point,float);xy=np.asarray(points,float);a=xy[:-1];v=xy[1:]-a
    length=(v*v).sum(axis=1)
    u=np.clip(((p-a)*v).sum(axis=1)/np.maximum(length,1e-12),0,1)
    return float(np.linalg.norm(p-(a+u[:,None]*v),axis=1).min())


def eligible_pattern(value,grasp,size,min_confidence=.65):
    graph=value['before_structure'];lookup={e['id']:e for e in graph['entities']}
    g=value['grasp_entity_id'];m=value['moving_surface_id']
    reasons=[]
    if g not in lookup or m not in lookup:return ['missing visible grasp boundary or moving surface']
    if lookup[g]['kind'] not in ('free_edge','opening_rim','occlusion_edge'):reasons.append('grasp is not an exposed boundary')
    if lookup[m]['kind']!='surface':reasons.append('moving entity is not a surface')
    if point_edge_distance(grasp,lookup[g]['points'])>.05*min(size):reasons.append('boundary does not explain recorded grasp')
    triples={(r['source'],r['target'],r['kind']) for r in graph['relations']}
    if (g,m,'bounds') not in triples:reasons.append('missing grasp-boundary to moving-surface relation')
    if len(graph['relations'])<2:reasons.append('edge plus surface alone does not specify a fold/coverage context')
    if not any(r['kind'] in ('covers','folds_at') for r in graph['relations']):
        reasons.append('no visible fold or layer-coverage relation; plain edge similarity insufficient')
    # A motif must be connected to the actual grasp, not an unrelated fold elsewhere.
    reachable={g}
    while True:
        expanded=reachable|{r['target'] for r in graph['relations'] if r['source'] in reachable}|{r['source'] for r in graph['relations'] if r['target'] in reachable}
        if expanded==reachable:break
        reachable=expanded
    if reachable!=set(lookup):reasons.append('disconnected context entities')
    if any(x['confidence']<min_confidence for x in graph['entities']+graph['relations']):
        reasons.append('required structure evidence below pilot confidence threshold')
    if value['operation'] in ('reposition','unclear'):reasons.append('no demonstrated unfolding operation established')
    if value['effect'] not in ('coverage_reduced','boundary_exposed') or not value['effect_evidence']:
        reasons.append('demonstrated local improvement unresolved')
    return reasons


def match_subgraphs(pattern,scene,min_confidence=.65,max_matches=40):
    """Injective typed relation matching. No RGB, text embeddings, or node-state IDs.

    Additional CURRENT relations among mapped entities are rejected to avoid
    ignoring contradictory coverage. Relations to outside context are retained
    for subsequent visual review, not assumed absent in the demonstration.
    """
    ps={e['id']:e for e in pattern['entities']}
    qs={e['id']:e for e in scene['entities'] if e['confidence']>=min_confidence}
    if any(e['confidence']<min_confidence for e in ps.values()) or any(r['confidence']<min_confidence for r in pattern['relations']):return []
    pr={(r['source'],r['target'],r['kind']) for r in pattern['relations']}
    qr={(r['source'],r['target'],r['kind']):r for r in scene['relations'] if r['confidence']>=min_confidence}
    pools={p:[q for q in qs if qs[q]['kind']==e['kind']] for p,e in ps.items()}
    order=sorted(ps,key=lambda p:(len(pools[p]),-sum(p in r[:2] for r in pr)))
    results=[]
    def search(mapping):
        if len(results)>=max_matches:return
        if len(mapping)==len(order):
            mapped={(mapping[s],mapping[t],k) for s,t,k in pr};members=set(mapping.values())
            actual={r for r in qr if r[0] in members and r[1] in members}
            if actual!=mapped:return
            confidence=min([qs[q]['confidence'] for q in members]+[qr[r]['confidence'] for r in mapped])
            results.append(dict(mapping=dict(mapping),evidence_confidence=confidence,
                matched_relations=[dict(reference=[s,t,k],current=[mapping[s],mapping[t],k]) for s,t,k in sorted(pr)]));return
        p=order[len(mapping)]
        for q in pools[p]:
            if q in mapping.values():continue
            mapping[p]=q
            if all(s not in mapping or t not in mapping or (mapping[s],mapping[t],k) in qr for s,t,k in pr):search(mapping)
            del mapping[p]
    search({})
    return results


def render_structure(image,graph,destination,title,highlight=()):
    with Image.open(image) as im:canvas=im.convert('RGB')
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    palette={'surface':'#f4cf69','free_edge':'#38ed8a','fold_line':'#ff843f',
             'occlusion_edge':'#f064ff','opening_rim':'#56d9ff','table_region':'#b7c6d4'}
    for entity in graph['entities']:
        pts=[tuple(p) for p in entity['points']];color=palette[entity['kind']]
        line=pts+[pts[0]] if entity['kind'] in ('surface','table_region') else pts
        draw.line(line,fill=color,width=5 if entity['id'] in highlight else 3)
        draw.text(pts[0],entity['id']+': '+entity['kind'],fill=color,font=font,stroke_width=2,stroke_fill='black')
    label=Image.new('RGB',(canvas.width,110),'#101827');d=ImageDraw.Draw(label)
    d.text((12,8),title,font=font,fill='white')
    d.text((12,38),'Green: free edge | Orange: fold | Purple: occlusion | Yellow: visible surface',font=font,fill='white')
    d.text((12,68),'Model-extracted visible hypotheses; inspect boundaries and layer relations.',font=font,fill='#a9bdd9')
    board=Image.new('RGB',(canvas.width,canvas.height+110));board.paste(label);board.paste(canvas,(0,110));board.save(destination)


def build(source,edges,output,model='gpt-6-astra',timeout=180.):
    lib=load_library(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ids=edges.split(',')
    if len(set(ids))!=len(ids) or any(e not in lib['actions'] for e in ids):raise ValueError('unknown or duplicate pilot action')
    requests=requests_at(output,model,timeout,2*len(ids));records={}
    for eid in ids:
        action=lib['actions'][eid];folder=output/eid;folder.mkdir()
        before=checked_file(action['before']);after=checked_file(action['after'])
        with Image.open(before) as im:size=im.size
        def validate(value):
            validate_graph(value['before_structure'],size,6)
        value=requests.ask('demo_structure',dict(reference_start_px=action['reference_start_px'],
            reference_target_px=action['reference_target_px'],image_size=list(size),
            evidence_limitations=action['source_evidence_limitations']),[('BEFORE',before),('AFTER',after)],validator=validate)
        reasons=eligible_pattern(value,action['reference_start_px'],size)
        record=dict(edge_id=eid,action=action,structure=value,eligible=not reasons,rejection_reasons=reasons)
        records[eid]=record;save(folder/'pattern.json',record)
        render_structure(before,value['before_structure'],folder/'structure.png',eid+' | demonstrated local structure',
                         [value['grasp_entity_id'],value['moving_surface_id']])
        save(output/'structure_library.json',dict(protocol=PROTOCOL,goal='unfold',
            source_library=evidence_file(Path(source)/'library.json'),records=records,planned_edges=ids,
            complete=len(records)==len(ids),min_evidence_confidence=.65,model_metrics=requests.metrics(),robot_calls=0))
        print(json.dumps(dict(edge_id=eid,eligible=not reasons,rejection_reasons=reasons),ensure_ascii=False),flush=True)
    return records


def load_patterns(path):
    lib=json.loads(Path(path).read_text())
    if lib['protocol']!=PROTOCOL or not lib['complete']:raise ValueError('incomplete structure library')
    checked_file(lib['source_library'])
    for record in lib['records'].values():
        checked_file(record['action']['before']);checked_file(record['action']['after'])
    return lib


def comparison_image(reference,current,pattern,scene,candidate,folder):
    """Expose correspondence before action transfer; no resemblance score hidden in UI."""
    from PIL import ImageOps
    folder=Path(folder);folder.mkdir(exist_ok=True)
    a=folder/'reference_structure.png';b=folder/'current_structure.png'
    render_structure(reference,pattern,a,'REFERENCE '+candidate['edge_id'],candidate['mapping'].keys())
    render_structure(current,scene,b,'CURRENT: matched entities',candidate['mapping'].values())
    board=Image.new('RGB',(1600,780),'#101827');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    for x,path in [(0,a),(800,b)]:
        with Image.open(path) as im:shown=ImageOps.contain(im,(790,570))
        board.paste(shown,(x+(800-shown.width)//2,5))
    mapping=' | '.join(f'{s} -> {t}' for s,t in candidate['mapping'].items())
    draw.text((20,580),candidate['candidate_id']+' / '+candidate['edge_id']+'   '+mapping,font=font,fill='white')
    for i,r in enumerate(candidate['matched_relations'][:6]):
        s,t,k=r['reference'];a,b,_=r['current']
        draw.text((20,610+i*25),f'{s} --{k}--> {t}     matches     {a} --{k}--> {b}',font=font,fill='#8de7b6')
    board.save(folder/'comparison.png')


def validate_ground(value,candidates,scene,size):
    if value['decision']=='STOP':
        if value['candidate_id'] or value['grasp_px'] is not None or value['target_px'] is not None:raise ValueError('STOP must have no action')
        return
    matches=[c for c in candidates if c['candidate_id']==value['candidate_id']]
    if len(matches)!=1:raise ValueError('not a matched structure candidate')
    if not value['correspondence_evidence'] or not value['expected_change'].strip():raise ValueError('missing transfer evidence')
    for key in ('grasp_px','target_px'):
        xy=value[key]
        if xy is None or not all(math.isfinite(v) for v in xy) or not (0<=xy[0]<size[0] and 0<=xy[1]<size[1]):raise ValueError('invalid original-image anchors')
    boundary=next(e for e in scene['entities'] if e['id']==matches[0]['current_grasp_entity'])
    if point_edge_distance(value['grasp_px'],boundary['points'])>.04*min(size):raise ValueError('grasp does not belong to mapped boundary')
    if np.linalg.norm(np.asarray(value['grasp_px'])-value['target_px'])<5:raise ValueError('degenerate action')


def inspect_scene(image,output,model='gpt-6-astra',timeout=180.):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    requests=requests_at(output,model,timeout,2)
    with Image.open(image) as im:size=im.size
    scene=requests.ask('scene_structure',dict(image_size=list(size)),[('CURRENT',Path(image))],
                       validator=lambda v:validate_graph(v['structure'],size))
    save(output/'scene.json',scene)
    save(output/'scene_source.json',dict(protocol=PROTOCOL,image=evidence_file(image),
         scene=evidence_file(output/'scene.json'),model_metrics=requests.metrics(),robot_calls=0))
    render_structure(image,scene['structure'],output/'current_structure.png','CURRENT | independently parsed structure')
    return scene


def query(library,image,output,model='gpt-6-astra',timeout=180.,transfer=True,scene_dir=None):
    lib=load_patterns(library);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    requests=requests_at(output,model,timeout,4);started=time.perf_counter()
    with Image.open(image) as im:size=im.size
    if scene_dir:
        source=json.loads((Path(scene_dir)/'scene_source.json').read_text())
        if source['protocol']!=PROTOCOL or source['image']['sha256']!=file_hash(image):raise ValueError('scene belongs to different query image/protocol')
        scene=json.loads(checked_file(source['scene']).read_text())
        jsonschema.validate(scene,SCENE);validate_graph(scene['structure'],size)
        save(output/'scene_source.json',source)
    else:
        scene=requests.ask('scene_structure',dict(image_size=list(size)),[('CURRENT',Path(image))],
                           validator=lambda v:validate_graph(v['structure'],size))
    save(output/'scene.json',scene)
    render_structure(image,scene['structure'],output/'current_structure.png','CURRENT | independently parsed structure')
    candidates=[];attempts=[];match_started=time.perf_counter()
    for eid,record in lib['records'].items():
        if not record['eligible']:
            attempts.append(dict(edge_id=eid,status='REFERENCE_INELIGIBLE',reasons=record['rejection_reasons']));continue
        pattern=record['structure'];matches=match_subgraphs(pattern['before_structure'],scene['structure'],lib['min_evidence_confidence'])
        attempts.append(dict(edge_id=eid,status='STRUCTURE_MATCH' if matches else 'NO_STRUCTURAL_CORRESPONDENCE',count=len(matches)))
        for match in matches:
            candidates.append(dict(candidate_id=f'c{len(candidates):03d}',edge_id=eid,**match,
                current_grasp_entity=match['mapping'][pattern['grasp_entity_id']],
                current_moving_surface=match['mapping'][pattern['moving_surface_id']]))
    match_s=time.perf_counter()-match_started
    candidates.sort(key=lambda c:(-len(c['matched_relations']),-c['evidence_confidence'],c['candidate_id']))
    for candidate in candidates[:4]:
        record=lib['records'][candidate['edge_id']]
        comparison_image(checked_file(record['action']['before']),image,record['structure']['before_structure'],
                         scene['structure'],candidate,output/candidate['candidate_id'])
    save(output/'structural_matches.json',dict(candidates=candidates,attempts=attempts,match_s=match_s,
        method='injective typed induced-subgraph correspondence',appearance_similarity_used=False,
        caveat='Confidence reflects extracted evidence, not calibrated action success. Additional context still needs visual review.'))
    result=dict(status='STRUCTURE_MATCHES' if candidates else 'NO_STRUCTURAL_MATCH',robot_calls=0,
        graph_state_matching=False,appearance_retrieval=False,candidate_count=len(candidates),
        query=evidence_file(image),library=evidence_file(library),completion=scene['completion'],scene_precomputed=bool(scene_dir))
    if scene['completion']=='unfolded':result['status']='MODEL_REPORTED_UNFOLDED'
    elif candidates and transfer:
        top=candidates[:4];images=[('CURRENT',Path(image))];payload=[]
        for c in top:
            r=lib['records'][c['edge_id']];action=r['action']
            images.extend([(c['candidate_id']+'_BEFORE',checked_file(action['before'])),(c['candidate_id']+'_AFTER',checked_file(action['after']))])
            payload.append(dict(**c,reference_structure=r['structure'],reference_start_px=action['reference_start_px'],
                reference_target_px=action['reference_target_px'],source_evidence_limitations=action['source_evidence_limitations']))
        value=requests.ask('structure_transfer',dict(candidates=payload,current_structure=scene['structure'],image_size=list(size)),images,
                           validator=lambda v:validate_ground(v,top,scene['structure'],size))
        result['proposal']=value;result['status']='OFFLINE_STRUCTURE_PROPOSAL' if value['decision']=='PROPOSE' else 'STRUCTURE_TRANSFER_REJECTED'
        save(output/'proposal.json',value)
        if value['decision']=='PROPOSE':
            c=next(c for c in candidates if c['candidate_id']==value['candidate_id']);r=lib['records'][c['edge_id']]
            render_structure(image,scene['structure'],output/'matched_current.png','Matched CURRENT: '+c['edge_id'],c['mapping'].values())
            render_structure(checked_file(r['action']['before']),r['structure']['before_structure'],output/'matched_reference.png',
                             'Matched REFERENCE: '+c['edge_id'],c['mapping'].keys())
            with Image.open(image) as im:im=im.convert('RGB')
            d=ImageDraw.Draw(im);d.line([tuple(value['grasp_px']),tuple(value['target_px'])],fill='yellow',width=3)
            for key,color in [('grasp_px','red'),('target_px','cyan')]:
                x,y=value[key];d.ellipse((x-8,y-8,x+8,y+8),outline=color,width=4)
            im.save(output/'proposal_preview.png')
    result.update(model_metrics=requests.metrics(),total_s=time.perf_counter()-started,structure_match_s=match_s)
    save(output/'result.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('build');b.add_argument('--source-library',type=Path,required=True)
    b.add_argument('--edges',default='e085,e103,e104');b.add_argument('--output',type=Path,required=True)
    q=s.add_parser('query');q.add_argument('--library',type=Path,required=True);q.add_argument('--image',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--no-transfer',action='store_true')
    q.add_argument('--scene-dir',type=Path,help='reuse verified independent parse of exactly this image')
    i=s.add_parser('inspect');i.add_argument('--image',type=Path,required=True);i.add_argument('--output',type=Path,required=True)
    for x in (b,q,i):x.add_argument('--model',default='gpt-6-astra');x.add_argument('--timeout-s',type=float,default=180.)
    a=p.parse_args(argv)
    if a.command=='build':
        records=build(a.source_library,a.edges,a.output,a.model,a.timeout_s)
        result=dict(patterns=len(records),eligible=sum(r['eligible'] for r in records.values()),robot_calls=0)
    elif a.command=='inspect':result=inspect_scene(a.image,a.output,a.model,a.timeout_s)
    else:result=query(a.library,a.image,a.output,a.model,a.timeout_s,not a.no_transfer,a.scene_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
