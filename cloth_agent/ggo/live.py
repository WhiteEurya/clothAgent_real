"""GGO-assisted contextual planner, with automatic or manual landmarks.

All motion stays in the existing episode Host. This module only reviews a frozen
observation, computes geometry, and makes one action-model request.
"""
import json
from pathlib import Path
import secrets
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from PIL import Image
from cloth_agent import restricted_context_one_shot as original, geometry_report, hem_sleeve_pull_rule
from cloth_agent import preference_one_shot as shared
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.dual_arm.observation import Observation
from . import cli
from .observer import observe,Settings,LANDMARKS
from .assessment import write_outputs

SCHEMA=original.SCHEMA
REPORTS_PREVIOUS_EFFECT=False
PROMPT='\n'.join(line for line in original.PROMPT.splitlines() if not line.startswith(('几何报告仅作辅助','结构距离报告')))+'''
GGO Geometry Observation 是用上半身锚点配准后计算的区域证据，不是真值；自动识别的锚点可能有误，配准通过也不保证部位识别正确。
优先减轻主要衣身结构偏差，不为了普通小皱纹反复拉动。几何报告没有确定真正下摆角；UNVERIFIED不能当作已确认材料角点。
UNKNOWN 表示不可测，不是已展开。结合干净原图选可见衣袖或下摆；无可靠动作则ABSTAIN。保持原有JSON格式，只输出一个Pa→Pb和简短理由。
'''
validate=original.validate


def automatic_annotations(root,folder,before,reference,seed,headers,timeout_s,direct):
    folder.mkdir(parents=True,exist_ok=False)
    point={'anyOf':[{'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2},{'type':'null'}]}
    schema={'type':'object','properties':{k:point for k in ('N','AL','AR')},
            'required':['N','AL','AR'],'additionalProperties':False}
    prompt=('只识别第一张当前图的三个上半身锚点，不规划动作。N=领口中心，AL/AR=两侧袖子与衣身连接的腋下凹角。'
            '第二张是同一衣服展平参考图，给定参考坐标用于明确各标签对应的材料部位；保持标签对应，不能简单按当前图左右命名。'
            '输出当前原图整数像素坐标，遮挡、不可见或证据不足必须输出null，不得猜测。参考坐标：'
            +json.dumps({k:seed['reference'][k] for k in ('N','AL','AR')}))
    body=shared.make_payload(root,folder,before,prompt,schema,assess_previous_effect=False)
    body['text']['format']['name']='ggo_upper_anchors'
    io.save(folder/'request.json',body)
    output,metrics=direct(folder,body,headers,timeout_s)
    io.save(folder/'response.json',output);io.save(folder/'model_metrics.json',metrics)
    io.jsonschema.validate(output,schema)
    data=dict(reference=seed['reference'],current=output,reference_rgb_sha256=io.sha(reference.image),
              current_rgb_sha256=io.sha(before.image),provenance='automatic model current anchors; fixed reference seed; not human verified')
    if seed.get('reference_rgb_sha256')!=data['reference_rgb_sha256']:
        raise ValueError('reference seed image hash mismatch')
    validate_annotations(data,data['reference_rgb_sha256'],data['current_rgb_sha256'],Image.open(reference.image).size,Image.open(before.image).size)
    io.save(folder/'annotations.json',data)
    return data,metrics


def validate_annotations(data,reference_hash,current_hash,reference_size,current_size):
    if data.get('reference_rgb_sha256')!=reference_hash or data.get('current_rgb_sha256')!=current_hash:
        raise ValueError('图像已变化，标注不能跨图复用')
    for key,names,size in [('reference',LANDMARKS,reference_size),('current',('N','AL','AR'),current_size)]:
        if set(data.get(key,{}))!=set(names):raise ValueError('锚点字段不完整')
        for p in data[key].values():
            if p is None:
                if key=='reference':raise ValueError('参考图需要完整的7个标注点')
                continue
            if not isinstance(p,list) or len(p)!=2 or any(type(v)!=int for v in p) or not(0<=p[0]<size[0] and 0<=p[1]<size[1]):
                raise ValueError('标注点超出图像或不是整数像素')
    return data


def review(before,reference,seed,folder,port,timeout=1800):
    cli.annotate(reference.directory,before.directory,folder)
    template=io.read(folder/'annotation_template.json');template['reference']=seed['reference']
    template['provenance']='operator confirmed live upper-body and reference landmarks'
    token=secrets.token_urlsafe(24);event=threading.Event();result={}
    page=(folder/'index.html').read_text()
    start=page.index('const data=')+len('const data=');end=page.index(';const canvas=',start)
    page=page[:start]+json.dumps(template)+page[end:]
    page=page.replace('下载 annotations.json','确认锚点，计算几何并执行 GPT 选择的一个动作')
    start=page.index("document.getElementById('download').onclick=");end=page.index(';load();</script>',start)
    handler="""document.getElementById('download').onclick=async()=>{if(!confirm('确认这些标注对应当前画面？提交后将规划并执行一个动作。'))return;let res=await fetch('/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:__TOKEN__,annotations:data})});let text=await res.text();document.getElementById('status').textContent=text;if(res.ok){document.getElementById('download').disabled=true;}}""".replace('__TOKEN__',json.dumps(token))
    page=page[:start]+handler+page[end:]
    page=page.replace('<h1>','<p style="color:#b21">机器人正停在观测位。标注期间不要移动衣物。识别不清的当前锚点设为未知。停止本次请使用批次控制页。</p><h1>',1)
    (folder/'index.html').write_text(page)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,body,kind='text/plain; charset=utf-8'):
            b=body if isinstance(body,bytes) else body.encode();self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)
        def do_GET(self):
            route=self.path.split('?')[0]
            if route in ('/','/index.html'):return self.send(200,page,'text/html; charset=utf-8')
            if route in ('/reference.png','/current.png'):return self.send(200,(folder/route[1:]).read_bytes(),'image/png')
            self.send(404,'not found')
        def do_POST(self):
            if self.path!='/submit' or self.headers.get('Origin')!=f'http://127.0.0.1:{port}':return self.send(403,'origin rejected')
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<20000:raise ValueError('invalid size')
                body=json.loads(self.rfile.read(n))
                if body.get('token')!=token or event.is_set():raise ValueError('标注已提交或页面已过期')
                data=validate_annotations(body['annotations'],template['reference_rgb_sha256'],template['current_rgb_sha256'],Image.open(reference.image).size,Image.open(before.image).size)
                data['provenance']=template['provenance'];result['data']=data;event.set();self.send(200,'已接收。正在计算几何并规划；请勿移动衣服。')
            except (ValueError,KeyError,TypeError) as e:self.send(400,str(e))
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    print(f'GGO WAITING_ANCHORS http://127.0.0.1:{port} current={before.image}',flush=True)
    try:
        if not event.wait(timeout):raise TimeoutError('GGO manual annotation timeout; no action planned')
        io.save(folder/'confirmed_annotations.json',result['data']);return result['data']
    finally:server.shutdown();server.server_close()


def short_observation(report):
    # Compact numeric evidence, with no presumed right-side target on a new scene.
    assessment=__import__('cloth_agent.ggo.assessment',fromlist=['assessment']).assessment(report)
    value=dict(reference='Same garment spread reference, aligned using upper-body landmarks',
               annotation_provenance=report.get('annotation_provenance'),
               registration=assessment['registration'],lower_body={},asymmetry=assessment['asymmetry'],
               interpretation=assessment['interpretation'],semantic_corners=assessment['semantic_corners'],
               limitations=assessment['limitations'])
    value['registration_reason']=report.get('registration',{}).get('reason')
    value['anchor_sampling']=report.get('anchor_diagnostics',{}).get('current',{})
    for side,v in assessment['lower_body'].items():
        value['lower_body'][side]={k:v[k] for k in ('projected_deficit','missing_ratio','missing_area_mm2','boundary_profile')}
    value['section_curve']=assessment['section_curve']
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def plan_round(root,folder,before,headers,timeout_s,direct,previous_context=None,**kwargs):
    config=io.read(root/'ggo_live_config.json');ref=Observation.load(config['reference'])
    if ref.meta['calibration_id']!=before.meta['calibration_id']:raise ValueError('GGO reference/current calibration mismatch')
    if io.sha(ref.image)!=io.sha(root/'goal/target.png'):raise ValueError('GGO reference/goal mismatch')
    began=time.perf_counter();seed=io.read(root/'ggo_reference_seed.json')
    automatic=config.get('automatic_anchors',False)
    anchor_metrics={}
    if automatic:
        print('GGO AUTO_ANCHORS: recognizing current upper-body landmarks',flush=True)
        annotations,anchor_metrics=automatic_annotations(root,folder/'ggo_annotation',before,ref,seed,headers,timeout_s,direct)
    else:
        annotations=review(before,ref,seed,folder/'ggo_annotation',config['port'])
    anchor_s=time.perf_counter()-began
    wait_s=0. if automatic else anchor_s
    started=time.perf_counter()
    report,art=observe(np.asarray(Image.open(ref.image).convert('RGB')),ref.xyz,np.asarray(Image.open(before.image).convert('RGB')),before.xyz,annotations)
    report['sources']=dict(reference=str(ref.directory),current=str(before.directory),reference_sha256=ref.meta['sha256'],current_sha256=before.meta['sha256'])
    report['compute_wall_s']=time.perf_counter()-started
    geometry_dir=folder/'ggo';geometry_dir.mkdir();cli.save(geometry_dir/'report.json',report);cli.save(geometry_dir/'annotations.json',annotations)
    np.savez_compressed(geometry_dir/'geometry.npz',**art);write_outputs(geometry_dir,report);cli.render(geometry_dir,report,art,before,ref,annotations)
    body=shared.make_payload(root,folder,before,hem_sleeve_pull_rule.apply(root,PROMPT),SCHEMA,previous_context=previous_context,assess_previous_effect=False)
    if kwargs.get('view_change'):
        body['input'][0]['content'].append(dict(type='input_text',text='抓点深度不可用后已移动相机；没有执行衣物动作。按当前新图重新选择Pa/Pb，不复用旧像素。'))
    if kwargs.get('rejection_feedback'):
        body['input'][0]['content'].append(dict(type='input_text',text='上一候选执行前被Host拒绝，未执行。根据错误重新规划，不重复被拒绝坐标：'+json.dumps(kwargs['rejection_feedback'],ensure_ascii=False)))
    evidence=short_observation(report);(folder/'geometry_observation.txt').write_text(evidence)
    body['input'][0]['content'].append(dict(type='input_text',text='Geometry Observation:\n'+evidence))
    body['text']['format']['name']='ggo_live_single_action'
    io.save(folder/'request_manifest.json',dict(model=io.MODEL,effort=io.EFFORT,model_calls=2 if automatic else 1,geometry_model_calls=int(automatic),
        automatic_anchors=automatic,anchor_wall_s=anchor_s if automatic else 0.,
        current_sha256=io.sha(before.image),annotation_wait_s=wait_s,geometry_wall_s=report['compute_wall_s'],
        ggo_report='ggo/report.json',input_image_count=3 if previous_context else 2,overlays_input=False))
    output,metrics=direct(folder,body,headers,timeout_s)
    metrics.update(annotation_wait_s=wait_s,anchor_wall_s=anchor_s if automatic else 0.,geometry_wall_s=report['compute_wall_s'])
    metrics['action_model_wall_s']=metrics.get('wall_latency_s',0.)
    metrics['wall_latency_s']=metrics.get('wall_latency_s',0.)+anchor_metrics.get('wall_latency_s',0.)
    io.save(folder/'model_metrics.json',metrics);io.save(folder/'decision_evidence.json',output)
    plan=validate(output,Image.open(before.image).size,bool(previous_context))
    for name in ('plan.json','fast_plan.json'):io.save(folder/name,plan)
    io.overlay(before.image,plan,folder/'proposed_action_overlay.png','GGO ONE ACTION')
    shutil.copyfile(folder/'proposed_action_overlay.png',folder/'fast_action_overlay.png');render_round(folder)
    return plan,metrics


def replan_rejected(root,folder,before,headers,timeout_s,direct,previous_context,feedback):
    attempt=folder/'target_replan'/'retry_01'
    attempt.mkdir(parents=True,exist_ok=False)
    shutil.copytree(before.directory,attempt/'before')
    plan,metrics=plan_round(root,attempt,before,headers,timeout_s,direct,
                           previous_context=previous_context,rejection_feedback=feedback)
    for path in attempt.iterdir():
        if path.is_dir():shutil.copytree(path,folder/path.name,dirs_exist_ok=True)
        else:shutil.copyfile(path,folder/path.name)
    render_round(folder)
    return plan,metrics


def render_round(folder):
    original.render_round(folder)
    path=folder/'decision.html'
    if path.exists():
        path.write_text(path.read_text()+'<h2>GGO 几何与标注</h2><a href="ggo/index.html">查看几何报告</a><img src="ggo/rgb_overlay.png"><img src="ggo/body_cross_sections.png">')
