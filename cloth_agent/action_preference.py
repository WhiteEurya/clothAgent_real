"""Human choice of Ps + alpha*Pt candidates; persistent scalar preference learning.

No hardware dependencies. A choice is a preference label, never an outcome label.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import urlparse

from .experiments import one_shot_goal_planner_probe_v1 as io

DEFAULT_STATE = Path('data/experiments/one_shot_preference/alpha.json')
LEARNING = dict(initial_alpha=1., min_alpha=0., max_alpha=5., learning_rate=.2,
                max_update=.25, temperature=.1, score_scale=100.)


def atomic_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        import os
        os.fsync(f.fileno())
        name = f.name
    Path(name).replace(path)


@contextmanager
def state_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def _load_state(path):
    if Path(path).exists():
        state = io.read(path)
        if state['schema_version'] != 1 or state['learning'] != LEARNING:
            raise ValueError('alpha state learning protocol mismatch')
        if not math.isfinite(state['alpha']) or not 0 <= state['alpha'] <= 5:
            raise ValueError('invalid saved alpha')
        return state
    return dict(schema_version=1, alpha=1., revision=0, learning=LEARNING, feedback={})


def load_state(path=DEFAULT_STATE):
    with state_lock(path):
        state = _load_state(path)
        if not Path(path).exists(): atomic_save(path, state)
        return state


def ranked(rows, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 5:
        raise ValueError('alpha must be finite and within [0,5]')
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('duplicate action ids')
    for r in rows:
        if any(isinstance(r[k], bool) or not isinstance(r[k], (int, float)) or
               not math.isfinite(r[k]) or not 0 <= r[k] <= 100 for k in ('Ps', 'Pt')):
            raise ValueError('Ps and Pt must be finite scores in [0,100]')
    return sorted([dict(r, P=r['Ps']+alpha*r['Pt']) for r in rows], key=lambda r: (-r['P'], r['id']))


def loss_gradient(rows, chosen, alpha):
    denom = LEARNING['score_scale']*LEARNING['temperature']
    logits = [(r['Ps']+alpha*r['Pt'])/denom for r in rows]
    peak = max(logits); weights = [math.exp(v-peak) for v in logits]; total = sum(weights)
    index = next(i for i,r in enumerate(rows) if r['id'] == chosen)
    loss = peak+math.log(total)-logits[index]
    gradient = (sum(w*r['Pt'] for w,r in zip(weights, rows))/total-rows[index]['Pt'])/denom
    return loss, gradient


def learn(rows, chosen, alpha):
    """One bounded softmax-choice gradient step over displayed alternatives only."""
    ranked(rows, alpha)
    if chosen not in {r['id'] for r in rows}: raise ValueError('choice was not displayed')
    before, gradient = loss_gradient(rows, chosen, alpha)
    step = max(-.25, min(.25, -.2*gradient))
    new = max(0., min(5., alpha+step))
    # Backtrack to avoid increasing this label's loss near a scalar optimum.
    for _ in range(30):
        if loss_gradient(rows, chosen, new)[0] <= before+1e-12: break
        new = (alpha+new)/2
    chosen_row = next(r for r in rows if r['id']==chosen)
    low, high = 0., 5.
    for other in rows:
        ds, dt = chosen_row['Ps']-other['Ps'], chosen_row['Pt']-other['Pt']
        if abs(dt) < 1e-12:
            if ds < 0: low, high = 1., 0.; break
        elif dt > 0: low = max(low, -ds/dt)
        else: high = min(high, -ds/dt)
    possible = low <= high
    return dict(alpha_before=alpha, alpha_after=new, gradient=gradient,
        loss_before=before, loss_after=loss_gradient(rows, chosen, new)[0],
        chosen_can_be_top1_with_scalar=possible,
        top1_feasibility_scope='displayed Top10 only',
        top1_alpha_interval=[low, high] if possible else None,
        ranks_after=[r['id'] for r in ranked(rows, new)],
        supervision='chosen preferred over other displayed actions; not physical success')


def apply_feedback(path, manifest, chosen):
    rows = manifest['displayed']
    if chosen not in {r['id'] for r in rows}: raise ValueError('choice not in frozen Top10')
    key = manifest['review_id']
    with state_lock(path):
        state = _load_state(path)
        if key in state['feedback']:
            event = state['feedback'][key]
            if event['chosen'] != chosen: raise ValueError('review already has another choice')
            return event  # Idempotent after a process crash/repeated delivery.
        if state['revision'] != manifest['state_revision'] or state['alpha'] != manifest['alpha']:
            raise ValueError('alpha changed since ranking; stale feedback rejected')
        update = learn(rows, chosen, state['alpha'])
        event = dict(review_id=key, chosen=chosen, at=io.utc(), **update)
        state.update(alpha=update['alpha_after'], revision=state['revision']+1)
        state['feedback'][key] = event
        atomic_save(path, state)
    return event


def make_review(folder, rows, state, images, *, simulated=False, ranked_rows=None, ranking_context=None):
    """Freeze display and generate only user-facing arrows, never model inputs."""
    import shutil
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    order = ranked(rows, state['alpha']) if ranked_rows is None else ranked_rows
    if len(order) < 10: raise ValueError('fewer than 10 eligible actions; no automatic substitution')
    for name, source in images.items(): shutil.copyfile(source, folder/name)
    binding = dict(rows=order, alpha=state['alpha'], state_revision=state['revision'],
                   image_hashes={n:io.sha(p) for n,p in images.items()}, nonce=secrets.token_hex(16))
    if ranking_context is not None:
        if ranking_context['image_sha256'] != binding['image_hashes']['current.png']:
            raise ValueError('residual features do not match the reviewed image')
        if ranking_context['alpha_fixed'] != state['alpha']:
            raise ValueError('residual fixed alpha does not match base')
        binding['ranking_context'] = ranking_context
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    manifest = dict(review_id=digest, alpha=state['alpha'], state_revision=state['revision'],
                    displayed=order[:10], all_candidates=order, image_hashes=binding['image_hashes'],
                    simulated=simulated, token=secrets.token_urlsafe(32))
    if ranking_context is not None: manifest['ranking_context'] = ranking_context
    atomic_save(folder/'manifest.json', manifest)
    for r in manifest['displayed']:
        io.overlay(folder/'current.png', plan_for(r), folder/(r['id']+'.png'),
                   (f"{r['id']} | base {r['base']:.2f} + R {r['correction']:+.2f} = S {r['S']:.2f}"
                    if ranking_context is not None else
                    f"{r['id']} | Ps {r['Ps']} + {state['alpha']:.3f} * Pt {r['Pt']} = {r['P']:.2f}"))
    render_review(folder, manifest)
    return manifest


def plan_for(row):
    return dict(decision='ACTION', grasp_px=row['grasp_px'], target_px=row['target_px'],
                motion_type='lift_pull_place', grasp_angle_deg=row['grasp_angle_deg'],
                short_reason=row['trajectory_reason'])


def render_review(folder, manifest):
    if manifest.get('ranking_context', {}).get('mode') == 'residual':
        from .ranking_residual import render_review
        return render_review(folder, manifest)
    cards=[]
    for i,r in enumerate(manifest['displayed'],1):
        cards.append(f'''<article><h2>第 {i} · {r['id']} · {html.escape(r['point_id'])}</h2>
<img src="{r['id']}.png"><p><b>Ps {r['Ps']} + α {manifest['alpha']:.3f} × Pt {r['Pt']} = {r['P']:.2f}</b></p>
<p>Pa {r['grasp_px']} → Pb {r['target_px']}；抓前角度 {r['grasp_angle_deg']}°</p>
<p>抓点：{html.escape(r['point_reason'])}</p><p>动作：{html.escape(r['trajectory_reason'])}</p>
<button class="choose" data-id="{r['id']}">选择 {r['id']}</button></article>''')
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>选择下一步动作 · Ps + αPt</title>
<style>body{font-family:system-ui;background:#f3f5f9;color:#182235;margin:24px}article,section{background:white;padding:18px;border-radius:12px;margin-bottom:18px}p{line-height:1.6}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%}button{padding:12px;font:inherit;cursor:pointer}#controls{position:sticky;top:0;z-index:2;border:2px solid #b77d00}#status{white-space:pre-wrap}.selected{outline:3px solid #d19800}@media(max-width:900px){.cards{grid-template-columns:1fr}}</style>
<h1>Top 10 动作：P = Ps + α × Pt</h1><p>__MODE__</p>
<section><p>当前 α = __ALPHA__。Ps：抓点本身的可抓性；Pt：成功抓住后沿该方向展开的预期收益。两者均为模型0–100主观分，总分不是概率。同一个抓点可对应多个方向。</p>
<details><summary>查看干净原图与目标</summary><img src="current.png"><img src="goal.png"></details></section>
<section id="controls"><b id="selected">尚未选择</b><p><label><input id="unchanged" type="checkbox">确认拍照后衣物和桌面没有被挪动</label></p>
<button id="submit" disabled>确认选择，更新 α 并执行此动作</button> <button id="stop">都不合适，停止本轮</button><p id="status">等待你的选择；不会自动执行排名第一的动作。提交后仍需通过原有 Host 检查。</p></section>
<div class="cards">__CARDS__</div>
<script>const context=__CONTEXT__;let chosen=null;const status=document.querySelector('#status');
document.querySelectorAll('.choose').forEach(b=>b.onclick=()=>{chosen=b.dataset.id;document.querySelector('#selected').textContent='已选 '+chosen;document.querySelector('#submit').disabled=false;document.querySelectorAll('article').forEach(a=>a.classList.toggle('selected',a.contains(b)));});
async function send(stop){if(location.protocol==='file:'){status.textContent='这是保存的页面。请使用运行终端打印的 http://127.0.0.1 地址提交。';return;}
if(!stop&&(!chosen||!document.querySelector('#unchanged').checked)){status.textContent='请选择动作并确认衣物没有被挪动。';return;}
document.querySelectorAll('button').forEach(b=>b.disabled=true);
try{const res=await fetch('/choice',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({review_id:context.review_id,token:context.token,decision:stop?'STOP':'SELECT',candidate_id:stop?null:chosen,scene_unchanged:!stop})});const data=await res.json();if(!res.ok)throw Error(data.error);status.textContent=data.message;}catch(e){status.textContent=String(e)+'；请查看运行终端，勿重复启动机器人。';}}
document.querySelector('#submit').onclick=()=>send(false);document.querySelector('#stop').onclick=()=>send(true);
if(context.simulated){document.querySelector('#submit').textContent='演示：记录选择（不执行机器人）';}
</script></html>'''
    context={k:manifest[k] for k in ('review_id','token','simulated')}
    text=text.replace('__MODE__','模拟界面，分数仅为测试数据，不是新的模型结果。' if manifest['simulated'] else
                      '真实候选。先选择，再确认；仅选中的动作进入执行检查。')
    text=text.replace('__ALPHA__',str(manifest['alpha'])).replace('__CARDS__',''.join(cards))
    text=text.replace('__CONTEXT__',json.dumps(context).replace('<','\\u003c'))
    (folder/'index.html').write_text(text)


def validate_choice(body, manifest):
    if body.get('review_id') != manifest['review_id'] or not secrets.compare_digest(str(body.get('token','')),manifest['token']):
        raise ValueError('stale or invalid review token')
    if body.get('decision')=='STOP': return dict(decision='STOP', review_id=manifest['review_id'])
    if body.get('decision')!='SELECT' or body.get('scene_unchanged') is not True:
        raise ValueError('explicit selection and unchanged-scene confirmation required')
    if body.get('candidate_id') not in {r['id'] for r in manifest['displayed']}:
        raise ValueError('candidate not in displayed Top10')
    choice = dict(decision='SELECT',candidate_id=body['candidate_id'],review_id=manifest['review_id'],scene_unchanged=True)
    if manifest.get('ranking_context', {}).get('mode') == 'residual':
        from .ranking_residual import feedback_groups
        if body.get('ranking_confirmed') is not True:
            raise ValueError('explicit confirmation of the full ranking is required')
        groups = feedback_groups(body, {r['id'] for r in manifest['displayed']})
        choice['human_order'] = [i for group in groups for i in group]
        if 'human_groups' in body: choice['human_groups'] = groups
        choice['ranking_confirmed'] = True
    return choice


def wait_for_choice(folder, manifest, cancel, *, port=8767, timeout_s=1800):
    """Loopback HTTP only; no execution/learning in a request handler."""
    folder=Path(folder); result=[]
    files={'/':'index.html','/index.html':'index.html',
           **{f'/{name}':name for name in ('current.png','goal.png')},
           **{f'/{r["id"]}.png':r['id']+'.png' for r in manifest['displayed']}}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def reply(self,code,value):
            body=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            name=files.get(urlparse(self.path).path)
            if name is None:return self.reply(404,dict(error='not found'))
            content=(folder/name).read_bytes();self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8' if name.endswith('.html') else 'image/png')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(content)))
            self.end_headers();self.wfile.write(content)
        def do_POST(self):
            if self.path!='/choice':return self.reply(404,dict(error='not found'))
            expected=f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Origin') not in (None,expected):return self.reply(403,dict(error='origin rejected'))
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('invalid request size')
                if result:raise ValueError('choice already accepted')
                body=json.loads(self.rfile.read(length))
                choice=validate_choice(body,manifest)
                # Durable only; main thread decides learning and execution later.
                atomic_save(folder/'choice.json',dict(**choice,at=io.utc()))
                result.append(choice)
            except (ValueError,TypeError,KeyError) as error:return self.reply(400,dict(error=str(error)))
            self.reply(200,dict(message='已收到停止请求。' if choice['decision']=='STOP' else
                ('已收到排名和选择。先记录训练前表现，再训练 Rθ；本轮只执行你已选的动作，无需再次确认。'
                 if manifest.get('ranking_context') else
                 '已收到选择。接下来更新 α，并检查当前观测和动作；执行及更新结果会写入本轮记录和终端。')))
    server=HTTPServer(('127.0.0.1',port),Handler);server.timeout=.25
    # Prevent slow/incomplete local requests from blocking cancellation indefinitely.
    server.get_request_original=server.get_request
    def get_request():
        sock,address=server.get_request_original();sock.settimeout(2.);return sock,address
    server.get_request=get_request
    started=time.monotonic()
    atomic_save(folder/'review_status.json',dict(status='WAITING',url=f'http://127.0.0.1:{server.server_port}',review_id=manifest['review_id']))
    print(f'HUMAN CHOICE WAITING http://127.0.0.1:{server.server_port} | {folder}',flush=True)
    try:
        while not result:
            if cancel.is_set():raise KeyboardInterrupt('human review cancelled')
            if time.monotonic()-started>=timeout_s:raise TimeoutError('human review timed out; no auto-choice')
            server.handle_request()
        if cancel.is_set():raise KeyboardInterrupt('cancelled after human choice')
        return result[0],time.monotonic()-started
    finally:
        server.server_close()
        atomic_save(folder/'review_status.json',dict(status='CLOSED',choice=result[0] if result else None,
                                                   wait_s=time.monotonic()-started))
