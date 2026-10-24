"""Loopback-only review UI. Human labels only; no feature/model calls."""
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import json
import mimetypes
from urllib.parse import unquote,urlsplit

from .local_dense_proposal_v2 import read,save,verify,summarize,freeze_groups

HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Local experience review</title>
<style>
body{font:16px system-ui;background:#101827;color:#e6edf7;margin:20px auto;max-width:1500px}button,input,select{font:inherit;padding:9px;border-radius:6px;border:1px solid #52627c;margin:4px}button{cursor:pointer;background:#25334d;color:white}button:hover{background:#385174}input{background:#172338;color:white}h2{margin:8px 0}.bar{position:sticky;top:0;background:#101827;padding:10px;z-index:2;border-bottom:1px solid #425573}.panel{background:#19263b;border-radius:10px;padding:14px;margin:12px 0}img{max-width:100%;object-fit:contain}.columns{display:grid;grid-template-columns:350px 1fr;gap:12px}.muted{color:#afc0d8}.ok{color:#68efb1}#card{width:100%}#crop{width:320px}#full{max-height:650px;width:100%}#status{white-space:pre-wrap}a{color:#99cdff}.group{width:340px}
</style>
<div class="bar"><h2>Top-10 候选预览</h2>
<button onclick="mode='candidate';idx=0;render()">看 Top-10</button>
<select id="method" onchange="idx=0;render()"><option value="A">A · single patch</option><option value="B">B · 3×3 core</option><option value="C">C · core + context</option></select>
<label><input id="review" type="checkbox" onchange="render()">需要时标注</label>
<span id="reviewTools" style="display:none"><button onclick="mode='group';idx=0;render()">同类操作分组</button><label>标注者 <input id="name" placeholder="姓名 / 代号"></label></span><span id="progress"></span>
<a id="sheet" target="_blank">打开 Top-10 总览图</a></div>
<div class="panel"><strong>固定示范</strong><div class="muted">判断局部操作及其效果是否可复用；不要求描述完整 cloth state。</div><img id="demo"></div>
<div class="panel"><button onclick="move(-1)">上一项</button><button onclick="move(1)">下一项</button><select id="jump" onchange="idx=+this.value;render()"></select><strong id="title"></strong>
<p id="hint" class="muted"></p>
<div id="candidateView"><div class="columns"><img id="crop"><img id="full"></div>
<div id="labelTools" style="display:none"><div id="decision"></div><button onclick="labelCandidate('YES')">YES · 此位置可复用</button><button onclick="labelCandidate('NO')">NO · 不可复用</button><button onclick="labelCandidate('MAYBE')">MAYBE · 不能确定</button><button onclick="labelCandidate('UNREVIEWED')">清除判断</button></div></div>
<div id="groupView"><img id="card"><p id="evidence"></p>
<label>同一种局部操作填同一个组名：<input id="group" class="group" placeholder="例如 open_panel_1（自由命名）"></label>
<label><input id="confirmed" type="checkbox">我确认此步属于该类经验</label>
<button onclick="saveGroup()">保存分组</button><button onclick="freezeGroups()">冻结已确认分组</button>
<p class="muted">只选少量明显同类操作即可，其余留空。请跨 session 选择至少两条。assisted 模式的 start 可能只是接近点，暂不作为定位 GT。</p></div>
<p id="status"></p></div>
<script>
let data,mode='candidate',idx=0;
async function refresh(){data=await(await fetch('/api')).json();if(!document.getElementById('name').value)document.getElementById('name').value=data.candidate_labels.annotator||data.groups.annotator||'';render()}
function ranked(){return (data.rankings[document.getElementById('method').value]||[]).slice(0,10)}
function items(){return mode==='candidate'?ranked().map(c=>c.candidate_id):data.manifest.steps.map(s=>s.step_id)}
function render(){
let review=document.getElementById('review').checked;
if(!review)mode='candidate';
let method=document.getElementById('method').value;
document.getElementById('reviewTools').style.display=review?'inline':'none';
document.getElementById('labelTools').style.display=review?'block':'none';
document.getElementById('sheet').href='/discovery/'+method+'_top50_01.jpg';
document.getElementById('sheet').style.display=mode==='candidate'?'inline':'none';
let ids=items();idx=Math.max(0,Math.min(idx,ids.length-1));let id=ids[idx];document.getElementById('candidateView').style.display=mode==='candidate'?'block':'none';document.getElementById('groupView').style.display=mode==='group'?'block':'none';document.getElementById('demo').src='/step_cards/'+data.manifest.demo.step_id+'.jpg';let j=document.getElementById('jump');j.replaceChildren();ids.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=mode==='candidate'?'Top '+(i+1)+' · '+x:x;o.selected=i===idx;j.append(o)});document.getElementById('title').textContent=ids.length?(idx+1)+' / '+ids.length+' · '+id:'尚无候选，请先运行 discover';let labeled=Object.values(data.candidate_labels.labels).filter(x=>x!=='UNREVIEWED').length;let grouped=Object.values(data.groups.steps).filter(x=>x.confirmed).length;document.getElementById('progress').textContent=review?'候选已审 '+labeled+'/'+Object.keys(data.candidates).length+'；已确认动作 '+grouped:method+' · 相似度从高到低 · 无需打分';document.getElementById('status').textContent='';if(!id)return;
if(mode==='candidate'){let c=ranked()[idx];document.getElementById('crop').src='/candidate_cards/'+id+'_crop.png';document.getElementById('full').src='/candidate_cards/'+id+'_full.jpg';document.getElementById('decision').textContent='当前判断：'+(data.candidate_labels.labels[id]||'UNREVIEWED');document.getElementById('hint').textContent=method+' · Top '+(idx+1)+' · cosine '+c.score.toFixed(4)+' · 红点为候选位置。图像 '+c.image_id+'，原图位置 '+c.xy.map(x=>x.toFixed(1)).join(', ')}
else{let s=data.manifest.steps.find(x=>x.step_id===id),g=data.groups.steps[id];document.getElementById('card').src='/step_cards/'+id+'.jpg';document.getElementById('group').value=g.group;document.getElementById('confirmed').checked=g.confirmed;document.getElementById('group').disabled=!s.anchor_eligible||data.groups_frozen;document.getElementById('confirmed').disabled=!s.anchor_eligible||data.groups_frozen;document.getElementById('evidence').textContent=s.anchor_evidence;document.getElementById('hint').textContent=s.session+' | '+s.trajectory_unit+' | '+s.collection_mode+(data.groups_frozen?' | 已冻结':'')}
}
async function post(body){body.annotator=document.getElementById('name').value.trim();let res=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let obj=await res.json();if(!res.ok)throw Error(obj.error);return obj}
async function labelCandidate(label){try{await post({action:'candidate',id:items()[idx],label});await refresh();document.getElementById('status').textContent='已保存 '+label}catch(e){document.getElementById('status').textContent=e.message}}
async function saveGroup(){try{await post({action:'group',id:items()[idx],group:document.getElementById('group').value.trim(),confirmed:document.getElementById('confirmed').checked});await refresh();document.getElementById('status').textContent='分组已保存'}catch(e){document.getElementById('status').textContent=e.message}}
async function freezeGroups(){if(!confirm('冻结当前人工确认的分组？不会启动模型或机器人。'))return;try{await post({action:'freeze'});await refresh();document.getElementById('status').textContent='已冻结，可单独运行 localize'}catch(e){document.getElementById('status').textContent=e.message}}
function move(d){if(mode==='group'&&items().length){let g=data.groups.steps[items()[idx]];if(g.group!==document.getElementById('group').value.trim()||g.confirmed!==document.getElementById('confirmed').checked){if(!confirm('当前分组未保存，离开？'))return}}idx+=d;render()}
refresh();
</script></html>'''


def state(root):
    manifest=read(root/'manifest.json');result=root/'discovery/results.json'
    discovery=read(result) if result.exists() else {}
    return dict(manifest=manifest,candidates=discovery.get('unique_candidates',{}),rankings=discovery.get('rankings',{}),
        candidate_labels=read(root/'candidate_labels.json'),groups=read(root/'experience_groups.draft.json'),
        groups_frozen=(root/'experience_groups.frozen.json').exists())


def apply_review(root,body):
    name=body.get('annotator','').strip()
    if not name:raise ValueError('请填写标注者姓名 / 代号')
    action=body.get('action')
    if action=='candidate':
        s=state(root);cid=body.get('id');label=body.get('label')
        if cid not in s['candidates'] or label not in ('YES','NO','MAYBE','UNREVIEWED'):raise ValueError('无效候选/标签')
        a=s['candidate_labels'];a['annotator']=name;a['labels'][cid]=label;save(root/'candidate_labels.json',a)
    elif action=='group':
        if (root/'experience_groups.frozen.json').exists():raise ValueError('已冻结分组不可覆盖')
        s=state(root);sid=body.get('id');step=next((x for x in s['manifest']['steps'] if x['step_id']==sid),None)
        group=body.get('group','').strip();confirmed=body.get('confirmed')
        if step is None or not step['anchor_eligible'] or not isinstance(confirmed,bool) or len(group)>80 or confirmed and not group:raise ValueError('需要有效的 human Pa、组名和人工确认')
        a=s['groups'];a['annotator']=name;a['steps'][sid]=dict(group=group,confirmed=confirmed);save(root/'experience_groups.draft.json',a)
    elif action=='freeze':freeze_groups(root)
    else:raise ValueError('unknown review action')
    summarize(root)


def serve(root,port=8767):
    root=Path(root).resolve();verify(root)
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,data,content_type='application/json'):
            self.send_response(code);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            path=unquote(urlsplit(self.path).path)
            if path=='/':return self.send(200,HTML.encode(),'text/html; charset=utf-8')
            if path=='/api':return self.send(200,json.dumps(state(root),ensure_ascii=False).encode())
            target=(root/path.lstrip('/')).resolve()
            if root not in target.parents or not target.is_file():return self.send(404,b'{}')
            self.send(200,target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
        def do_POST(self):
            # Loopback UI only, reject cross-origin browser writes.
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                return self.send(403,b'{"error":"cross-origin writes rejected"}')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length>100000:raise ValueError('request too large')
                apply_review(root,json.loads(self.rfile.read(length)))
                self.send(200,b'{"saved":true}')
            except Exception as e:self.send(400,json.dumps({'error':str(e)},ensure_ascii=False).encode())
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'Review UI: http://127.0.0.1:{port}  (Ctrl-C to stop)',flush=True)
    try:server.serve_forever()
    finally:server.server_close()
