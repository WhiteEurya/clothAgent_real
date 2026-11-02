"""Human endpoint coordinates first; condition-blind pick adjudication second."""
import argparse,json,secrets,threading,os
from datetime import datetime,timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from .core import read,save,validate_annotation,PARTS

HTML='''<!doctype html><meta charset="utf-8"><title>四端部核验</title><style>*{box-sizing:border-box}body{font:16px system-ui;margin:0;background:#f3f5f9;color:#20304a}header,main{padding:20px;max-width:1500px;margin:auto}h1{font-size:25px}button,input{font:inherit}button{padding:9px 12px;border:1px solid #b6c3d7;border-radius:7px;background:white;cursor:pointer;margin:3px}.active,.primary{background:#2654c8;color:white}nav{display:flex;flex-wrap:wrap}input{padding:8px}#layout{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px}.picture{position:relative;background:#111;align-self:start}.picture img{display:block;width:100%}.picture canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none}.card{padding:14px;background:white;border-radius:10px;margin-bottom:10px}.card b{display:block;margin-bottom:8px}.muted{color:#596982;font-size:14px}#message{white-space:pre-wrap}#reviewer{width:160px}@media(max-width:950px){#layout{grid-template-columns:1fr}}</style><header><h1 id="heading">四端部坐标核验 · 阶段1</h1><p id="instructions">袖口：可见袖口开口边缘的中点；下摆：下摆边与侧缝相交的真实衣角。先选部位，再在图上点一下。不确定时不要猜。</p><p class="muted">左右按穿着者定义：衣服正面朝你且领口向上时，衣服左侧在画面右边。衣服旋转后保持材料部位身份。此处不判断抓取是否能成功展开。</p><label>核验人 <input id="reviewer" placeholder="名字或代号"></label> <span id="progress"></span><p id="phasehint" class="muted"></p><nav id="nav"></nav></header><main><h2 id="case"></h2><div id="layout"><div class="picture"><img id="rgb"><canvas id="canvas"></canvas></div><div><div id="controls"></div><div class="card"><p id="message">加载中…</p><button id="prev">上一张</button><button class="primary" id="confirm">确认并下一张</button><button id="next">下一张</button><p class="muted">选择会保存为草稿。确认后再修改，需要重新确认。可用浏览器缩放放大图片。</p></div></div></div></main><script>
const TOKEN=__TOKEN__,PARTS=__PARTS__,KEYS=Object.keys(PARTS),COLORS=['#ffed4a','#44d9ff','#ff83cc','#6bff83'];let data,index=0,selected='LS',timer,queue=Promise.resolve(),reviewer='';const $=id=>document.getElementById(id);
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','X-Token':TOKEN},body:body?JSON.stringify(body):undefined});const d=await r.json();if(!r.ok)throw Error(d.error||r.status);return d}
function current(){return data.items[index]}function draft(){const k=current().id;return data.drafts[k]||(data.drafts[k]=data.phase==='points'?{points:Object.fromEntries(KEYS.map(k=>[k,{visibility:'',px:null}])),reviewer:reviewer}:{judgment:'',part:null,reviewer:reviewer})}
function paint(){const img=$('rgb'),c=$('canvas');if(!img.naturalWidth)return;c.width=img.naturalWidth;c.height=img.naturalHeight;const ctx=c.getContext('2d');if(data.phase==='blind'){const p=current().pick_px;ctx.strokeStyle='#ffeb30';ctx.lineWidth=3;ctx.beginPath();ctx.arc(p[0],p[1],9,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(p[0]-16,p[1]);ctx.lineTo(p[0]+16,p[1]);ctx.moveTo(p[0],p[1]-16);ctx.lineTo(p[0],p[1]+16);ctx.stroke();return}KEYS.forEach((k,i)=>{const p=draft().points[k].px;if(!p)return;ctx.strokeStyle=COLORS[i];ctx.fillStyle=COLORS[i];ctx.lineWidth=3;ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.stroke();ctx.font='bold 20px sans-serif';ctx.fillText(k,p[0]+12,p[1]-12)})}
function progress(){$('progress').textContent=`已确认 ${data.items.filter(x=>data.confirmed[x.id]).length} / ${data.items.length}`}
function button(text,fn,active){const b=document.createElement('button');b.textContent=text;b.onclick=fn;b.className=active?'active':'';return b}
function render(){if(!data.items.length){$('case').textContent='没有待复核的ACTION';return}const item=current(),d=draft();$('reviewer').value=d.reviewer||reviewer;$('case').textContent=`${index+1}/${data.items.length} · ${item.id}`;$('rgb').src='/image/'+item.case+'.png';$('controls').replaceChildren();if(data.phase==='points'){KEYS.forEach((k,i)=>{const box=document.createElement('div');box.className='card';box.style.borderLeft='5px solid '+COLORS[i];const title=document.createElement('b');title.textContent=PARTS[k]+' ('+k+')';box.append(title);box.append(button(selected===k?'正在点此部位':'选择并在图上点击',()=>{selected=k;render()},selected===k));for(const [v,n] of [['HIDDEN','不可见'],['UNCERTAIN','看不准']])box.append(button(n,()=>{d.points[k]={visibility:v,px:null};changed();render()},d.points[k].visibility===v));const p=document.createElement('p');p.className='muted';p.textContent=d.points[k].px?'坐标：'+d.points[k].px.join(', '):(d.points[k].visibility||'尚未核验');box.append(p);$('controls').append(box)})}else{const box=document.createElement('div');box.className='card';for(const [v,n] of [['TRUE_ENDPOINT','真实四端部附近'],['FALSE_FOLD_CORNER','折叠形成的假角点'],['OTHER_CLOTH','其他布料位置'],['OFF_CLOTH','衣物之外'],['UNCERTAIN','看不准']])box.append(button(n,()=>{d.judgment=v;if(v!=='TRUE_ENDPOINT')d.part=null;changed();render()},d.judgment===v));if(d.judgment==='TRUE_ENDPOINT'){const p=document.createElement('p');p.textContent='确认所属语义部位：';box.append(p);KEYS.forEach(k=>box.append(button(PARTS[k],()=>{d.part=k;changed();render()},d.part===k)))}$('controls').append(box)}$('nav').replaceChildren();data.items.forEach((it,i)=>$('nav').append(button((i+1)+(data.confirmed[it.id]?' ✓':''),()=>go(i),i===index)));$('message').textContent=data.confirmed[item.id]?'已确认':'草稿，确认后才用于实验。';progress();paint()}
function collect(confirm=false){const d=draft();d.reviewer=$('reviewer').value.trim();reviewer=d.reviewer;return {id:current().id,confirm,...structuredClone(d)}}
function enqueue(body){queue=queue.catch(()=>{}).then(()=>api('/save',body)).then(v=>{data.confirmed=v.confirmed;progress();$('message').textContent=body.confirm?'已确认保存':'草稿已保存'}).catch(e=>{$('message').textContent='保存失败：'+e.message;throw e});return queue}
function changed(){delete data.confirmed[current().id];clearTimeout(timer);const body=collect();timer=setTimeout(()=>enqueue(body).catch(()=>{}),300);progress()}
async function go(i){if(i<0||i>=data.items.length)return;clearTimeout(timer);try{await enqueue(collect());index=i;render()}catch(e){}}
$('canvas').onclick=e=>{if(data.phase!=='points')return;const r=e.target.getBoundingClientRect(),img=$('rgb');const x=Math.min(img.naturalWidth-1,Math.max(0,Math.round((e.clientX-r.left)*img.naturalWidth/r.width))),y=Math.min(img.naturalHeight-1,Math.max(0,Math.round((e.clientY-r.top)*img.naturalHeight/r.height)));draft().points[selected]={visibility:'VISIBLE',px:[x,y]};changed();render()};$('rgb').onload=paint;$('reviewer').oninput=changed;$('prev').onclick=()=>go(index-1);$('next').onclick=()=>go(index+1);$('confirm').onclick=async()=>{const body=collect(true);if(!body.reviewer){alert('请填写核验人');return}clearTimeout(timer);try{await enqueue(body);if(index<data.items.length-1)index++;render();if(data.items.every(x=>data.confirmed[x.id]))$('message').textContent=data.phase==='points'?'全部坐标已确认，可告诉我开始72次离线请求。':'全部独立复核已确认，可生成统计和对照报告。'}catch(e){}};
api('/data').then(d=>{data=d;if(d.phase==='blind'){$('heading').textContent='端部选择独立复核 · 阶段2';$('instructions').textContent='仅看黄色抓点，判断它是否位于真实语义端部。组别、动作理由和候选坐标均隐藏。不是端部不自动等于假角点。'}$('phasehint').textContent=d.phase==='points'?'先核验坐标，模型尚未运行。阶段2会隐藏条件名称复核抓点。':'同一图可能出现多次，请逐次判断，不猜组别。';render()}).catch(e=>{$('case').textContent=e.message});
</script>'''

class Store:
    def __init__(self,root,phase='points'):
        self.root=Path(root).resolve();self.phase=phase;self.lock=threading.Lock();self.cases={c['id']:c for c in read(self.root/'cases.json')}
        self.items=[{'id':c,'case':c} for c in self.cases] if phase=='points' else [{'id':x['review_id'],'case':x['case'],'pick_px':x['pick_px']} for x in read(self.root/'blind_manifest_private.json')]
        self.keys={x['id'] for x in self.items};self.formal='annotations.json' if phase=='points' else 'blind_labels.json';self.drafts='annotation_drafts.json' if phase=='points' else 'blind_drafts.json'
    def load(self,name):return read(self.root/name) if (self.root/name).exists() else {}
    def atomic(self,name,data):
        f=self.root/name;tmp=f.with_suffix('.tmp');save(tmp,data);os.replace(tmp,f)
    def data(self):
        with self.lock:return {'phase':self.phase,'items':self.items,'drafts':self.load(self.drafts),'confirmed':{k:True for k,v in self.load(self.formal).items() if v.get('confirmed_by')}}
    def save(self,body):
        key=body.get('id');confirm=body.get('confirm') is True;reviewer=str(body.get('reviewer','')).strip()[:100]
        if key not in self.keys:raise ValueError('unknown item')
        if self.phase=='points' and (self.root/'frozen/oracle_annotations.json').exists():raise ValueError('coordinates frozen for model experiment; do not change this study')
        if self.phase=='points':
            points=body.get('points',{});draft={'reviewer':reviewer,'points':points};record={'confirmed_by':reviewer,'points':points,'rgb_sha256':self.cases[key]['rgb_sha256']}
            if set(points)!=set(PARTS):raise ValueError('four endpoint fields required')
            if confirm:validate_annotation(record,self.cases[key]['size'])
        else:
            judgment=body.get('judgment','');part=body.get('part');draft={'reviewer':reviewer,'judgment':judgment,'part':part};record={'confirmed_by':reviewer,'judgment':judgment,'part':part}
            if judgment not in ('','TRUE_ENDPOINT','FALSE_FOLD_CORNER','OTHER_CLOTH','OFF_CLOTH','UNCERTAIN'):raise ValueError('invalid judgment')
            if confirm and (not judgment or (judgment=='TRUE_ENDPOINT' and part not in PARTS)):raise ValueError('请选择判断；真实端部需确认所属部位')
        if confirm and not reviewer:raise ValueError('reviewer required')
        with self.lock:
            drafts=self.load(self.drafts);records=self.load(self.formal);old=drafts.get(key);changed=old!=draft
            if confirm:record['confirmed_at']=datetime.now(timezone.utc).isoformat();records[key]=record
            elif changed and key in records:records[key]['confirmed_by']=None
            drafts[key]=draft
            with (self.root/'review_audit.jsonl').open('a') as f:f.write(json.dumps({'phase':self.phase,'id':key,'confirm':confirm,'draft':draft,'time':datetime.now(timezone.utc).isoformat()},ensure_ascii=False)+'\n')
            self.atomic(self.drafts,drafts);self.atomic(self.formal,records);return {'confirmed':{k:True for k,v in records.items() if v.get('confirmed_by')}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['points','blind'],default='points');p.add_argument('--port',type=int,default=8795);a=p.parse_args();store=Store(a.root,a.phase);token=secrets.token_urlsafe(24);page=HTML.replace('__TOKEN__',json.dumps(token)).replace('__PARTS__',json.dumps(PARTS,ensure_ascii=False)).encode()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,mime='application/json; charset=utf-8'):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':return self.send(200,page,'text/html; charset=utf-8')
            if path=='/data':return self.send(200,store.data())
            if path.startswith('/image/') and path.endswith('.png'):
                key=path[len('/image/'):-4]
                if key in store.cases:return self.send(200,(store.root/'images'/f'{key}.png').read_bytes(),'image/png')
            self.send(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/save' or self.headers.get('X-Token')!=token:return self.send(403,{'error':'invalid session'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<20000:raise ValueError('invalid request size')
                self.send(200,store.save(json.loads(self.rfile.read(n))))
            except (ValueError,KeyError,TypeError) as exc:self.send(400,{'error':str(exc)})
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);print(f'ORACLE_REVIEW http://127.0.0.1:{a.port} phase={a.phase}',flush=True);server.serve_forever()
if __name__=='__main__':main()
