"""Local human annotation UI; drafts are separate from confirmed oracle labels."""
import argparse,json,secrets,threading,os
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .evaluation.evaluator import validate_label

FIELDS=[('grasp_status','是否抓住衣服',[('SUCCESS','抓住'),('MISS','抓空'),('SLIPPED','中途掉落'),('UNKNOWN','不确定')]),('area_change','可见投影面积',[('INCREASED','增加'),('UNCHANGED','基本不变'),('DECREASED','减少'),('UNKNOWN','不确定')]),('motion_type','主要运动',[('RIGID_DOMINANT','整体平移 / 旋转'),('NONRIGID','局部变形'),('MIXED','两者都有'),('UNKNOWN','不确定')]),('local_shape_change','局部形状',[('CHANGED','有明显变化'),('NO_CLEAR_CHANGE','无明显变化'),('UNKNOWN','不确定')]),('visible_unfolding_progress','展开效果',[('IMPROVED','改善'),('NO_CLEAR_PROGRESS','无明显进展'),('WORSE','变差'),('UNKNOWN','不确定')])]
HTML='''<!doctype html><meta charset="utf-8"><title>衣物操作 · 人工标注</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f4f8;color:#192337;font:16px system-ui}header{background:#fff;padding:20px 4%;border-bottom:1px solid #dae0ea}h1{font-size:25px;margin:0 0 10px}main{max-width:1500px;margin:auto;padding:22px}button,input,textarea{font:inherit}button{cursor:pointer;border:1px solid #c4ccdb;border-radius:8px;padding:10px 16px;background:white;color:#233650}button.active{background:#2454cf;color:white;border-color:#2454cf}.nav{display:flex;gap:8px;flex-wrap:wrap;margin:15px 0}.photos{display:grid;grid-template-columns:1fr 1fr;gap:14px}.photo{background:white;padding:12px;border-radius:12px}.photo img{width:100%;cursor:zoom-in}.tag{font-weight:700;margin-bottom:8px}.card{background:white;padding:22px;border-radius:12px;margin-top:16px}.field{display:flex;align-items:center;gap:12px;padding:15px 0;border-bottom:1px solid #edf0f5}.field>b{width:160px;flex-shrink:0}.choices{display:flex;gap:9px;flex-wrap:wrap}textarea{width:100%;height:85px;margin-top:10px;padding:12px;border:1px solid #bdc8da;border-radius:8px}input{padding:8px;border:1px solid #bdc8da;border-radius:6px}.actions{display:flex;gap:12px;align-items:center;margin:20px 0}.primary{background:#2454cf;color:white;border:0}.muted{color:#5a6780;font-size:14px}.ok{color:#14704c}.bad{color:#b33030}dialog{max-width:95vw;border:0;padding:12px;border-radius:10px}dialog img{max-width:90vw;max-height:85vh}@media(max-width:800px){.photos{grid-template-columns:1fr}.field{display:block}.choices{margin-top:10px}}
</style><header><h1>衣物操作 · 人工结果标注</h1><div>比较真实动作前后，只判断发生了什么，不需要选择下一步抓点。</div><p class="muted">看不清请选「不确定」。机器人执行完成不代表抓住衣服；仅靠前后照片不能确定抓取时，也选「不确定」。</p><label>标注人 <input id="reviewer" placeholder="你的名字或代号"></label> <span id="progress"></span></header><main><nav class="nav" id="nav"></nav><h2 id="title"></h2><p id="action"></p><div class="photos"><div class="photo"><div class="tag">动作前</div><img id="before"></div><div class="photo"><div class="tag">动作后</div><img id="after"></div></div><section class="card"><div id="fields"></div><label><p>可选：你看到的依据（例如“右下方叠住的布料打开了”）</p><textarea id="note" placeholder="只描述实际结果，不填写下一步应该抓哪里"></textarea></label><p id="saved" class="muted"></p><div class="actions"><button id="prev">上一组</button><button class="primary" id="confirm">确认本组并下一组</button><button id="next">下一组</button></div><p class="muted">修改会自动保存为草稿；只有点击「确认本组」才会写入正式人工标签。确认后再修改会撤销本组确认，需要重新确认。</p></section></main><dialog id="zoom"><button onclick="document.querySelector('#zoom').close()">关闭大图</button><br><img></dialog><script>
const token=__TOKEN__, fields=__FIELDS__;let state,idx=0,timer,queue=Promise.resolve();const $=id=>document.getElementById(id);
async function api(path,data){const r=await fetch(path,{method:data?'POST':'GET',headers:{'Content-Type':'application/json','X-Review-Token':token},body:data?JSON.stringify(data):undefined});const v=await r.json();if(!r.ok)throw Error(v.error||r.status);return v}
function current(){return state.cases[idx]}function draft(){const c=current();return state.drafts[c.id]||(state.drafts[c.id]={labels:{},note:'',reviewer:state.reviewer||''})}
function progress(){$('progress').textContent=`已确认 ${state.cases.filter(c=>state.confirmed[c.id]).length} / ${state.cases.length} 组`}
function render(){const c=current(),d=draft();$('reviewer').value=d.reviewer||state.reviewer||'';$('title').textContent=`第 ${idx+1} / ${state.cases.length} 组 · ${c.id}`;$('action').textContent=`实际动作 Pa ${c.plan.grasp_px.join(', ')} → Pb ${c.plan.target_px.join(', ')}；控制器状态 ${c.execution.status}`;$('before').src=`/image/${c.id}/before.png`;$('after').src=`/image/${c.id}/after.png`;$('note').value=d.note||'';$('fields').replaceChildren();for(const [key,title,opts] of fields){const row=document.createElement('div');row.className='field';const label=document.createElement('b');label.textContent=title;row.append(label);const options=document.createElement('div');options.className='choices';for(const [value,name] of opts){const b=document.createElement('button');b.textContent=name;b.className=d.labels[key]===value?'active':'';b.onclick=()=>{draft().labels[key]=value;change();render()};options.append(b)}row.append(options);$('fields').append(row)}$('nav').replaceChildren();state.cases.forEach((c,i)=>{const b=document.createElement('button');b.textContent=`${i+1}${state.confirmed[c.id]?' ✓':''}`;b.className=i===idx?'active':'';b.onclick=()=>go(i);$('nav').append(b)});$('saved').textContent=state.confirmed[c.id]?'本组已确认，可继续修改。':'草稿：五项均需选择，也可以全部选不确定。';progress()}
function enqueue(data){queue=queue.catch(()=>{}).then(()=>api('/save',data)).then(v=>{state.confirmed=v.confirmed;progress();$('saved').textContent=data.confirm?'正式标签已保存':'草稿已保存';$('saved').className='ok'}).catch(e=>{$('saved').textContent='保存失败：'+e.message;$('saved').className='bad';throw e});return queue}
function capture(confirm=false){const d=draft();d.reviewer=$('reviewer').value.trim();state.reviewer=d.reviewer;d.note=$('note').value;return {id:current().id,...structuredClone(d),confirm}}
function change(){delete state.confirmed[current().id];clearTimeout(timer);const data=capture();timer=setTimeout(()=>enqueue(data).catch(()=>{}),350);progress()}
async function flush(){clearTimeout(timer);await enqueue(capture())}
async function go(i){if(i<0||i>=state.cases.length)return;try{await flush();idx=i;render()}catch(e){}}
$('note').oninput=change;$('reviewer').oninput=change;$('prev').onclick=()=>go(idx-1);$('next').onclick=()=>go(idx+1);
$('confirm').onclick=async()=>{const d=capture(true);if(!d.reviewer){alert('请先填写标注人名字或代号');return}if(fields.some(([k])=>!d.labels[k])){alert('请完成五项选择；看不清可以选“不确定”');return}clearTimeout(timer);try{await enqueue(d);if(idx<state.cases.length-1)idx++;render();if(state.cases.every(c=>state.confirmed[c.id])){$('saved').textContent='全部标注已确认并保存。可以告诉我继续跑 D 组。';$('saved').className='ok'}}catch(e){}};
for(const id of ['before','after'])$(id).onclick=()=>{$('zoom').querySelector('img').src=$(id).src;$('zoom').showModal()};
api('/data').then(v=>{state=v;render()}).catch(e=>{$('title').textContent='加载失败：'+e.message});
</script>'''

class Store:
    def __init__(self,root):
        self.root=Path(root).resolve();self.lock=threading.Lock()
        selected=json.loads((self.root/'abcd/protocol_ABCD.json').read_text())['selected_cases']
        pairs=json.loads((self.root/'pairs.json').read_text());self.cases=[p for p in pairs if p['id'] in selected];self.ids={p['id'] for p in self.cases}
    def read(self,name,default):
        p=self.root/name;return json.loads(p.read_text()) if p.exists() else default
    def atomic(self,name,data):
        p=self.root/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));os.replace(tmp,p)
    def data(self):
        with self.lock:
            labels=self.read('oracle_labels.json',{});drafts=self.read('review_drafts.json',{})
            for k in self.ids:
                if k not in drafts and labels.get(k,{}).get('confirmed_by'):
                    v=labels[k];drafts[k]={'labels':v['labels'],'note':v.get('note',''),'reviewer':v['confirmed_by']}
            return {'cases':self.cases,'drafts':drafts,'confirmed':{k:True for k,v in labels.items() if v.get('confirmed_by')},'reviewer':''}
    def save(self,data):
        key=data.get('id');confirm=data.get('confirm',False);labels=data.get('labels',{})
        if key not in self.ids:raise ValueError('unknown case')
        if type(confirm) is not bool:raise ValueError('invalid confirmation')
        allowed={k:{v for v,_ in opts} for k,_,opts in FIELDS}
        if not isinstance(labels,dict) or any(k not in allowed or v not in allowed[k] for k,v in labels.items()):raise ValueError('invalid labels')
        reviewer=str(data.get('reviewer','')).strip()[:100];note=str(data.get('note',''))[:2000]
        if confirm and (set(labels)!=set(allowed) or not reviewer):raise ValueError('complete all five labels and reviewer')
        now=datetime.now(timezone.utc).isoformat()
        with self.lock:
            drafts=self.read('review_drafts.json',{});old=drafts.get(key);draft={'labels':labels,'reviewer':reviewer,'note':note}
            records=self.read('oracle_labels.json',{});previous=records.get(key,{})
            changed=old is None or any(old.get(k)!=v for k,v in draft.items())
            if confirm:
                record={'confirmed_by':reviewer,'confirmed_at':now,'evidence':[key+'/before.png',key+'/after.png'],'labels':labels,'note':note}
                validate_label(record);records[key]=record
            elif changed and previous.get('confirmed_by'):
                records[key]={**previous,'confirmed_by':None,'superseded_by_draft_at':now}
            drafts[key]=draft
            with (self.root/'human_review_history.jsonl').open('a') as f:f.write(json.dumps({'time':now,'id':key,'confirm':confirm,'changed':changed,'previous_record':previous,'draft':draft},ensure_ascii=False)+'\n')
            self.atomic('review_drafts.json',drafts);self.atomic('oracle_labels.json',records)
            return {'ok':True,'confirmed':{k:True for k,v in records.items() if v.get('confirmed_by')}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8793);a=p.parse_args();store=Store(a.root);token=secrets.token_urlsafe(24)
    page=HTML.replace('__TOKEN__',json.dumps(token)).replace('__FIELDS__',json.dumps(FIELDS,ensure_ascii=False)).encode()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,kind='application/json; charset=utf-8'):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':return self.send(200,page,'text/html; charset=utf-8')
            if path=='/data':return self.send(200,store.data())
            bits=path.split('/')
            if len(bits)==4 and bits[1]=='image' and bits[2] in store.ids and bits[3] in ('before.png','after.png'):
                return self.send(200,(store.root/bits[2]/bits[3]).read_bytes(),'image/png')
            self.send(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/save' or self.headers.get('X-Review-Token')!=token:return self.send(403,{'error':'invalid session'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<20000:raise ValueError('invalid request size')
                result=store.save(json.loads(self.rfile.read(size)));self.send(200,result)
            except (ValueError,TypeError,KeyError) as e:self.send(400,{'error':str(e)})
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);print(f'REVIEW http://127.0.0.1:{a.port}',flush=True);server.serve_forever()
if __name__=='__main__':main()
