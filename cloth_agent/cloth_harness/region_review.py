"""Independent region annotations for the prospective A/D repeated study."""
import argparse,json,math,secrets,threading
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from .review_app import Store,FIELDS,HTML

class RegionStore(Store):
    def __init__(self,root):
        super().__init__(root);self.lock=threading.RLock()
    def save(self,data):
        target=data.get('next_decision','');boxes=data.get('regions',[])
        if target not in ('','CONTINUE','DONE','UNCERTAIN'):raise ValueError('invalid next decision')
        if not isinstance(boxes,list) or len(boxes)>30:raise ValueError('too many regions')
        for b in boxes:
            if b.get('kind') not in ('allow','avoid'):raise ValueError('invalid region type')
            rect=b.get('rect')
            if not isinstance(rect,list) or len(rect)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in rect):raise ValueError('invalid normalized rectangle')
            x,y,w,h=rect
            if w<=0 or h<=0 or x+w>1.000001 or y+h>1.000001:raise ValueError('rectangle outside image')
        for a in boxes:
            for b in boxes:
                if a['kind']==b['kind']:continue
                x,y,w,h=a['rect'];X,Y,W,H=b['rect']
                if min(x+w,X+W)>max(x,X) and min(y+h,Y+H)>max(y,Y):raise ValueError('绿色与红色区域不能重叠，请撤销后重画')
        if data.get('confirm') and (not target or (target=='CONTINUE' and not any(b['kind']=='allow' for b in boxes))):raise ValueError('继续操作时请至少框一个允许抓取区域；无法判断请选择不确定')
        with self.lock:
            key=data.get('id');old=self.read('review_drafts.json',{}).get(key,{});changed=(old.get('regions',[])!=boxes or old.get('next_decision','')!=target)
            super().save(data)
            drafts=self.read('review_drafts.json',{});drafts[key].update(next_decision=target,regions=boxes);self.atomic('review_drafts.json',drafts)
            records=self.read('oracle_labels.json',{});evaluation=self.read('next_action_labels.json',{})
            if data.get('confirm'):
                evaluation[key]={'confirmed_by':data['reviewer'],'confirmed_at':datetime.now(timezone.utc).isoformat(),'next_decision':target,'regions':boxes,'coordinates':'normalized current-after image','scope':'partial acceptable/unacceptable pick regions, not complete trajectory correctness'}
            elif changed or not records.get(key,{}).get('confirmed_by'):
                if key in evaluation:evaluation[key]['confirmed_by']=None
                if key in records:records[key]['confirmed_by']=None
            self.atomic('next_action_labels.json',evaluation);self.atomic('oracle_labels.json',records)
            with (self.root/'region_review_history.jsonl').open('a') as f:f.write(json.dumps({'id':key,'confirm':data.get('confirm',False),'next_decision':target,'regions':boxes},ensure_ascii=False)+'\n')
            return {'ok':True,'confirmed':{k:True for k,v in records.items() if v.get('confirmed_by') and evaluation.get(k,{}).get('confirmed_by')}}
    def data(self):
        result=super().data();evaluation=self.read('next_action_labels.json',{})
        result['confirmed']={k:v for k,v in result['confirmed'].items() if evaluation.get(k,{}).get('confirmed_by')}
        return result

EXTRA='''
const originalRender=render;
let drawing=null,mode='allow';
function nextControls(){const d=draft();d.regions=d.regions||[];const panel=$('nextDecision');panel.replaceChildren();for(const [v,n] of [['CONTINUE','还需要操作'],['DONE','可以 DONE'],['UNCERTAIN','看不准']]){const b=document.createElement('button');b.textContent=n;b.className=d.next_decision===v?'active':'';b.onclick=()=>{draft().next_decision=v;change();nextControls()};panel.append(b)}$('regionCount').textContent=`已画 ${d.regions.length} 个区域`;paint()}
render=function(){originalRender();nextControls();$('saved').textContent=state.confirmed[current().id]?'本组已确认':'先判断上一动作，再标动作后图中的合理抓取区域。';};
const originalCapture=capture;capture=function(confirm=false){const v=originalCapture(confirm);v.next_decision=draft().next_decision||'';v.regions=structuredClone(draft().regions||[]);return v};
function paint(temp){const image=$('after'),canvas=$('regions');if(!image.naturalWidth)return;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;const ctx=canvas.getContext('2d');for(const b of [...(draft().regions||[]),...(temp?[temp]:[])]){const [x,y,w,h]=b.rect;ctx.strokeStyle=b.kind==='allow'?'#10c45e':'#ff4949';ctx.fillStyle=b.kind==='allow'?'#10c45e33':'#ff494933';ctx.lineWidth=4;ctx.fillRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height);ctx.strokeRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height)}}
$('after').onload=()=>paint();
function xy(e){const r=$('regions').getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
function box(end){return {kind:mode,rect:[Math.min(drawing[0],end[0]),Math.min(drawing[1],end[1]),Math.abs(drawing[0]-end[0]),Math.abs(drawing[1]-end[1])]}}
$('regions').onpointerdown=e=>{drawing=xy(e);e.target.setPointerCapture(e.pointerId);e.preventDefault()};$('regions').onpointermove=e=>{if(drawing)paint(box(xy(e)))};$('regions').onpointerup=e=>{if(!drawing)return;const b=box(xy(e));drawing=null;if(b.rect[2]>.005&&b.rect[3]>.005){draft().regions=draft().regions||[];draft().regions.push(b);change()}nextControls()};$('regions').onpointercancel=()=>{drawing=null;paint()};
$('allowMode').onclick=()=>{mode='allow';$('allowMode').className='active';$('avoidMode').className=''};$('avoidMode').onclick=()=>{mode='avoid';$('avoidMode').className='active';$('allowMode').className=''};$('undoRegion').onclick=()=>{draft().regions=(draft().regions||[]).slice(0,-1);change();nextControls()};$('clearRegions').onclick=()=>{draft().regions=[];change();nextControls()};$('largeAfter').onclick=()=>{$('zoom').querySelector('img').src=$('after').src;$('zoom').showModal()};
'''

def page(token):
    s=HTML.replace('__TOKEN__',json.dumps(token)).replace('__FIELDS__',json.dumps(FIELDS,ensure_ascii=False))
    s=s.replace('比较真实动作前后，只判断发生了什么，不需要选择下一步抓点。','两步标注：① 上一动作发生了什么；② 在动作后图上框出下一步可接受／明显不合适的抓取区域。')
    s=s.replace('<img id="after">','<div style="position:relative"><img id="after" style="display:block"><canvas id="regions" style="position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair"></canvas></div><p><button id="allowMode" class="active">绿色：可以抓</button> <button id="avoidMode">红色：不适合抓</button> <button id="undoRegion">撤销一框</button> <button id="clearRegions">清空框</button> <button id="largeAfter">放大原图</button></p><span id="regionCount"></span>')
    s=s.replace('<div id="fields"></div>','<h3>① 上一动作结果（提供给D组）</h3><div id="fields"></div><h3>② 下一步评价标准（不会提供给GPT）</h3><div id="nextDecision" class="choices"></div><p class="muted">选择绿色或红色模式，在右侧动作后图上拖动框选。允许多个合理区域；不用框完整衣服。没有标记的地方表示尚未评价，不会自动算错。绿色和红色不可重叠。还需要操作时，至少框一个绿色区域；无法判断可选“看不准”。本轮先评价抓点区域，不能证明整条轨迹正确。</p>')
    s=s.replace('api(\'/data\').then(v=>{state=v;render()})',EXTRA+"\napi('/data').then(v=>{state=v;render()})")
    s=s.replace('全部标注已确认并保存。可以告诉我继续跑 D 组。','全部8组已确认保存。可以告诉我开始 A/D 各三次对照。')
    return s.encode()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=8794);args=p.parse_args();store=RegionStore(args.root);token=secrets.token_urlsafe(24);content=page(token)
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,mime='application/json; charset=utf-8'):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':return self.send(200,content,'text/html; charset=utf-8')
            if path=='/data':return self.send(200,store.data())
            bits=path.split('/')
            if len(bits)==4 and bits[1]=='image' and bits[2] in store.ids and bits[3] in ('before.png','after.png'):return self.send(200,(store.root/bits[2]/bits[3]).read_bytes(),'image/png')
            self.send(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/save' or self.headers.get('X-Review-Token')!=token:return self.send(403,{'error':'invalid session'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<30000:raise ValueError('invalid body size')
                self.send(200,store.save(json.loads(self.rfile.read(n))))
            except (ValueError,TypeError,KeyError) as e:self.send(400,{'error':str(e)})
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);print(f'REGION_REVIEW http://127.0.0.1:{args.port}',flush=True);server.serve_forever()
if __name__=='__main__':main()
