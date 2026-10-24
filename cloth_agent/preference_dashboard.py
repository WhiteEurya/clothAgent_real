"""Persistent loopback review portal; never connects to or commands hardware.

The episode's short-lived review server remains the authority for accepting choices.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError
import argparse
import json
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PORT = 8768
SERVICE = 'cloth-preference-dashboard-v1'
LOCAL = build_opener(ProxyHandler({}))


def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return {}


def upstream(url):
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port or parsed.path not in ('', '/'):
        raise ValueError('invalid review endpoint')
    return f'http://127.0.0.1:{parsed.port}'


def status(reports, check_connection=True):
    run = None
    for path in sorted(Path(reports).glob('one_shot_goal_full_episode_v1_20*'), reverse=True):
        protocol = read(path/'protocol.json')
        if protocol.get('human_preference', {}).get('enabled'):
            run = path
            break
    if run is None:
        return dict(phase='IDLE', message='等待下一次 rollout 启动。', ready=False, run=None, review_id=None)
    summary = read(run/'episode_summary.json')
    result = dict(run=run.name, ready=False, review_id=None, round=None,
                  progress=(run/'learning_progress.md').read_text() if (run/'learning_progress.md').exists() else '')
    if summary.get('termination_reason'):
        return dict(result, phase='FINISHED', message=f"本次运行已结束：{summary['termination_reason']}。{summary.get('termination_detail','')}")
    rounds = sorted(run.glob('round_[0-9][0-9][0-9]'))
    if not rounds:
        return dict(result, phase='INITIALIZING', message='正在初始化：采图、随机抓起并放下，随后生成第一轮候选。')
    folder = rounds[-1]; result['round'] = folder.name
    review = read(folder/'preference_review/review_status.json')
    if review.get('status') == 'WAITING' and not (folder/'preference_review/choice.json').exists():
        try:
            address = upstream(review['url'])
            if check_connection:
                with LOCAL.open(address+'/', timeout=.5) as response:
                    body = response.read().decode()
                    if review['review_id'] not in body:
                        raise ValueError('review server does not match persisted review')
            return dict(result, phase='WAITING', message='候选已就绪：设置优先分组（组内并列），选择本轮执行动作，然后提交一次。',
                        ready=True, review_id=review['review_id'], upstream=address)
        except (ValueError, OSError, URLError, KeyError):
            return dict(result, phase='RECONNECTING', message='审核服务暂未连接，页面会自动重试。')
    if (folder/'preference_review/choice.json').exists():
        return dict(result, phase='AFTER_CHOICE', message='已提交本轮反馈，正在训练、检查或执行。下一轮候选就绪后自动显示。')
    if (folder/'residual_features').exists():
        return dict(result, phase='FEATURES', message='正在提取视觉特征并计算修正排名。')
    if (folder/'trajectory_scores').exists():
        return dict(result, phase='PT', message='正在评分 Pa→Pb 的 Pt。')
    return dict(result, phase='PS', message='正在采图并生成抓点 Pa 与 Ps。')


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One-shot 常驻审核</title><style>body{margin:0;font-family:system-ui;background:#f3f5f9;color:#182235}
header{padding:18px 24px;background:white;border-bottom:1px solid #ccd3dc}h1{font-size:22px;margin:0 0 8px}
#detail{color:#637080;font-size:14px}#message{font-weight:600}iframe{width:100%;height:calc(100vh - 180px);border:0;background:#f3f5f9}
#framewrap{position:relative}#cover{display:none;position:absolute;inset:0;background:#f3f5f9e8;z-index:5;padding:36px;font-size:20px;pointer-events:auto}
pre{white-space:pre-wrap;max-height:330px;overflow:auto;font-size:14px}details{margin-top:10px}a{color:#1457a6}</style>
<header><h1>One-shot 常驻审核</h1><div id="message">正在连接…</div><p id="detail"></p>
<details><summary>查看已完成轮次的训练前效果</summary><pre id="progress">尚无评估。</pre></details></header>
<div id="framewrap"><iframe id="review" title="本轮候选动作"></iframe><div id="cover"></div></div>
<script>
let shown=null,loading=false;const frame=document.querySelector('#review'),cover=document.querySelector('#cover');
let frameObserver=null;
frame.addEventListener('load',()=>{
 if(frameObserver)frameObserver.disconnect();
 const doc=frame.contentDocument;
 if(shown&&!doc.querySelector('.cards')){shown=null;return;}
 if(doc&&doc.body){frameObserver=new ResizeObserver(()=>{frame.style.height=(doc.body.scrollHeight+30)+'px';});frameObserver.observe(doc.body);}
});
async function poll(){if(loading)return;loading=true;
 try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('status unavailable');const s=await response.json();
 document.querySelector('#message').textContent=s.message;
 document.querySelector('#detail').textContent=[s.run,s.round,'每2秒检查状态；正在编辑时不会刷新候选'].filter(Boolean).join(' · ');
 document.querySelector('#progress').textContent=s.progress||'尚无后续新状态评估。';
 if(s.ready){
  cover.style.display='none';
  if(shown!==s.review_id){shown=s.review_id;frame.src='/review/'+encodeURIComponent(s.review_id)+'/';window.scrollTo(0,0);}
 }else{cover.style.display='block';cover.textContent=s.message;}
 }catch(error){document.querySelector('#message').textContent='连接暂时中断，正在自动重连；已编辑的候选页面保留。';}
 finally{loading=false;}}
poll();setInterval(poll,2000);
</script></html>'''


def handler(reports):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def send(self, code, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes): body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':return self.send(200,PAGE.encode(),'text/html; charset=utf-8')
            if path=='/health':return self.send(200,dict(service=SERVICE,reports=str(Path(reports).resolve())))
            if path=='/api/status':return self.send(200,status(reports))
            match=re.fullmatch(r'/review/([a-f0-9]{64})/(|index\.html|current\.png|goal\.png|P\d{2}[RDLU]\.png)',path)
            if not match:return self.send(404,dict(error='not found'))
            current=status(reports,check_connection=False)
            if not current['ready'] or match[1]!=current['review_id']:
                return self.send(409,dict(error='review is closed or replaced; wait for next round'))
            try:
                with LOCAL.open(current['upstream']+'/'+match[2],timeout=2) as response:
                    body=response.read();content_type=response.headers.get('Content-Type','application/octet-stream')
                if match[2] in ('','index.html'):
                    # Bind relative image requests to this review, including across transitions.
                    base=f'<base href="/review/{match[1]}/">'.encode()
                    body=body.replace(b'<meta charset="utf-8">',b'<meta charset="utf-8">'+base,1)
                return self.send(200,body,content_type)
            except (OSError,URLError):return self.send(503,dict(error='review server unavailable'))

        def do_POST(self):
            origin=f'http://127.0.0.1:{self.server.server_port}'
            if self.path!='/choice':return self.send(404,dict(error='not found'))
            if self.headers.get('Origin')!=origin:
                return self.send(403,dict(error='origin rejected'))
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('invalid request size')
                body=self.rfile.read(length);choice=json.loads(body)
                current=status(reports,check_connection=False)
                if not current['ready'] or choice.get('review_id')!=current['review_id']:
                    raise ValueError('stale or closed review')
                request=Request(current['upstream']+'/choice',data=body,
                    headers={'Content-Type':'application/json','Origin':current['upstream']},method='POST')
                # The original server validates token/order/scene and persists exactly one choice.
                with LOCAL.open(request,timeout=3) as response:
                    return self.send(response.status,response.read())
            except HTTPError as error:return self.send(error.code,error.read())
            except (ValueError,TypeError,AttributeError):return self.send(400,dict(error='invalid or stale choice'))
            except (OSError,URLError):return self.send(503,dict(error='connection interrupted; inspect status before retrying'))
    return Handler


def ensure(port, reports):
    expected=dict(service=SERVICE,reports=str(Path(reports).resolve()))
    def healthy():
        try:
            with LOCAL.open(f'http://127.0.0.1:{port}/health',timeout=.5) as response:
                data=json.loads(response.read())
            if data!=expected:raise RuntimeError('dashboard port belongs to a different service')
            return True
        except (OSError,URLError,ValueError):return False
    if healthy():return
    folder=Path(reports)/'preference_dashboard';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'server.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-B','-u','-m','cloth_agent.preference_dashboard',
            '--port',str(port),'--reports',str(Path(reports).resolve())],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    for _ in range(50):
        if healthy():return
        if child.poll() is not None:break
        time.sleep(.1)
    raise RuntimeError('persistent dashboard did not start; see reports/preference_dashboard/server.log')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=PORT)
    parser.add_argument('--reports',type=Path,default=ROOT/'reports')
    parser.add_argument('--ensure',action='store_true')
    args=parser.parse_args()
    if args.ensure:
        ensure(args.port,args.reports)
        print(f'PERSISTENT REVIEW http://127.0.0.1:{args.port}',flush=True)
        return
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(args.reports))
    server.daemon_threads=True
    original=server.get_request
    def get_request():
        sock,address=original();sock.settimeout(3);return sock,address
    server.get_request=get_request
    print(f'PERSISTENT REVIEW http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever(poll_interval=.25)
    finally:server.server_close()


if __name__=='__main__':main()
