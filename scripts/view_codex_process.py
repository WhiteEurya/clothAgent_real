#!/usr/bin/env python3
"""Read-only live view of public Codex outputs and robot run artifacts."""
from pathlib import Path
import argparse
import base64
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    try:return path.read_text(errors='replace')
    except FileNotFoundError:return ''


def data(path):
    try:return json.loads(read(path))
    except (ValueError,OSError):return {}


def block(title,text):
    return f'<details><summary>{html.escape(title)}</summary><pre>{html.escape(text)}</pre></details>'


def picture(path,label):
    if not path.is_file():return f'<figure><figcaption>{html.escape(label)}：等待图像</figcaption></figure>'
    encoded=base64.b64encode(path.read_bytes()).decode()
    return f'<figure><figcaption>{html.escape(label)}</figcaption><img src="data:image/png;base64,{encoded}"></figure>'


def render(root):
    sections=[]
    for folder in sorted(root.glob('round_*')):
        outcome=data(folder/'outcome.json');review=data(folder/'reflection.json')
        stats=data(folder/'planning_stats.json')
        section=f'<section><h2>{html.escape(folder.name)}</h2>'
        section+=f'<p>执行：{html.escape(outcome.get("status","等待结果"))} · 已完成指令：{outcome.get("completed_commands","—")} · 细分/下发：{stats.get("checked_segments","—")}/{stats.get("dispatched_segments","—")}</p>'
        section+='<div class="images">'+picture(folder/'before/rgb.png','动作前')+picture(folder/'after/rgb.png','动作后')+'</div>'
        for role in ('planner','reflection'):
            metrics=data(folder/role/'result.json')
            section+=f'<h3>{"规划" if role=="planner" else "反思"} · {html.escape(str(metrics.get("status","进行中或尚未开始")))} · {metrics.get("duration_s",0):.1f}s</h3>'
            for line in read(folder/role/'events.jsonl').splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('type')!='item.completed':continue
                item=event.get('item',{})
                if item.get('type')=='agent_message':
                    section+='<pre class="message">'+html.escape(item.get('text',''))+'</pre>'
                elif item.get('type')=='command_execution':
                    section+=block('工具调用 · '+str(item.get('exit_code','')),item.get('command','')+'\n\n'+item.get('aggregated_output',''))
            section+=block('本轮完整提示词',read(folder/role/'prompt.txt'))
            section+=block('耗时及 token 用量',json.dumps(metrics,ensure_ascii=False,indent=2))
        if review:section+=block('结构化反思',json.dumps(review,ensure_ascii=False,indent=2))
        section+=block('模型动作代码',read(folder/'motion.py'))
        if (folder/'effective_motion.py').exists():
            section+=block('规划器实际接收的动作代码',read(folder/'effective_motion.py'))
            section+=block('抓取高度调整记录',read(folder/'grasp_height_policy.json'))
        section+=block('实际执行结果',read(folder/'outcome.json'))
        for f in sorted(folder.glob('gripper_*.json')):section+=block(f.name,read(f))
        section+=block('控制器事件',read(folder/'controller/events.jsonl'))
        section+='</section>';sections.append(section)
    refs=data(root/'references.json')
    images=''
    if isinstance(refs,list):
        for row in refs:
            # Serve only staged reference images, never arbitrary manifest paths.
            p=root/'references'/Path(row['path']).name
            images+=picture(p,row['role'])
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Astra 过程查看</title>
<style>body{font:16px system-ui;background:#f3f5f7;color:#192333;max-width:1400px;margin:24px auto;padding:0 20px}section{background:white;padding:22px;margin:20px 0;border-radius:12px}.images{display:flex;gap:16px;flex-wrap:wrap}figure{flex:1;min-width:260px;margin:0}img{width:100%;max-height:550px;object-fit:contain;background:#111}figcaption{padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;background:#f5f7fa;padding:12px}summary{cursor:pointer;padding:10px}.message{border-left:3px solid #3678cd}button{padding:8px 16px}</style>
<h1>Astra 可观察过程</h1><p>这里显示图像、公开过程说明、工具调用、动作代码和实测记录，不是模型内部隐藏推理。图像是前后快照，不是动作视频。</p>
<button onclick="location.reload()">刷新</button> <label><input id="auto" type="checkbox">每 10 秒刷新</label>
<script>const a=document.getElementById('auto');a.checked=sessionStorage.auto==='1';a.onchange=()=>sessionStorage.auto=a.checked?'1':'0';setInterval(()=>{if(a.checked)location.reload()},10000)</script>
'''+'<p>'+html.escape(str(root))+'</p><div class="images">'+images+'</div>'+''.join(sections)+'</html>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--port',type=int,default=8769)
    parser.add_argument('--export',type=Path)
    args=parser.parse_args()
    runs=sorted((ROOT/'results/codex_astra').glob('loop_*'))
    root=(args.run or (runs[-1] if runs else ROOT/'results/codex_astra')).resolve()
    if args.export:
        args.export.write_text(render(root));print(args.export);return
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/','/index.html'):
                self.send_error(404);return
            content=render(root).encode()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(content)))
            self.end_headers();self.wfile.write(content)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'http://127.0.0.1:{args.port}  run={root}',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
