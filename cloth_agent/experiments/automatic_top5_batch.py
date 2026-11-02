"""Finite automatic Top5 episodes, operator-controlled stop and next reset."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import random
import math
from pathlib import Path
import secrets
import shutil
import signal
import sys
import threading
import time
from urllib.parse import unquote, urlparse

from . import one_shot_goal_continuous_v1 as base

PAGE=r'''<!doctype html><meta charset="utf-8"><title>自动展开采集</title>
<style>body{font:19px system-ui;max-width:1200px;margin:32px auto;padding:20px;background:#eef2f6;color:#182235}button{font:inherit;margin:8px;padding:16px;border:0;border-radius:8px;cursor:pointer}#stop,#all{background:#bd2338;color:white}#next{background:#127450;color:white}button:disabled{opacity:.35;cursor:default}pre{white-space:pre-wrap;background:white;padding:20px}iframe{width:100%;height:900px;border:0}a{display:block;margin:10px}</style>
<h1>自动展开 · 3次采集</h1><p>执行GPT选择，不使用学习参数。每次结束后，手动整理并确认才开始下一次随机初始化。</p>
<button id="stop" onclick="act('stop_current')">终止当前次</button><button id="all" onclick="act('stop_all')">结束整批</button>
<button id="next" onclick="act('next')">已整理好，开始下一次随机初始化</button>
<p>“终止当前次”会请求软件停止并等待机器人清理，不是硬件急停；紧急情况使用现场急停。</p><p id="notice"></p><pre id="status"></pre><div id="links"></div><iframe id="view"></iframe>
<script>let state={}, shown='';
async function refresh(){try{state=await(await fetch('/api/status',{cache:'no-store'})).json();
const phases={STARTING:'正在准备',RUNNING:'运行中',STOPPING:'正在停止并清理',WAITING_READY:'等待你整理并确认',STARTING_NEXT:'准备开始随机初始化',FINISHED:'整批已结束'};
document.querySelector('#status').textContent=[`状态：${phases[state.phase]||state.phase}`,`输入：${state.stateful_context?'当前图＋目标图＋上一轮动作前图及已执行动作':'当前图＋目标图'}`,state.episode===0?'尚未开始第1次':`当前：第 ${state.episode} / ${state.episodes_limit} 次`,`本次已记录轮数：${state.active_rounds||0}；已派发动作：${state.actions_dispatched||0}`,state.last_termination?`上次结束原因：${state.last_termination}`:'',...(state.completed||[]).map(r=>`第${r.episode}次：${r.termination}，${r.actions}个动作${r.detail?'；'+r.detail:''}`)].filter(Boolean).join('\n');
document.querySelector('#next').textContent=state.episode===0?'现场与控制器已就绪，开始第1次随机初始化':'已整理好，开始下一次随机初始化';
document.querySelector('#stop').disabled=!['RUNNING','STOPPING'].includes(state.phase);
document.querySelector('#next').disabled=state.phase!=='WAITING_READY';document.querySelector('#all').disabled=state.phase==='FINISHED';
let links=document.querySelector('#links');links.replaceChildren();for(let r of state.reports||[]){let a=document.createElement('a');a.href=r.url;a.textContent=r.label;a.target='_blank';links.appendChild(a)}
if(state.latest_review&&state.latest_review!==shown){shown=state.latest_review;document.querySelector('#view').src=shown;}
}catch(e){document.querySelector('#notice').textContent='连接暂时不可用：'+e}}
async function act(command){try{let r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,token:state.token,episode:state.episode})});let data=await r.json();document.querySelector('#notice').textContent=data.message||data.error;await refresh()}catch(e){document.querySelector('#notice').textContent=String(e)}}
refresh();setInterval(refresh,2000);</script>'''


class Control(base.Session):
    def __init__(self,root,limit,stateful_context=False):
        super().__init__(root)
        self.lock=threading.RLock();self.ready=threading.Event();self.limit=limit
        self.stateful_context=bool(stateful_context)
        self.phase='STARTING';self.episode=0;self.token=secrets.token_urlsafe(24)
        self.records=[];self.last_stop=None;self.episode_stop=False

    def transition(self,phase,episode=None):
        with self.lock:
            self.phase=phase
            if episode is not None:self.episode=episode
            self.ready.clear();self.token=secrets.token_urlsafe(24)

    def command(self,body):
        with self.lock:
            if body.get('token')!=self.token or body.get('episode')!=self.episode:
                raise ValueError('页面状态已更新，请刷新后重试')
            cmd=body.get('command')
            if cmd=='stop_all':
                self.request_stop(signal.SIGINT);self.ready.set()
                self.event('OPERATOR_STOP_ALL');return '已请求结束整批，等待停止与清理。'
            if cmd=='stop_current':
                if self.phase not in ('RUNNING','STOPPING'):raise ValueError('当前没有运行中的采集')
                self.episode_stop=True;self.phase='STOPPING';self.stop_signal=signal.SIGINT
                self.forward_stop();self.event('OPERATOR_STOP_CURRENT',episode=self.episode)
                return '已请求终止当前次；清理完成后等待你整理衣服。'
            if cmd=='next':
                if self.phase!='WAITING_READY' or self.episode>=self.limit:raise ValueError('当前不能开始下一次')
                self.phase='STARTING_NEXT';self.ready.set();self.token=secrets.token_urlsafe(24)
                self.event('OPERATOR_READY_NEXT',after_episode=self.episode)
                return '已确认整理完成，将从随机初始化开始下一次。'
            raise ValueError('未知指令')

    def forward_stop(self):
        # The launcher calls this on cancellation; current-stop also calls it.
        super().forward_stop()

    def status(self):
        with self.lock:
            result=dict(phase=self.phase,episode=self.episode,episodes_limit=self.limit,
                token=self.token,last_termination=self.last_stop,completed=self.records,
                alpha_used=False,residual_used=False,learning=False,
                stateful_context=self.stateful_context)
        reports=[]
        for ep in sorted(self.root.glob('episode_[0-9][0-9][0-9][0-9]')):
            for f in sorted(list(ep.glob('round_*/top5.html'))+list(ep.glob('round_*/decision.html'))):
                reports.append(dict(label=ep.name+' / '+f.parent.name,url='/files/'+str(f.relative_to(self.root))))
        result['reports']=reports
        if reports:
            # Refresh same-round evidence after execution, without stealing browser focus.
            f=self.root/reports[-1]['url'].removeprefix('/files/')
            result['latest_review']=reports[-1]['url']+'?v='+str(f.stat().st_mtime_ns)
        current=self.root/f'episode_{self.episode:04d}'/'episode_summary.json'
        try:
            record=base.json.loads(current.read_text());result['active_rounds']=record['total_rounds'];result['actions_dispatched']=record['physical_actions']
        except (OSError,ValueError,KeyError):pass
        return result


def serve(control,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,body,kind='application/json; charset=utf-8'):
            data=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            route=urlparse(self.path).path
            if route=='/':return self.send(200,getattr(control,'page',PAGE).replace('3次采集',f'{control.limit}次采集').encode(),'text/html; charset=utf-8')
            if route=='/api/status':return self.send(200,control.status())
            if route.startswith('/files/'):
                file=(control.root/unquote(route[7:])).resolve()
                if control.root not in file.parents or not file.is_file() or file.suffix not in ('.html','.png','.jpg','.txt'):
                    return self.send(404,dict(error='not found'))
                return self.send(200,file.read_bytes(),mimetypes.guess_type(file)[0] or 'application/octet-stream')
            self.send(404,dict(error='not found'))
        def do_POST(self):
            if self.path!='/api/control':return self.send(404,dict(error='not found'))
            if self.headers.get('Origin')!=f'http://127.0.0.1:{self.server.server_port}':return self.send(403,dict(error='origin rejected'))
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<4096:raise ValueError('invalid size')
                msg=control.command(json.loads(self.rfile.read(n)));self.send(200,dict(message=msg))
            except (ValueError,TypeError) as exc:self.send(409,dict(error=str(exc)))
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server


def run(args):
    if getattr(args,"depth_image",False) and not getattr(args,"rgb_only",False):raise ValueError("--depth-image requires --rgb-only")
    if getattr(args,"rgb_only",False) and (not getattr(args,"visible_sleeve_hem",False) or getattr(args,"cv_feedback",False) or getattr(args,"ggo_reference_annotations",None)):
        raise ValueError("--rgb-only requires original sleeve/hem mode without CV feedback or GGO")
    if getattr(args,'hem_sleeve_pull_rule',False) and not getattr(args,'visible_sleeve_hem',False):
        raise ValueError('--hem-sleeve-pull-rule requires --visible-sleeve-hem')
    radius_range=getattr(args,'initialize_center_radius_range_mm',None)
    if radius_range is not None and not (len(radius_range)==2 and 0<=radius_range[0]<=radius_range[1]<=80):
        raise ValueError('reset radius range must satisfy 0 <= min <= max <= 80 mm')
    if getattr(args,'ggo_reference_annotations',None) and (not args.visible_sleeve_hem or (args.max_rounds!=1 and not getattr(args,'ggo_auto_anchors',False))):
        raise ValueError('GGO batch requires sleeve/hem mode and one action per episode')
    if getattr(args,'visible_sleeve_hem',False):
        args.stateful_context=True
        if getattr(args,'resume_first_from',None):raise ValueError('sleeve/hem experiment must start a new episode')
    if getattr(args,'max_actions',0) and args.max_rounds<args.max_actions+1:
        raise ValueError('max-rounds must include final action assessment')
    hours=getattr(args,'max_hours',0.)
    min_free=getattr(args,'min_free_gb',0.)
    if not math.isfinite(hours) or hours<0 or not math.isfinite(min_free) or min_free<0:raise ValueError('invalid duration or disk limit')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    control=Control(root,args.episodes,getattr(args,'stateful_context',False));server=serve(control,args.port)
    if getattr(args,'ggo_reference_annotations',None) and not getattr(args,'ggo_auto_anchors',False):
        control.page=PAGE.replace('<h1>自动展开 · 3次采集</h1>',f'<h1>GGO 几何辅助 · 3次单步实验</h1><p><a target="_blank" href="http://127.0.0.1:{args.ggo_port}">打开本轮锚点标注页（拍照后可用）</a></p>')
    if getattr(args,'auto_next',False):
        control.page=PAGE.replace('每次结束后，手动整理并确认才开始下一次随机初始化。','DONE或达到动作上限后自动reset；异常或人工停止会结束整批。').replace('自动展开 · 3次采集','自动连续展开 · 3次采集')
    resume_first=getattr(args,'resume_first_from',None)
    if resume_first is not None:
        resume_first=resume_first.resolve()
        source_summary=resume_first/'episode_summary.json'
        if not source_summary.is_file() or not (resume_first/'goal/target.png').is_file():
            raise ValueError('resume source lacks episode summary or goal image')
        if base.json.loads(source_summary.read_text()).get('termination_reason')!='SAFETY_STOP':
            raise ValueError('first episode continuation requires SAFETY_STOP source')
    (root/'goal').mkdir();shutil.copyfile(resume_first/'goal/target.png' if resume_first else args.goal_image,
                                          root/'goal/target.png')
    files=base.SOURCE_FILES+['cloth_agent/geometry_report.py','cloth_agent/one_shot_depth_recovery.py','cloth_agent/experiments/arm6_depth_recovery.py']+['cloth_agent/restricted_context_one_shot.py','cloth_agent/automatic_top5.py','cloth_agent/preference_one_shot.py','cloth_agent/action_preference.py','cloth_agent/experiments/automatic_top5_batch.py']
    if getattr(args,'ggo_reference_annotations',None):
        files += [str(p.relative_to(base.ROOT)) for p in (base.ROOT/'cloth_agent/ggo').glob('*.py')]
    if getattr(args,'cv_feedback',False):
        files += [str(p.relative_to(base.ROOT)) for sub in ('runtime','cv_harness') for p in (base.ROOT/'cloth_agent/cloth_harness'/sub).glob('*.py')]
    if getattr(args,'systematic_report',False):files+=['cloth_agent/experiments/overnight_analysis.py']
    files+=['cloth_agent/random_fold_reset.py','cloth_agent/hem_sleeve_pull_rule.py','cloth_agent/depth_image_input.py']
    hashes={str(base.ROOT/p):base.digest(base.ROOT/p) for p in files}
    if getattr(args,'ggo_reference_annotations',None):hashes[str(args.ggo_reference_annotations.resolve())]=base.digest(args.ggo_reference_annotations)
    hashes[str(args.config.resolve())]=base.digest(args.config)
    hashes[str(root/'goal/target.png')]=base.digest(root/'goal/target.png')
    if resume_first is not None:
        hashes[str(source_summary)]=base.digest(source_summary)
    schedule_seed=getattr(args,'schedule_seed',None)
    if schedule_seed is None:schedule_seed=secrets.randbits(63)
    rng=random.Random(schedule_seed)
    schedule=[dict(episode=i,radius_mm=rng.uniform(*radius_range) if radius_range else getattr(args,'initialize_center_radius_mm',20.),reset_seed=rng.getrandbits(63)) for i in range(1,args.episodes+1)]
    base.save(root/'reset_schedule.json',dict(seed=schedule_seed,schedule=schedule,physical_state_reproducible_from_seed=False))
    base.save(root/'protocol.json',dict(depth_image=getattr(args,'depth_image',False),rgb_only=getattr(args,'rgb_only',False),hem_sleeve_pull_rule=getattr(args,'hem_sleeve_pull_rule',False),cv_feedback=getattr(args,'cv_feedback',False),max_hours=hours,min_free_gb=min_free,schedule_seed=schedule_seed,systematic_report=getattr(args,'systematic_report',False),ggo_manual_geometry=bool(getattr(args,'ggo_reference_annotations',None)) and not getattr(args,'ggo_auto_anchors',False),ggo_auto_anchors=getattr(args,'ggo_auto_anchors',False),created_at=base.utc(),episodes=args.episodes,max_rounds=args.max_rounds,max_actions=getattr(args,'max_actions',0),
        mode='context one-shot, visible sleeve/hem only, no TopN' if getattr(args,'visible_sleeve_hem',False) else 'automatic Top5, explicit GPT selection',between_episodes='automatic after normal completion; stop on error' if getattr(args,'auto_next',False) else 'operator confirmation required',
        wait_before_first=getattr(args,'wait_before_first',False),reset_enabled=not getattr(args,'no_reset',False),
        resume_first_from=str(resume_first) if resume_first else None,
        stateful_context=bool(getattr(args,'stateful_context',False)),
        initialize_center_radius_mm=float(getattr(args,'initialize_center_radius_mm',20.)),
        initialize_center_radius_range_mm=radius_range,
        press_after_drop=bool(getattr(args,'press_after_drop',False)),
        no_learning=True,source_sha256=hashes,control_url=f'http://127.0.0.1:{args.port}'))
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in handlers:signal.signal(s,control.request_stop)
    print(f'CONTROL http://127.0.0.1:{args.port} OUTPUT {root}',flush=True)
    def checkpoint():
        st=control.status();st.pop('token',None);base.save(root/'batch_summary.json',st)
    try:
        if getattr(args,'wait_before_first',False):
            control.transition('WAITING_READY',0);checkpoint()
            print('WAITING_READY: 尚未连接机器人；确认现场与控制器就绪后，从随机初始化开始第1次。',flush=True)
            while not control.ready.wait(.5):
                if control.cancel.is_set():break
        run_started=time.monotonic()
        for index in range(1,args.episodes+1):
            if hours and time.monotonic()-run_started>=hours*3600:
                control.last_stop='DURATION_LIMIT';control.event('BATCH_DURATION_LIMIT');break
            free_gb=shutil.disk_usage(root).free/1024**3
            if min_free and free_gb<min_free:
                control.last_stop='DISK_RESERVE_STOP';control.event('DISK_RESERVE_STOP',free_gb=free_gb);break
            if control.cancel.is_set():break
            if any(base.digest(p)!=v for p,v in hashes.items()):raise RuntimeError('batch source/input changed; no next dispatch')
            control.episode_stop=False;control.stop_signal=None;control.transition('RUNNING',index);checkpoint()
            name=f'episode_{index:04d}';folder=root/name
            command=[sys.executable,'-u','-m','cloth_agent.experiments.one_shot_goal_full_episode_v1','--execute',
                '--visible-sleeve-hem' if getattr(args,'visible_sleeve_hem',False) else '--automatic-top5','--output',str(folder),'--goal-image',str(root/'goal/target.png'),
                '--config',str(args.config.resolve()),'--max-rounds',str(args.max_rounds)]
            if getattr(args,'ggo_reference_annotations',None):
                command+=['--ggo-reference-annotations',str(args.ggo_reference_annotations.resolve()),'--ggo-port',str(args.ggo_port)]
            if getattr(args,'depth_image',False):command.append('--depth-image')
            if getattr(args,'rgb_only',False):command.append('--rgb-only')
            if getattr(args,'hem_sleeve_pull_rule',False):command.append('--hem-sleeve-pull-rule')
            if getattr(args,'cv_feedback',False):command.append('--cv-feedback')
            if getattr(args,'no_reset',False):command.append('--no-reset')
            if getattr(args,'ggo_auto_anchors',False):command.append('--ggo-auto-anchors')
            if getattr(args,'max_actions',0):command+=['--max-actions',str(args.max_actions)]
            if index==1 and resume_first is not None:command+=['--continue-from',str(resume_first)]
            if getattr(args,'stateful_context',False):command.append('--stateful-context')
            selected=schedule[index-1];radius=selected['radius_mm']
            command+=['--initialize-seed',str(selected['reset_seed'])]
            base.save(root/(name+'.reset_radius.json'),dict(enabled=not getattr(args,'no_reset',False),radius_mm=radius,range_mm=radius_range,reset_seed=selected['reset_seed'],
                distribution='uniform radius bound' if radius_range else 'fixed radius bound'))
            print('RESET disabled; using manually arranged state' if getattr(args,'no_reset',False) else f'RESET radius={radius:.2f} mm',flush=True)
            command+=['--initialize-center-radius-mm',str(radius)]
            if getattr(args,'press_after_drop',False):command.append('--press-after-drop')
            # Handle a stop pressed between state publication and child creation.
            def watch_stop():
                while control.phase in ('RUNNING','STOPPING') and control.child is None and not control.cancel.is_set():time.sleep(.02)
                if control.episode_stop:control.forward_stop()
            threading.Thread(target=watch_stop,daemon=True).start()
            episode_started=time.monotonic()
            control.event('EPISODE_START',episode=index,radius_mm=radius,reset_seed=selected['reset_seed'],free_gb=free_gb)
            code=base.launch_episode(command,root/(name+'.run.log'),control)
            record=base.json.loads((folder/'episode_summary.json').read_text()) if (folder/'episode_summary.json').exists() else {}
            control.last_stop=record.get('termination_reason','MISSING_SUMMARY')
            control.records.append(dict(episode=index,termination=control.last_stop,detail=record.get('termination_detail'),
                actions=record.get('physical_actions',0),rounds=record.get('total_rounds',0),returncode=code,
                operator_stopped_current=control.episode_stop,cleanup_error=record.get('cleanup_error')))
            control.records[-1].update(radius_mm=radius,reset_seed=selected['reset_seed'],elapsed_s=time.monotonic()-episode_started)
            control.event('EPISODE_END',**control.records[-1])
            checkpoint()
            if getattr(args,'systematic_report',False):
                from .overnight_analysis import analyze
                analyze(root)
            if control.cancel.is_set() or index==args.episodes:break
            if not record.get('finished_at') or record.get('cleanup_error'):
                raise RuntimeError('episode cleanup not confirmed; batch stopped')
            if getattr(args,'auto_next',False):
                if code!=0 or control.episode_stop or control.last_stop not in ('ROUND_LIMIT','DONE'):
                    print('AUTO BATCH STOP: '+str(control.last_stop),flush=True)
                    break
                print('AUTO NEXT: starting next random initialization',flush=True)
                continue
            control.transition('WAITING_READY');checkpoint()
            print('WAITING_READY: 手动整理后，点击控制页开始下一次随机初始化。',flush=True)
            while not control.ready.wait(.5):
                if control.cancel.is_set():break
            if control.cancel.is_set():break
    except BaseException as exc:
        control.request_stop(signal.SIGINT);control.last_stop=f'{type(exc).__name__}: {exc}'
        base.save(root/'batch_error.json',dict(error=control.last_stop))
        raise
    finally:
        control.transition('FINISHED');checkpoint()
        if getattr(args,'systematic_report',False):
            try:
                from .overnight_analysis import analyze
                analyze(root)
            except Exception as exc:base.save(root/'analysis_error.json',dict(error=f'{type(exc).__name__}: {exc}'))
        for s,h in handlers.items():signal.signal(s,h)
        server.shutdown();server.server_close()
        print('BATCH FINISHED '+str(root),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=3);p.add_argument('--max-rounds',type=int,default=12)
    p.add_argument('--max-hours',type=float,default=0.,help='do not start a new episode after this duration; finish current episode')
    p.add_argument('--min-free-gb',type=float,default=0.)
    p.add_argument('--schedule-seed',type=int)
    p.add_argument('--systematic-report',action='store_true')
    p.add_argument('--max-actions',type=int,default=0)
    p.add_argument('--no-reset',action='store_true',help='skip random initialization in every episode; use --episodes 1 for a manual-state trial')
    p.add_argument('--wait-before-first',action='store_true',help='serve control page; do not connect robot until operator confirms first start')
    p.add_argument('--resume-first-from',type=Path,help='continue episode 1 from a stopped automatic Top5 run without another reset')
    p.add_argument('--ggo-reference-annotations',type=Path)
    p.add_argument('--depth-image',action='store_true')
    p.add_argument('--rgb-only',action='store_true')
    p.add_argument('--hem-sleeve-pull-rule',action='store_true')
    p.add_argument('--cv-feedback',action='store_true')
    p.add_argument('--ggo-auto-anchors',action='store_true')
    p.add_argument('--auto-next',action='store_true',help='continue normal episodes automatically; stop batch on errors')
    p.add_argument('--ggo-port',type=int,default=8781)
    p.add_argument('--visible-sleeve-hem',action='store_true',help='contextual single-action sleeve/hem experiment without TopN')
    p.add_argument('--stateful-context',action='store_true',help='give Ps and Pt the previous executed action and pre-action RGB from round 2')
    p.add_argument('--initialize-center-radius-mm',type=float,default=20.,
                   help='random initialization grasp radius around the visible cloth center (default 20 mm)')
    p.add_argument('--initialize-center-radius-range-mm',type=float,nargs=2,metavar=('MIN','MAX'),
                   help='sample the reset radius bound uniformly per episode; overrides fixed radius')
    p.add_argument('--press-after-drop',action='store_true')
    p.add_argument('--port',type=int,default=8769);p.add_argument('--goal-image',type=Path,default=base.DEFAULT_GOAL)
    p.add_argument('--config',type=Path,default=base.ROOT/'config/dual_arm.local.json')
    a=p.parse_args()
    if not 1<=a.episodes<=1000 or not 1<=a.max_rounds<=50 or not 1024<=a.port<=65535:p.error('invalid limits')
    if not 0<=a.initialize_center_radius_mm<=80:p.error('initialization center radius must be in [0,80] mm')
    if a.output.exists() or not a.goal_image.is_file() or not a.config.is_file():p.error('invalid output, goal or config')
    run(a)


if __name__=='__main__':main()
