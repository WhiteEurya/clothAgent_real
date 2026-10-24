"""Read-only archive of October 20 interrupted unfolding experiments and images."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import html
import json
import os
from PIL import Image, ImageDraw, ImageFont

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'reports/interrupted_experiments_20261020_archive'
TZ=timezone(timedelta(hours=8))


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def link(path):
    return os.path.relpath(path,OUT) if path and path.exists() else None


def local(value):
    return datetime.fromisoformat(value).astimezone(TZ).isoformat()


def entry(folder,label,kind='round'):
    result=read(folder/('execution.json' if kind=='round' else 'result.json'),{})
    plan=read(folder/'plan.json',{})
    before=folder/'before/rgb.png';after=folder/'after/rgb.png'
    tracks=[]
    for directory in sorted(folder.glob('**/execution_images')):
        frames=sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg'))
        if frames:tracks.append(dict(label=str(directory.parent.relative_to(folder)) or '.',frames=[link(p) for p in frames]))
    return dict(label=label,kind=kind,status=result.get('status','UNKNOWN'),result=result,
        before=link(before),after=link(after),overlay=link(folder/'proposed_action_overlay.png') or link(folder/'grasp_preview.png'),
        plan=plan,reason=plan.get('short_reason') or plan.get('short_rationale') or '',
        tracks=tracks,frame_count=sum(len(t['frames']) for t in tracks),
        evidence=link(folder/('execution.json' if kind=='round' else 'result.json')),
        plan_path=link(folder/'plan.json'),source=str(folder))


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=[];automatic=[]
    for f in sorted((BASE/'reports').glob('**/episode_summary.json')):
        if 'source_snapshot' in str(f):continue
        s=read(f,{})
        try:start=local(s['started_at'])
        except (KeyError,ValueError,TypeError):continue
        if not start.startswith('2026-10-20'):continue
        root=f.parent
        exception=(root/'exception.txt').read_text() if (root/'exception.txt').exists() else ''
        interrupted=s.get('termination_reason')=='USER_STOP' or 'KeyboardInterrupt' in exception
        if interrupted:
            entries=[]
            if (root/'initialization').exists():entries.append(entry(root/'initialization','初始化','initialization'))
            entries.extend(entry(p,p.name) for p in sorted(root.glob('round_*')) if p.is_dir())
            method='one-shot' if root.name.startswith('one_shot') else 'trajectory memory' if root.name.startswith('trajectory_memory') else 'goal-only'
            note=s.get('termination_detail') or 'exception.txt 记录 KeyboardInterrupt；episode 总结未正常写完。'
            if method=='trajectory memory':note='7 个展开动作完成；round_007 在动作前采图阶段中断，未调用模型或下发动作。'
            elif start[11:16]=='22:48':note='初始化抓放已发生，返回 perception 时中断；没有干净 after 图，episode 总结未正常写完。'
            elif start[11:16]=='18:15':note='初始化阶段中断；抓放记录显示 executed=true，但没有干净 after 图。'
            elif start[11:16]=='23:43':note='初始化移动阶段即中断；没有本次当前观测或过程帧。'
            runs.append(dict(name=root.name,start=start,method=method,note=note,entries=entries,
                summary=link(f),exception=link(root/'exception.txt'),goal=link(root/'goal/target.png')))
        elif s.get('termination_reason')=='SAFETY_STOP':
            folders=[p for p in sorted(root.glob('round_*')) if p.is_dir()]
            last=folders[-1] if folders else root/'initialization'
            automatic.append(dict(name=root.name,start=start,reason=s.get('termination_detail'),
                summary=link(f),image=link(last/'before/rgb.png'),overlay=link(last/'proposed_action_overlay.png'),
                round=last.name,completed=s.get('physical_actions',0)))
    # Earlier local-action experiment uses result.json, not episode_summary.json.
    root=BASE/'reports/arm6_local_policy_20261019T163743980525Z'
    result=read(root/'result.json',{})
    observation=read(root/'initial_observation/observation.json',{})
    if result.get('status')=='INTERRUPTED' and local(observation['captured_at']).startswith('2026-10-20'):
        plan=result.get('last_proposal',{})
        e=dict(label='step_0000（提议未执行）',kind='proposal',status='INTERRUPTED_BEFORE_EXECUTION',result=result,
            before=link(root/'step_0000/before.png'),after=None,overlay=link(root/'step_0000/proposal_preview.png'),
            plan=plan,reason=plan.get('reason',''),tracks=[],frame_count=0,evidence=link(root/'result.json'),plan_path=None,source=str(root/'step_0000'))
        runs.append(dict(name=root.name,start='2026-10-20T00:37:43.980525+08:00',method='local morphology action',
            note='在动作方案审阅时取消，physical_actions=0；已有当前观测和候选动作预览，未执行。开始时间来自目录中的 UTC 时间，观测时间也已核对。',
            entries=[e],summary=link(root/'result.json'),exception=link(root/'cli_exception.txt'),goal=None))
    runs.sort(key=lambda r:r['start']);automatic.sort(key=lambda r:r['start'])
    inventory=[];gallery=[]
    for ri,r in enumerate(runs):
        r['index']=ri
        r['frame_count']=sum(e['frame_count'] for e in r['entries'])
        r['completed_rounds']=sum(e['kind']=='round' and e['status']=='EXECUTED' for e in r['entries'])
        for ei,e in enumerate(r['entries']):
            e['index']=ei
            for role in ('before','after'):
                if not e[role]:continue
                path=(OUT/e[role]).resolve();sha=hashlib.sha256(path.read_bytes()).hexdigest()
                item=dict(run=r['name'],start=r['start'],entry=e['label'],role=role,status=e['status'],path=str(path),sha256=sha)
                inventory.append(item)
                gallery.append(dict(run_index=ri,entry_index=ei,path=e[role],caption=f"{r['start'][11:16]} · {e['label']} · {role}",sha256=sha))
            for track in e['tracks']:
                for path in track['frames']:
                    inventory.append(dict(run=r['name'],start=r['start'],entry=e['label'],role='execution_frame:'+track['label'],
                        status=e['status'],path=str((OUT/path).resolve()),sha256=''))
    clean=[i for i in inventory if i['role'] in ('before','after')]
    stats=dict(interrupted_runs=len(runs),completed_rounds=sum(r['completed_rounds'] for r in runs),
        clean_image_files=len(clean),unique_clean_image_hashes=len({i['sha256'] for i in clean}),
        execution_frames=sum(r['frame_count'] for r in runs),automatic_stop_runs=len(automatic))
    data=dict(date='2026-10-20',timezone='Asia/Singapore',runs=runs,automatic_stops=automatic,gallery=gallery,stats=stats,
        scope='Unfolding runs with explicit user-stop/KeyboardInterrupt evidence. Reference library images, goals and rendered overlays excluded from source-image counts. Safety stops listed separately.')
    (OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    with (OUT/'image_inventory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['run','start','entry','role','status','path','sha256']);writer.writeheader();writer.writerows(inventory)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    complete=next(r for r in runs if r['method']=='trajectory memory')
    pairs=[e for e in complete['entries'] if e['kind']=='round' and e['before'] and e['after']]
    sheet=Image.new('RGB',(1000,len(pairs)*315),'white');draw=ImageDraw.Draw(sheet)
    for row,e in enumerate(pairs):
        for col,role in enumerate(('before','after')):
            im=Image.open(OUT/e[role]).convert('RGB');im.thumbnail((490,277));x=col*500;y=row*315
            draw.text((x+5,y+3),e['label']+' '+role+' | EXECUTED',fill='black',font=font);sheet.paste(im,(x+5,y+30))
    sheet.save(OUT/'seven_completed_rounds.png')
    thumbs=[]
    for i,g in enumerate(gallery):
        im=Image.open(OUT/g['path']).convert('RGB');im.thumbnail((400,225));dest=OUT/f'thumb_{i:02d}.jpg';im.save(dest,quality=85)
        thumbs.append(f'<button class="thumb" onclick="choose({g["run_index"]},{g["entry_index"]});document.getElementById(\'viewer\').scrollIntoView()"><img src="{dest.name}"><span>{html.escape(g["caption"])}</span></button>')
    summary_rows=[]
    for r in runs:
        clean_count=sum(bool(e['before'])+bool(e['after']) for e in r['entries'])
        summary_rows.append(f'<tr><td>{r["start"][11:19]}</td><td>{r["method"]}</td><td>{r["completed_rounds"]}</td><td>{clean_count}</td><td>{r["frame_count"]}</td><td>{html.escape(r["note"])}</td></tr>')
    auto_cards=[]
    for a in automatic:
        im=f'<a href="{a["image"]}"><img src="{a["image"]}" loading="lazy"></a>' if a['image'] else '<p>无已保存当前图</p>'
        auto_cards.append(f'<article>{im}<b>{a["start"][11:19]} · {html.escape(a["name"])}</b><p>{html.escape(a["reason"] or "")}</p><p>{a["round"]} · 已执行动作数 {a["completed"]}</p><a href="{a["summary"]}">原始记录</a></article>')
    default_run=complete['index'];default_entry=next(e['index'] for e in complete['entries'] if e['label']=='round_000')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>10 月 20 日中断实验与图像档案</title>
<style>body{font-family:system-ui,sans-serif;background:#f3f5f9;color:#182235;margin:24px}p{line-height:1.6}.panel,article{padding:18px;background:white;border-radius:12px;margin-bottom:18px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}select,button{padding:8px;cursor:pointer}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.pair img{width:100%;display:block}.gallery{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.thumb{text-align:left;border:0;background:white}.thumb img,article img{width:100%}.thumb span{display:block;padding:8px 0}.process{max-width:1000px}.process img{width:100%}input[type=range]{width:100%}.empty{padding:40px;background:#edf0f5;color:#59657a}a{color:#2059a3}#source{overflow-wrap:anywhere}.badge{background:#fff2ce;padding:6px 10px}.automatic{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}article b{display:block;overflow-wrap:anywhere}@media(max-width:900px){.pair{grid-template-columns:1fr}.gallery,.automatic{grid-template-columns:1fr 1fr}}</style>
<h1>2026-10-20：中断的展开实验与剩余图像</h1>
<p>按新加坡时间归档。这里区分主动停止、初始化中断与自动安全停止；“动作完成”不等于成功夹取或成功展开。没有新增模型推理或机器人执行。</p>
<div class="panel"><p><b>'''+f"{stats['interrupted_runs']} 次有主动中断证据的实验 · {stats['completed_rounds']} 组完整动作前后图 · {stats['clean_image_files']} 张静态观测文件（{stats['unique_clean_image_hashes']} 个不同文件哈希） · {stats['execution_frames']} 张执行过程帧"+'''</b></p>
<table><thead><tr><th>开始时间</th><th>方法</th><th>完成展开动作</th><th>静态观测</th><th>过程帧</th><th>停止位置与证据</th></tr></thead><tbody>'''+''.join(summary_rows)+'''</tbody></table>
<p><a href="seven_completed_rounds.png">17:40 实验：7 组前后图总览</a> · <a href="image_inventory.csv">全部原始图像索引 CSV</a> · <a href="data.json">结构化档案</a></p></div>
<div class="panel" id="viewer"><p>实验 <select id="run"></select> 阶段 <select id="entry"></select> <button id="previous">上一阶段</button> <button id="next">下一阶段</button></p><h2 id="title"></h2><p id="note"></p><p><span id="status" class="badge"></span></p><p id="reason"></p><p id="source"></p>
<div class="pair"><div><h3>动作前 / 当前观测</h3><div id="before"></div><button id="overlay">切换动作标注</button></div><div><h3>动作后静态观测</h3><div id="after"></div></div>
<details id="process"><summary>查看执行过程帧（可能有机械臂、视角变化或模糊，不是干净 after）</summary><div class="process"><p><select id="track"></select> <span id="frame-info"></span></p><input id="frame" type="range" min="0" value="0"><img id="frame-image"><p id="frame-link"></p></div></details></div>
<h2>保存下来的静态观测</h2><p>点击缩略图跳转对应实验与阶段。17:40 实验的 before/after 是同一轮的成对观测；初始化未完成 after 的记录不会用过程帧补成结果图。</p><div class="gallery">'''+''.join(thumbs)+'''</div>
<h2>补充：同日自动安全停止的记录</h2><p>下列记录也保存了可继续分析的图，但没有被归为用户主动中断。下图是停止阶段的当前观测，不是被拒绝动作执行后的结果。</p><div class="automatic">'''+''.join(auto_cards)+'''</div>
<script>const data='''+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+''';let runIndex=0,entryIndex=0,annotated=false;
const byId=id=>document.getElementById(id);function option(select,value,text){let o=document.createElement('option');o.value=value;o.textContent=text;select.append(o)}
data.runs.forEach((r,i)=>option(byId('run'),i,r.start.slice(11,19)+' · '+r.method));
function imageInto(id,path,empty){const el=byId(id);el.replaceChildren();if(path){let a=document.createElement('a');a.href=path;a.target='_blank';let im=document.createElement('img');im.src=path;a.append(im);el.append(a)}else{let d=document.createElement('div');d.className='empty';d.textContent=empty;el.append(d)}}
function links(items){byId('source').replaceChildren();for(const [label,path] of items){if(!path)continue;let a=document.createElement('a');a.href=path;a.textContent=label;byId('source').append(a,document.createTextNode('　'))}}
function show(){let r=data.runs[runIndex],e=r.entries[entryIndex];byId('title').textContent=r.start.slice(11,19)+' · '+e.label;byId('note').textContent=r.note;byId('status').textContent=e.status+' · 过程帧 '+e.frame_count;byId('reason').textContent=e.reason?'原模型理由：'+e.reason:'';links([['执行/初始化记录',e.evidence],['原计划',e.plan_path],['episode 记录',r.summary],['中断堆栈',r.exception],['该实验目标',r.goal]]);imageInto('before',annotated&&e.overlay?e.overlay:e.before,'此阶段没有保存静态当前观测');imageInto('after',e.after,'没有保存 after；不能据此判定动作后的状态');byId('overlay').disabled=!e.overlay;byId('process').hidden=!e.tracks.length;byId('track').replaceChildren();e.tracks.forEach((t,i)=>option(byId('track'),i,t.label+' · '+t.frames.length+' 帧'));if(e.tracks.length)loadTrack();}
function choose(r,e){runIndex=r;entryIndex=e;annotated=false;byId('run').value=r;byId('entry').replaceChildren();data.runs[r].entries.forEach((v,i)=>option(byId('entry'),i,v.label+' · '+v.status));byId('entry').value=e;show();}
function loadTrack(){const t=data.runs[runIndex].entries[entryIndex].tracks[Number(byId('track').value)];byId('frame').max=t.frames.length-1;byId('frame').value=0;showFrame()}
function showFrame(){let t=data.runs[runIndex].entries[entryIndex].tracks[Number(byId('track').value)],i=Number(byId('frame').value);byId('frame-image').src=t.frames[i];byId('frame-info').textContent=`第 ${i+1} / ${t.frames.length} 帧`;byId('frame-link').replaceChildren();let a=document.createElement('a');a.href=t.frames[i];a.target='_blank';a.textContent='打开当前过程帧原图';byId('frame-link').append(a)}
byId('run').onchange=e=>choose(Number(e.target.value),0);byId('entry').onchange=e=>choose(runIndex,Number(e.target.value));byId('overlay').onclick=()=>{annotated=!annotated;show()};byId('track').onchange=loadTrack;byId('frame').oninput=showFrame;byId('previous').onclick=()=>choose(runIndex,Math.max(0,entryIndex-1));byId('next').onclick=()=>choose(runIndex,Math.min(data.runs[runIndex].entries.length-1,entryIndex+1));choose('''+f'{default_run},{default_entry}'+''');</script></html>'''
    (OUT/'index.html').write_text(page)
    print(json.dumps(stats,ensure_ascii=False));print(OUT/'index.html')


if __name__=='__main__':build()
