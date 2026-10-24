"""Visual review of explicit manual stops only, grouped by actual local date."""
from pathlib import Path
import json
import os
import html
from PIL import Image,ImageDraw,ImageFont

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'reports/manual_stop_visual_review_20261022'
OLD=BASE/'reports/interrupted_experiments_20261020_archive'


def read(p):return json.loads(p.read_text())
def relative(p):return os.path.relpath(p,OUT) if p and p.exists() else None
def oldpath(p):return (OLD/p).resolve() if p else None


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[]
    for suffix,rd in [('230743_643820346','round_002'),('232513_053641787','round_004')]:
        run=BASE/'reports'/('one_shot_goal_full_episode_v1_20261021_'+suffix);p=run/rd
        execution=read(p/'execution.json');assert execution['status']=='USER_INTERRUPTED'
        plan=read(p/'plan.json');frames=sorted((p/'execution_images').glob('*.jpg'))
        cases.append(dict(date='2026-10-21',title=f"10/21 {suffix[:2]}:{suffix[2:4]} · {rd} · one-shot",kind='during_execution',
            note='该轮已开始执行，日志明确为 USER_INTERRUPTED / EXECUTION。没有 after 图；右图仅为录制结束前最后保存的过程帧，不证明动作完成或机器人已停稳。',
            reason=plan['short_reason'],status=execution,source=relative(p/'execution.json'),
            panels=[dict(label='中断轮：动作前原图',path=relative(p/'before/rgb.png')),
                    dict(label='中断轮：计划 Pa → Pb（未确认完成）',path=relative(p/'proposed_action_overlay.png')),
                    dict(label=f'最后保存的过程帧 · {frames[-1].name}',path=relative(frames[-1]))],
            frames=[relative(f) for f in frames],source_round=str(p)))
    old=read(OLD/'data.json')
    for r in old['runs']:
        entries=r['entries']
        e=next((e for e in reversed(entries) if e['before']),entries[-1])
        is_memory=r['method']=='trajectory memory'
        if is_memory:
            last_image=e['after'];last_label='最后完整轮：动作后；随后采图被中断'
            before_label='最后完整轮 round_006：动作前'
            overlay_label='最后完整轮 round_006：已执行动作'
            frames=[]
        else:
            tracks=e['tracks'];track=next((t for t in reversed(tracks) if 'return_perception' in t['label']),tracks[-1] if tracks else None)
            frames=track['frames'] if track else []
            last_image=frames[-1] if frames else None
            last_label='中断阶段最后保存的过程帧；不是 after'
            before_label='本次保存的静态当前图'
            overlay_label='已有抓点/方案预览；完成状态见说明'
        panels=[dict(label=before_label,path=relative(oldpath(e['before']))),dict(label=overlay_label,path=relative(oldpath(e['overlay']))),
                dict(label=last_label,path=relative(oldpath(last_image)))]
        evidence=entries[-1]['evidence'] or r['summary']
        cases.append(dict(date='2026-10-20',title=f"10/20 {r['start'][11:16]} · {r['method']}",kind='before_or_between_actions',
            note=r['note'],reason=e['reason'],status=entries[-1]['result'],source=relative(oldpath(evidence)),
            panels=panels,frames=[relative(oldpath(f)) for f in frames],source_round=e['source'],
            sequence=relative(OLD/'seven_completed_rounds.png') if is_memory else None))
    (OUT/'data.json').write_text(json.dumps(dict(cases=cases,automatic_stops_included=False),ensure_ascii=False,indent=2))
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(font_path,19);small=ImageFont.truetype(font_path,16)
    sheet=Image.new('RGB',(1500,2*380),'white');draw=ImageDraw.Draw(sheet)
    for row,c in enumerate(cases[:2]):
        draw.text((12,row*380+5),c['title']+'：动作执行中手动中断',font=font,fill='#172033')
        for col,p in enumerate(c['panels']):
            im=Image.open(OUT/p['path']).convert('RGB');im.thumbnail((490,276));x=col*500;y=row*380
            sheet.paste(im,(x+5,y+40));draw.text((x+8,y+325),p['label'],font=small,fill='#172033')
    sheet.save(OUT/'manual_execution_stops.png')
    oct20=cases[2:]
    sheet20=Image.new('RGB',(1500,len(oct20)*390),'white');draw20=ImageDraw.Draw(sheet20)
    for row,c in enumerate(oct20):
        y=row*390;draw20.text((12,y+5),c['title']+'：手动停止',font=font,fill='#172033')
        for col,p in enumerate(c['panels']):
            x=col*500
            if p['path']:
                im=Image.open(OUT/p['path']).convert('RGB');im.thumbnail((490,276));sheet20.paste(im,(x+5,y+40))
            else:
                draw20.rectangle((x+5,y+40,x+495,y+316),fill='#edf0f5')
                draw20.text((x+40,y+160),'此阶段未保存对应图像',font=font,fill='#647086')
            label=p['label'];draw20.text((x+8,y+325),label[:28],font=small,fill='#172033')
            if len(label)>28:draw20.text((x+8,y+350),label[28:],font=small,fill='#172033')
    sheet20.save(OUT/'manual_stops_oct20.png')
    cards=[]
    for i,c in enumerate(cases):
        panels=[]
        for p in c['panels']:
            image=f'<a href="{p["path"]}" target="_blank"><img src="{p["path"]}"></a>' if p['path'] else '<div class="empty">此阶段未保存对应图像</div>'
            panels.append('<figure>'+image+'<figcaption>'+html.escape(p['label'])+'</figcaption></figure>')
        playback=''
        if c['frames']:
            playback=f'''<details><summary>浏览此中断阶段的 {len(c['frames'])} 张过程帧</summary><p>过程帧可能包含机械臂、视角变化或运动模糊，不能替代完整 after。</p>
<input class="slider" data-case="{i}" type="range" min="0" max="{len(c['frames'])-1}" value="{len(c['frames'])-1}"><span id="count-{i}">{len(c['frames'])} / {len(c['frames'])}</span>
<img class="frame" id="frame-{i}" src="{c['frames'][-1]}"></details>'''
        sequence=f'<details><summary>展开该实验全部 7 组已完成动作的前后图</summary><a href="{c["sequence"]}" target="_blank"><img class="sequence" src="{c["sequence"]}"></a></details>' if c.get('sequence') else ''
        cards.append(f'''<section data-date="{c['date']}"><h2>{html.escape(c['title'])}</h2><p class="notice">{html.escape(c['note'])}</p>
<div class="panels">{''.join(panels)}</div><p>{('原模型理由：'+html.escape(c['reason'])) if c['reason'] else ''}</p><a href="{c['source']}">查看原始中断/执行记录</a>{playback}{sequence}</section>''')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>手动中断实验 · 图像可视化</title>
<style>body{font-family:system-ui,sans-serif;background:#f2f5fa;color:#172033;margin:24px}section{padding:20px;margin:24px 0;background:white;border-radius:12px}.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0}figure img{width:100%;display:block}figcaption{padding:10px 0}p{line-height:1.65}.notice{background:#fff2cf;padding:12px;border-radius:8px}.empty{padding:75px 15px;background:#f0f2f5;color:#647086}a{color:#2059a3}.frame{display:block;width:100%;max-width:1000px;margin-top:10px}.sequence{width:100%;max-width:1100px}details{margin-top:16px}summary{cursor:pointer;font-weight:600}input{width:75%}select{padding:8px}@media(max-width:1000px){.panels{grid-template-columns:1fr}}</style>
<h1>明确由你手动中断的实验</h1><p>只收录 USER_STOP、USER_INTERRUPTED 或明确取消的记录，按实际日期区分；不含自动安全停止。</p>
<p><b>10/21 23:07、23:25：</b>one-shot 动作执行中中断。<b>10/20：</b>初始化、采图或方案审阅阶段手动停止。</p>
<p>筛选日期：<select id="date"><option value="all">全部手动中断</option><option value="2026-10-20" selected>10 月 20 日（前天）</option><option value="2026-10-21">10 月 21 日（昨天）</option></select>　<a href="manual_stops_oct20.png">10/20 手动停止图像总览</a> · <a href="manual_execution_stops.png">10/21 两次 one-shot 中断动作对照图</a> · <a href="data.json">原图索引与证据</a></p>
'''+''.join(cards)+'''<script>const cases='''+json.dumps(cases,ensure_ascii=False).replace('<','\\u003c')+''';
document.getElementById('date').onchange=e=>{document.querySelectorAll('section').forEach(s=>s.hidden=e.target.value!=='all'&&s.dataset.date!==e.target.value)};
document.querySelectorAll('.slider').forEach(e=>e.oninput=()=>{let i=Number(e.dataset.case),n=Number(e.value);document.getElementById('frame-'+i).src=cases[i].frames[n];document.getElementById('count-'+i).textContent=(n+1)+' / '+cases[i].frames.length;});document.getElementById('date').dispatchEvent(new Event('change'));</script></html>'''
    (OUT/'index.html').write_text(page)
    print('SAVED',OUT/'index.html','manual cases',len(cases))


if __name__=='__main__':build()
