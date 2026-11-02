"""Read-only selection from completed bank; image report, no new model calls."""
from pathlib import Path
import json,html,shutil,math,hashlib
from PIL import Image,ImageDraw,ImageFont
REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'reports/intention_outcome_hypothesis_bank_v1'
OUT=REPO/'reports/hypothesis_bank_positive_images_20261102'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
GOALS={'Reveal(LS)':'穿着者左袖口','Reveal(RS)':'穿着者右袖口','Reveal(LH)':'穿着者左下摆角','Reveal(RH)':'穿着者右下摆角'}
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def esc(x):return html.escape(str(x))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wrap(draw,text,font,width):
    lines=[]
    for paragraph in text.split('\n'):
        line=''
        for ch in paragraph:
            if line and draw.textlength(line+ch,font=font)>width:lines.append(line);line=ch
            else:line+=ch
        lines.append(line)
    return lines
def textblock(draw,text,xy,width,size=32,color='#172b42'):
    font=ImageFont.truetype(FONT,size);x,y=xy
    for line in wrap(draw,text,font,width):draw.text((x,y),line,font=font,fill=color);y+=int(size*1.5)
    return y
def overlay(before,action,path):
    im=Image.open(before).convert('RGB');d=ImageDraw.Draw(im);a=action.get('grasp_px');b=action.get('target_px')
    if a and b:
        d.line([tuple(a),tuple(b)],fill='yellow',width=6)
        ang=math.atan2(b[1]-a[1],b[0]-a[0]);d.polygon([tuple(b)]+[(b[0]-20*math.cos(ang+t),b[1]-20*math.sin(ang+t)) for t in [-.45,.45]],fill='yellow')
        for p,c,label in [(a,'lime','Pa'),(b,'red','Pb')]:
            d.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),fill=c)
            d.text((p[0]+10,p[1]-25),label,font=ImageFont.truetype(FONT,24),fill=c,stroke_width=2,stroke_fill='black')
    im.save(path)

def main():
    OUT.mkdir(exist_ok=True)
    observations=read(SOURCE/'experience_observations.json');entries=[]
    for e in observations:
        if e['status']!='COMPLETE':continue
        corners=[c for c in e['observation']['corners'] if c['change'] in ['REVEAL','VISIBILITY_IMPROVED']]
        if not corners:continue
        rid=e['experience_id'];dest=OUT/'evidence'/rid;dest.mkdir(parents=True,exist_ok=True)
        payload=read(SOURCE/'inputs'/rid/'payload.json')
        for part in ['before','after']:shutil.copyfile(SOURCE/'inputs'/rid/(part+'.png'),dest/(part+'.png'))
        overlay(dest/'before.png',payload['indexed_executed_action'],dest/'action.png')
        record={'id':rid,'source':e['source'],'corners':corners,'original_intention':payload['original_intention'],
                'action':payload['indexed_executed_action'],'whole_garment_changes':e['observation']['observed_effect']['whole_garment_changes'],
                'new_occlusions':e['observation']['observed_effect']['new_occlusions'],
                'original_observation':e['observation'],'source_hashes':{k:sha(SOURCE/'inputs'/rid/(k+'.png')) for k in ['before','after']}}
        save(dest/'record.json',record);entries.append(record)
    byid={e['id']:e for e in entries}
    primary=[byid[r] for r in ['X0518','X0311']]
    external=[byid['X0331']]
    visibility=[e for e in entries if all(c['change']!='REVEAL' for c in e['corners'])]
    revealed=[e for e in entries if any(c['change']=='REVEAL' for c in e['corners'])]
    assert len(revealed)==3 and len(visibility)==58 and len(entries)==61
    confirmed=sum(c['change']=='REVEAL' and c['evidence_level']=='CONFIRMED' for e in observations if e['status']=='COMPLETE' for c in e['observation']['corners'])
    assert confirmed==0
    protocol={'source':str(SOURCE),'source_observations_sha256':sha(SOURCE/'experience_observations.json'),
              'selection':'All corner change=REVEAL or VISIBILITY_IMPROVED in completed original observations; no new AI scoring.',
              'strong_confirmed_reveal_events':confirmed,'provisional_reveal_operations':3,'visibility_improved_operations':58,
              'visibility_improved_events':sum(c['change']=='VISIBILITY_IMPROVED' for e in entries for c in e['corners']),
              'primary_self_occlusion_candidates':[e['id'] for e in primary],
              'external_occluder_separate':['X0331'],
              'external_reason':'Original X0331 alternative_explanation explicitly says cloth may move out from behind foreground tool; do not equate with release of self-fold.',
              'physical_success_verified':False,'interpretation':'Positive-observation picture report, not verified successful-action dataset. No task completion or skill success rate.'}
    save(OUT/'selection.json',protocol);save(OUT/'positive_cases.json',entries)
    pages=[]
    def card(e,main=False):
        rid=e['id'];base=f'evidence/{rid}';intention=e['original_intention'].get('short_reason','原意图缺失')
        action=e['action'];title=' / '.join(GOALS[c['goal']] for c in e['corners'])
        body=f'<article id="{rid}"><h3>{rid} · {esc(title)}</h3><p class="source">{esc(e["source"].replace(str(REPO)+"/reports/",""))}</p>'
        body+='<div class="pair">'+''.join(f'<figure><figcaption>{label} · 干净原图</figcaption><a href="{base}/{part}.png"><img loading="lazy" src="{base}/{part}.png"></a></figure>' for part,label in [('before','操作前'),('after','操作后')])+'</div>'
        body+=f'<p><b>原本想做：</b>{esc(intention)}</p><p><b>实际命令：</b>Pa {esc(action.get("grasp_px"))} → Pb {esc(action.get("target_px"))}；记录水平距离 {action.get("distance_mm",0):.1f} mm，索引抬升 {action.get("lift_height_mm","未知")} mm（不是实测布料抬升）。</p>'
        for c in e['corners']:
            body+=f'<p><b>已有分析记录的正向变化：</b>{esc(c["after_evidence"])}</p><p class="caveat"><b>{esc(c["change"])} / {esc(c["evidence_level"])}</b> · {esc(c["alternative_explanation"])}</p>'
        body+=f'<details><summary>查看 Pa → Pb 标注及伴随变化</summary><a href="{base}/action.png"><img src="{base}/action.png" loading="lazy"></a><ul>'+''.join('<li>'+esc(s)+'</li>' for s in e['whole_garment_changes'])+'</ul></details>'
        body+=f'<p><a href="{base}/record.json">完整来源与分析</a></p></article>'
        return body
    for number,e in enumerate(primary+external,1):
        rid=e['id'];c=next(x for x in e['corners'] if x['change']=='REVEAL');action=e['action']
        im=Image.new('RGB',(2680,1840),'#f7f9fc');d=ImageDraw.Draw(im)
        external_case=rid=='X0331'
        label='外部遮挡混淆：不能算作自折叠恢复' if external_case else '袖口显露候选：PROVISIONAL，尚非确认成功'
        textblock(d,f'{number}/3  {rid}  |  '+GOALS[c['goal']],(40,30),2600,50)
        textblock(d,label,(40,115),2600,34,'#ad531b')
        textblock(d,'操作前 · 干净原图',(40,195),1260,32);textblock(d,'操作后 · 干净原图',(1360,195),1260,32)
        for part,x in [('before',40),('after',1360)]:
            with Image.open(OUT/'evidence'/rid/(part+'.png')) as src:im.paste(src.convert('RGB').resize((1280,720)),(x,250))
        y=textblock(d,'原本想做：'+e['original_intention'].get('short_reason','UNKNOWN'),(40,1000),2600,32)
        y=textblock(d,'记录的正向变化：'+c['after_evidence'],(40,y+12),2600,32)
        y=textblock(d,'证据边界：'+c['alternative_explanation'],(40,y+12),2600,30,'#9b4c21')
        y+=30
        with Image.open(OUT/'evidence'/rid/'action.png') as action_image:im.paste(action_image.resize((720,405)),(40,y))
        textblock(d,'黄色箭头为历史实际命令 Pa → Pb\n'+str(action['grasp_px'])+' → '+str(action['target_px'])+'\n水平距离约 '+f'{action.get("distance_mm",0):.1f}'+' mm；索引抬升 '+str(action.get('lift_height_mm'))+' mm。\n抓取材料身份与层数仍未确认。\n局部显露不等于整件衣服展平成功。',(800,y),1820,30)
        textblock(d,e['source'].replace(str(REPO)+'/',''),(40,1765),2600,23,'#526479')
        im.save(OUT/(rid+'_comparison.jpg'),quality=94);pages.append(im)
    pages[0].save(OUT/'positive_candidates_3pages.pdf','PDF',save_all=True,append_images=pages[1:],resolution=180)
    body='<h1>先看图片：有正向变化的操作</h1><p>从532条已完成记录中提取；只展示已有分析中的正向可见性变化，不重新调用AI，也不改原标签。</p>'
    body+='<div class="notice"><b>确认Corner Reveal：0例。</b> 以下两例袖口显露为暂定证据；另有1例外部遮挡混淆、58例仅可见性改善。不能把它们合计为动作成功率。</div>'
    body+='<p><a class="button" href="positive_candidates_3pages.pdf">下载3页重点图片PDF</a> · <a href="selection.json">筛选口径</a> · <a href="positive_cases.json">全部61例来源</a></p><nav><a href="#primary">2例袖口显露候选</a> · <a href="#external">1例外部遮挡混淆</a> · <a href="#visibility">58例可见性改善</a></nav>'
    body+='<h2 id="primary">两例值得优先看的袖口显露候选</h2>'+''.join(card(e,True) for e in primary)
    body+='<h2 id="external">单列：移出工具遮挡，不能算作解除自折叠</h2>'+''.join(card(e,True) for e in external)
    body+='<details id="visibility"><summary><b>展开附录：58次端部可见性改善（66个端点事件）</b></summary><p>这些记录只支持更易辨认，不证明端点原本隐藏。每例均保留Before/After和原记录的局限。</p>'+''.join(card(e) for e in visibility)+'</details>'
    body+='<footer>报告由已有结构化观察机械筛选生成。主报告突出两例自遮挡候选，并根据X0331原文中的工具遮挡解释将其单列；未增加成功标签，没有合并或升级任何假设。</footer>'
    style='body{font:18px/1.65 system-ui;max-width:1700px;margin:30px auto;padding:20px;color:#193047;background:#f7f9fc}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}img{max-width:100%;display:block}figcaption{font-weight:bold;padding:8px}article{background:white;padding:24px;margin:25px 0;border:1px solid #dce3eb;border-radius:12px}.notice,.caveat{padding:16px;background:#fff2d9}.source{font-size:14px;overflow-wrap:anywhere;color:#61748a}summary{cursor:pointer;padding:12px}a{color:#176a9c}.button{display:inline-block;padding:10px;background:#e1edf8}footer{margin:35px 0;color:#526479}nav{position:sticky;top:0;background:#f7f9fc;padding:12px}@media(max-width:900px){.pair{grid-template-columns:1fr}}'
    (OUT/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>正向变化图片报告</title><style>'+style+'</style>'+body+'</html>')
    print(json.dumps({'report':str(OUT/'review.html'),'pdf':str(OUT/'positive_candidates_3pages.pdf'),'positive_cases':len(entries),'confirmed_reveal':0},ensure_ascii=False))

if __name__=='__main__':main()
