"""Known-page screening diagnostic: one independent session, exactly 20 pairs.
Only reads original contact sheet and its membership metadata. No intent/skill.
"""
from pathlib import Path
import json,sys,hashlib,time,csv,html
from PIL import Image
CODE=Path(__file__).resolve().parent;REPO=CODE.parents[1]
sys.path.insert(0,str(CODE))
from transport import call,read,save,sha
OUT=REPO/'reports/effect_screen_page05_diagnostic_20261029'
SOURCE=REPO/'reports/hindsight_fast_test_20261029/contact_sheets.json'
S={'type':'string'}
def enum(*a):return {'type':'string','enum':list(a)}
def obj(d):return {'type':'object','properties':d,'required':list(d),'additionalProperties':False}
OBS=enum('PRESENT','NOT_OBSERVED','UNCERTAIN')
PROMPT='''你是一个全新独立Codex会话中的视觉初筛器。本次是已知Contact Sheet页面的筛选器诊断，不是独立盲发现实验。只看随附四组图像，不读取此前筛选结果、人工案例结论、动作意图、抓取点或技能分析。
共20对Before/After，按原页面逐行从左到右的顺序拆成4组，每组5对。每组从上到下5行，每行左Before、右After，原始编号位于该行上方。必须逐对检查全部20对，按给定编号顺序输出20条，不得只报最显眼几对，不得遗漏或合并。
任务只是在初筛阶段寻找值得进一步放大核验的语义端部显露现象。对每对检查：原来不明显的袖部或下摆是否变清楚；After是否出现新显露或更独立的袖状轮廓、袖口边缘、腋下凹口、下摆交角。四个feature字段中PRESENT表示存在前后对比支持的新显露或更清楚线索，而不只是After图里本来就有这个结构；NOT_OBSERVED表示没有观察到这种变化，不是证明它不存在；拿不准为UNCERTAIN。
每对必须给一个标签：
POSSIBLE_REVEAL：至少存在一条具体前后视觉线索，可能是语义端部显露，值得放大核验。允许端点材料身份、左右和隐藏层序不明。
UNCERTAIN：有疑似变化但缩略图质量、遮挡或对应关系不足以明确归类；仍保留复查可能。
NO_EVIDENCE：未观察到与端部更清楚有关的正向线索，例如只有整体位置改变、一般布面变化或相关部位并未更清楚。不能仅因语义身份不确定就使用NO_EVIDENCE。
不要求证明同一隐藏材料点，也不要求确定完整布层关系。不要因其他区域变皱而忽略局部显露。面积增长本身不是线索，但袖状轮廓、袖口边、腋下凹口或下摆交角更独立可构成初筛线索。
仅记录可见前后差异和简短疑点；不解释物理机制，不分析抓点规律，不生成动作或技能。每条visible_evidence用1–2个简短中文句子描述局部位置及Before/After差别，uncertainty用一句。四组均完成后只输出统一JSON。'''

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sheet=next(x for x in read(SOURCE) if Path(x['path']).name=='page_05.jpg');ids=sheet['ids'];assert len(ids)==20 and len(set(ids))==20
    page=Path(sheet['path']);assert sha(page)==sheet['sha256'];im=Image.open(page).convert('RGB');assert im.size==(1200,2000)
    fields={'id':{'type':'string','enum':ids},'status':enum('POSSIBLE_REVEAL','UNCERTAIN','NO_EVIDENCE'),'sleeve_or_hem_clearer':enum('YES','NO','UNCERTAIN'),'new_sleeve_like_contour':OBS,'cuff_edge_clearer':OBS,'axillary_notch_clearer':OBS,'hem_corner_clearer':OBS,'visible_evidence':S,'uncertainty':S}
    schema=obj({'judgments':{'type':'array','items':obj(fields),'minItems':20,'maxItems':20},'screening_limitations':S})
    protocol={'task':'known-page diagnostic, not blind discovery and not discovery-rate evaluation','source_page':str(page),'source_page_sha256':sha(page),'membership_metadata_sha256':sha(SOURCE),'order':ids,'grouping':'4 consecutive groups of5; original row-major order; each original 600x200 cell copied without scaling into a vertical 600x1000 strip','model':'gpt-6-astra/high','independent_sessions':1,'model_requests':1,'request_timeout_s':900,'retry':False,'input_scope':'four image strips and ID/group list only; no previous results, human conclusions, action intent or raw actions','aggregation':'after all20 complete: POSSIBLE_REVEAL candidates first; UNCERTAIN review queue separately; preserve original order within each, no top-k cap','skill_induction':False,'physical_actions':0,'code_sha256':sha(__file__),'transport_sha256':sha(CODE/'transport.py'),'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest()}
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    groups=[]
    for g in range(4):
        out=Image.new('RGB',(600,1000),'white');cells=[]
        for j in range(5):
            i=g*5+j;x=(i%2)*600;y=(i//2)*200;box=(x,y,x+600,y+200);out.paste(im.crop(box),(0,j*200));cells.append({'id':ids[i],'original_index_1based':i+1,'source_box':box})
        p=OUT/f'group_{g+1:02d}.png';out.save(p);groups.append({'group':g+1,'path':str(p),'sha256':sha(p),'cells':cells})
    save(OUT/'groups.json',groups)
    save(OUT/'progress.json',{'stage':'REVIEW_ALL20_ONE_SESSION','time':time.time()});print('Reviewing all20 in one independent session',flush=True)
    payload={'image_order':[{'image':i+1,'group':g['group'],'ids_top_to_bottom':[c['id'] for c in g['cells']]} for i,g in enumerate(groups)],'required_output_order':ids,'all_pairs_required':20,'known_page_diagnostic':True}
    res=call(OUT,'all20_one_session',PROMPT,payload,[Path(g['path']) for g in groups],schema)
    if res['status']!='COMPLETE':save(OUT/'progress.json',{'stage':'API_FAILED','error':res.get('error')});raise RuntimeError(res.get('error'))
    rows=res['output']['judgments'];assert [x['id'] for x in rows]==ids,'Missing, duplicate or reordered judgment'
    save(OUT/'judgments.json',res['output'])
    candidates=[x['id'] for x in rows if x['status']=='POSSIBLE_REVEAL'];uncertain=[x['id'] for x in rows if x['status']=='UNCERTAIN'];none=[x['id'] for x in rows if x['status']=='NO_EVIDENCE']
    save(OUT/'candidate_queue.json',{'possible_reveal':candidates,'uncertain_for_optional_enlargement':uncertain,'no_evidence':none,'order':'original page order within each category','followup_started':False,'not_confirmed_corner_reveal':True})
    m=read(OUT/'calls/all20_one_session/attempt_0/manifest.json');assert {x['sha256'] for x in m['images']}=={g['sha256'] for g in groups}
    save(OUT/'audit.json',{'status':'PASS','expected':20,'received':len(rows),'ids_and_order_match':True,'all_four_groups_supplied':True,'calls':1,'tool_calls':res['tool_calls'],'old_results_or_intent_supplied':False,'physical_actions':0,'skill_induction':False,'orchestrator_has_prior_chat_context':True,'diagnostic_not_independent_blind_discovery':True,'not_pooled_with_previous_discovery_rates':True})
    with (OUT/'judgments.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fields));w.writeheader();w.writerows(rows)
    esc=lambda x:html.escape(str(x))
    body='<h1>page_05 · Effect Discovery 漏检诊断</h1><p>已知页面诊断，不能与全量实验混算发现率。一次隔离Codex会话、四组各五对、20条逐对结果。未提供旧筛选结果、人工案例结论或动作意图。</p><p>POSSIBLE_REVEAL '+str(len(candidates))+'；UNCERTAIN '+str(len(uncertain))+'；NO_EVIDENCE '+str(len(none))+'。均为初筛标签，不是确认的Corner Reveal。</p><p><a href="judgments.json">完整20条JSON</a> · <a href="judgments.csv">CSV</a> · <a href="candidate_queue.json">待放大候选</a> · <a href="audit.json">完整性及隔离审计</a> · <a href="protocol.json">固定协议</a></p>'
    labels={'new_sleeve_like_contour':'袖状轮廓','cuff_edge_clearer':'袖口边','axillary_notch_clearer':'腋下凹口','hem_corner_clearer':'下摆交角'}
    for gi,g in enumerate(groups):
        body+='<section><h2>第'+str(gi+1)+'组 · 原序号'+str(gi*5+1)+'–'+str(gi*5+5)+'</h2><a href="'+Path(g['path']).name+'"><img class="strip" src="'+Path(g['path']).name+'"></a><table><tr><th>序号 / ID</th><th>判断</th><th>变化线索</th><th>可见证据</th><th>不确定性</th></tr>'
        for j,x in enumerate(rows[gi*5:(gi+1)*5]):
            feats='；'.join(v+': '+x[k] for k,v in labels.items());body+='<tr>'+''.join('<td>'+esc(t)+'</td>' for t in [str(gi*5+j+1)+' / '+x['id'],x['status'],feats,x['visible_evidence'],x['uncertainty']])+'</tr>'
        body+='</table></section>'
    body+='<h2>自动整理的后续核验队列</h2><p>优先放大：'+esc(', '.join(candidates) or '无')+'</p><p>保留不确定复查：'+esc(', '.join(uncertain) or '无')+'</p><p>本轮停止于初筛，未启动放大核验或Skill Induction。</p><p>'+esc(res['output']['screening_limitations'])+'</p>'
    (OUT/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Effect screening diagnostic</title><style>body{font:17px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}section{margin:25px 0;padding:20px;background:#f5f7f9}.strip{width:600px;max-width:100%;height:auto}td{overflow-wrap:anywhere}</style>'+body+'</html>')
    summary={'stage':'COMPLETE','reviewed':20,'possible_reveal':len(candidates),'uncertain':len(uncertain),'no_evidence':len(none),'latency_s':res['latency_s'],'physical_actions':0,'skill_induction':False};save(OUT/'progress.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
