"""Render saved artifacts; no model calls or changes to experimental decisions."""
from pathlib import Path
import json,html,shutil
def render(root):
    root=Path(root)
    def read(name,default):return json.loads((root/name).read_text()) if (root/name).exists() else default
    esc=lambda x:html.escape(str(x))
    dump=lambda x:'<pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
    summary=read('summary.json',{})
    rows=read('experience_index.json',[])
    intents=read('intended_effects.json',{})
    effects=read('observed_effects.json',{})
    comparisons=read('mismatch_comparisons.json',{})
    body=['<h1>Intention–Outcome Mismatch Driven Skill Discovery v0</h1>',
          '<p>按照2026-10-30保存的研究说明运行。独立意图提取 → 全衣物效果观察 → 语义比较 → 意外收益判据 → 目标重标记 → 假设与技能 → 主动物理验证。</p>',
          '<p class="notice">本次初始化使用已有17条候选的原始记录，属于选择过的历史批次。未输入此前模型结论、人工几何规则或指定成功案例。不能将本批候选比例解释为全部历史数据的发现率。</p>',
          '<h2>结果与三个层次</h2>',dump(summary),
          '<p>Discovery的模型证据不是独立材料真值；生成代码不等于Acquisition。离线新图推理和像素合同检查不等于Host安全检查或物理成功。</p>',
          '<h2>全部经验：原始意图与四角效果</h2>']
    for r in rows:
        id=r['id'];d=root/'evidence'/id;d.mkdir(parents=True,exist_ok=True)
        for k in ['before','after']:shutil.copyfile(r[k],d/(k+'.png'))
        body+=['<details><summary>'+id+' · '+esc(r['source'])+'</summary><div class="pair">']
        for k in ['before','after']:
            path=f'evidence/{id}/{k}.png';body+=['<figure><a href="'+path+'"><img loading="lazy" src="'+path+'"></a><figcaption>'+k+'</figcaption></figure>']
        body+=['</div><h3>原始理由与实际参数</h3>',dump({'intention':r['raw_intention'],'action':r['action'],'execution':r['execution'],'provenance':r['intention_provenance']}),
               '<h3>独立意图提取（未看After）</h3>',dump(intents.get(id,'MISSING')),
               '<h3>独立整件衣物观察（未看意图）</h3>',dump(effects.get(id,'MISSING')),
               '<h3>冻结结果之间的比较</h3>',dump(comparisons.get(id,'MISSING')),'</details>']
    for f,title in [('corner_reveal_candidates.json','Corner Reveal候选：包含预期内及不确定的变化'),
                    ('hindsight_events.json','符合意外收益判据的事件'),
                    ('ranking.json','预先固定规则的自动排序'),
                    ('hindsight_relabel.json','Hindsight Goal Relabeling'),
                    ('skill_hypotheses.json','竞争机制、预测与反证'),
                    ('skill_synthesis.json','候选函数与推理规则')]:
        if (root/f).exists():body+=['<h2>'+title+'</h2>',dump(read(f,None))]
    for t in read('smoke_results.json',[]):
        body+=['<h2>新图接口测试 '+t['id']+'</h2><p>这不是Active Validation；只使用Before RGB，没有输入After和历史动作。</p><a href="smoke/'+t['id']+'/overlay.png"><img src="smoke/'+t['id']+'/overlay.png"></a>',dump(t)]
    body+=['<h2>Active Validation & Skill Refinement</h2>',dump(read('validation_record.json',{})),
           '<h2>信息隔离与局限</h2><p>模型分析均使用独立的codex -p api临时会话。意图通道没有After；观察通道没有原始意图；比较通道只读冻结结果；后续假设使用本轮事件。不读取旧技能分析报告。主协调会话保留历史聊天，不能声称整个过程完全无研究先验。原始历史动作也可能受到当时人工规则影响。</p>',
           '<p>原始dispatch_intent是生产者的执行前记录；本批文件没有独立时间戳证明，保留来源与哈希，不事后补写预期。视觉端点身份、具体夹持层和自遮挡转换仍可能不确定。</p>',
           '<h2>模型调用</h2><table><tr><th>请求</th><th>状态</th><th>秒</th><th>Token</th></tr>']
    for p in sorted((root/'calls').glob('*/result.json')):
        x=json.loads(p.read_text());body+=['<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [p.parent.name,x['status'],round(x.get('latency_s',0),1),x.get('usage')])+'</tr>']
    body+=['</table><h2>文件与复现</h2><p>']
    for name in ['specification.md','protocol.json','experience_index.json','source_audit.json','intended_effects.json','observed_effects.json','mismatch_comparisons.json','hindsight_events.json','ranking.json','candidate_skill.py','skill_freeze.json','validation_record.json','incremental_ledger.json','summary.json']:
        if (root/name).exists():body+=['<a href="'+name+'">'+name+'</a> · ']
    body+=['</p>']
    (root/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Intention–Outcome Skill Discovery</title><style>body{font:17px/1.65 system-ui;max-width:1450px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf1f6;padding:16px}.pair{display:flex;gap:12px}.pair figure{width:50%;margin:0}img{width:100%}details{border:1px solid #ccd4de;padding:16px;margin:18px 0}summary{cursor:pointer;font-weight:bold}td,th{border:1px solid #ccd4de;padding:8px}table{border-collapse:collapse;width:100%}.notice{background:#eaf1f8;padding:16px}</style>'+''.join(body)+'</html>')
if __name__=='__main__':
    import sys
    render(Path(sys.argv[1]))
