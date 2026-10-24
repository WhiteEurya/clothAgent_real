"""Compare frozen Ps/Pt/base/residual rankings from one observed state; no hardware."""
from pathlib import Path
import argparse
import csv
import html
import json
import sys
from itertools import combinations

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import spearmanr, kendalltau, rankdata
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cloth_agent import action_preference as pref

KEYS=['Ps','Pt','base','correction','S']
LABELS=['Ps','Pt','base','R','S']


def compare(rows):
    result={}
    for a,b in combinations(KEYS,2):
        x=np.array([r[a] for r in rows]); y=np.array([r[b] for r in rows])
        varying=np.ptp(x)>0 and np.ptp(y)>0
        pairs=[(np.sign(x[i]-x[j]),np.sign(y[i]-y[j])) for i,j in combinations(range(len(rows)),2)]
        strict=[(u,v) for u,v in pairs if u and v]
        result[a+':'+b]=dict(spearman=float(spearmanr(x,y).statistic) if varying else None,
            kendall_tau_b=float(kendalltau(x,y).statistic) if varying else None,
            strict_pair_agreement=float(sum(u==v for u,v in strict)/len(strict)) if strict else None,
            strict_pairs=len(strict),opposite_pairs=int(sum(u!=v for u,v in strict)),
            pairs_tied_in_either=len(pairs)-len(strict))
    return result


def order(rows,key):return sorted(rows,key=lambda r:(-r[key],r['id']))


def analyze(round_path,out):
    if (out/'index.html').exists():raise ValueError('report already exists')
    out.mkdir(parents=True,exist_ok=True)
    manifest=pref.io.read(round_path/'preference_review/manifest.json')
    rows=manifest['all_candidates'];alpha=manifest['alpha'];context=manifest['ranking_context']
    for row in rows:
        assert np.isclose(row['base'],row['Ps']+alpha*row['Pt'])
        assert np.isclose(row['S'],row['base']+row['correction'])
    groups={i:[r for r in rows if r['point_id']==i] for i in sorted({r['point_id'] for r in rows})}
    points=[];directions=[]
    for identifier,group in groups.items():
        assert len({r['Ps'] for r in group})==1
        points.append(dict(id=identifier,n=len(group),grasp_px=group[0]['grasp_px'],
            **{k:max(r[k] for r in group) for k in KEYS}))
        maximum_pt=max(r['Pt'] for r in group); selected=order(group,'S')[0]
        directions.append(dict(point_id=identifier,n=len(group),S_best=selected['id'],
            Pt_best=[r['id'] for r in group if r['Pt']==maximum_pt],
            S_winner_is_Pt_max=selected['Pt']==maximum_pt,
            base_best=order(group,'base')[0]['id'],
            correlations=compare(group)))
    action_corr=compare(rows);point_corr=compare(points)
    action_orders={k:[r['id'] for r in order(rows,k)] for k in KEYS}
    point_orders={k:[r['id'] for r in order(points,k)] for k in KEYS}
    overlap={k:len(set(action_orders[k][:10])&set(action_orders['S'][:10])) for k in ('Pt','base','correction')}
    scales={k:dict(min=min(r[k] for r in rows),max=max(r[k] for r in rows),
        std=float(np.std([r[k] for r in rows]))) for k in KEYS}
    scales['alpha_Pt']=dict(std=alpha*scales['Pt']['std'])
    timing={name:pref.io.read(round_path/name) for name in ('model_metrics.json','residual_features/features.json') if (round_path/name).exists()}
    fm=timing.get('residual_features/features.json',{})
    metrics=dict(source=str(round_path.resolve()),alpha=alpha,residual_revision=context['revision'],
        image_sha256=context['image_sha256'],n_actions=len(rows),n_eligible_points=len(points),
        action_correlations=action_corr,point_correlations=point_corr,action_orders=action_orders,
        point_orders=point_orders,top10_overlap_with_S=overlap,scales=scales,within_point=directions,
        point_aggregation='Ps once per eligible Pa; other columns independently maximize over its eligible directions; may use different directions',
        ties='average ranks for Spearman; tau-b for Kendall; displayed tied IDs sorted by ID only',
        timing=dict(model_wall_s=timing.get('model_metrics.json',{}).get('wall_latency_s'),
                    feature_wall_s=fm.get('wall_s'),feature_forward_s=fm.get('timing',{}).get('forward_and_transfer_s')),
        scope='single new state, frozen pre-feedback scores; descriptive correlations, not success or preference-alignment evidence')
    pref.atomic_save(out/'analysis.json',metrics)
    for filename,data in [('action_rankings.csv',rows),('point_rankings.csv',points)]:
        ranks={k:rankdata([-r[k] for r in data],method='average') for k in KEYS}
        with (out/filename).open('w') as file:
            writer=csv.DictWriter(file,fieldnames=['id']+KEYS+['rank_'+k for k in KEYS]);writer.writeheader()
            for i,row in enumerate(data):writer.writerow(dict(id=row['id'],**{k:row[k] for k in KEYS},**{'rank_'+k:float(ranks[k][i]) for k in KEYS}))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,corr,title in zip(axes,[action_corr,point_corr],[f'All {len(rows)} actions',f'{len(points)} grasp points (best direction)']):
        matrix=np.eye(5)
        for i,j in combinations(range(5),2):matrix[i,j]=matrix[j,i]=corr[KEYS[i]+':'+KEYS[j]]['spearman']
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(5),LABELS);ax.set_yticks(range(5),LABELS);ax.set_title(title+'\nSpearman correlation')
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='black')
    fig.colorbar(im,ax=axes,shrink=.7);fig.savefig(out/'correlations.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
    pranks={k:rankdata([-r[k] for r in points],method='average') for k in KEYS}
    for i,row in enumerate(points):
        yy=[pranks[k][i] for k in KEYS];ax.plot(range(5),yy,'o-',label=row['id'])
        ax.annotate(row['id'],(4,yy[-1]),xytext=(8,0),textcoords='offset points',fontsize=8)
    ax.set_xticks(range(5),LABELS);ax.set_ylabel('Rank (1 = highest; ties averaged)');ax.invert_yaxis()
    ax.set_title('Grasp point ranks: independent best eligible direction per score');ax.legend(ncol=5,loc='lower center');ax.grid(alpha=.25)
    fig.savefig(out/'point_rank_trends.png',dpi=160);plt.close(fig)
    with Image.open(round_path/'preference_review/current.png') as source: image=source.convert('RGB')
    draw=ImageDraw.Draw(image);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',20)
    for rank,row in enumerate(order(points,'Ps'),1):
        x,y=row['grasp_px'];draw.ellipse((x-6,y-6,x+6,y+6),fill='yellow',outline='black',width=2)
        draw.text((x+8,y-15),f"{row['id']} Ps={row['Ps']}",font=font,fill='yellow',stroke_width=2,stroke_fill='black')
    image.save(out/'grasp_points.png')
    for key in ('Pt','base','S'):
        for i,row in enumerate(order(rows,key)[:5],1):
            pref.io.overlay(round_path/'preference_review/current.png',pref.plan_for(row),out/f'{key}_{i}.png',
                f"{key} #{i}: {row['id']} | Ps={row['Ps']} Pt={row['Pt']} R={row['correction']:.2f} S={row['S']:.2f}")
    summary=['单次重置后的评分趋势比较',str(round_path.resolve()),
        f"alpha={alpha:.7f}；R版本={context['revision']}；{len(rows)}个动作，{len(points)}个有效抓点。",
        '以下使用本轮人工反馈之前的冻结评分。Ps按同一Pa重复赋给各方向，因此另做每抓点一次的汇总。','',
        '全部动作 Spearman：']
    for key in ('Ps:S','Pt:S','base:S','correction:S','Ps:correction','Pt:correction','Ps:Pt'):
        value=action_corr[key]['spearman'];summary.append(f'{key}: {value:.4f}' if value is not None else f'{key}: 未定义（常量评分）')
    summary+=['','每个抓点一次（Pt/base/R/S各自取该点的最高分方向）：']
    for key in ('Ps:S','Pt:S','base:S','Ps:correction','Pt:correction'):
        value=point_corr[key]['spearman'];summary.append(f'{key}: {value:.4f}' if value is not None else f'{key}: 未定义')
    summary+=['','各评分第一名：']+[f'{k}: {action_orders[k][0]}' for k in ('Pt','base','correction','S')]
    summary += [f"抓点Ps第一：{point_orders['Ps'][0]}",f"S与base的Top10重合：{overlap['base']}/10；S与Pt：{overlap['Pt']}/10。",
        f"S相对base翻转严格比较对数：{action_corr['base:S']['opposite_pairs']}/{action_corr['base:S']['strict_pairs']}。",
        f"同一个Pa内，S第一方向也是Pt最高方向：{sum(d['S_winner_is_Pt_max'] for d in directions)}/{len(directions)}。",
        f"分数标准差：Ps={scales['Ps']['std']:.2f}，alpha*Pt={scales['alpha_Pt']['std']:.2f}，R={scales['correction']['std']:.2f}。",
        '', '抓点排序（并列显示以ID排序，不意味着真实高低）：']+[f'{k}: '+ ' > '.join(point_orders[k]) for k in KEYS]
    summary += ['', '同一状态只能观察分数是否冗余，不能证明偏好模型有效或无效，也不能判断真实展开成功率。',
        '所有候选而非只看最终Top10；相关系数处理并列，不使用这些非独立候选的显著性p值。',
        '按抓点取max会受可执行方向数量影响，各点n和逐动作原始分数见CSV/JSON。']
    (out/'report.txt').write_text('\n'.join(summary)+'\n')
    cards=''
    for i in range(1,6):
        for key in ('Pt','base','S'):cards+=f'<article><h3>{key} 第{i}</h3><img src="{key}_{i}.png"></article>'
    page='<!doctype html><meta charset="utf-8"><title>Ps / Pt / 偏好排名趋势</title><style>body{font-family:system-ui;margin:28px;color:#182235}pre{white-space:pre-wrap;line-height:1.7}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}article{background:#f3f5f9;padding:10px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>'
    page+='<h1>重置后的一轮：抓点、轨迹、偏好评分趋势</h1><a href="report.txt">纯文字报告</a> · <a href="analysis.json">完整指标</a>'
    page+='<pre>'+html.escape('\n'.join(summary))+'</pre><h2>10个抓点与Ps</h2><img src="grasp_points.png"><h2>相关性与抓点排名趋势</h2><img src="correlations.png"><img src="point_rank_trends.png"><h2>相同候选按Pt / base / S取前5</h2><div class="grid">'+cards+'</div>'
    (out/'index.html').write_text(page)
    print('\n'.join(summary[:32]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('round',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.round,args.output)
