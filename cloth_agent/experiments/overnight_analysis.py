"""Read recorded episodes, summarize outcomes and prepare independent human review."""
import argparse,csv,html,json,math,statistics
from collections import Counter
from pathlib import Path


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {} if default is None else default


def csv_write(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def wilson(k,n):
    if not n:return None
    z=1.96;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]


def collect(root):
    root=Path(root);episodes=[];rounds=[]
    completed={f'episode_{r["episode"]:04d}':r for r in read(root/'batch_summary.json').get('completed',[])}
    for ep in sorted(root.glob('episode_[0-9][0-9][0-9][0-9]')):
        s=read(ep/'episode_summary.json')
        if not s:continue
        p=read(ep/'protocol.json');radius=read(root/(ep.name+'.reset_radius.json'));init=read(ep/'initialization/result.json');finished=bool(s.get('finished_at'))
        grasp=read(ep/'initialization/grasp.json');release=read(ep/'initialization/release.json')
        actual_offset=None
        if grasp.get('world_xyz_mm') and grasp.get('center_world_xyz_mm'):
            actual_offset=math.dist(grasp['world_xyz_mm'][:2],grasp['center_world_xyz_mm'][:2])
        reason=s.get('termination_reason');clean=not s.get('cleanup_error')
        record=completed.get(ep.name,{})
        eligible=finished and clean and record.get('returncode',0)==0 and not record.get('operator_stopped_current',False) and reason in ('DONE','ROUND_LIMIT','NO_IMPROVEMENT')
        row=dict(episode=ep.name,termination=reason,finished=finished,eligible_for_done_rate=eligible,model_done=bool(eligible and reason=='DONE'),
                 radius_mm=radius.get('radius_mm',init.get('center_radius_mm',p.get('initialization_center_radius_mm'))),
                 reset_seed=init.get('seed',radius.get('reset_seed')),actions=s.get('physical_actions',0),
                 reset_actual_offset_mm=actual_offset,reset_sampling=grasp.get('sampling','RANDOM_ACCEPTED' if grasp else None),
                 reset_release_height_mm=release.get('lift_height'),elapsed_s=record.get('elapsed_s'),
                 initialization_status=init.get('status','SKIPPED' if not init else None),
                 planning_s=s.get('total_planning_s',0),robot_s=s.get('total_robot_execution_s',0),
                 started_at=s.get('started_at'),finished_at=s.get('finished_at'),detail=s.get('termination_detail'),cleanup_error=s.get('cleanup_error'))
        episodes.append(row)
        for folder in sorted(ep.glob('round_*')):
            if not (folder/'before/rgb.png').exists():continue
            plan=read(folder/'plan.json');ex=read(folder/'execution.json');tm=read(folder/'timing.json');m=read(folder/'model_metrics.json');g=read(folder/'ggo/report.json')
            pa,pb=plan.get('grasp_px'),plan.get('target_px');dx=pb[0]-pa[0] if pa and pb else None;dy=pb[1]-pa[1] if pa and pb else None
            rounds.append(dict(episode=ep.name,round=folder.name,decision=plan.get('decision'),grasp_px=json.dumps(pa),target_px=json.dumps(pb),dx_px=dx,dy_px=dy,
                reason=plan.get('short_reason'),action_purpose=plan.get('action_purpose'),expected_stop_state=plan.get('expected_stop_state'),
                is_smoothing=plan.get('is_smoothing'),execution=ex.get('status'),
                acquisition=ex.get('acquisition','NOT_VERIFIED'),planning_s=tm.get('planning_s',m.get('wall_latency_s')),
                anchor_s=m.get('anchor_wall_s'),action_model_s=m.get('action_model_wall_s'),geometry_s=m.get('geometry_wall_s'),
                robot_s=tm.get('robot_execution_s'),total_s=tm.get('total_round_s'),
                registration=g.get('registration',{}).get('status','NOT_RECORDED'),registration_reason=g.get('registration',{}).get('reason'),
                deformation=g.get('deformation',{}).get('status'),
                depth_recovery=(folder/'depth_recovery').exists(),target_replan=(folder/'target_replan').exists(),
                before=str(folder/'before/rgb.png'),after=str(folder/'after/rgb.png'),action_image=str(folder/'proposed_action_overlay.png')))
    return episodes,rounds


def stats(episodes,rounds):
    eligible=[r for r in episodes if r['eligible_for_done_rate']];k=sum(r['model_done'] for r in eligible);n=len(eligible)
    executed=[r for r in rounds if r['execution']=='EXECUTED'];g=Counter(r['registration'] for r in rounds)
    return dict(attempts=len(episodes),finished=sum(r['finished'] for r in episodes),eligible_completed=n,model_done=k,
                model_done_rate=k/n if n else None,model_done_wilson95=wilson(k,n),
                terminations=dict(Counter(str(r['termination']) for r in episodes)),excluded_from_done_rate=[r['episode'] for r in episodes if not r['eligible_for_done_rate']],
                actions_executed=len(executed),registration=dict(g),registration_ok_rate=g['OK']/len(rounds) if rounds else None,
                mean_actions_per_completed=statistics.mean(r['actions'] for r in eligible) if eligible else None,
                mean_planning_s=statistics.mean(r['planning_s'] for r in rounds if r['planning_s'] is not None) if any(r['planning_s'] is not None for r in rounds) else None,
                leftward_actions=sum(r['dx_px'] is not None and r['dx_px']<0 for r in executed),
                rightward_actions=sum(r['dx_px'] is not None and r['dx_px']>0 for r in executed),
                note='DONE is a model decision, not independently verified unfolding success. Commands completed do not prove cloth acquisition.')


def analyze(root,historical=None):
    root=Path(root).resolve();out=root/'analysis';out.mkdir(exist_ok=True)
    episodes,rounds=collect(root);summary=stats(episodes,rounds)
    labels={}
    labelpath=out/'human_review.csv'
    if labelpath.exists():
        with labelpath.open() as f:labels={r['episode']:r for r in csv.DictReader(f)}
    review=[]
    for r in episodes:
        previous=labels.get(r['episode'],{})
        review.append(dict(episode=r['episode'],human_done=previous.get('human_done',''),notes=previous.get('notes','')))
    csv_write(labelpath,review,['episode','human_done','notes'])
    scored=[r for r in review if r['human_done'] in ('DONE','NOT_DONE')];true=sum(r['human_done']=='DONE' for r in scored)
    summary['human_review']=dict(labeled=len(scored),done=true,rate=true/len(scored) if scored else None,
                                pending_or_uncertain=len(review)-len(scored),allowed_labels=['DONE','NOT_DONE','UNCERTAIN'])
    bins=[]
    for lo,hi in [(0,10),(10,20),(20,30),(30,40)]:
        selected=[r for r in episodes if r['radius_mm'] is not None and lo<=r['radius_mm'] and (r['radius_mm']<hi or hi==40 and r['radius_mm']==hi)]
        bins.append(dict(radius_bin=f'[{lo},{hi}{"]" if hi==40 else ")"}',**stats(selected,[])))
    summary['radius_bins']=bins
    if historical:
        comparisons=[]
        for path in historical:
            eps,rr=collect(Path(path).resolve());comparisons.append(dict(source=str(Path(path).resolve()),statistics=stats(eps,rr),protocol=read(Path(path)/'protocol.json')))
        summary['historical_comparison']=comparisons
        summary['comparison_limitations']='Observational comparison only: randomized radius distribution, initial cloth state, timestamps and prompts may differ. Compare human-reviewed outcomes and matched radius bins; no causal improvement claim.'
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if episodes:csv_write(out/'episodes.csv',episodes,list(episodes[0]))
    if rounds:csv_write(out/'rounds.csv',rounds,list(rounds[0]))
    rate=f'{100*summary["model_done_rate"]:.1f}%' if summary['model_done_rate'] is not None else '尚无完整样本'
    md=f'''# 连续展开实验分析

已记录 {len(episodes)} 次，完成且纳入DONE统计 {summary['eligible_completed']} 次，模型DONE {summary['model_done']} 次，比例 {rate}。

这是模型DONE率，不是人工验证成功率。异常、人工停止、未结束及清理失败单独列出，不隐藏。DONE或动作上限后才自动reset；达到时长或磁盘保留条件后不开始新试验。

人工复核：{len(scored)} 次，确认展开 {true} 次。填写 human_review.csv 的 human_done 列（DONE / NOT_DONE / UNCERTAIN）后重新运行分析，不覆盖已有标签。

逐次明细：episodes.csv；逐轮Pa/Pb、理由、执行、几何、恢复和耗时：rounds.csv。初态以每次实际RGB-D为准，种子不能复现布料物理状态。

几何配准统计：{summary['registration']}。运动命令完成不等于夹住且未掉落；acquisition未验证时不当作抓取成功。

左右拉动次数仅为描述，不能据此认定无改善。未调用额外在线评审模型。

历史比较只有相关性证据：新旧初态与半径分布不一致，不能直接把比例变化归因于代码。应结合人工复核和相近半径分组比较。
'''
    if summary.get('historical_comparison'):
        md+='\n## 历史对比（模型DONE率，非因果结论）\n\n|记录|已结束纳入统计|模型DONE|比例|\n|---|---:|---:|---:|\n'
        for item in summary['historical_comparison']:
            st=item['statistics'];rr=f"{st['model_done_rate']*100:.1f}%" if st['model_done_rate'] is not None else '无有效样本'
            md+=f"|{item['source']}|{st['eligible_completed']}|{st['model_done']}|{rr}|\n"
    (out/'report.md').write_text(md)
    parts=['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><style>body{font:18px system-ui;margin:25px}img{max-width:48%}pre{white-space:pre-wrap}section{border-top:1px solid #aaa;padding:15px}</style><h1>连续展开实验报告</h1><pre>'+html.escape(md)+'</pre>']
    for ep in episodes:
        er=[r for r in rounds if r['episode']==ep['episode']]
        if not er:continue
        first,last=er[0],er[-1];final=Path(last['after']) if Path(last['after']).exists() else Path(last['before'])
        parts.append('<section><h2>'+ep['episode']+'</h2><p>'+html.escape(str(ep['termination'])+'；reset半径 '+str(ep['radius_mm'])+' mm；动作数 '+str(ep['actions']))+'</p>')
        def rel(p):return '../'+str(Path(p).relative_to(root))
        parts.append('<img src="'+rel(first['before'])+'"><img src="'+rel(final)+'"><p>左：实际初态；右：最后观测（不等于成功标签）。</p>')
        for rr in er:
            prefix='../'+rr['episode']+'/'+rr['round']
            parts.append('<details><summary>'+html.escape(rr['round']+' '+str(rr['reason'])+' · '+str(rr['execution']))+'</summary><a href="'+prefix+'/decision.html">本轮证据</a><img src="'+prefix+'/proposed_action_overlay.png"><img src="'+prefix+'/after/rgb.png"></details>')
        parts.append('</section>')
    (out/'index.html').write_text('\n'.join(parts))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--historical',type=Path,nargs='*');a=p.parse_args()
    summary=analyze(a.root,a.historical);print(json.dumps({k:summary[k] for k in ['attempts','eligible_completed','model_done','model_done_rate','human_review']},ensure_ascii=False))

if __name__=='__main__':main()
