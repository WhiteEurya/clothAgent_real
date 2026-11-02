"""Label-free measurement replay + explicitly synthetic transform controls."""
import argparse,copy,html,json,shutil,time
from dataclasses import asdict,replace
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .runtime.dataset import load_observation
from .runtime.interfaces import Action
from .cv_harness.outcome_observer import observe,execution_status
from .cv_harness.segmentation import segment
from .experiment import read,save,sha,overlay


def action(plan):return Action(tuple(plan['grasp_px']),tuple(plan['target_px']),plan.get('grasp_angle_deg'),plan.get('action_purpose'))
def fmt(value,percent=False):return 'UNKNOWN' if value is None else f'{value*100 if percent else value:.3f}'+('%' if percent else '')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-selection',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    # Preserve membership to replay the same experiment; discard all annotations/groups.
    selected=[{'id':f'S{i+1:02d}','source':str(Path(x['source']).resolve())} for i,x in enumerate(read(args.source_selection))]
    save(root/'selection.json',selected)
    snapshot=root/'source_snapshot';shutil.copytree(Path(__file__).parent,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(Path(__file__).parents[1]/'geometry_report.py',snapshot/'external_geometry_report.py')
    save(root/'protocol.json',{'human_labels_used':False,'human_groups_used':False,'source_membership':'same prior diagnostic sample; selection originally included human screening, so not an unbiased automated sample',
        'CV':'existing deployed measurement functions, unchanged','controls':['exact self-image','image translation dx20 dy-10','image rotation5deg + translation15 -8'],'synthetic_is_not_robot_experiment':True,'static_camera_noise_calibration':False,'semantic_outcome_evaluation':False,'model_calls':0,'robot_actions':0})
    rows=[];controls=[]
    for item in selected:
        folder=root/item['id'];folder.mkdir();source=Path(item['source']);log=read(source/'execution.json');r={**item,'execution':execution_status(log),'raw_execution_log':log};save(folder/'execution.json',log)
        for side in ('before','after'):
            f=source/side/'rgb.png'
            if f.exists():shutil.copyfile(f,folder/(side+'.png'));r[side+'_sha256']=sha(f)
        if not (source/'after/observation.json').exists():
            r.update(status='NO_POST_OBSERVATION',reason='no measured after RGB-D; no substitution from another step');save(folder/'result.json',r);rows.append(r);continue
        before=load_observation(source/'before');after=load_observation(source/'after');plan=read(source/'dispatch_intent.json')['plan'];act=action(plan)
        started=time.perf_counter();out,mb,ma=observe(before,act,after,log);r.update(status='MEASURED',outcome=asdict(out),call_wall_ms=(time.perf_counter()-started)*1000)
        save(folder/'outcome.json',asdict(out))
        for side,rgb,mask in [('before',before.rgb,mb),('after',after.rgb,ma)]:Image.fromarray(overlay(rgb,mask)).save(folder/(side+'_mask.png'))
        if out.motion['status']=='OK':
            warp=cv2.warpAffine(mb.astype('uint8'),np.array(out.motion['matrix']),ma.shape[::-1],flags=cv2.INTER_NEAREST).astype(bool);view=(after.rgb*.5).astype('uint8');view[ma&~warp]=[0,255,100];view[warp&~ma]=[255,60,60];Image.fromarray(view).save(folder/'aligned_difference.png')
        save(folder/'result.json',r);rows.append(r)
        h,w=before.rgb.shape[:2];ys,xs=np.where(mb)
        if len(xs)==0:continue
        center=np.array([xs.mean(),ys.mean()]);rotation=cv2.getRotationMatrix2D(tuple(center),5,1);rotation[:,2]+=[15,-8]
        transforms={'self':np.float64([[1,0,0],[0,1,0]]),'translate':np.float64([[1,0,20],[0,1,-10]]),'rotate_translate':rotation}
        samples=np.c_[xs[::max(1,len(xs)//1000)],ys[::max(1,len(ys)//1000)],np.ones(len(xs[::max(1,len(xs)//1000)]))]
        for name,matrix in transforms.items():
            generated=cv2.warpAffine(before.rgb,matrix,(w,h),flags=cv2.INTER_LINEAR);expected_mask=cv2.warpAffine(mb.astype('uint8'),matrix,(w,h),flags=cv2.INTER_NEAREST)
            synthetic=replace(before,rgb=generated,depth=None,world_xyz_mm=None,timestamp=before.timestamp+1,provenance=copy.deepcopy(before.provenance))
            synth_before=replace(before,depth=None,world_xyz_mm=None,provenance=copy.deepcopy(before.provenance))
            result,_,_=observe(synth_before,act,synthetic,{'status':'SYNTHETIC_CONTROL','physical_execution':False})
            c={'case':item['id'],'control':name,'known_transform':matrix.tolist(),'status':result.motion['status'],'reason':result.motion.get('reason'),'cv_ms':result.latency_ms,'expected_raster_area_delta':float((int(expected_mask.sum())-int(mb.sum()))/max(1,int(mb.sum()))),'measured_area_delta':result.coverage['delta_ratio'],'aligned_iou':result.motion.get('aligned_iou')}
            if result.motion['status']=='OK':
                estimate=np.asarray(result.motion['matrix']);error=samples@estimate.T-samples@matrix.T
                c['transform_rmse_px']=float(np.sqrt(np.mean(np.sum(error*error,axis=1))))
                expected_center=matrix@np.r_[center,1];estimated_center=estimate@np.r_[center,1];c['center_error_px']=float(np.linalg.norm(expected_center-estimated_center))
            controls.append(c)
            if item['id']=='S01':Image.fromarray(generated).save(root/f'synthetic_example_{name}.png')
    save(root/'results.json',rows);save(root/'synthetic_controls.json',controls)
    paired=[r['outcome'] for r in rows if r['status']=='MEASURED'];summary={'cases':len(rows),'real_pairs':len(paired),'missing_after':len(rows)-len(paired),'real_area_available':sum(o['coverage']['delta_ratio'] is not None for o in paired),'real_registration_available':sum(o['motion']['status']=='OK' for o in paired),'mean_cv_ms':float(np.mean([o['latency_ms'] for o in paired])),'p95_cv_ms':float(np.percentile([o['latency_ms'] for o in paired],95)),'semantic_accuracy':'NOT_EVALUATED_NO_GROUND_TRUTH','GPT_experiment':'NOT_STARTED: synthetic transform validity does not establish real unfolding/drag discrimination'}
    for name in ('self','translate','rotate_translate'):
        group=[c for c in controls if c['control']==name];ok=[c for c in group if c['status']=='OK']
        summary[name]={'total':len(group),'registration_available':len(ok),'mean_transform_rmse_px':float(np.mean([c['transform_rmse_px'] for c in ok])) if ok else None,'max_transform_rmse_px':max((c['transform_rmse_px'] for c in ok),default=None)}
    save(root/'summary.json',summary);save(root/'source_hashes.json',{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')});render(root,rows,controls,summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


def render(root,rows,controls,summary):
    intro='同一批历史案例重新运行现有CV。没有加载人工结果标签用于评价，也没有沿用展开/拖动/错误动作的人工分组。样本成员来自上次筛选，因此并非完全自动抽样。真实数据仅报告测量和缺测；另有明确已知变换的合成控制，不是真机实验，也不是静态相机噪声标定。'
    parts=['<!doctype html><meta charset="utf-8"><title>无人工真值 CV 重跑</title><style>body{font:17px system-ui;margin:24px}.pair{display:flex;gap:10px}.pair img{width:49%}pre{white-space:pre-wrap}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}section{border-top:1px solid #aaa;padding:15px 0}</style><h1>CV测量重跑：不用人工分类真值</h1><p>'+intro+'</p><a href="report.md">文字报告</a><h2>汇总</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><h2>真实操作测量</h2><table><tr><th>案例</th><th>执行状态</th><th>面积变化</th><th>配准</th><th>对齐IoU</th><th>形状残差</th></tr>']
    md=['# 无人工真值 CV 重跑','',intro,'','## 真实操作测量','','|案例|执行状态|面积变化|配准|对齐IoU|形状残差|','|---|---|---|---|---|---|']
    for r in rows:
        o=r.get('outcome',{});m=o.get('motion',{});cells=[r['id'],r['execution']['status'],fmt(o.get('coverage',{}).get('delta_ratio'),True),m.get('status','无后图'),fmt(m.get('aligned_iou')),fmt(m.get('shape_change'))]
        parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>');md.append('|'+ '|'.join(cells)+'|')
    parts.append('</table><h2>合成控制：有已知变换，不使用人工标签</h2><p>原图自比较不代表重复拍照；全图数字变换不代表真实布料运动。误差只在有估计的样本上计算，UNKNOWN单列。</p><table><tr><th>案例</th><th>变换</th><th>状态</th><th>变换RMSE(px)</th><th>原因</th></tr>')
    for c in controls:parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [c['case'],c['control'],c['status'],fmt(c.get('transform_rmse_px')),c.get('reason','')])+'</tr>')
    parts.append('</table>')
    for r in rows:
        key=r['id'];parts.append('<section><h2>'+key+'</h2><p>'+html.escape(r['source'])+'</p><div class="pair"><img src="'+key+'/before.png">'+('<img src="'+key+'/after.png">' if r.get('after_sha256') else '<p>缺少实际后图，不补全。</p>')+'</div>')
        if r['status']=='MEASURED':
            parts.append(f'<details><summary>mask和测量详情</summary><div class="pair"><img src="{key}/before_mask.png"><img src="{key}/after_mask.png"></div><pre>'+html.escape(json.dumps(r['outcome'],ensure_ascii=False,indent=2))+'</pre>')
            if (root/key/'aligned_difference.png').exists():parts.append(f'<p>绿：配准后新出现区域；红：消失区域，颜色不表示好坏。</p><img style="max-width:90%" src="{key}/aligned_difference.png">')
            parts.append('</details>')
        parts.append('<pre>'+html.escape(json.dumps(r['raw_execution_log'],ensure_ascii=False,indent=2))+'</pre></section>')
    (root/'index.html').write_text('\n'.join(parts));md+=['','## 汇总','',json.dumps(summary,ensure_ascii=False,indent=2),'','不能根据这些数据报告展开识别准确率。真实的静止重复采集、受控整体运动，以及明确物理量的参考测量仍未完成。未调用GPT、未执行机器人、未修改生产CV。'];(root/'report.md').write_text('\n'.join(md))

if __name__=='__main__':main()
