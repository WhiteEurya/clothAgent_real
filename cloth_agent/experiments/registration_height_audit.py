"""Camera-only height-layer audit, reusing the unchanged registration collector."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cloth_agent.experiments import registration_field as rf
from cloth_agent.experiments.registration_field_report import metric, truth

DEFAULT_BASELINE=rf.ROOT/'runs/registration_field/20261016T080154_133094Z'

def read_session(path):
    path=Path(path).resolve()
    manifest=json.loads((path/'session.json').read_text())
    camera=json.loads((path/'camera.json').read_text())
    with (path/'all_samples.csv').open() as f:rows=list(csv.DictReader(f))
    return manifest,camera,rows

def check_camera(a,b):
    # This comparison must not attribute changed stream/calibration settings to height.
    for key in ('serial','profiles','extrinsics','depth_scale','depth_capture_options'):
        if a.get(key)!=b.get(key):raise ValueError(f'Camera configuration differs: {key}; comparison blocked')

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def compare(baseline,current,out,nominal_height_mm):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bm,bc,br=read_session(baseline);cm,cc,cr=read_session(current);check_camera(bc,cc)
    stats=[];binned=[];combined=[]
    for label,path,rows in [('table',baseline,br),('raised',current,cr)]:
        for row in rows:
            combined.append(dict(row,height_layer=label,nominal_raise_mm=0 if label=='table' else nominal_height_mm,source_session=str(Path(path).resolve())))
        for placement in rf.PLACEMENTS:
            selected=[r for r in rows if r['placement']==placement]
            for method in ('direct','stereo'):
                s=metric(selected,method)
                stats.append(dict(layer=label,placement=placement,method=method,**s))
    # Compare only common fixed RGB tiles; placement medians alone change image coverage.
    # Repeated frames are collapsed per tile/repeat, then per layer, not independent points.
    for method in ('direct','stereo'):
        layers=[]
        for rows in (br,cr):
            groups={}
            for r in rows:
                if r['repeat_status']!='VALID' or not truth(r.get(method+'_valid')):continue
                u,v=float(r['rgb_u']),float(r['rgb_v'])
                key=(int(u//64),int(v//64),r['placement'],r['repeat'])
                z=r.get('native_depth_z_mm')
                if not z:continue
                groups.setdefault(key,[]).append([u,v,float(r[method+'_du']),float(r[method+'_dv']),float(z)])
            tiles={}
            for key,values in groups.items():tiles.setdefault(key[:2],[]).append(np.median(values,axis=0))
            layers.append({k:np.median(v,axis=0) for k,v in tiles.items()})
        for key in sorted(layers[0].keys()&layers[1].keys()):
            a,b=layers[0][key],layers[1][key]
            binned.append(dict(method=method,tile_u=key[0],tile_v=key[1],baseline_u=a[0],baseline_v=a[1],raised_u=b[0],raised_v=b[1],coverage_center_distance_px=float(np.linalg.norm(b[:2]-a[:2])),baseline_z_mm=a[4],raised_z_mm=b[4],delta_z_mm=b[4]-a[4],baseline_du=a[2],baseline_dv=a[3],raised_du=b[2],raised_dv=b[3],delta_du=b[2]-a[2],delta_dv=b[3]-a[3]))
    write_csv(out/'layer_samples.csv',combined);write_csv(out/'layer_statistics.csv',stats)
    write_csv(out/'common_rgb_tiles.csv',binned)
    result=dict(baseline=str(Path(baseline).resolve()),current=str(Path(current).resolve()),baseline_status=bm['status'],current_status=cm['status'],nominal_raise_mm=nominal_height_mm,nominal_height_is_not_measured_depth=True,calibration_changed=False,correction_fitted=False,robot_connected=False,statistics=stats,common_tiles=binned)
    rf.save(out/'comparison.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,method in enumerate(('direct','stereo')):
        for layer,marker in [('table','o'),('raised','s')]:
            ss=[s for s in stats if s['method']==method and s['layer']==layer and s['corners']]
            for j,component in enumerate(('du','dv')):
                axes[i,j].plot([s['median_rgb_u'] for s in ss],[s[f'median_{component}_px'] for s in ss],marker=marker,label=layer)
                axes[i,j].set(xlabel='Median RGB u (px)',ylabel=f'{component} (px)',title=f'{method}: placement medians');axes[i,j].grid(alpha=.3);axes[i,j].legend()
    fig.tight_layout();fig.savefig(out/'height_comparison.png',dpi=150);plt.close(fig)
    lines=['# Camera B 多高度对照','',f'桌面基线：{baseline}',f'新增层：{current}',f'采集状态：{bm["status"]} / {cm["status"]}',f'操作者声明抬高约 {nominal_height_mm:g} mm；这不是测得的Z变化。','',
        '| 层 | 位置 | 方法 | 样本角点 | 实测native Z mm | du px | dv px | error median px |','|---|---|---|---:|---:|---:|---:|---:|']
    for s in stats:
        if s['corners']:
            z=s['median_native_depth_z_mm'];zs=f'{z:.2f}' if z is not None else 'NA'
            lines.append(f'| {s["layer"]} | {s["placement"]} | {s["method"]} | {s["corners"]} | {zs} | {s["median_du_px"]:.3f} | {s["median_dv_px"]:.3f} | {s["median_error_px"]:.3f} |')
    lines+=['','## 解释限制','',
        '- 原始两方法、24共同角点门槛、1 px运动门槛、设置恢复均复用原采集器。',
        '- layer_samples.csv保留被拒绝记录；统计仅使用VALID且对应方法有效的角点。',
        '- common_rgb_tiles.csv为固定64×64 RGB网格的共有覆盖比较；先按每个repeat聚合，再取中位数。格内坐标并不完全相同，必须查看coverage_center_distance_px，不能当作精确同像素深度对照。',
        '- 三位置整体中位数可能受空间覆盖/角点缺失影响；不能仅凭其差异唯一判定内参、旋转或平移外参。',
        '- 如果实际Z没有拉开，本次没有形成有效距离对照。不要用名义50 mm代替测量。',
        '- 尚未进行候选标定拟合或留出验证；本入口只收集和描述证据。',
        '- PARTIAL/FAILED的session只供查看，不能宣称完整三点采集完成。','', '![高度对照](height_comparison.png)']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result

class HeightUI:
    """Forward existing UI, delaying final notification until comparison is saved."""
    def __init__(self,ui,height):
        object.__setattr__(self,'ui',ui);object.__setattr__(self,'height',height);object.__setattr__(self,'end',None)
    def __getattr__(self,name):return getattr(self.ui,name)
    def __setattr__(self,name,value):setattr(self.ui,name,value)
    def update(self,message,count,total):
        if message.startswith('请将'):
            message=message+f' 板用稳固垫块整体抬高约{self.height:g} mm，保持平放，不要手持。'
        self.ui.update(message+f' ｜抬高层约{self.height:g} mm',count,total)
    def finished(self,status,session,error):object.__setattr__(self,'end',(status,session,error))

def run(args,ui):
    root=args.output/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True,exist_ok=False)
    rf.save(root/'height_session.json',dict(status='STARTED',baseline=str(args.baseline.resolve()),nominal_raise_mm=args.raise_mm,robot_connected=False,calibration_changed=False))
    proxy=HeightUI(ui,args.raise_mm)
    # UI widgets are only updated by the main Tk thread; worker uses existing state callbacks.
    native=SimpleNamespace(output=root/'capture',config=args.config,repeats=args.repeats,max_attempts=args.max_attempts)
    session=rf.run_session(native,proxy)
    status,_,error=proxy.end
    try:
        compare(args.baseline,session,root,args.raise_mm)
    except Exception as exc:
        error=f'{error} Comparison failed: {type(exc).__name__}: {exc}';status='COMPARISON_FAILED'
    rf.save(root/'height_session.json',dict(status=status,capture_session=str(session),baseline=str(args.baseline.resolve()),nominal_raise_mm=args.raise_mm,error=error,robot_connected=False,calibration_changed=False))
    ui.finished(status,str(root),error)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE);r.add_argument('--raise-mm',type=float,default=50)
    r.add_argument('--output',type=Path,default=rf.ROOT/'runs/registration_height_audit');r.add_argument('--config',type=Path,default=rf.ROOT/'config/dual_arm.local.json')
    r.add_argument('--repeats',type=int,default=10);r.add_argument('--max-attempts',type=int,default=40)
    c=sub.add_parser('compare');c.add_argument('--baseline',type=Path,required=True);c.add_argument('--current',type=Path,required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--raise-mm',type=float,required=True)
    args=parser.parse_args()
    if not np.isfinite(args.raise_mm) or args.raise_mm<=0:parser.error('raise-mm must be finite and positive')
    bm,_,_=read_session(args.baseline)
    if bm['status']!='COMPLETE':parser.error('Baseline must be a COMPLETE session')
    if args.command=='compare':compare(args.baseline,args.current,args.output,args.raise_mm);return 0
    if args.repeats<1 or args.max_attempts<args.repeats:parser.error('Require max-attempts >= repeats >= 1')
    print(f'仅相机实验：将板稳固垫高约{args.raise_mm:g} mm，保持平放、斜置约30°。按界面依次采L/C/R。基线只读：{args.baseline}',flush=True)
    from cloth_agent.experiments.registration_field_ui import launch
    return launch(lambda ui:run(args,ui))

if __name__=='__main__':raise SystemExit(main())
