"""Descriptive registration field reports only; no fitted correction model."""
import csv,json
from pathlib import Path
import numpy as np

PLACEMENTS=('L-M','C-M','R-M')
METHODS=('stereo','direct')

def truth(value):return str(value).lower()=='true'
def metric(rows,method):
    valid=[r for r in rows if r.get('repeat_status')=='VALID' and truth(r.get(method+'_valid'))]
    if not valid:return dict(corners=0)
    vals=lambda key:np.array([float(r[key]) for r in valid])
    error=vals(method+'_error_px');du=vals(method+'_du');dv=vals(method+'_dv')
    depth=[float(r['native_depth_z_mm']) for r in valid if r.get('native_depth_z_mm') not in (None,'')]
    return dict(corners=len(valid),median_error_px=float(np.median(error)),p95_error_px=float(np.percentile(error,95)),
        median_du_px=float(np.median(du)),median_dv_px=float(np.median(dv)),
        median_native_depth_z_mm=float(np.median(depth)) if depth else None,
        median_rgb_u=float(np.median(vals('rgb_u'))),median_rgb_v=float(np.median(vals('rgb_v'))))

def generate(session):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((session/'session.json').read_text())
    samples=session/'all_samples.csv'
    rows=[]
    if samples.exists():
        with samples.open() as f:rows=list(csv.DictReader(f))
    if not samples.exists():
        from cloth_agent.experiments.registration_field import FIELDS
        with samples.open('w',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS).writeheader()
    summary={};repeat_rows=[]
    for placement in PLACEMENTS:
        selected=[r for r in rows if r['placement']==placement]
        accepted={r['repeat'] for r in selected if r['repeat_status']=='VALID'}
        attempts=[a for a in manifest.get('attempts',[]) if a['placement']==placement]
        summary[placement]=dict(valid_repeats=len(accepted),rejected_attempts=sum(not a['accepted'] for a in attempts),
            methods={m:metric(selected,m) for m in METHODS})
        for repeat in sorted(accepted,key=int):
            for method in METHODS:
                stats=metric([r for r in selected if r['repeat']==repeat],method)
                repeat_rows.append(dict(placement=placement,repeat=int(repeat),method=method,**stats))
        for method in METHODS:
            rep=[r for r in repeat_rows if r['placement']==placement and r['method']==method and r['corners']]
            if rep:
                summary[placement]['methods'][method]['repeat_median_du_range_px']=[min(r['median_du_px'] for r in rep),max(r['median_du_px'] for r in rep)]
                summary[placement]['methods'][method]['repeat_median_dv_range_px']=[min(r['median_dv_px'] for r in rep),max(r['median_dv_px'] for r in rep)]
    (session/'summary.json').write_text(json.dumps(dict(status=manifest['status'],placements=summary,repeat_statistics=repeat_rows),indent=2))
    if repeat_rows:
        with (session/'repeat_summary.csv').open('w',newline='') as f:
            keys=list(dict.fromkeys(k for r in repeat_rows for k in r))
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(repeat_rows)
    camera=json.loads((session/'camera.json').read_text()) if (session/'camera.json').exists() else None
    width=camera['profiles']['rgb']['width'] if camera else 1280
    height=camera['profiles']['rgb']['height'] if camera else 720
    colors={'L-M':'tab:blue','C-M':'tab:orange','R-M':'tab:green'}
    fig,axes=plt.subplots(1,2,figsize=(15,6),sharex=True,sharey=True)
    for ax,method in zip(axes,METHODS):
        for placement in PLACEMENTS:
            valid=[r for r in rows if r['placement']==placement and r['repeat_status']=='VALID' and truth(r.get(method+'_valid'))]
            if valid:
                q=ax.quiver([float(r['rgb_u']) for r in valid],[float(r['rgb_v']) for r in valid],
                    [float(r[method+'_du']) for r in valid],[float(r[method+'_dv']) for r in valid],
                    color=colors[placement],angles='xy',scale_units='xy',scale=.2,alpha=.22,width=.002)
                ax.scatter([],[],color=colors[placement],label=placement)
        ax.set(xlim=(0,width),ylim=(height,0),xlabel='Observed RGB u (px)',ylabel='Observed RGB v (px)',
            title=f'{method}: accepted corners; arrows x5')
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    fig.suptitle('Residual = projected RGB - observed RGB; common full-image coordinates')
    fig.tight_layout();fig.savefig(session/'residual_field.png',dpi=150);plt.close(fig)
    # Photographic overlays, one actual RGB per placement; all accepted vectors overlaid.
    from PIL import Image
    fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
    for column,placement in enumerate(PLACEMENTS):
        evidence=None
        for attempt in manifest.get('attempts',[]):
            if attempt['placement']==placement and attempt.get('accepted') and 'attempt' in attempt:
                candidate=session/placement/f'attempt_{attempt["attempt"]:03d}'/'off_last.png'
                if candidate.exists():evidence=np.asarray(Image.open(candidate));break
        for row_index,method in enumerate(METHODS):
            ax=axes[row_index,column]
            if evidence is not None:ax.imshow(evidence)
            else:ax.set_facecolor('#eeeeee')
            valid=[r for r in rows if r['placement']==placement and r['repeat_status']=='VALID' and truth(r.get(method+'_valid'))]
            if valid:
                ax.quiver([float(r['rgb_u']) for r in valid],[float(r['rgb_v']) for r in valid],
                    [float(r[method+'_du']) for r in valid],[float(r[method+'_dv']) for r in valid],
                    color='red',angles='xy',scale_units='xy',scale=.2,width=.004,alpha=.3)
            ax.set(xlim=(0,width),ylim=(height,0),title=f'{placement} / {method}',xlabel='u',ylabel='v')
    fig.suptitle('Actual RGB overlays; accepted residuals x5; missing placements left blank')
    fig.tight_layout();fig.savefig(session/'residual_overlays.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    extent=1.
    for ax,method in zip(axes,METHODS):
        for placement in PLACEMENTS:
            s=summary[placement]['methods'][method]
            if not s['corners']:continue
            du=s['median_du_px'];dv=s['median_dv_px'];extent=max(extent,abs(du),abs(dv))
            ax.quiver([0],[0],[du],[dv],angles='xy',scale_units='xy',scale=1,color=colors[placement])
            ax.scatter([du],[dv],color=colors[placement],label=f'{placement}: ({du:.2f}, {dv:.2f})')
        ax.set(title=method,xlabel='du (px), right positive',ylabel='dv (px), down positive')
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.grid(alpha=.3)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    for ax in axes:ax.set_xlim(-extent*1.3,extent*1.3);ax.set_ylim(extent*1.3,-extent*1.3);ax.set_aspect('equal')
    fig.suptitle('Placement median residual vectors (no fitted correction)')
    fig.tight_layout();fig.savefig(session/'median_vectors.png',dpi=150);plt.close(fig)
    lines=['# Camera B registration_field','',f'状态：{manifest["status"]}',manifest.get('error',''),
        '', '只测量 RGB↔IR/depth 残差；没有修改 calibration、Graph、grasp、replay，也没有连接机器人。',
        'L/C/R 指原始 RGB 图像左/中/右，M 指中排。板平放同一桌面；实际 native Z 另行记录。',
        'du/dv = factory 投影 RGB 像素 − 实际 RGB 角点。正 du 向右，正 dv 向下。',
        '', '| Placement | 有效 repeats | 拒绝 attempts | 方法 | 角点数 | error median/P95 px | median du/dv px | native Z median mm |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for placement,s in summary.items():
        for method,stats in s['methods'].items():
            if not stats['corners']:
                lines.append(f'| {placement} | {s["valid_repeats"]} | {s["rejected_attempts"]} | {method} | 0 | N/A | N/A | N/A |')
                continue
            z=stats['median_native_depth_z_mm'];zs=f'{z:.2f}' if z is not None else 'N/A'
            lines.append(f'| {placement} | {s["valid_repeats"]} | {s["rejected_attempts"]} | {method} | {stats["corners"]} | {stats["median_error_px"]:.3f}/{stats["p95_error_px"]:.3f} | {stats["median_du_px"]:.3f}/{stats["median_dv_px"]:.3f} | {zs} |')
    lines+=['','## 观测规律']
    for method in METHODS:
        stats=[summary[p]['methods'][method] for p in PLACEMENTS]
        if not all(s['corners'] for s in stats):
            lines.append(f'- {method}：三个位置尚未全部有有效数据，不判断跨位置规律。');continue
        vectors=np.array([[s['median_du_px'],s['median_dv_px']] for s in stats])
        d=vectors[-1]-vectors[0]
        spread=max(np.linalg.norm(a-b) for a in vectors for b in vectors)
        lines.append(f'- {method}：右减左的 median residual 为 ({d[0]:+.3f}, {d[1]:+.3f}) px；三位置 median 向量最大间距 {spread:.3f} px。')
        ranges=[s.get('repeat_median_du_range_px') for s in stats]
        lines.append(f'  每处 repeat 中位 du 范围：{ranges}；完整 du/dv repeat 范围见 summary.json。这里只描述差异，不自动判定固定平移或拟合模型。')
    lines+=['','## 统计口径与限制',
        '- placement median/P95 使用全部 accepted repeat 的 method-valid 角点；同一角点的 repeats 是相关观测，不视为独立样本置信区间。',
        '- paired-valid 至少 24 个角点才算有效 repeat；不使用被测 RGB residual 大小作接受阈值。',
        '- 所有 rejected attempts 均保留在 all_samples.csv 和各 attempt/result.json 中，不进入有效统计。',
        '- RGB motion RMS = sqrt(mean(du²+dv²))，比较 emitter-on/off/恢复后的角点，任一检查 >1 px 拒绝；匹配不足也拒绝。',
        '- 5×5 native depth 至少 18 个有效像素，取中位数，与历史检查一致；记录 Z spread 但不套用 grasp 的 spread 门控。IR 三角化检查正深度及 IR 重投影 ≤2 px。',
        '- 切换曝光沿用现有相机 adapter：start 保存/设置、stop 恢复；每个 repeat 在 finally 恢复 emitter 并核验。',
        '- 两条残差计算不使用 board 实际尺寸或 PnP；142 mm 板模型只用于摆放姿态检查。',
        '- M 只有一排，不能据此宣称已测得整个 r(u,v,Z) 的距离依赖。九点网格尚未启用，需先人工审核三点流程。',
        '- residual_field.png 的箭头放大 5 倍；median_vectors.png 使用真实残差像素坐标。没有 correction fitting 或部署。']
    (session/'summary.md').write_text('\n'.join(lines)+'\n')
    return summary
