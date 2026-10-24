"""Offline recorded grasp geometry audit. No model, robot, or runtime mutation."""
from __future__ import annotations
import argparse
from collections import deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.dual_arm.geometry import DualArmError

ROOT=Path('reports/host_grasp_geometry_audit_v2')
SOURCE=Path('reports/one_shot_goal_full_episode_v1_continued_summary/per_round_metrics.csv')
RADII=(3,5,10,15)
FIT_WINDOWS=(5,31)
LIMIT=8.
PROTECTED=['cloth_agent/dual_arm/observation.py','cloth_agent/dual_arm/kinematics.py',
 'cloth_agent/gamepad_collect_runtime.py','cloth_agent/codex_native_runtime.py','config/dual_arm.local.json',
 'cloth_agent/experiments/one_shot_goal_planner_probe_v1.py','cloth_agent/experiments/one_shot_goal_full_episode_v1.py']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csv_save(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else clean(v) for k,v in row.items()})

def depth_stats(obs,point):
    x,y=map(int,point);h,w=obs.xyz.shape[:2]
    result=dict(pixel=[x,y],supported_interior=2<=x<w-2 and 2<=y<h-2)
    if not result['supported_interior']:
        return dict(result,passed=False,error='grasp pixel outside supported image interior')
    patch=obs.xyz[y-2:y+3,x-2:x+3];valid=np.isfinite(patch).all(axis=2);v=patch[valid]
    result.update(valid_count=int(valid.sum()),center_valid=bool(np.isfinite(obs.xyz[y,x]).all()),
        world_z_min_mm=float(v[:,2].min()) if len(v) else None,
        world_z_max_mm=float(v[:,2].max()) if len(v) else None,
        world_z_range_mm=float(np.ptp(v[:,2])) if len(v) else None)
    try:
        result.update(sample_world_xyz_mm=obs.sample([x,y],LIMIT).tolist(),passed=True,error=None)
    except DualArmError as exc:
        result.update(sample_world_xyz_mm=None,passed=False,error=str(exc))
    # Independent literal reproduction must agree with the current production function.
    manual=bool(valid.sum()>=18 and np.isfinite(obs.xyz[y,x]).all() and np.ptp(v[:,2])<=LIMIT)
    if manual!=result['passed']:raise AssertionError('literal rule disagrees with Observation.sample')
    return result

def fit_plane(points):
    """Huber IRLS in centered/scaled world XY, vertical world-Z residuals, no pass gate."""
    p=np.asarray(points,dtype=float)
    if len(p)<18:return dict(status='INSUFFICIENT_POINTS',n=len(p)),None,None
    origin=p.mean(axis=0);scale=p[:,:2].std(axis=0)
    if np.any(scale<1e-9):return dict(status='DEGENERATE_XY',n=len(p)),None,None
    A=np.c_[(p[:,:2]-origin[:2])/scale,np.ones(len(p))];z=p[:,2]-origin[2]
    if np.linalg.matrix_rank(A)<3:return dict(status='DEGENERATE_XY',n=len(p)),None,None
    beta=np.linalg.lstsq(A,z,rcond=None)[0]
    weights=np.ones(len(p));iterations=0
    for i in range(50):
        residual=z-A@beta
        robust_scale=max(1.4826*float(np.median(np.abs(residual-np.median(residual)))),1e-6)
        u=np.abs(residual)/(1.345*robust_scale)
        weights=np.ones(len(p));np.divide(1.,u,out=weights,where=u>1)
        updated=np.linalg.lstsq(A*np.sqrt(weights[:,None]),z*np.sqrt(weights),rcond=None)[0]
        iterations=i+1
        if np.linalg.norm(updated-beta)<1e-9:
            beta=updated;break
        beta=updated
    slopes=beta[:2]/scale;c=origin[2]+beta[2]-slopes@origin[:2]
    prediction=p[:,:2]@slopes+c;residual=p[:,2]-prediction
    return dict(status='FIT',n=len(p),a=float(slopes[0]),b=float(slopes[1]),c=float(c),
        plane_slope=float(np.linalg.norm(slopes)),plane_slope_degrees=float(np.degrees(np.arctan(np.linalg.norm(slopes)))),
        raw_z_range_mm=float(np.ptp(p[:,2])),residual_rmse_mm=float(np.sqrt(np.mean(residual**2))),
        median_absolute_residual_mm=float(np.median(np.abs(residual))),
        residual_p90_p10_mm=float(np.percentile(residual,90)-np.percentile(residual,10)),
        max_absolute_residual_mm=float(np.max(np.abs(residual))),iterations=iterations,
        xy_design_condition_number=float(np.linalg.cond(A)),residual_definition='measured world Z minus fitted z=aX+bY+c; all valid points, including downweighted points'),prediction,residual

def patch_fit(obs,point,window,folder):
    x,y=point;r=window//2;h,w=obs.xyz.shape[:2]
    x0,x1=max(0,x-r),min(w,x+r+1);y0,y1=max(0,y-r),min(h,y+r+1)
    xyz=obs.xyz[y0:y1,x0:x1];valid=np.isfinite(xyz).all(axis=2);p=xyz[valid]
    metrics,pred,res=fit_plane(p)
    metrics.update(window_px=window,patch_pixels=int(valid.size),valid_fraction=float(valid.mean()),bounds=[x0,y0,x1,y1])
    residual=np.full(valid.shape,np.nan);plane=np.full(valid.shape,np.nan)
    if pred is not None:
        residual[valid]=res;plane[valid]=pred
        jumps=[]
        for delta in (np.diff(residual,axis=0),np.diff(residual,axis=1)):
            jumps.extend(np.abs(delta[np.isfinite(delta)]).tolist())
        metrics['max_adjacent_residual_difference_mm']=max(jumps) if jumps else None
    np.savez(folder/f'plane_{window:02d}.npz',world_xyz=xyz,valid=valid,fitted_world_z=plane,residual_world_z=residual)
    point_rows=[]
    for yy,xx in np.ndindex(valid.shape):
        point_rows.append(dict(x=x0+xx,y=y0+yy,valid=bool(valid[yy,xx]),
          world_X_mm=float(xyz[yy,xx,0]),world_Y_mm=float(xyz[yy,xx,1]),world_Z_mm=float(xyz[yy,xx,2]),
          fitted_Z_mm=float(plane[yy,xx]),residual_Z_mm=float(residual[yy,xx])))
    csv_save(folder/f'plane_{window:02d}_per_point.csv',point_rows)
    return metrics

def continuity_path(obs,rgb,origin,candidate,radius):
    """4-connected measured-depth path. Reuse 8 mm solely as an edge screen; no new residual rule."""
    ox,oy=origin;cx,cy=candidate;xyz=obs.xyz
    finite=np.isfinite(xyz).all(axis=2)
    origin_valid=bool(finite[oy,ox]);anchors=[]
    if origin_valid:anchors=[(ox,oy)]
    else:
        for y in range(oy-2,oy+3):
            for x in range(ox-2,ox+3):
                if finite[y,x]:anchors.append((x,y))
        if anchors:
            distance=min((x-ox)**2+(y-oy)**2 for x,y in anchors)
            anchors=[p for p in anchors if (p[0]-ox)**2+(p[1]-oy)**2==distance]
    h,w=finite.shape;parent={p:None for p in anchors};q=deque(anchors)
    while q:
        x,y=q.popleft()
        if (x,y)==(cx,cy):break
        for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if (not(0<=nx<w and 0<=ny<h) or (nx-ox)**2+(ny-oy)**2>radius**2 or
                    (nx,ny) in parent or not finite[ny,nx]):continue
            if abs(float(xyz[ny,nx,2])-float(xyz[y,x,2]))>LIMIT:continue
            parent[(nx,ny)]=(x,y);q.append((nx,ny))
    if (cx,cy) not in parent:return dict(connected=False,origin_center_measured=origin_valid)
    path=[];node=(cx,cy)
    while node is not None:path.append(node);node=parent[node]
    path.reverse();zs=np.array([xyz[y,x,2] for x,y in path],dtype=float)
    colors=np.array([rgb[y,x] for x,y in path],dtype=float)
    oldrgb=np.median(rgb[oy-2:oy+3,ox-2:ox+3].reshape(-1,3),axis=0)
    newrgb=np.median(rgb[cy-2:cy+3,cx-2:cx+3].reshape(-1,3),axis=0)
    return dict(connected=True,origin_center_measured=origin_valid,anchor_pixel=list(path[0]),path_pixels=path,
      missing_origin_bridge_px=float(np.linalg.norm(np.array(path[0])-origin)),
      max_adjacent_world_z_step_mm=float(np.max(np.abs(np.diff(zs)))) if len(zs)>1 else 0.,
      path_world_z_range_mm=float(np.ptp(zs)),
      max_adjacent_rgb_l2=float(np.max(np.linalg.norm(np.diff(colors,axis=0),axis=1))) if len(colors)>1 else 0.,
      patch_median_rgb_l2=float(np.linalg.norm(newrgb-oldrgb)),
      interpretation='Measured connectivity screen only; missing center remains unobserved. RGB continuity is recorded, not thresholded. Does not certify same fabric layer or execution safety.')

def snap_search(obs,rgb,pa,pb,round_id,mode):
    old=depth_stats(obs,pa);target=depth_stats(obs,pb)
    if mode=='SEPARATE_IK_FAILURE':return [],[]
    all_candidates=[]
    ox,oy=pa
    for dy in range(-15,16):
        for dx in range(-15,16):
            if dx*dx+dy*dy>225:continue
            pixel=[ox+dx,oy+dy];stats=depth_stats(obs,pixel)
            if not stats['passed']:continue
            point=np.array(stats['sample_world_xyz_mm'])
            target_distance=float(np.linalg.norm(point[:2]-np.array(target['sample_world_xyz_mm'])[:2])) if target['passed'] else None
            valid_target=target_distance is not None and 10<=target_distance<=150
            all_candidates.append(dict(pixel=pixel,pixel_displacement=float(np.hypot(dx,dy)),stats=stats,
                target_unchanged=list(pb),target_translation_mm=target_distance,existing_action_distance_pass=valid_target))
    all_candidates.sort(key=lambda c:(c['pixel_displacement'],c['pixel'][1],c['pixel'][0]))
    results=[]
    local=obs.xyz[oy-2:oy+3,ox-2:ox+3];v=local[np.isfinite(local).all(axis=2)]
    proxy=np.median(v,axis=0) if len(v) else None
    for radius in RADII:
        candidates=[c for c in all_candidates if c['pixel_displacement']<=radius]
        selected=None;paths=[]
        for cand in candidates:
            if not cand['existing_action_distance_pass']:continue
            continuity=continuity_path(obs,rgb,pa,cand['pixel'],radius)
            if continuity['connected']:
                selected=dict(cand,continuity=continuity);break
        row=dict(global_round=round_id,radius_px=radius,original_pa=pa,unchanged_pb=pb,
            original_depth_stats=old,host_depth_valid_candidate_count=len(candidates),
            nearest_host_depth_valid_pixel=candidates[0]['pixel'] if candidates else None,
            nearest_host_depth_valid_distance_px=candidates[0]['pixel_displacement'] if candidates else None,
            found=selected is not None,selected_pixel=None,selected_pixel_displacement=None,
            world_xy_displacement_mm=None,host_sample_xy_displacement_mm=None,
            original_center_world_z_delta_mm=None,original_patch_median_proxy_xy_displacement_mm=None)
        if selected:
            x,y=selected['pixel'];newpoint=obs.xyz[y,x];newhost=np.array(selected['stats']['sample_world_xyz_mm'])
            row.update(selected_pixel=selected['pixel'],selected_pixel_displacement=selected['pixel_displacement'],
                new_depth_stats=selected['stats'],continuity=selected['continuity'],
                target_translation_mm=selected['target_translation_mm'],existing_action_distance_pass=True,
                # Never invent the missing original center's XYZ.
                world_xy_displacement_mm=float(np.linalg.norm(newpoint[:2]-obs.xyz[oy,ox,:2])) if old['center_valid'] else None,
                host_sample_xy_displacement_mm=float(np.linalg.norm(newhost[:2]-np.array(old['sample_world_xyz_mm'])[:2])) if old['passed'] else None,
                original_center_world_z_delta_mm=float(newpoint[2]-obs.xyz[oy,ox,2]) if old['center_valid'] else None,
                original_patch_median_proxy_xy_displacement_mm=float(np.linalg.norm(newhost[:2]-proxy[:2])) if proxy is not None else None,
                original_patch_median_proxy_world_z_delta_mm=float(newhost[2]-proxy[2]) if proxy is not None else None,
                validity_scope='Recorded depth + unchanged-target translation + local measured connectivity only; IK/workspace/collision/acquisition NOT certified')
        results.append(row)
    return results,all_candidates

def plot_round(obs,rgb,pa,folder,metrics,snaps,number):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    x,y=pa;data=np.load(folder/'plane_31.npz');small=np.load(folder/'plane_05.npz')
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('#171717')
    ax=axs[0,0];ax.imshow(rgb);ax.plot(x,y,'r+',ms=16,mew=2)
    for s in snaps:
        if s['found']:
            xx,yy=s['selected_pixel'];ax.plot(xx,yy,'o',mfc='none',mec='lime',ms=12,mew=2);break
    ax.set_title(f'Round {number}: RGB + Pa(red) / snap(green)');ax.axis('off')
    b=metrics[1]['bounds'];extent=[b[0]-.5,b[2]-.5,b[3]-.5,b[1]-.5]
    ax=axs[0,1];z=data['world_xyz'][...,2].copy();z[~data['valid']]=np.nan
    im=ax.imshow(z,cmap=cmap,interpolation='nearest',extent=extent)
    ax.add_patch(Rectangle((x-2.5,y-2.5),5,5,fill=False,ec='red',lw=2))
    ax.set_title('31 x 31 measured world Z (mm)');fig.colorbar(im,ax=ax)
    ax=axs[0,2];z5=small['world_xyz'][...,2].copy();z5[~small['valid']]=np.nan
    im=ax.imshow(z5,cmap=cmap,interpolation='nearest')
    for yy,xx in np.ndindex(z5.shape):
        ax.text(xx,yy,f'{z5[yy,xx]:.2f}' if np.isfinite(z5[yy,xx]) else 'NaN',ha='center',va='center',color='white',fontsize=9,
                bbox=dict(facecolor='black',alpha=.4,pad=1,edgecolor='none'))
    ax.add_patch(Rectangle((1.5,1.5),1,1,fill=False,ec='red',lw=2));ax.set_title('Actual 5 x 5 world Z (mm), center red')
    ax.set_xticks(range(5),range(x-2,x+3));ax.set_yticks(range(5),range(y-2,y+3))
    ax=axs[1,0];im=ax.imshow(data['fitted_world_z'],cmap=cmap,interpolation='nearest',extent=extent)
    ax.set_title('31 x 31 robust fitted plane at measured X/Y');fig.colorbar(im,ax=ax)
    ax=axs[1,1];res=data['residual_world_z'];lim=max(float(np.nanmax(np.abs(res))) if np.isfinite(res).any() else 0.,1e-6)
    cmapr=plt.colormaps['coolwarm'].copy();cmapr.set_bad('#171717')
    im=ax.imshow(res,cmap=cmapr,vmin=-lim,vmax=lim,interpolation='nearest',extent=extent)
    ax.add_patch(Rectangle((x-2.5,y-2.5),5,5,fill=False,ec='black',lw=2));ax.set_title('31 x 31 detrended residual (mm)');fig.colorbar(im,ax=ax)
    ax=axs[1,2];res=small['residual_world_z'];lim=max(float(np.nanmax(np.abs(res))) if np.isfinite(res).any() else 0.,1e-6)
    im=ax.imshow(res,cmap=cmapr,vmin=-lim,vmax=lim,interpolation='nearest')
    for yy,xx in np.ndindex(res.shape):
        ax.text(xx,yy,f'{res[yy,xx]:.3f}' if np.isfinite(res[yy,xx]) else 'N/A',ha='center',va='center',fontsize=9)
    m=metrics[0]
    ax.set_title('5 x 5 detrended residual (mm)\n'+(f"RMSE {m['residual_rmse_mm']:.3f}, slope {m['plane_slope_degrees']:.1f} deg" if m['status']=='FIT' else m['status']))
    fig.colorbar(im,ax=ax)
    fig.suptitle('Offline geometry only | no interpolation / new residual threshold / execution certification',fontsize=14)
    fig.savefig(folder/'diagnostic.png',dpi=150);plt.close(fig)
    # Per-row profiles make a gradual slope distinguishable from a spatial step visually.
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for yy in range(5):
        axs[0].plot(range(x-2,x+3),z5[yy],'-o',label=f'y={y-2+yy}')
        axs[1].plot(range(x-2,x+3),small['residual_world_z'][yy],'-o')
    axs[0].set_title('Measured world Z: horizontal profiles');axs[1].set_title('Detrended residual: horizontal profiles')
    for ax in axs:ax.set_xlabel('image x');ax.set_ylabel('mm');ax.grid(alpha=.3)
    axs[0].legend(fontsize=8);fig.savefig(folder/'patch_profiles.png',dpi=150);plt.close(fig)

def distribution(values):
    return dict(n=len(values),minimum=float(np.min(values)),median=float(np.median(values)),p90=float(np.percentile(values,90)),maximum=float(np.max(values))) if values else dict(n=0)

def render_summary(out):
    metrics=read(out/'per_round_metrics.json');by_round={m['global_round']:m for m in metrics}
    refs=read(out/'executed_reference_distributions.json')
    reviews=read(out/'interpretation.json') if (out/'interpretation.json').exists() else {}
    labels=reviews.get('round_classifications',{})
    lines=['# host_grasp_geometry_audit_v2', '',
        '**全部 12/12 轮与保存的 runtime 深度结果一致。** 8 个 EXECUTED 对照、3 个 depth rejection、'
        '1 个单列 IK failure。仅使用已保存 RGB、world XYZ 和 calibration；未调用模型、连接机器人或改变线上文件。', '',
        '## 1. 第 9、10 轮：坡面还是断层？', '',
        '|Round|分类（仅原 5×5 支持区域）|raw Z range mm|坡度 °|residual RMSE mm|median abs mm|p90−p10 mm|max abs mm|',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for n in (9,10):
        f=by_round[n]['plane_fits'][0]
        lines.append(f"|{n}|{labels.get(str(n),'UNRESOLVED')}|{f['raw_z_range_mm']:.3f}|{f['plane_slope_degrees']:.2f}|"
                     f"{f['residual_rmse_mm']:.3f}|{f['median_absolute_residual_mm']:.3f}|"
                     f"{f['residual_p90_p10_mm']:.3f}|{f['max_absolute_residual_mm']:.3f}|")
    lines += ['', '5×5 使用 actual world X/Y/Z 拟合 z=aX+bY+c，不是 image-x/image-y。'
        '分类是结合数值、空间剖面和诊断图作出的局部几何解释，不是新 classifier 或已验证安全标签。'
        '本次不设置 residual threshold。31×31 作为更大范围的曲率/折边上下文，不替代 5×5 主分析。', '',
        '## 2. 第 7 轮：局部缺深度与 snap', '']
    s=by_round[7]['nearest_candidate'];new=s['new_depth_stats']
    lines += [f"最近满足当前完整深度检查的像素为 **{s['selected_pixel']}**，距原 Pa=[418,203] "
        f"**{s['selected_pixel_displacement']:.3f} px**；固定测试半径中首先在 **3 px** 找到，5/10/15 px 返回同一点。"
        f"原点中心无深度、16/25 valid；候选中心有效、**{new['valid_count']}/25 valid**，"
        f"raw range **{new['world_z_range_mm']:.3f} mm**，符合原 8 mm 限制。", '',
        f"原 Pa 中心 XYZ 缺失，因此真实 world-XY displacement 是 **null/不可知**。"
        f"相对原 5×5 有效点 XYZ 中位数的 Host-sample 代理位移为 **{s['original_patch_median_proxy_xy_displacement_mm']:.3f} mm**；"
        '这不是原中心坐标的实测位移，未补洞或伪造原点。', '',
        '支持“非常局部的 depth hole，可用小范围候选修复深度门槛”的证据；'
        '**不等于已经证明 executable grasp**。候选尚未验证 workspace/IK/collision 或 acquisition，且 18/25 正好位于最低有效数门槛。'
        '原中心缺深度，无法证明穿过空洞处的实际几何连续性；仅能记录邻近已测点和 RGB 的连续性证据。', '',
        '|Round|原 Pa|候选 Pa|移动 px|实测中心 XY 位移 mm|新 valid|新 range mm|首个测试半径|',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for n in (7,9,10):
        m=by_round[n];s=m['nearest_candidate'];xy=s['world_xy_displacement_mm'];d=s['new_depth_stats']
        lines.append(f"|{n}|{m['grasp_px']}|{s['selected_pixel']}|{s['selected_pixel_displacement']:.3f}|"
            f"{'未知' if xy is None else f'{xy:.3f}'}|{d['valid_count']}/25|{d['world_z_range_mm']:.3f}|{m['first_radius_with_candidate']} px|")
    lines += ['', '候选搜索保持 Pb 不变，并额外验证原 10–150 mm 平移范围。所有原本 EXECUTED 点的最近合法点均为自身（0 px）。'
        '搜索是欧氏圆盘，距离相同按 y、x 排序。4-connected 有效深度路径上相邻 world-Z 跳变不得超过现有 8 mm 数值，'
        '这只是离线连续性筛查，不是线上新增规则；RGB 邻域差异只记录、不设阈值。'
        '第 7 轮采用距空洞中心最近的已测像素作路径锚点，因此路径单点的 0 mm 跳变不代表空洞内深度已知。', '',
        '## 3. raw-range gate 是否存在 false positive？', '',
        '对“真实断层”的几何含义而言，**存在明显疑似 false positive**：第 9/10 轮主要变化来自连续坡度，'
        '去坡度残差与已放行样本重叠，原检查却因略超 8 mm 而拒绝。'
        '这不是代码与阈值不一致，代码严格按规则工作；也不能称为已证实的“安全可执行动作误拒”。'
        '8 个 EXECUTED 的 raw range 全部 ≤8 mm，没有出现历史放行与当前规则相矛盾的情况。'
        '其 acquisition 均 UNKNOWN，无法据此建立实际抓取成功/失败的因果关系。', '',
        '## 4. 去坡度 residual 是否更能区分平滑面和真断层？', '',
        '本组斜面例子中，它更能分离局部坡度与未被平面解释的变化。'
        '**对真正断层的分类优越性仍未验证**：本数据没有经人工确认的真断层负例，不能报告 sensitivity、specificity 或选择最优阈值。'
        '小窗口也可能把很窄的阶跃近似为坡面，robust fit 会下调异常点权重；因此报告所有有效点 residual，'
        '包括 max absolute residual 和相邻 residual jump，而非只报告 inlier 误差。', '',
        '## 5. EXECUTED 对照及误拒风险', '',
        '|指标（8 个对照，5×5）|min|median|p90|max|', '|---|---:|---:|---:|---:|']
    for key in ('raw_z_range_mm','plane_slope_degrees','residual_rmse_mm','median_absolute_residual_mm','residual_p90_p10_mm','max_absolute_residual_mm'):
        d=refs['5'][key];lines.append(f"|{key}|{d['minimum']:.3f}|{d['median']:.3f}|{d['p90']:.3f}|{d['maximum']:.3f}|")
    lines += ['', '第 9/10 轮 residual RMSE 均落在上述 EXECUTED 分布内。'
        '没有设定新阈值，所以**不能计算新误拒率，也不能保证不会误拒**。'
        '部分已放行点（如第 8 轮）的残差高于第 9/10 轮；若根据被拒例强行设置过紧上限，会排斥对照。'
        f"31×31 对照 RMSE 为 {refs['31']['residual_rmse_mm']['minimum']:.3f}–"
        f"{refs['31']['residual_rmse_mm']['maximum']:.3f} mm，"
        '更大邻域包含曲率/折边，不能把 5×5 的数值尺度直接套用到 31×31。'
        '所有对照均标记 UNKNOWN_NOT_HUMAN_CONFIRMED，不称作抓取成功。', '',
        '## 6. 下一步是否值得做？', '',
        '**A 和 B 都值得下一步独立验证，但本轮不实现或部署线上规则。**', '',
        '- A：slope-aware secondary depth check 有明确的局部斜面证据。仍需要真实断层负例与独立阈值验证；'
        '不应把 low residual 直接等同于安全抓取。',
        '- B：small-radius Host-side Pa snap 有直接的深度可用性证据，第 7 轮仅需 1.414 px。'
        '后续需独立核验保持抓取语义、深度边界和完整 Host 几何检查；本次没有验证执行性。', '',
        '## IK 独立记录', '',
        '第 11 轮：**SEPARATE_IK_FAILURE**；Pa=[235,286]，Pb=[245,130]。'
        'Pa 深度 25/25、range=6.105 mm；IK residual=1.173 mm / 0.467°，原要求 0.5 mm / 0.2°。'
        '不纳入 plane/snap classifier 数据，不尝试 IK repair。'
        '原 trajectory 路径、motion、points、geometry、plan、execution 已保存在 [round_11](round_11/separate_ik_failure.json)。', '',
        '## 文件与可复核性', '',
        '- [audit.csv](audit.csv)：12 轮线上复现与完整 depth stats。',
        '- [per_round_metrics.json](per_round_metrics.json)、[plane_fit_metrics.csv](plane_fit_metrics.csv)、[snap_search.csv](snap_search.csv)。',
        '- 每轮 plane_05/31_per_point.csv 保存 world XYZ、拟合 Z 和每点 residual；npz 保存全部格点与 invalid mask。',
        '- [对照分布图](reference_distribution.png)；[协议](protocol.json)；[输入哈希](input_hashes.json)；[最终校验](verification.json)。', '',
        '拟合预先固定：Huber IRLS k=1.345，MAD×1.4826，最多 50 次迭代，至少 18 个有效点，'
        '以中心化/标准化 world XY 改善数值条件；1e-6 mm 是计算 scale 下限，不是判别阈值。'
        '没有更改 descriptor、radius、crop 或线上安全阈值来迎合结果。', '']
    for n in (7,9,10):
        lines += [f'### Round {n}', '', f'![round {n}](round_{n:02d}/diagnostic.png)', '',
                  f'[局部剖面](round_{n:02d}/patch_profiles.png) · [snap 放大图](round_{n:02d}/snap_detail.png)', '']
    (out/'summary.md').write_text('\n'.join(lines))

def run(out,source):
    out.mkdir(parents=True,exist_ok=False)
    protected={p:sha(p) for p in PROTECTED}
    rows=list(csv.DictReader(source.open()))
    protocol=dict(experiment='host_grasp_geometry_audit_v2',source=str(source.resolve()),expected_rounds=12,
        radii_px=list(RADII),radius_metric='Euclidean inclusive circle; deterministic tie break distance, y, x',
        plane_windows_px=list(FIT_WINDOWS),fit='Huber IRLS z=a world_X+b world_Y+c; minimum 18 valid XYZ; rank 3 required',
        huber_k=1.345,mad_multiplier=1.4826,numerical_scale_floor_mm=1e-6,max_iterations=50,
        residual_acceptance_threshold=None,raw_gate_mm=LIMIT,
        continuity='4-connected finite-XYZ path inside radius, adjacent World-Z jumps <= existing 8 mm; missing-origin nearest measured anchors in original 5x5; RGB stats only, no new RGB threshold',
        missing_original_xyz='true center XY displacement is null; separately labeled median-original-patch proxy, never used as actual center',
        execution_certification=False,ik_repair=False,robot=False,gpt=False,training=False,source_sha256=sha(__file__),
        protected_sources=protected)
    save(out/'protocol.json',protocol);shutil.copyfile(__file__,out/'source_snapshot.py')
    hashes={str(source.resolve()):sha(source)};audits=[];observations=[]
    if len(rows)!=12 or {int(r['global_round']) for r in rows}!=set(range(1,13)):
        raise ValueError('expected all 12 distinct rounds')
    config=read('config/dual_arm.local.json')
    if config['limits']['max_surface_spread_mm']!=LIMIT:raise ValueError('unexpected current depth threshold')
    # Stage 1 must finish and match EVERY round before any fit or snap is performed.
    for row in rows:
        p=Path(row['directory']);number=int(row['global_round']);obs=Observation.load(p/'before')
        if obs.meta['calibration_id']!=config['calibration_id'] or obs.meta.get('synthetic') is not False:
            raise ValueError('saved calibration or real-observation provenance mismatch')
        for path in [p/'before/observation.json',p/'before/rgb.png',p/'before/world_xyz_mm.npy',p/'plan.json',p/'execution.json',p.parent/'protocol.json']:
            hashes[str(path.resolve())]=sha(path)
        old=read(p.parent/'protocol.json')
        if old['protected_sources']['config/dual_arm.local.json']!=sha('config/dual_arm.local.json'):
            raise ValueError('current config does not match recorded config; cannot claim exact threshold reproduction')
        plan=read(p/'plan.json');execution=read(p/'execution.json');pa=depth_stats(obs,plan['grasp_px']);pb=depth_stats(obs,plan['target_px'])
        err=execution.get('error','');ik='IK rejected target' in err
        expected=execution['status']=='EXECUTED' or ik
        match=pa['passed']==expected and pb['passed']
        if not expected:match=match and pa['error'] is not None and pa['error'] in err
        audit=dict(global_round=number,source_directory=str(p),runtime_status=execution['status'],runtime_error=err,
            group='SEPARATE_IK_FAILURE' if ik else 'EXECUTED_REFERENCE' if expected else 'DEPTH_REJECTED',
            acquisition='UNKNOWN_NOT_HUMAN_CONFIRMED',grasp_px=plan['grasp_px'],target_px=plan['target_px'],
            valid_count=pa.get('valid_count'),center_valid=pa.get('center_valid'),world_z_min_mm=pa.get('world_z_min_mm'),
            world_z_max_mm=pa.get('world_z_max_mm'),world_z_range_mm=pa.get('world_z_range_mm'),
            reproduced_depth_result='PASS' if pa['passed'] else 'REJECT',runtime_match=match,grasp_depth_stats=pa,target_depth_stats=pb)
        audits.append(audit);observations.append((obs,plan,row))
    csv_save(out/'audit.csv',audits);save(out/'runtime_reproduction.json',dict(all_match=all(a['runtime_match'] for a in audits),rounds=audits))
    save(out/'input_hashes.json',hashes)
    if not all(a['runtime_match'] for a in audits):
        (out/'summary.md').write_text('# STOPPED\n\nOffline reproduction does not match recorded runtime. No plane/snap analysis performed.\n')
        raise RuntimeError('offline/runtime mismatch; stopped before further diagnostics')
    print('REPRODUCTION MATCHED 12/12; beginning frozen plane/snap diagnostics',flush=True)
    metrics=[];planes=[];snaps=[]
    for audit,(obs,plan,row) in zip(audits,observations):
        n=audit['global_round'];folder=out/f'round_{n:02d}';folder.mkdir();rgb=np.asarray(Image.open(obs.image).convert('RGB'))
        if audit['group']=='SEPARATE_IK_FAILURE':
            save(folder/'separate_ik_failure.json',dict(classification='SEPARATE_IK_FAILURE',source_trajectory=audit['source_directory'],plan=plan,
                position_residual_mm=1.173,orientation_residual_deg=.467,required_position_mm=.5,required_orientation_deg=.2,
                exact_failure_waypoint='not persisted; do not infer',depth_checks=audit,excluded_from_depth_classifier=True,ik_repair_attempted=False))
            for name in ['motion.py','geometry.json','points.json','execution.json','plan.json']:
                if (Path(row['directory'])/name).exists():shutil.copyfile(Path(row['directory'])/name,folder/name)
            metrics.append(dict(**audit,plane_fits=[],snap_search=[],classification='SEPARATE_IK_FAILURE'));continue
        fits=[]
        for window in FIT_WINDOWS:
            fit=patch_fit(obs,plan['grasp_px'],window,folder);fits.append(fit)
            planes.append(dict(global_round=n,group=audit['group'],**fit))
        found,candidates=snap_search(obs,rgb,plan['grasp_px'],plan['target_px'],n,audit['group'])
        snaps.extend(found);save(folder/'snap_candidates.json',candidates);save(folder/'snap_search.json',found)
        item=dict(**audit,plane_fits=fits,snap_search=found,first_radius_with_candidate=next((s['radius_px'] for s in found if s['found']),None),
                  nearest_candidate=next((s for s in found if s['found']),None))
        save(folder/'metrics.json',item);metrics.append(item)
        plot_round(obs,rgb,plan['grasp_px'],folder,fits,found,n)
        print(f"ROUND {n:02d} fit/snap complete",flush=True)
    csv_save(out/'plane_fit_metrics.csv',planes);csv_save(out/'snap_search.csv',snaps);save(out/'per_round_metrics.json',metrics)
    refs=[p for p in planes if p['group']=='EXECUTED_REFERENCE' and p['status']=='FIT']
    distributions={str(w):{key:distribution([p[key] for p in refs if p['window_px']==w]) for key in
        ('raw_z_range_mm','plane_slope_degrees','residual_rmse_mm','median_absolute_residual_mm','residual_p90_p10_mm','max_absolute_residual_mm')} for w in FIT_WINDOWS}
    save(out/'executed_reference_distributions.json',distributions)
    # Diagnostic comparison, no threshold optimization and no fabricated successful labels.
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,window in zip(axs,FIT_WINDOWS):
        for p in planes:
            if p['window_px']!=window or p['status']!='FIT':continue
            ax.scatter(p['raw_z_range_mm'],p['residual_rmse_mm'],c='#2685b5' if p['group']=='EXECUTED_REFERENCE' else '#e15d35')
            ax.annotate(str(p['global_round']),(p['raw_z_range_mm'],p['residual_rmse_mm']),xytext=(4,4),textcoords='offset points')
        if window==5:ax.axvline(8,c='gray',ls='--',label='existing raw-range gate 8 mm');ax.legend()
        ax.set(xlabel='Raw world-Z range (mm)',ylabel='Plane residual RMSE (mm)',title=f'{window} x {window}: blue=EXECUTED, orange=depth rejected')
        ax.grid(alpha=.2)
    fig.savefig(out/'reference_distribution.png',dpi=150);plt.close(fig)
    verification=dict(runtime_reproduction_matches=12,total_rounds=12,executed_references=8,depth_rejections=3,separate_ik_failures=1,
        all_input_hashes_unchanged=all(sha(p)==v for p,v in hashes.items()),protected_sources_unchanged={p:sha(p)==v for p,v in protected.items()},
        plane_threshold_selected=False,snap_deployed=False,model_calls=0,robot_connections=0)
    save(out/'verification.json',verification)
    from .host_grasp_geometry_audit_v2_figures import render_extra_figures
    render_extra_figures(out)
    render_summary(out)
    print(json.dumps(dict(distributions=distributions,focus=[m for m in metrics if m['global_round'] in (7,9,10)]),ensure_ascii=False)[:2000],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--report-only',action='store_true',help='render saved metrics and interpretation only; no refit')
    a=p.parse_args()
    if a.report_only:render_summary(a.output)
    else:run(a.output,a.source)

if __name__=='__main__':main()
