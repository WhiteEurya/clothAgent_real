"""Offline RGB lookup audit. No camera connection, calibration write or robot API."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import cv2,numpy as np
from cloth_agent.experiments.registration_field import ROOT,validate_metadata
from cloth_agent.experiments.camb_cross_stream_audit import transform
from cloth_agent.experiments.camb_geometry_root_cause import optical_z_in_color

def align_native(depth_m,kd,kc,T,shape):
    """SDK generic-style pixel footprint/min-native-Z raster, then explicit RGB Z.

    No interpolation/hole filling. Raster agrees with generic align.cpp semantics;
    optimized SDK implementations may differ at quantized footprint boundaries.
    """
    depth=np.asarray(depth_m,float);kd=np.asarray(kd,float);kc=np.asarray(kc,float);T=np.asarray(T,float)
    if depth.ndim!=2 or T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('Invalid depth/transform')
    if not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(T[:3,:3])<0:raise ValueError('Transform must be rigid')
    h,w=shape;v,u=np.where(np.isfinite(depth)&(depth>0));z=depth[v,u]
    bounds=[];front=np.ones(len(z),bool)
    for offset in [-.5,.5]:
        rays=np.linalg.solve(kd,np.array([u+offset,v+offset,np.ones(len(u))])).T
        points=(rays*z[:,None])@T[:3,:3].T+T[:3,3];front &= points[:,2]>0
        projected=points@kc.T
        with np.errstate(divide='ignore',invalid='ignore'):uv=projected[:,:2]/projected[:,2,None]
        front &= np.isfinite(uv).all(axis=1)
        # C++ static_cast<int>(pixel + .5), not numpy bankers-round.
        bounds.append(np.trunc(np.nan_to_num(uv,nan=-1e8,posinf=1e8,neginf=-1e8)+.5).astype(np.int64))
    lo,hi=bounds
    valid=front&(lo[:,0]>=0)&(lo[:,1]>=0)&(hi[:,0]<w)&(hi[:,1]<h)&(hi[:,0]>=lo[:,0])&(hi[:,1]>=lo[:,1])
    lo,hi,z=lo[valid],hi[valid],z[valid]
    source=(v*depth.shape[1]+u)[valid]
    if len(z) and np.max(hi-lo)>32:raise ValueError('Unexpected pixel magnification >32; refuse pathological raster')
    flat=np.full(h*w,np.inf);owner=np.full(h*w,-1,np.int64)
    if len(z):
        for dy in range(int(np.max(hi[:,1]-lo[:,1]))+1):
            for dx in range(int(np.max(hi[:,0]-lo[:,0]))+1):
                use=(lo[:,0]+dx<=hi[:,0])&(lo[:,1]+dy<=hi[:,1]);idx=(lo[use,1]+dy)*w+lo[use,0]+dx
                np.minimum.at(flat,idx,z[use])
        # Deterministic source provenance: lowest native index among equal-depth winners.
        owner[:]=np.iinfo(np.int64).max
        for dy in range(int(np.max(hi[:,1]-lo[:,1]))+1):
            for dx in range(int(np.max(hi[:,0]-lo[:,0]))+1):
                use=(lo[:,0]+dx<=hi[:,0])&(lo[:,1]+dy<=hi[:,1]);idx=(lo[use,1]+dy)*w+lo[use,0]+dx;win=z[use]==flat[idx]
                np.minimum.at(owner,idx[win],source[use][win])
    valid=np.isfinite(flat);owner[~valid]=-1;flat[~valid]=np.nan;native_z=flat.reshape(h,w)
    vv,uu=np.indices((h,w));rays=np.linalg.solve(kc,np.array([uu.ravel(),vv.ravel(),np.ones(h*w)])).T.reshape(h,w,3)
    # Reuse the audited optical-frame conversion, rather than treating native Z as RGB Z.
    color_z=optical_z_in_color(native_z,rays,T[:3,:3],T[:3,3])
    color_z[~np.isfinite(color_z)|(color_z<=0)]=np.nan
    return dict(source_native_z_m=native_z,rgb_z_m=color_z,rgb_xyz_m=rays*color_z[:,:,None],legacy_xyz_m=rays*native_z[:,:,None],source_index=owner.reshape(h,w))

def sample(xyz,u,v):
    x,y=int(u),int(v);h,w=xyz.shape[:2]
    if not(2<=x<w-2 and 2<=y<h-2):return None
    patch=xyz[y-2:y+3,x-2:x+3].reshape(-1,3);valid=np.isfinite(patch).all(1)
    if valid.sum()<18 or not np.isfinite(xyz[y,x]).all():return None
    return dict(center_xyz_mm=(xyz[y,x]*1000).tolist(),median_xyz_mm=(np.median(patch[valid],axis=0)*1000).tolist(),valid_pixels=int(valid.sum()),z_spread_mm=float(np.ptp(patch[valid,2])*1000))

def source_correspondence(folder,rows,aligned,native_width):
    """Independent planar-marker reference for which native pixel a query selects."""
    tags=json.loads((folder/'detections.json').read_text());checks=[]
    for row in rows:
        key=str(row['tag'])
        if key not in tags['ir1'] or key not in tags['rgb']:continue
        H=cv2.getPerspectiveTransform(np.float32(tags['rgb'][key]),np.float32(tags['ir1'][key]))
        x,y=int(row['u']),int(row['v']);h,w=aligned['factory']['source_index'].shape
        if not(0<=x<w and 0<=y<h):continue
        expected=cv2.perspectiveTransform(np.float32([[[x,y]]]),H)[0,0]
        ids={name:int(a['source_index'][y,x]) for name,a in aligned.items()}
        if any(i<0 for i in ids.values()):continue
        item=dict(tag=key,point=row['point'],rgb_query=[x,y],expected_ir=expected.tolist())
        for name,i in ids.items():
            actual=np.array([i%native_width,i//native_width]);item[name+'_source_ir']=actual.tolist();item[name+'_error_ir_px']=float(np.linalg.norm(actual-expected))
        checks.append(item)
    summary=dict(paired_queries=len(checks),reference='Observed per-tag RGB/IR homography; planar and quantized, not XYZ ground truth')
    for name in aligned:
        if checks:
            e=[r[name+'_error_ir_px'] for r in checks];summary[name+'_median']=float(np.median(e));summary[name+'_p95']=float(np.percentile(e,95))
    return dict(summary=summary,queries=checks)

def audit_case(folder,meta,candidate,out,detector):
    validate_metadata(meta);kd=np.array(meta['profiles']['depth']['K']);kc=np.array(meta['profiles']['rgb']['K'])
    if not np.allclose(kd,candidate['K_ir1']) or not np.allclose(kc,candidate['K_target']):raise ValueError('Candidate changed intrinsics; T-only required')
    stack=np.load(folder/'native_depth.npz')['depth'];depth=stack[-1].astype(float)*meta['depth_scale'];depth[stack[-1]==0]=np.nan
    rgb=cv2.imread(str(folder/'on_last.png'));h,w=rgb.shape[:2]
    transforms=dict(factory=transform(meta,'rgb'),candidate=np.array(candidate['T_ir1_target']))
    aligned={name:align_native(depth,kd,kc,t,(h,w)) for name,t in transforms.items()}
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'aligned_maps.npz',**{name+'_'+k:v.astype(np.float32) if k!='source_index' else v for name,a in aligned.items() for k,v in a.items() if k!='legacy_xyz_m'})
    # Detect RGB tag points, not predicted points: queries are identical in both modes.
    from cloth_agent.experiments.registration_field import detect
    tags=detect(detector,cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB));rows=[]
    mask=np.zeros((h,w),np.uint8)
    for tag,corners in tags.items():
        cv2.fillConvexPoly(mask,np.rint(corners).astype(np.int32),1)
        # Projective tag center; not the arithmetic average under perspective.
        H=cv2.getPerspectiveTransform(np.float32([[0,0],[1,0],[1,1],[0,1]]),corners.astype(np.float32))
        center=cv2.perspectiveTransform(np.float32([[[.5,.5]]]),H)[0,0]
        for label,uv in [('center',center)]+[(f'corner_{j}',p) for j,p in enumerate(corners)]:
            samples={name:sample(a['rgb_xyz_m'],*uv) for name,a in aligned.items()};legacy=sample(aligned['factory']['legacy_xyz_m'],*uv)
            row=dict(tag=tag,point=label,u=float(uv[0]),v=float(uv[1]),factory_valid=samples['factory'] is not None,candidate_valid=samples['candidate'] is not None)
            for name,s in samples.items():
                if s:
                    row.update({name+'_'+axis+'_mm':s['median_xyz_mm'][i] for i,axis in enumerate(['x','y','z'])});row[name+'_spread_mm']=s['z_spread_mm']
            if all(samples.values()):
                delta=np.array(samples['candidate']['median_xyz_mm'])-samples['factory']['median_xyz_mm'];row.update(dict(zip(['delta_x_mm','delta_y_mm','delta_z_mm'],delta.tolist())))
                if legacy:row['factory_frame_semantics_delta_z_mm']=samples['factory']['median_xyz_mm'][2]-legacy['median_xyz_mm'][2]
            rows.append(row)
    af=np.isfinite(aligned['factory']['rgb_z_m']);ac=np.isfinite(aligned['candidate']['rgb_z_m']);common=af&ac;roi=mask.astype(bool)
    delta=1000*(aligned['candidate']['rgb_z_m']-aligned['factory']['rgb_z_m'])
    summary=dict(source=str(folder),depth_frame='last emitter-on frame, not temporal hole fill',tag_count=len(tags),factory_roi_valid_fraction=float(af[roi].mean()) if roi.any() else None,candidate_roi_valid_fraction=float(ac[roi].mean()) if roi.any() else None,roi_lost_pixels=int((roi&af&~ac).sum()),roi_gained_pixels=int((roi&ac&~af).sum()),roi_common_pixels=int((roi&common).sum()),roi_delta_z_median_mm=float(np.median(delta[roi&common])) if (roi&common).any() else None,roi_abs_delta_z_p95_mm=float(np.percentile(abs(delta[roi&common]),95)) if (roi&common).any() else None,factory_query_valid=sum(r['factory_valid'] for r in rows),candidate_query_valid=sum(r['candidate_valid'] for r in rows),query_count=len(rows),robot_connected=False,world_xyz_generated=False,ground_truth_available=False)
    panels=[]
    for name,a in aligned.items():
        image=rgb.copy();invalid=roi&~np.isfinite(a['rgb_z_m']);image[invalid]=[0,0,255];cv2.putText(image,name+' / missing depth in tag ROI = red',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2);panels.append(image)
    cv2.imwrite(str(out/'coverage.png'),np.concatenate(panels,axis=1))
    # Fixed +/-10 mm display range across all cases; invalid gray.
    colors=cv2.applyColorMap(np.uint8(np.clip((np.nan_to_num(delta)/10+1)*127.5,0,255)),cv2.COLORMAP_TURBO);colors[~common]=[100,100,100];cv2.imwrite(str(out/'delta_z_minus10_to_plus10_mm.png'),colors)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'queries.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    correspondence=source_correspondence(folder,rows,aligned,depth.shape[1])
    (out/'source_correspondence.json').write_text(json.dumps(correspondence,indent=2))
    summary['source_correspondence']=correspondence['summary']
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def report(out,results):
    lines=['# Host原生depth→RGB对齐离线验证','',
        '相机未连接，未执行机器人动作，未修改任何正式标定。factory和candidate使用同一个末帧native depth；没有时域填补、空间补洞或额外Z补偿。',
        '', '| 数据 | factory取源误差 IR px | candidate取源误差 IR px | ROI覆盖率 factory/candidate | RGB Z变化中位 mm |','|---|---:|---:|---:|---:|']
    for s in results:
        c=s['source_correspondence'];label=str(Path(s['source']).parent)
        lines.append(f'| {label} | {c.get("factory_median",float("nan")):.3f} | {c.get("candidate_median",float("nan")):.3f} | {s["factory_roi_valid_fraction"]:.2%}/{s["candidate_roi_valid_fraction"]:.2%} | {s["roi_delta_z_median_mm"]:.3f} |')
    lines+=['','## 方法与限制','',
        '- 取源误差：在相同RGB整数查询点，检查实际选中的原生depth像素与该tag的观测RGB→IR单应对应点之间的距离。不是此前的RGB重投影误差，也不是机器人XYZ真值。',
        '- SDK通用align风格的像素footprint映射，遮挡冲突取最小native Z；没有插值或补洞。保留source_index作为取点来源证据。与优化版SDK可能有量化边界差异。',
        '- SDK aligned depth保存的是native Z；本实验显式用已有optical_z_in_color换算RGB optical Z，再按RGB射线构建XYZ。旧流程解释下的Z差另记于queries.csv。',
        '- 5×5查询记录18像素有效性、中心有效性、Z跨度；没有调用机器人grasp，不代表通过线上workspace/IK/collision或抓取深度跨度检查。',
        '- 没有world XYZ，没有camera→robot变换，也没有Z补偿。RGB Z变化不可直接作为机器人下降量。',
        '- 这些是平面标记板，不是衣服边界；不能用本结果宣称已证明衣服边缘不会误取背景。还需要衣物/深度断层样本。',
        '- intrinsics及候选transform保持冻结；输入有非零distortion或depth/IR1坐标系不兼容时拒绝。',
        '', '## 输出','',
        '每个case包含aligned_maps.npz（source_native_z、rgb_z、rgb_xyz、source_index）、queries.csv、source_correspondence.json、coverage.png、delta_z_minus10_to_plus10_mm.png。',
        '对齐规则来源：https://raw.githubusercontent.com/realsenseai/librealsense/v2.58.4/src/proc/align.cpp']
    (out/'report.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/host_depth_alignment_audit')
    p.add_argument('--candidate',type=Path,default=ROOT/'reports/camb_multiview_analysis/candidate_models_OFFLINE_ONLY.json');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);blob=a.candidate.read_bytes();candidate=json.loads(blob)['models']['rgb_T_only']
    (a.output/'frozen_candidate.json').write_text(json.dumps(dict(sha256=hashlib.sha256(blob).hexdigest(),candidate=candidate,deployed=False),indent=2))
    sessions=[('table',ROOT/'runs/registration_field/20261016T080154_133094Z'),('raised',ROOT/'runs/registration_height_audit/20261016T082418_512885Z/capture/20261016T082418_513124Z')]
    from pupil_apriltags import Detector
    detector=Detector(families='tag36h11',quad_decimate=1,nthreads=2)
    results=[]
    for label,path in sessions:
        meta=json.loads((path/'camera.json').read_text());manifest=json.loads((path/'session.json').read_text())
        for pl in ['L-M','C-M','R-M']:
            att=next(r for r in manifest['attempts'] if r['placement']==pl and r['accepted']);folder=path/pl/f'attempt_{att["attempt"]:03d}'
            s=audit_case(folder,meta,candidate,a.output/(label+'_'+pl),detector);results.append(s);print(label,pl,json.dumps(s),flush=True)
    (a.output/'summary.json').write_text(json.dumps(results,indent=2))
    report(a.output,results)
if __name__=='__main__':main()
