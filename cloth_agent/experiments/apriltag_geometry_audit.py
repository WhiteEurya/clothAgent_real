"""Read-only AprilTag geometry audit using the existing online Camera B pipeline.

No NativeRuntime, robot controller preparation, mode changes, motion, gripper
commands or calibration writes. XArmAPI is used ONLY for joint/state getters.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
import fcntl
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import yaml

from cloth_agent.apriltag_diagnostic import A4_SCALE, load_board, solve_board_pose, transform_points
from cloth_agent.dual_arm.observation import Observation, capture, create_camera
from cloth_agent.experiments.grasp_acquire_baseline import save, raw_cloud, patch_stats

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOARD = ROOT/'config/apriltag_board_4x4_tag48mm.yaml'


def board_config(path, scale):
    data = yaml.safe_load(Path(path).read_text())
    required = [('family',), ('geometry','tag_size_mm'), ('geometry','pitch_mm'),
                ('layout','rows'), ('layout','cols'), ('tags',), ('detection_corner_order',)]
    missing = []
    for parts in required:
        value = data
        for key in parts:
            if not isinstance(value, dict) or key not in value:
                missing.append('.'.join(parts));break
            value = value[key]
    if missing:
        raise ValueError('Missing board parameters: '+', '.join(missing)+
            '; provide a complete --board YAML with measured family/size/pitch/layout/corner order; no defaults guessed')
    if data['layout']['rows'] != 4 or data['layout']['cols'] != 4 or len(data['tags']) != 16:
        raise ValueError('This audit requires the configured 4x4 board with 16 tag IDs')
    board = load_board(path, scale)
    board['pitch_mm'] = float(data['geometry']['pitch_mm']*scale)
    board['black_edge_spacing_mm'] = board['pitch_mm']-board['tag_size_mm']
    for points in board['corners'].values():
        lengths = np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)*1000
        if not np.allclose(lengths,board['tag_size_mm'],atol=1e-5):
            raise ValueError('Board corner geometry disagrees with tag size')
    return board


def distortion_coefficients(metadata):
    coeff = np.asarray(metadata['distortion_coefficients'],float)
    if coeff.shape != (5,) or not np.isfinite(coeff).all():
        raise ValueError('Missing/invalid live distortion coefficients')
    model = metadata['distortion_model'].lower()
    if np.max(np.abs(coeff)) == 0:
        return coeff  # Includes factory inverse Brown with all-zero coefficients.
    if model not in ('brown_conrady','distortion.brown_conrady'):
        raise ValueError(f'Nonzero {model} distortion cannot be treated as OpenCV Brown; no silent approximation')
    return coeff


def project_rgb(pose, points_m, k, distortion):
    rvec = cv2.Rodrigues(pose[:3,:3])[0]
    return cv2.projectPoints(np.asarray(points_m,float),rvec,pose[:3,3],k,distortion)[0].reshape(-1,2)


def fit_pose(objects, pixels, k, distortion):
    # Reuse existing board PnP and its planar ambiguity handling; only normalize
    # the RGB distortion model for its documented pinhole input.
    ideal = cv2.undistortPoints(np.asarray(pixels,float).reshape(-1,1,2),k,distortion,P=k).reshape(-1,2)
    return solve_board_pose(objects,ideal,k)


def physical_transform(obs):
    return np.asarray(obs.meta.get('uncorrected_world_from_camera_mm',obs.meta['world_from_camera_mm']),float)


def correction_transform(obs):
    for key in ('support_height_patch','world_tilt_correction'):
        if obs.meta.get(key):
            return np.asarray(obs.meta[key]['matrix_mm'],float)
    return np.eye(4)


def vector_columns(row, prefix, value):
    for axis, v in zip('xyz',value):
        row[f'{prefix}_{axis}_mm'] = float(v)


def delta_columns(row, prefix, delta):
    for axis, v in zip('xyz',delta):
        row[f'{prefix}d{axis}_mm'] = float(v)
    row[prefix+'xy_error_mm'] = float(np.linalg.norm(delta[:2]))
    row[prefix+'xyz_error_mm'] = float(np.linalg.norm(delta))


def distribution(values):
    values=np.asarray([v for v in values if v is not None],float)
    values=values[np.isfinite(values)]
    return dict(count=int(len(values)), median=float(np.median(values)) if len(values) else None,
        max=float(np.max(values)) if len(values) else None)


def region_metrics(rows, reprojections):
    result={}
    for name in ('all','left','center','right'):
        r=[v for v in rows if name=='all' or v['image_region']==name]
        q=[v for v in reprojections if name=='all' or v['image_region']==name]
        result[name]=dict(tag_centers=len(r),valid_depth=sum(v.get('depth_status')=='VALID' for v in r),
            reprojection_px=distribution([v['error_px'] for v in q]),
            xy_error_mm=distribution([v.get('xy_error_mm') for v in r]),
            abs_z_error_mm=distribution([abs(v['dz_mm']) for v in r if v.get('dz_mm') is not None]),
            signed_z_error_mm=distribution([v.get('dz_mm') for v in r]),
            online_world_xy_error_mm=distribution([v.get('online_world_xy_error_mm') for v in r]),
            online_world_abs_z_error_mm=distribution([abs(v['online_world_dz_mm']) for v in r if v.get('online_world_dz_mm') is not None]))
    return result


def image_region(u, width):
    return ('left','center','right')[min(2,max(0,int(3*u/width)))]


def analyze(obs, intrinsics, metadata, board, detector, max_spread, position):
    rgb=np.asarray(Image.open(obs.image).convert('RGB'))
    height,width=rgb.shape[:2]
    k=np.asarray(intrinsics,float)
    distortion=distortion_coefficients(metadata)
    detected=detector.detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),estimate_tag_pose=False)
    detections=[];accepted={};seen=set()
    for d in detected:
        tag_id=int(d.tag_id)
        reason=None
        if tag_id not in board['corners']: reason='ID_NOT_IN_BOARD'
        elif tag_id in seen: raise ValueError(f'Duplicate board tag ID {tag_id}; multiple boards are not supported')
        elif d.hamming != 0 or d.decision_margin < 20: reason='LOW_DETECTION_QUALITY'
        elif np.asarray(d.corners).shape != (4,2) or not np.isfinite(d.corners).all(): reason='INVALID_CORNERS'
        seen.add(tag_id)
        detections.append(dict(tag_id=tag_id,hamming=int(d.hamming),decision_margin=float(d.decision_margin),
            center=np.asarray(d.center).tolist(),corners=np.asarray(d.corners).tolist(),rejection=reason))
        if reason is None: accepted[tag_id]=d
    result=dict(status='INSUFFICIENT_TAGS',position=position,detected_tags=len(detected),
        accepted_tags=len(accepted),accepted_corners=len(accepted)*4,expected_tags=16,
        missing_ids=sorted(set(board['corners'])-set(accepted)),detections=detections,
        difference_sign='depth minus RGB/PnP board',rows=[],reprojections=[],
        region_definition='Fixed image thirds by RGB u; placement label is separate, not inferred world left/right',
        max_surface_spread_mm=max_spread)
    if len(accepted)<4:
        result['reason']='Need at least four clear configured board tags for reliable whole-board fit'
        result['regions']=region_metrics([],[])
        return result
    ids=sorted(accepted)
    objects=np.concatenate([board['corners'][i] for i in ids])
    pixels=np.concatenate([accepted[i].corners for i in ids])
    pose,ambiguous=fit_pose(objects,pixels,k,distortion)
    projected=project_rgb(pose,objects,k,distortion)
    errors=np.linalg.norm(projected-pixels,axis=1)
    train,held=ids[::2],ids[1::2]
    held_pose,held_ambiguous=fit_pose(np.concatenate([board['corners'][i] for i in train]),
        np.concatenate([accepted[i].corners for i in train]),k,distortion)
    held_err=np.linalg.norm(project_rgb(held_pose,np.concatenate([board['corners'][i] for i in held]),k,distortion)
        -np.concatenate([accepted[i].corners for i in held]),axis=1)
    rms=float(np.sqrt(np.mean(errors**2)));held_rms=float(np.sqrt(np.mean(held_err**2)))
    result.update(status='MEASURED' if not (ambiguous or held_ambiguous) and max(rms,held_rms)<=2. else 'RGB_POSE_UNRELIABLE',
        T_camera_board_m=pose.tolist(),planar_pose_ambiguous=bool(ambiguous or held_ambiguous),
        corner_reprojection_rms_px=rms,held_out_tag_ids=held,held_out_reprojection_rms_px=held_rms)
    physical=physical_transform(obs);inverse=np.linalg.inv(physical);correction=correction_transform(obs)
    raw=raw_cloud(obs)
    raw_obs=Observation(obs.directory,obs.meta,raw)
    # The online 5x5 sampler rejects on corrected-world Z spread. Apply that
    # gate once to obs; raw counterpart uses identical finite pixels without
    # inventing a second raw-Z spread gate.
    raw_camera=transform_points(inverse,raw.reshape(-1,3)).reshape(raw.shape)
    corrected_camera=transform_points(inverse,obs.xyz.reshape(-1,3)).reshape(raw.shape)
    result.update(T_world_camera_mm=physical.tolist(),world_correction_affine_mm=correction.tolist())
    for n,tag_id in enumerate(ids):
        d=accepted[tag_id];center=np.asarray(d.center,float);u,v=center
        pa=[int(round(u)),int(round(v))]
        center_obj=np.mean(board['corners'][tag_id],axis=0)
        center_pred=project_rgb(pose,[center_obj],k,distortion)[0]
        center_error=float(np.linalg.norm(center_pred-center))
        region=image_region(u,width)
        for index,(actual,predicted,error) in enumerate(zip(d.corners,projected[n*4:n*4+4],errors[n*4:n*4+4])):
            result['reprojections'].append(dict(tag_id=tag_id,point=f'corner_{index}',u=float(actual[0]),v=float(actual[1]),
                projected_u=float(predicted[0]),projected_v=float(predicted[1]),error_px=float(error),image_region=image_region(actual[0],width)))
        result['reprojections'].append(dict(tag_id=tag_id,point='center',u=float(u),v=float(v),
            projected_u=float(center_pred[0]),projected_v=float(center_pred[1]),error_px=center_error,image_region=region))
        p_board=transform_points(pose,[center_obj])[0]*1000
        board_world=transform_points(physical,[p_board])[0]
        board_corrected=transform_points(correction,[board_world])[0]
        row=dict(tag_id=tag_id,position=position,u=float(u),v=float(v),sample_u=pa[0],sample_v=pa[1],image_region=region,
            reprojection_error_px=center_error,corner_reprojection_median_px=float(np.median(errors[n*4:n*4+4])),
            corner_reprojection_max_px=float(np.max(errors[n*4:n*4+4])),depth_status='INVALID')
        vector_columns(row,'board_camera',p_board)
        vector_columns(row,'board_world_raw',board_world)
        vector_columns(row,'board_world_corrected',board_corrected)
        try:
            depth_world=obs.sample(pa,max_spread)
            raw_world=raw_obs.sample(pa,float('inf'))
            p_depth=transform_points(inverse,[raw_world])[0]
            effective_camera=transform_points(inverse,[depth_world])[0]
            row.update(depth_status='VALID',raw_depth_mm=float(raw_camera[pa[1],pa[0],2]),
                corrected_depth_mm=float(corrected_camera[pa[1],pa[0],2]),
                raw_sampled_camera_z_mm=float(p_depth[2]),corrected_sampled_camera_z_mm=float(effective_camera[2]),
                current_surface_z_mm=float(depth_world[2]),
                depth_valid_ratio=patch_stats(raw_camera[:,:,2],pa,valid=np.isfinite(raw).all(axis=2))['valid_ratio'])
            vector_columns(row,'depth_camera',p_depth)
            vector_columns(row,'depth_world_raw',raw_world)
            vector_columns(row,'depth_world_corrected',depth_world)
            delta_columns(row,'',p_depth-p_board)
            delta_columns(row,'raw_world_',raw_world-board_world)
            delta_columns(row,'online_world_',depth_world-board_world)
            delta_columns(row,'common_corrected_world_',depth_world-board_corrected)
        except Exception as exc:
            row['depth_failure']=f'{type(exc).__name__}: {exc}'
        result['rows'].append(row)
    result['regions']=region_metrics(result['rows'],result['reprojections'])
    if result['regions']['all']['valid_depth']==0:
        result['status']='NO_VALID_DEPTH'
    result['depth_semantics']='raw_depth_mm: temporally aggregated aligned sensor optical Z; corrected_depth_mm: corrected world point expressed back in physical camera frame, NOT a new sensor depth measurement'
    result['world_comparison']='raw_world: both physical rigid transform; online_world: online corrected depth versus uncorrected physical board; common_corrected_world: same compensation on both, consistency only'
    return result


def csv_write(path, rows, default_fields):
    keys=list(dict.fromkeys([*default_fields,*(key for row in rows for key in row)]))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def conclusions(result):
    notes=[]
    if result['status']!='MEASURED':
        notes.append(f'当前状态 {result["status"]}，不能据此认定几何链条正常或给出确定误差归因。')
    if result.get('corner_reprojection_rms_px',0)>2 or result.get('held_out_reprojection_rms_px',0)>2:
        notes.append('RGB/PnP 一致性偏弱：优先核对板平整度/打印比例/布局、角点检测、RGB 内参及畸变；不能直接归因于深度。')
    all_stats=result['regions']['all']
    xy=all_stats['xy_error_mm']['median'];z=all_stats['abs_z_error_mm']['median']
    if result['status']=='MEASURED' and xy is not None:
        if max(xy,z)>2:
            notes.append('RGB 拟合较好但 depth 与 PnP 存在毫米级差异：更值得检查 depth 尺度、RGB-depth 对齐及线上反投影。错误打印尺度也可产生相同现象，尚不能唯一定位。')
        else:
            notes.append('当前采样点 raw depth 与 RGB/PnP 的中位差异较小；这不是绝对精度认证，也不覆盖未采样工作区。')
    left=result['regions']['left'];right=result['regions']['right']
    if left['valid_depth'] and right['valid_depth']:
        delta=right['xy_error_mm']['median']-left['xy_error_mm']['median']
        notes.append(f'图像右侧相对左侧 XY median 差值 {delta:+.3f} mm；仅为本次描述性比较，不作显著性或系统偏差结论。')
    else:
        notes.append('左右图像区域覆盖不足，不能判断右侧误差是否系统性增大；需补采不同 board placement。')
    notes.append('同一外参同时作用于两套点：平移误差会抵消，刚体旋转保持 XYZ 差值范数。因此本实验不能独立认证 camera→world/base 外参、FK、TCP 或实际夹指中心。')
    notes.append('online_world 与 raw_world 的差异可显示现有补偿的影响，但没有独立世界坐标真值，不能据此认定补偿错误。未拟合或写入任何 correction。')
    return notes


def save_outputs(folder, obs, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb=np.asarray(Image.open(obs.image).convert('RGB'))
    overlay=rgb.copy()
    for det in result['detections']:
        corners=np.rint(det['corners']).astype(int)
        color=(255,160,0) if det['rejection'] else (0,255,0)
        cv2.polylines(overlay,[corners],True,color,2)
        uv=tuple(np.rint(det['center']).astype(int));cv2.circle(overlay,uv,4,color,-1)
        cv2.putText(overlay,str(det['tag_id']),uv,cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
    for p in result['reprojections']:
        a=np.array([p['u'],p['v']]);b=np.array([p['projected_u'],p['projected_v']])
        cv2.arrowedLine(overlay,tuple(np.rint(a).astype(int)),tuple(np.rint(a+10*(b-a)).astype(int)),(255,0,0),1,tipLength=.3)
    cv2.putText(overlay,'Red reprojection arrows: x10 (observed -> projected)',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,0,0),2)
    Image.fromarray(overlay).save(folder/'rgb_residuals.png')
    physical=physical_transform(obs)
    raw_cam=transform_points(np.linalg.inv(physical),raw_cloud(obs).reshape(-1,3)).reshape(obs.xyz.shape)
    depth=raw_cam[:,:,2]
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(np.ma.masked_invalid(depth),cmap='viridis')
    fig.colorbar(im,ax=ax,label='Aligned sensor camera Z (mm)');ax.set_title('Online temporally aggregated depth; invalid pixels masked')
    fig.tight_layout();fig.savefig(folder/'depth_visualization.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    valid=[r for r in result['rows'] if r['depth_status']=='VALID']
    for ax,prefix,title in zip(axes,['','online_world_'],['Camera XY residual','Online corrected world XY minus physical board']):
        ax.imshow(rgb)
        if valid:
            u=[r['u'] for r in valid];v=[r['v'] for r in valid]
            dx=[r[prefix+'dx_mm'] for r in valid];dy=[r[prefix+'dy_mm'] for r in valid]
            q=ax.quiver(u,v,dx,dy,angles='xy',scale_units='xy',scale=.05,color='red')
            ax.quiverkey(q,.82,1.04,1,'1 mm = 20 display px',coordinates='axes')
            for r in valid:ax.text(r['u'],r['v'],str(r['tag_id']),color='yellow')
        ax.set_title(title+'\nvector components, not image displacement')
        ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0)
    fig.tight_layout();fig.savefig(folder/'xy_error_field.png',dpi=150);plt.close(fig)
    csv_write(folder/'results.csv',result['rows'],['tag_id','position','u','v','depth_status','dx_mm','dy_mm','dz_mm','xy_error_mm','xyz_error_mm'])
    csv_write(folder/'reprojection.csv',result['reprojections'],['tag_id','point','u','v','error_px'])
    result['interpretation']=conclusions(result)
    save(folder/'analysis.json',result)
    text=[f'# AprilTag geometry audit: {result["position"]}', '', f'状态：{result["status"]}',
        f'实际检测 {result["detected_tags"]} tags；接受 {result["accepted_tags"]}/16 tags、{result["accepted_corners"]} corners。',
        '', '区域按完整 RGB 宽度三等分；placement 标签单独保存。差值方向：depth − board/PnP。', '',
        '| 图像区域 | tag / depth有效数 | reprojection median/max px¹ | camera XY median/max mm | camera abs Z median/max mm |',
        '|---|---:|---:|---:|---:|']
    def fmt(d):return 'N/A' if d['median'] is None else f'{d["median"]:.3f} / {d["max"]:.3f}'
    for name,st in result['regions'].items():
        text.append(f'| {name} | {st["tag_centers"]}/{st["valid_depth"]} | {fmt(st["reprojection_px"])} | {fmt(st["xy_error_mm"])} | {fmt(st["abs_z_error_mm"])} |')
    text+=['','¹ 重投影统计包含各 tag 的四个角点和中心。所有值都是拟合/一致性误差，不是独立人工真值。',
        '',f'RGB corner RMS: {result.get("corner_reprojection_rms_px","N/A")} px; held-out RMS: {result.get("held_out_reprojection_rms_px","N/A")} px.',
        '',*['- '+s for s in result['interpretation']], '',
        'raw_depth_mm 是原始对齐传感器光轴深度（线上时间聚合后）；corrected_depth_mm 是补偿后 world 点转换回物理 camera frame 的有效 Z，不是传感器测量。',
        'p_depth_camera 由线上原始 world cloud 的同一 5×5 有效点中位数逆变换得到；online world 点直接调用 Observation.sample。',
        '采集复用线上 temporal stabilization / RGB median；另保存最后一个同步对齐 RGB-D frame。不是机器人/相机硬件触发同步。',
        'Camera-only 指无机器人控制；只读关节反馈用于腕部相机当前外参。left arm 使用已有静态障碍姿态，不连接 arm6。',
        '板的打印比例/平整度和 PnP 平面歧义会影响毫米结果。当前诊断不验证真实指爪/TCP 对齐。']
    (folder/'summary.md').write_text('\n'.join(text)+'\n')


class AuditCamera:
    """Thin recording wrapper; all alignment and depth processing remain native."""
    def __init__(self, camera):
        self.camera=camera;self.last=None;self.read_count=0;self.depth_window=deque(maxlen=15)
    def __getattr__(self,name):return getattr(self.camera,name)
    def read(self):
        rgb,depth=self.camera.read();self.last=(rgb.copy(),depth.copy());self.read_count+=1
        self.depth_window.append(depth.copy())
        return rgb,depth


def readonly_snapshot(arm, config, left_joints):
    def get():
        code,q=arm.get_servo_angle(is_radian=False)
        if code!=0:raise RuntimeError(f'Read-only arm7 joint feedback failed: {code}')
        q=np.asarray(q,float)[:config.arms['right'].axis]
        if q.shape!=(config.arms['right'].axis,) or not np.isfinite(q).all():
            raise ValueError('Invalid live arm7 joint feedback')
        return dict(right=dict(joints=q.tolist(),state_source='READ_ONLY_LIVE'),
            left=dict(joints=list(left_joints),state_source='EXISTING_STATIC_POSE_NOT_CONNECTED'))
    return get


def capture_live(config_path, folder):
    from cloth_agent.dual_arm.config import DualConfig
    from cloth_agent.dual_arm.kinematics import ArmModel
    from cloth_agent.codex_camera_b import with_arm7_camera
    config=with_arm7_camera(DualConfig.load(config_path,root=ROOT,homing_only=True))
    models={k:ArmModel(a) for k,a in config.arms.items()}
    base_raw=json.loads((ROOT/'data/robot/arm6_parked_for_arm7.json').read_text())
    save(folder/'runtime_config.json',config.raw)
    arm=None;camera=None
    with open('/tmp/clothagent_codex_dual_arm.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            from xarm.wrapper import XArmAPI
            arm=XArmAPI(config.arms['right'].ip,do_not_open=True)
            arm.connect()  # No enable/mode/state/clear/gripper/motion setters anywhere.
            camera=AuditCamera(create_camera(config));camera.start()
            profile=camera.pipeline.get_active_profile().get_stream(camera.rs.stream.color).as_video_stream_profile()
            intr=profile.get_intrinsics()
            metadata=dict(K=camera.intrinsics.tolist(),width=intr.width,height=intr.height,
                distortion_model=str(intr.model),distortion_coefficients=list(intr.coeffs),
                depth_scale_m=camera.depth_scale,alignment='Existing RealSenseRGBD: rs.align(color)',
                online_deprojection='Existing dual_arm.observation.capture: pinhole K, no explicit distortion correction',
                camera_serial=camera.spec.serial,camera_label='B',robot_control_commands=False,
                max_surface_spread_mm=config.limits['max_surface_spread_mm'])
            save(folder/'camera.json',metadata)
            obs=capture(folder/'observation',config,models,readonly_snapshot(arm,config,base_raw['joint_angles_deg']),camera=camera)
            metadata['frame_reads']=camera.read_count
            save(folder/'camera.json',metadata)
            return obs,np.asarray(metadata['K']),metadata
        finally:
            try:
                if camera is not None:
                    try:
                        if camera.last is not None:
                            Image.fromarray(camera.last[0]).save(folder/'last_aligned_rgb.png')
                            np.save(folder/'last_aligned_depth_m.npy',camera.last[1],allow_pickle=False)
                            np.save(folder/'last_depth_window_m.npy',np.asarray(camera.depth_window),allow_pickle=False)
                    finally: camera.stop()
            finally:
                if arm is not None:arm.disconnect()


def aggregate(sessions, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];analyses=[];signatures=[]
    for path in sessions:
        path=Path(path).resolve();a=json.loads((path/'analysis.json').read_text());s=json.loads((path/'session.json').read_text())
        analyses.append(dict(session=str(path),position=a['position'],status=a['status'],regions=a['regions']))
        signatures.append((s['board']['source_sha256'],s['board']['scale']))
        rows.extend([dict(session=str(path),session_status=a['status'],**r) for r in a['rows']])
    if len(set(signatures))>1:raise ValueError('Cannot aggregate different board geometries/scales')
    csv_write(output/'results.csv',rows,['session','position','tag_id'])
    summary={}
    for position in ('left','center','right','unspecified'):
        r=[x for x in rows if x['position']==position and x['session_status']=='MEASURED']
        summary[position]=dict(sessions=sum(a['position']==position for a in analyses),tags=len(r),
            xy_error_mm=distribution([x.get('xy_error_mm') for x in r]),
            abs_z_error_mm=distribution([abs(x['dz_mm']) for x in r if x.get('dz_mm') is not None]),
            center_reprojection_px=distribution([x.get('reprojection_error_px') for x in r]))
    save(output/'comparison.json',dict(sessions=analyses,placement_summary=summary))
    lines=['# Placement comparison','', '按操作者指定的 placement 分组；不同 session 的衣板位置不是固定 world 真值。', '',
        '| placement | sessions | tags | XY median/max mm | abs Z median/max mm |','|---|---:|---:|---:|---:|']
    for key,r in summary.items():
        xy=r['xy_error_mm'];z=r['abs_z_error_mm']
        lines.append(f'| {key} | {r["sessions"]} | {r["tags"]} | {xy["median"]} / {xy["max"]} | {z["median"]} / {z["max"]} |')
    lines+=['','仅汇总描述性误差，不拟合 correction map；低质量 PnP session 的原始结果仍保留，但不进入汇总统计。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    parser.add_argument('--board',type=Path,default=DEFAULT_BOARD)
    parser.add_argument('--board-scale',type=float,default=A4_SCALE,help='Existing documented A3->A4 print scale; explicit override for measured print')
    parser.add_argument('--position',choices=('left','center','right','unspecified'),default='unspecified')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/apriltag_geometry_audit')
    parser.add_argument('--session',type=Path,help='Offline reanalysis of an audit session, no device connections')
    parser.add_argument('--aggregate',type=Path,nargs='+',help='Compare existing audit sessions; --output must be a new directory')
    args=parser.parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate,args.output);print(args.output/'summary.md');return 0
    board_path=args.session/'board.yaml' if args.session else args.board
    scale=json.loads((args.session/'session.json').read_text())['board']['scale'] if args.session else args.board_scale
    board=board_config(board_path,scale)
    from pupil_apriltags import Detector
    detector=Detector(families=board['family'],quad_decimate=1.,nthreads=2)
    folder=args.output/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    shutil_board=board_path.read_bytes();(folder/'board.yaml').write_bytes(shutil_board)
    manifest=dict(status='STARTED',created_at=datetime.now(timezone.utc).isoformat(),
        mode='OFFLINE' if args.session else 'READ_ONLY_CAPTURE',position=args.position,
        board={k:v for k,v in board.items() if k!='corners'},robot_motion=False,
        board_scale_source='Existing docs/apriltag_mapping_diagnostic.md A4 specification; actual print size must match')
    save(folder/'session.json',manifest)
    print(f'Audit: {folder}',flush=True)
    try:
        if args.session:
            shutil.copytree(args.session/'observation',folder/'observation')
            shutil.copy2(args.session/'camera.json',folder/'camera.json')
            obs=Observation.load(folder/'observation')
            metadata=json.loads((args.session/'camera.json').read_text());k=np.asarray(metadata['K'])
            manifest['position']=json.loads((args.session/'session.json').read_text())['position']
            manifest['source_session']=str(args.session.resolve())
        else:obs,k,metadata=capture_live(args.config,folder)
        result=analyze(obs,k,metadata,board,detector,metadata['max_surface_spread_mm'],manifest['position'])
        save_outputs(folder,obs,result)
        manifest.update(status=result['status'],detected_tags=result['detected_tags'],accepted_tags=result['accepted_tags'])
        print(json.dumps(dict(status=result['status'],detected=result['detected_tags'],accepted=result['accepted_tags'],regions=result['regions']),ensure_ascii=False),flush=True)
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        (folder/'summary.md').write_text(f'# Audit failed\n\n{manifest["error"]}\n\nNo geometry conclusion; no robot motion or calibration correction.\n')
        print(manifest['error'],flush=True)
    finally:save(folder/'session.json',manifest)
    print(folder/'summary.md',flush=True)
    return 0 if manifest['status']=='MEASURED' else 1


if __name__=='__main__':raise SystemExit(main())
