"""Fixed dual-wrist perception using the existing capture and motion adapters."""
import json
from pathlib import Path
import numpy as np
from .dual_arm.geometry import DualArmError


def load_fixed(path):
    from .dual_camera_tag import rigid
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg.get('mode')!='DUAL_REALSENSE_FIXED_PERCEPTION':raise ValueError('Expected fixed dual-wrist calibration')
    for key in ('left','right'):
        q=np.asarray(cfg['capture_joints_deg'][key],float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Invalid perception joints')
    if len(set(cfg['camera_serials'].values()))!=2:raise ValueError('Camera serials must differ')
    right_key='world_from_right_color_mm' if 'world_from_right_color_mm' in cfg else 'candidate_world_from_right_color_mm'
    for name in ('world_from_left_color_mm',right_key,'left_color_from_right_color_mm'):rigid(cfg[name])
    if not np.allclose(np.asarray(cfg['world_from_left_color_mm'])@cfg['left_color_from_right_color_mm'],cfg[right_key],atol=1e-6):raise ValueError('Inconsistent fixed camera transforms')
    return cfg


def assert_capture_pose(fixed,snapshot,tolerance=.1):
    for key in ('left','right'):
        if np.max(np.abs(np.asarray(snapshot[key]['joints'])-fixed['capture_joints_deg'][key]))>tolerance:
            raise DualArmError(f'{key}: fixed camera extrinsics require the recorded perception joints')


def capture_pair(runtime,directory,before,snapshots):
    from PIL import Image
    from .dual_cloud_observation import capture_secondary
    from .dual_arm.observation import save_observation,sha
    fixed=runtime.dual_wrist_config
    for key in fixed['sequence']:
        q=fixed['capture_joints_deg'][key];saved={'joints_deg':q,'pose_base_mm_deg':runtime.models[key].forward(q).tolist()}
        before=runtime._perception_before_capture(Path(directory).with_name(Path(directory).name+'_perception')/key,before,snapshots,arm=key,saved=saved)
    assert_capture_pose(fixed,before)
    captures=[]
    right_key='world_from_right_color_mm' if 'world_from_right_color_mm' in fixed else 'candidate_world_from_right_color_mm'
    for key,name in [('left','world_from_left_color_mm'),('right',right_key)]:
        cal=fixed['color_intrinsics'][key]
        cfg={'calibration':{'realsense_serial':fixed['camera_serials'][key],'realsense_K':cal,'world_from_realsense_color_mm':fixed[name]},'realsense_color_size_wh':[1280,720],'realsense_depth_size_wh':[1280,720],'realsense_warmup_frames':60,'depth_scale':{'realsense':1.}}
        recorder=getattr(runtime,'video_recorder',None)
        provider=recorder.read_native_frames if key=='left' and recorder is not None and hasattr(recorder,'read_native_frames') else None
        captures.append(capture_secondary(cfg,frame_provider=provider))
    zed_frame=None
    if getattr(runtime,'topdown_config',{}).get('zed_supplement',{}).get('enabled',False):
        from .perception import PerceptionConfig,capture_two_view_rgbd
        from .site_observation import ROOT
        perception=PerceptionConfig.load(ROOT,ROOT/'config/perception.zed_dual_xarm7_20261103.json')
        recorder=getattr(runtime,'video_recorder',None)
        if recorder is not None and hasattr(recorder,'camera_factory'):
            if getattr(recorder.camera,'rgb_only',False):raise DualArmError('ZED supplement requires measured RGB-D')
            frames=capture_two_view_rgbd(perception,camera_factory=recorder.camera_factory)
        else:frames=capture_two_view_rgbd(perception)
        if len(frames)!=1:raise DualArmError('expected one fixed ZED frame')
        zed_frame=frames[0]
        transform=np.array(zed_frame.X_base_camera,copy=True);transform[:3,3]*=1000
        if str(zed_frame.serial)!=str(runtime.config.raw['camera']['serial']) or not np.allclose(transform,runtime.config.raw['camera']['world_from_camera_mm'],atol=1e-6,rtol=0):
            raise DualArmError('ZED supplement calibration/serial differs from robot scene')
    after=snapshots();assert_capture_pose(fixed,after)
    for key in before:
        if np.max(np.abs(np.asarray(before[key]['joints'])-after[key]['joints']))>runtime.config.limits['stationary_tolerance_deg']:raise DualArmError('arm moved during dual-wrist capture')
    observations=[]
    # Operational transforms apply to original measured XYZ as well as RGB.
    # Archived display-only candidates remain observation-only.
    for key,cap in zip(('left','right'),captures):
        target=Path(directory) if key=='left' else Path(directory)/'right'
        extra=dict(cap['meta'],snapshot_before=before,snapshot_after=after,
                   common_frame='right_arm_base',topdown_config=runtime.topdown_config,
                   fixed_camera_mapping={name:fixed[name] for name in ('capture_joints_deg','world_from_left_color_mm',right_key,'left_color_from_right_color_mm')},
                   mapping_validation=fixed.get('validation',{}))
        if right_key=='candidate_world_from_right_color_mm':
            extra.update(execution_allowed=False,execution_block_reason='Archived display-only camera mapping')
        ob=save_observation(target,cap['rgb'],cap['xyz'],runtime.config,{k:v['joints'] for k,v in before.items()},extra=extra)
        Image.fromarray(cap['color_rgb']).save(ob.directory/'color_rgb.png');observations.append(ob)
    first,second=observations
    first.meta['dual_cloud']={'secondary_directory':'right','secondary_observation_sha256':sha(second.directory/'observation.json'),'config':{'mode':'dual_wrist_fixed'},'capture_policy':'sequential stationary fixed-pose dual RealSense'}
    if zed_frame is not None:
        from .site_observation import world_xyz_mm
        zed=save_observation(Path(directory)/'zed',zed_frame.rgb,world_xyz_mm(zed_frame),runtime.config,
            {k:v['joints'] for k,v in before.items()},extra=dict(camera_serial=zed_frame.serial,
                intrinsics=zed_frame.intrinsics.tolist(),world_from_camera_mm=transform.tolist(),
                snapshot_before=before,snapshot_after=after,common_frame='right_arm_base'))
        first.meta['dual_cloud']['additional_sources']=[dict(directory='zed',observation_sha256=sha(zed.directory/'observation.json'))]
        first.meta['dual_cloud']['capture_policy']='sequential stationary dual wrist + fixed ZED RGB-D'
    (first.directory/'observation.json').write_text(json.dumps(first.meta,indent=2),encoding='utf-8')
    first.secondary=second
    return first
