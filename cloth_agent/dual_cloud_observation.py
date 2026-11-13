"""Secondary RGB-D capture; wrist extrinsics use measured stationary arm poses. No motion commands."""
import json
from pathlib import Path
import numpy as np
from .dual_arm.geometry import DualArmError


def _acquire_frames(rs, config, serial, warmup_frames):
    """Retry acquisition only: two waits per stream, at most one stream restart."""
    from .experiments import unfold_console as console
    events=[]
    for attempt in range(2):
        pipe=rs.pipeline();started=False;phase='start'
        try:
            console.emit('相机',f'RealSense {serial} · 打开采集流 {attempt+1}/2')
            pipe.start(config);started=True
            phase='warmup';received=0;timeouts=0
            while received < max(1,warmup_frames):
                try:
                    frames=pipe.wait_for_frames(5000)
                except RuntimeError as exc:
                    if "Frame didn't arrive" not in str(exc):raise
                    timeouts+=1
                    events.append(dict(attempt=attempt+1,phase=phase,received=received,error=str(exc)))
                    if timeouts>=2:raise
                    console.emit('相机重试',f'RealSense {serial} · 5 秒未收到帧，继续等待一次',warning=True)
                    continue
                if not frames.get_depth_frame() or not frames.get_color_frame():
                    raise RuntimeError('incomplete RGB-D frameset')
                received+=1
            console.emit('相机',f'RealSense {serial} · RGB-D 采集完成')
            return pipe,frames,dict(stream_attempts=attempt+1,events=events)
        except BaseException as exc:
            if started:
                try:pipe.stop()
                except Exception as stop_error:
                    raise RuntimeError(f'RealSense {serial}: stream cleanup failed after {exc}: {stop_error}') from exc
            if not isinstance(exc,RuntimeError):raise
            events.append(dict(attempt=attempt+1,phase=phase,error=str(exc)))
            if attempt==1:
                raise RuntimeError(f'RealSense {serial}: {phase} failed after 2 stream attempts: {exc}') from exc
            console.emit('相机重连',f'RealSense {serial} · {phase} 失败：{exc}；关闭后重开采集流',warning=True)


def load_config(path):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    from .dual_camera_tag import rigid
    if cfg['schema_version']!=1:raise ValueError('Unknown dual cloud config')
    rigid(cfg['calibration']['world_from_zed_mm'])
    mount=cfg.get('realsense_mount')
    if mount:
        if mount['arm']!='right' or mount['link']!='link_eef':raise ValueError('Unsupported wrist mount')
        rigid(mount['eef_from_color_mm'])
    else:
        rigid(cfg['calibration']['world_from_realsense_color_mm'])
    for s in cfg['depth_scale'].values():
        if not np.isfinite(s) or not .9<s<1.1:raise ValueError('Invalid measured depth scale')
    return cfg


def scale_world(xyz, world_from_native, scale):
    # Uniform native optical depth scale leaves rays and rigid extrinsics unchanged.
    origin=np.asarray(world_from_native)[:3,3]
    return (np.asarray(xyz)-origin)*scale+origin


def validate_primary(frame, cfg):
    cal=cfg['calibration'];t=np.array(frame.X_base_camera,copy=True);t[:3,3]*=1000
    if str(frame.serial)!=cal['zed_serial'] or list(frame.rgb.shape[1::-1])!=cfg['zed_size_wh']:
        raise DualArmError('fusion primary camera serial/resolution mismatch')
    if not np.allclose(t,cal['world_from_zed_mm'],atol=1e-6,rtol=0):
        raise DualArmError('fusion and runtime ZED world calibration differ')


def resolve_wrist_color_transform(cfg, joints):
    """Compose world<-base<-eef<-color in millimetres from capture-time joints."""
    from .dual_arm.config import DualConfig
    from .dual_arm.kinematics import ArmModel
    from .site_observation import ROOT
    mount=cfg['realsense_mount']
    scene=DualConfig.load_geometry(ROOT/mount['robot_config'],root=ROOT)
    arm=scene.arms[mount['arm']]
    return arm.world_from_base @ ArmModel(arm).frame_at(joints,mount['link']) @ np.asarray(mount['eef_from_color_mm'])


def capture_secondary(cfg, *, wrist_joints=None, frame_provider=None):
    import pyrealsense2 as rs
    cal=cfg['calibration'];pipe=None;config=rs.config()
    mount=cfg.get('realsense_mount');readback=None;wrist_meta={}
    if mount:
        if wrist_joints is None:
            from .site_robot_check import read_arm
            from .site_observation import load_site
            profile=load_site()
            def readback():
                row=read_arm(profile,mount['arm'])
                if row['status']!='READ_ONLY_CHECKS_PASS':raise DualArmError('wrist capture requires stationary ready arm: '+str(row.get('checks')))
                return row['joints_deg']
            wrist_joints=readback()
        color_world=resolve_wrist_color_transform(cfg,wrist_joints)
        wrist_meta=dict(mount_arm=mount['arm'],mount_link=mount['link'],capture_joints_deg=list(wrist_joints))
    else:
        color_world=np.asarray(cal['world_from_realsense_color_mm'])
    config.enable_device(cal['realsense_serial'])
    config.enable_stream(rs.stream.color,*cfg['realsense_color_size_wh'],rs.format.rgb8,30)
    config.enable_stream(rs.stream.depth,*cfg['realsense_depth_size_wh'],rs.format.z16,30)
    started=False;acquisition=dict(source='external_frame_provider')
    try:
        if frame_provider is None:
            pipe,frames,acquisition=_acquire_frames(rs,config,cal['realsense_serial'],cfg['realsense_warmup_frames'])
            started=True
        else:
            frames=frame_provider()
        d,c=frames.get_depth_frame(),frames.get_color_frame()
        intr=d.profile.as_video_stream_profile().get_intrinsics()
        # Grasp angle projection uses a pinhole model; do not silently ignore distortion.
        if any(v!=0 for v in intr.coeffs):raise DualArmError('nonzero native depth distortion needs explicit projection support')
        ci=c.profile.as_video_stream_profile().get_intrinsics()
        if any(v!=0 for v in ci.coeffs):raise DualArmError('nonzero color distortion differs from current calibration')
        k=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1.]])
        ck=np.array([[ci.fx,0,ci.ppx],[0,ci.fy,ci.ppy],[0,0,1.]])
        if not np.allclose(ck,cal['realsense_K'],atol=.01,rtol=0):raise DualArmError('RealSense color intrinsics changed since calibration')
        pc=rs.pointcloud();pc.map_to(c);points=pc.calculate(d)
        native=np.asanyarray(points.get_vertices()).view(np.float32).reshape(intr.height,intr.width,3).copy()*1000
        uv=np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(intr.height,intr.width,2).copy()
        rgb=np.asanyarray(c.get_data()).copy();pix=np.rint(np.nan_to_num(uv)*[rgb.shape[1],rgb.shape[0]]).astype(int)
        valid=np.isfinite(uv).all(2)&np.isfinite(native).all(2)&(native[:,:,2]>.15*1000)&(native[:,:,2]<2000)
        valid&=(pix[:,:,0]>=0)&(pix[:,:,0]<rgb.shape[1])&(pix[:,:,1]>=0)&(pix[:,:,1]<rgb.shape[0])
        texture=np.zeros(native.shape,dtype=np.uint8);texture[valid]=rgb[pix[valid,1],pix[valid,0]]
        ext=d.profile.get_extrinsics_to(c.profile);dc=np.eye(4);dc[:3,:3]=np.array(ext.rotation).reshape(3,3,order='F');dc[:3,3]=np.array(ext.translation)*1000
        world_native=color_world@dc
        raw=native@world_native[:3,:3].T+world_native[:3,3];raw[~valid]=np.nan
        corrected=scale_world(raw,world_native,cfg['depth_scale']['realsense'])
        if readback is not None and np.max(np.abs(np.asarray(readback())-wrist_joints))>.1:
            raise DualArmError('wrist moved during secondary capture; observation discarded')
        # Downstream color projection receives the resolved transform for this
        # observation, never the former fixed-camera calibration.
        cal['world_from_realsense_color_mm']=color_world.tolist()
        return dict(rgb=texture,xyz=corrected,raw_xyz=raw,color_rgb=rgb,
            meta=dict(camera_serial=cal['realsense_serial'],capture_recovery=acquisition,world_from_camera_mm=world_native.tolist(),
            world_from_color_mm=color_world.tolist(),wrist_capture=wrist_meta,
            intrinsics=k.tolist(),color_intrinsics=ck.tolist(),source_grid='native_depth_pixels_with_color_texture',
            depth_to_color_mm=dc.tolist(),depth_scale=cfg['depth_scale']['realsense'],
            camera_timestamp_ms=c.get_timestamp(),depth_size_wh=[intr.width,intr.height]))
    finally:
        if started:pipe.stop()


def attach_secondary(before, capture, cfg, config):
    from .dual_arm.observation import save_observation,sha
    from PIL import Image
    directory=before.directory/'realsense'
    second=save_observation(directory,capture['rgb'],capture['xyz'],config,before.meta['joints_deg'],extra=capture['meta'])
    np.save(directory/'raw_world_xyz_mm.npy',capture['raw_xyz'],allow_pickle=False)
    Image.fromarray(capture['color_rgb']).save(directory/'color_rgb.png')
    before.meta['dual_cloud']={'secondary_directory':'realsense','secondary_observation_sha256':sha(directory/'observation.json'),
        'config':cfg,'capture_policy':'stationary sequential ZED then RealSense; arm snapshots enclose both'}
    (before.directory/'observation.json').write_text(json.dumps(before.meta,indent=2),encoding='utf-8')
    before.secondary=second
    return before
