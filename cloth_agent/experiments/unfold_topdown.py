"""Measured top-down planning view with explicit source-pixel execution mapping."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from cloth_agent.pixel_mapping_visualization import PixelMappingView
from cloth_agent.depth_stability import table_depth_mask
from .discovery_data import save
from . import one_shot_goal_planner_probe_v1 as io

CONFIG=Path(__file__).resolve().parents[2]/'config/topdown.unfold_zed.json'
def reconstruct(before, folder, *, config=None):
    folder=Path(folder)/'topdown';folder.mkdir(parents=True,exist_ok=True)
    report=dict(status='UNAVAILABLE',role='primary',coordinate_output='topdown_pixels',global_depth_coverage_gate=False)
    try:
        cfg=config if config is not None else json.loads(CONFIG.read_text(encoding='utf-8'))
        if not hasattr(before,'xyz') or not hasattr(before,'meta'):
            raise ValueError('observation has no measured point cloud')
        rgb=np.asarray(Image.open(before.image).convert('RGB'))
        if cfg['image_size_wh']!=[rgb.shape[1],rgb.shape[0]] or str(before.meta.get('camera_serial'))!=cfg['camera_serial']:
            raise ValueError('topdown ROI camera/size mismatch')
        valid=table_depth_mask(rgb.shape[:2],cfg['table_polygon_px']) & np.isfinite(before.xyz).all(axis=2)
        omitted_below_table=0
        if 'display_min_world_z_mm' in cfg:
            excluded=valid & (before.xyz[...,2]<cfg['display_min_world_z_mm'])
            omitted_below_table=int(excluded.sum());valid &= ~excluded
        view=PixelMappingView(rgb,before.xyz,valid,frame_label='WORLD XY TOP-DOWN',show_xy_grid=False,max_points=None)
        view.xy_image.save(folder/'rgb.png')
        view.measured_image.save(folder/'planning_rgb.png')
        np.save(folder/'source_xy.npy',view.source_xy,allow_pickle=False)
        np.save(folder/'world_xyz_mm.npy',view.projected_xyz,allow_pickle=False)
        view.render(show_grid=False).save(folder/'comparison.png')
        report.update(status='AVAILABLE',source_image=str(before.image),source_sha256=io.sha(before.image),
            observation_id=before.meta.get('observation_id'),valid_points=int(valid.sum()),
            omitted_below_table=omitted_below_table,display_min_world_z_mm=cfg.get('display_min_world_z_mm'),
            segmentation_profile=cfg.get('segmentation_profile','auto'),
            source_depth_sha256=before.meta.get('sha256',{}).get('world_xyz_mm.npy'),
            center_world_xy_mm=view.center.tolist(),span_mm=view.span,
            mm_per_pixel=view.span/view.plot_size,table_polygon_px=cfg['table_polygon_px'],
            projection='highest measured Z per XY raster cell; no hole filling; finite ROI points passing display-only height bound',
            limitation='Single-view occlusions remain unknown; highest measured is not guaranteed true top surface')
    except ValueError as exc:
        report['reason']=str(exc)
    save(folder/'view.json',report)
    return report


class PrimaryView:
    """Top-down planning pixels map to the actual measured source RGB pixel."""
    def __init__(self, source, directory):
        self.source=source
        self.directory=Path(directory)
        self.image=self.directory/'planning_rgb.png'
        metadata=json.loads((self.directory/'view.json').read_text(encoding='utf-8')) if (self.directory/'view.json').is_file() else {}
        self.segmentation_profile=metadata.get('segmentation_profile','auto')
        self.source_xy=np.load(self.directory/'source_xy.npy',allow_pickle=False)
        self.xyz=np.load(self.directory/'world_xyz_mm.npy',allow_pickle=False)
        self.unknown_mask=~np.isfinite(self.xyz).all(axis=2)

    def source_pixel(self, pixel):
        from cloth_agent.dual_arm.geometry import DualArmError
        x,y=map(int,pixel);h,w=self.source_xy.shape[:2]
        if not (0<=x<w and 0<=y<h) or self.unknown_mask[y,x]:
            raise DualArmError('selected topdown point has no measured source depth')
        return self.source_xy[y,x].tolist()

    def sample(self, pixel, spread):
        return self.source.sample(self.source_pixel(pixel),spread)

    def raw_plan(self, plan):
        result=dict(plan)
        if plan['decision']!='ACTION':return result
        result['grasp_px']=self.source_pixel(plan['grasp_px'])
        result['target_px']=self.source_pixel(plan['target_px'])
        angle=plan['grasp_angle_deg']
        if angle is not None:
            # Project the top-down opening axis into the source camera at the
            # actual grasp surface. Do not pass a world-plane angle as RGB yaw.
            a=np.asarray(self.sample(plan['grasp_px'],None),float)
            rad=np.deg2rad(angle)
            b=a+10*np.array([np.cos(rad),-np.sin(rad),0.])
            T=np.asarray(self.source.meta['world_from_camera_mm'],float)
            K=np.asarray(self.source.meta['intrinsics'],float)
            camera=(np.array([a,b])-T[:3,3])@T[:3,:3]
            if np.any(camera[:,2]<=0):raise ValueError('grasp axis projects behind camera')
            uv=camera@K.T;uv=uv[:,:2]/uv[:,2:]
            direction=uv[1]-uv[0]
            if np.linalg.norm(direction)<1e-6:raise ValueError('degenerate grasp axis projection')
            result['grasp_angle_deg']=float(np.degrees(np.arctan2(direction[1],direction[0]))%180)
        return result


PRIMARY_GUIDANCE='''【主规划视图：实测TOP-DOWN】
第一张current是当前RGB-D生成的俯视图，不是原始相机图。双相机图可能使用表面重采样和局部颜色补全以改善显示；执行仍只使用原始实测深度。所有边界、多边形、抓放点和图像夹爪角度均在第一张俯视图上输出。
+X向右、+Y向上；Host会把所选点回映到原始RGB的实测位置再执行，模型不得自行换坐标。
这张图允许残缺。暗色空白是未观测栅格，不代表衣物孔洞或裸桌面；不要求整件衣服深度完整，
不得仅因空白多、有效深度比例低或重建不完整而ABSTAIN，也不要求先补齐深度。
根据仍可见的结构规划，在有实测颜色的可见布料处抓取，选择有实测颜色的桌面放置。
不能捏造隐藏衣角；实际选中的点没有实测对应时可以换点。俯视图只包含最高已测表面，不保证恢复遮挡层。
历史图若提供也为当时的俯视图；目标图仅用于形态参考，不照搬坐标。
'''


def primary(before, folder, *, config=None):
    """Rebuild from this observation, with no global valid-depth ratio gate."""
    if config is None:
        config=getattr(before,'meta',{}).get('topdown_config')
    if getattr(before,'meta',{}).get('dual_cloud'):
        from .unfold_dual_topdown import reconstruct as reconstruct_dual
        cfg=config if config is not None else json.loads(CONFIG.read_text(encoding='utf-8'))
        return reconstruct_dual(before,folder,cfg)
    report=reconstruct(before,folder,config=config)
    if report['status']!='AVAILABLE':
        raise ValueError('topdown reconstruction unavailable: '+report.get('reason',''))
    output=Path(folder)/'topdown'
    report.update(role='primary',coordinate_output='topdown_pixels',global_depth_coverage_gate=False)
    save(output/'view.json',report)
    return PrimaryView(before,output)


def guidance_for(before):
    if getattr(before,'meta',{}).get('topdown_config',{}).get('appearance_render')=='rgb_mosaic':
        return '''【主规划视图：RGB俯视拼图】
第一张current由两台原始RGB投影拼合，用于观察衣物结构；显示不依赖深度完整性。所有边界、抓放点和图像夹爪角度均在第一张图输出，+X向右、+Y向上。
显示平面高度不是抓取高度；Host将选点回映到原相机，再读取实测深度。接缝、色差不代表衣物折叠；空白不代表衣物孔洞，不能仅因深度缺失而ABSTAIN。目标图只作形态参考，不照搬坐标。'''
    return PRIMARY_GUIDANCE
