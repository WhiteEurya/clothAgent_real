"""Optional measured world-Z image, with open/closed-gripper hole filling.

Fused data is prompt-only; robot execution continues using fresh measured depth.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent import hem_sleeve_pull_rule as settings


def enabled(root):
    p=Path(root)/settings.CONFIG
    return p.exists() and json.loads(p.read_text()).get('depth_image') is True


def fuse(first,second,rgb_a,rgb_b,meta_a,meta_b):
    """Never replace first-frame measurements; only fill locally consistent holes."""
    import cv2
    if first.shape!=second.shape or rgb_a.shape!=rgb_b.shape:raise ValueError('frame dimensions differ')
    output=first.copy();filled=np.zeros(first.shape[:2],bool)
    a=np.isfinite(first).all(2);b=np.isfinite(second).all(2)
    same=(meta_a.get('calibration_id')==meta_b.get('calibration_id') and
          np.allclose(meta_a['world_from_camera_mm'],meta_b['world_from_camera_mm'],atol=1e-5,rtol=0))
    if not same:return output,filled,{'status':'SINGLE_FRAME_POSE_MISMATCH'}
    # Local photometric agreement rejects changed foreground/cloth pixels. No global gate.
    changed=np.max(np.abs(rgb_a.astype(float)-rgb_b.astype(float)),axis=2)>30
    changed=cv2.dilate(changed.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
    filled=(~a)&b&(~changed);output[filled]=second[filled]
    return output,filled,{'status':'FUSED','first_valid':int(a.sum()),'second_valid':int(b.sum()),'filled_pixels':int(filled.sum()),'remaining_invalid':int((~np.isfinite(output).all(2)).sum()),'rgb_difference_threshold':30,'changed_neighborhood_px':5}


def render(xyz,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid=np.isfinite(xyz).all(2);cm=plt.get_cmap('turbo').copy();cm.set_bad('#808080')
    fig,ax=plt.subplots(figsize=(12.8,8))
    im=ax.imshow(np.ma.array(xyz[:,:,2],mask=~valid),cmap=cm,vmin=15,vmax=75,interpolation='nearest')
    ax.set(xlabel='Original RGB x (px)',ylabel='Original RGB y (px)',title='Absolute world Z (mm); gray = invalid')
    fig.colorbar(im,ax=ax,label='World Z (mm)',extend='both');fig.tight_layout();fig.savefig(path,dpi=100);plt.close(fig)


def capture_pair(runtime,before,folder):
    folder=Path(folder);dest=folder/'depth_pair';dest.mkdir(exist_ok=False)
    owner,target=runtime.perception_target()
    q=np.array(before.meta['joints_deg'][owner])
    if owner!=runtime.arm_key or np.max(np.abs(q-target))>runtime.config.limits['stationary_tolerance_deg']:
        raise DualArmError('depth pair requires stationary perception pose')
    grip=before.meta['snapshot_after'][owner]['gripper']
    if not grip.get('usable_for_completion') or grip.get('position_pulse',0)<800:
        raise DualArmError('depth pair requires initially open gripper at perception')
    connection=runtime.coordinator.connections[owner];record={'first_observation':str(before.directory),'prompt_only':True}
    try:
        if runtime.cancel.is_set():raise DualArmError('depth pair cancelled')
        record['close_feedback']=connection.gripper('close')
        second=runtime.capture_current(dest/'closed')
    finally:
        if not runtime.cancel.is_set():record['restore_open_feedback']=connection.gripper('open')
        (dest/'capture.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    xyz,filled,report=fuse(before.xyz,second.xyz,np.asarray(Image.open(before.image).convert('RGB')),np.asarray(Image.open(second.image).convert('RGB')),before.meta,second.meta)
    np.save(dest/'prompt_world_xyz_mm.npy',xyz);Image.fromarray(filled.astype('uint8')*255).save(dest/'filled_pixels.png')
    report.update(source_observation_id=before.meta['observation_id'],second_observation_id=second.meta['observation_id'],prompt_only=True)
    (dest/'fusion.json').write_text(json.dumps(report,indent=2));render(xyz,dest/'world_z.png')


def append_image(root,folder,before,body):
    if not enabled(root):return
    from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
    dest=Path(folder)/'depth_pair';report=dest/'fusion.json'
    if report.exists():
        if json.loads(report.read_text())['source_observation_id']!=before.meta['observation_id']:
            raise ValueError('depth image observation mismatch')
        path=dest/'world_z.png'
    else:
        # Recovery uses its new observation, never a heatmap from the old view.
        path=Path(folder)/'world_z.png';render(before.xyz,path)
    body['input'][0]['content'] += [dict(type='input_text',text='补充图为当前RGB对应的world Z深度可视化：毫米，蓝低红高，色标15–75 mm，超范围饱和，灰色无有效深度。坐标轴对应原RGB像素；Pa/Pb仍用第一张RGB坐标。Z不是离桌面高度。'),dict(type='input_image',image_url=io.image_data(path),detail='original')]
