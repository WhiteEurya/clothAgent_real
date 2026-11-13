"""Optional single post-reset visibility press through the existing RobotAPI chain."""
import numpy as np
from PIL import Image
from cloth_agent.dual_arm.geometry import DualArmError, matrix_pose, pose_matrix
from cloth_agent.replay_dual_arm_runtime import route_grasp
from .discovery_data import save

SCHEMA = dict(type='object',additionalProperties=False,properties={
    'decision':dict(type='string',enum=['PRESS','SKIP','UNCERTAIN']),
    'center_px':dict(anyOf=[dict(type='null'),dict(type='array',items=dict(type='integer',minimum=0),minItems=2,maxItems=2)]),
    'confidence':dict(type='number',minimum=0,maximum=1),
    'reason':dict(type='string',minLength=1)},required=['decision','center_px','confidence','reason'])
PROMPT = '''检查 reset 落下后的当前衣服，判断是否有明显竖起/拱起的布面遮挡其他衣物边缘或整体形态。
不是把所有皱纹都当成遮挡，也不要求衣服完全展平。仅在明确有此类遮挡、且凸起上有宽阔可见的布面可供一次竖直按压时返回 PRESS。
选凸起上可见布面的内部中心，不选尖角、窄边、洞口、桌面、机械臂或夹爪，不猜被遮挡区域。
若没有这种遮挡返回 SKIP；若无法确定遮挡或合适接触位置返回 UNCERTAIN。非 PRESS 的 center_px 必须为 null。
坐标为当前原始 RGB 像素。reason 简述所见遮挡及选择理由；不得声称已经消除遮挡。
你只选可见位置，不能决定按压力度、下降深度或发送机器人命令。只输出指定 JSON。'''


def run(runtime, before, folder, *, headers, timeout, direct, depth_mm):
    from . import one_shot_goal_planner_probe_v1 as planner
    folder.mkdir()
    report=dict(status='PREPARING',physical_execution=False,press_depth_mm=depth_mm)
    try:
        before.validate_for(runtime.config,live=True,check_age=False)
        with Image.open(before.image) as image:size=list(image.size)
        body=planner.payload(before.image,before.image,size)
        body['instructions']=PROMPT
        body['input'][0]['content']=[dict(type='input_text',text=f'CURRENT RGB size: {size}'),
            dict(type='input_image',image_url=planner.image_data(before.image),detail='high')]
        body['text']['format'].update(name='post_reset_occlusion',schema=SCHEMA)
        decision,metrics=direct(folder/'request',body,headers,timeout)
        save(folder/'decision.json',decision);save(folder/'model_metrics.json',metrics)
        planner.jsonschema.validate(decision,SCHEMA)
        report['decision']=decision
        if decision['decision']!='PRESS':
            if decision['center_px'] is not None:raise ValueError('non-PRESS must have null center')
            report['status']='SKIPPED';return report
        pixel=decision['center_px']
        if pixel is None or not (0<=pixel[0]<size[0] and 0<=pixel[1]<size[1]):raise ValueError('invalid press pixel')
        if decision['confidence']<.8:
            report['status']='SKIPPED';return report
        if depth_mm is None or not np.isfinite(depth_mm) or not 0<depth_mm<=15:
            raise ValueError('explicit press depth in (0,15] mm required')
        surface=np.asarray(before.sample(pixel,runtime.config.limits['max_surface_spread_mm']),float)
        joints=before.meta['joints_deg']
        tcp={k:(a.world_from_base @ pose_matrix(runtime.models[k].forward(joints[k])))[:3,3] for k,a in runtime.config.arms.items()}
        raw=np.r_[surface/1000,0.,0.,0.].tolist()
        _,assignment=route_grasp(f'motion.grasp(right={raw!r})\nmotion.move_tcp(right={raw!r})\n',tcp)
        active=assignment['active_arm'];arm=runtime.config.arms[active]
        rpy=matrix_pose(arm.world_from_base @ pose_matrix(runtime.models[active].forward(arm.home_joints)))[3:]
        above=surface+[0,0,50.]
        source=''.join(f'motion.{name}(right={np.r_[point/1000,np.radians(rpy)].tolist()!r})\n'
                       for name,point in [('grasp',surface),('move_tcp',above)])
        _,program,assignment=runtime.compile(source,before,press_depth_mm=depth_mm)
        save(folder/'geometry.json',dict(surface_world_mm=surface.tolist(),contact_world_mm=(surface-[0,0,depth_mm]).tolist(),
            assignment=assignment,policy='one position-controlled press, no force feedback',source_observation=str(before.directory)))
        (folder/'robotapi_preview.py').write_text(program,encoding='utf-8')
        with Image.open(before.image) as image:
            from PIL import ImageDraw
            preview=image.convert('RGB');draw=ImageDraw.Draw(preview);x,y=pixel
            draw.ellipse((x-8,y-8,x+8,y+8),outline='red',width=3);preview.save(folder/'press_point.png')
        report.update(runtime.execute(source,before,folder/'execution',press_depth_mm=depth_mm))
        return report
    except BaseException as exc:
        report.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save(folder/'result.json',report)
