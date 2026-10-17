"""Generate planning guidance from the installed URDF and calibration.

This is model context, not a new motion gate or a replacement for validation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def _boundaries(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def _arm6_downward(model):
    """Approximate outer bound for the installed xArm6 geometry only.

URDF angles approximate pi/2. Refuse this formula for a different chain/TCP,
and label it as a geometric outer bound, never a sufficient reachability test.
"""
    from scipy.spatial.transform import Rotation
    if model.config.axis!=6:return None
    positions=[[0,0,.267],[0,0,0],[.0535,-.2845,0],[.0775,.3425,0],[0,0,0],[.076,.097,0]]
    rolls=[0,-90,0,-90,90,-90]
    for name,xyz,roll in zip(model.names,positions,rolls):
        joint=model.urdf.joint_map[name]
        if (not np.allclose(joint.origin[:3,3],xyz,atol=1e-6,rtol=0)
            or not np.allclose(joint.origin[:3,:3],Rotation.from_euler('x',roll,degrees=True).as_matrix(),atol=5e-6,rtol=0)
            or not np.allclose(joint.axis,[0,0,1])):return None
    tcp=model.config.tcp_offset
    if not np.allclose(tcp[[0,1,3,4,5]],0):return None
    parents={j.child:j for j in model.urdf.robot.joints}
    link=model.flange
    while link!='link6':
        joint=parents.get(link)
        if joint is None or joint.type!='fixed' or not np.allclose(joint.origin,np.eye(4)):return None
        link=joint.parent
    upper=np.linalg.norm(model.urdf.joint_map['joint3'].origin[:3,3])*1000
    fore=np.linalg.norm(model.urdf.joint_map['joint4'].origin[:3,3])*1000
    horizontal=76.;vertical=float(tcp[2]+97.-267.);length=float(upper+fore)
    rows=[]
    for z in (0.,30.,60.,100.,140.,180.,220.):
        remaining=length**2-(z+vertical)**2
        rows.append(dict(tcp_base_z_mm=z,max_horizontal_radius_mm=float(horizontal+np.sqrt(remaining)) if remaining>=0 else None))
    return dict(assumption='TCP points vertically down in arm6 base frame; yaw arbitrary',
        shoulder_to_wrist_max_mm=length,horizontal_wrist_offset_mm=horizontal,
        vertical_offset_mm=vertical,table=rows)


def build_capabilities(config,models,active_arms=('left','right')):
    report=dict(schema_version=1,calibration_id=config.raw['calibration_id'],
        active_arms=list(active_arms),units='mm_deg',arms={})
    for key,arm in config.arms.items():
        model=models[key];joints=[]
        for name in model.names:
            j=model.urdf.joint_map[name]
            joints.append(dict(name=name,lower_deg=float(np.degrees(j.limit.lower)),
                upper_deg=float(np.degrees(j.limit.upper)),origin_xyz_mm=(j.origin[:3,3]*1000).tolist()))
        report['arms'][key]=dict(robot='arm6' if key=='left' else 'arm7',axis=arm.axis,
            urdf=str(arm.urdf),urdf_sha256=hashlib.sha256(arm.urdf.read_bytes()).hexdigest(),
            world_from_base_mm=arm.world_from_base.tolist(),tcp_offset_mm_deg=arm.tcp_offset.tolist(),
            workspace_base_mm=dict(min=_boundaries(arm.workspace['min_mm']),max=_boundaries(arm.workspace['max_mm'])),
            joints=joints,downward_outer_bound=_arm6_downward(model))
    report['fingerprint']=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
    return report


def render_capabilities(report):
    lines=['# 当前机械臂能力说明（由本机 URDF / 标定生成）',
        f'配置指纹：`{report["fingerprint"][:16]}`。本次允许动作的机械臂：{", ".join(report["active_arms"])}。',
        '这是选点和动作生成的参考，不新增硬性工作区限制。URDF 可达不等于两臂无碰撞，最终仍由宿主验证。',
        '## 坐标与姿态',
        '- motion.grasp / move_tcp 使用 World 米/弧度；下列说明使用毫米/度。目标是 TCP，不是相机或法兰。',
        '- T_base_tcp = inverse(T_world_base) @ T_world_tcp；必须同时转换位置和朝向。不能把 arm7 的 World X 当作它向前伸出的距离。',
        '- 当夹爪竖直朝下时，yaw 通常改变夹爪平面内方向；它不能补偿伸展距离。调整 roll/pitch 会改变 TCP 偏移对应的腕部位置。',
        '- 检查 IK 时使用本次动作契约对应的实际 TCP 目标。',
        '## 两臂模型']
    for key,arm in report['arms'].items():
        base=np.array(arm['world_from_base_mm'])
        lines += [f'### {arm["robot"]} / {key}：{arm["axis"]} 轴',
            f'- 基座 World 位置 mm：{np.round(base[:3,3],3).tolist()}；TCP 偏移 mm/deg：{arm["tcp_offset_mm_deg"]}。',
            '- 基座旋转矩阵（base → World）：`'+json.dumps(np.round(base[:3,:3],6).tolist())+'`。',
            '- 关节角范围 deg：'+', '.join(f'{j["name"]} [{j["lower_deg"]:.2f}, {j["upper_deg"]:.2f}]' for j in arm['joints'])+'。',
            '- 已有基座坐标工作区 min/max mm（null 表示该方向未设限，不代表机械臂无限长）：`'+json.dumps(arm['workspace_base_mm'])+'`。']
        bound=arm['downward_outer_bound']
        if bound:
            lines += ['夹爪沿本臂基座 −Z 竖直朝下时，水平半径 r = sqrt(x_base²+y_base²) 的近似几何外边界：',
                f'`r_max ≈ {bound["horizontal_wrist_offset_mm"]:.3f} + sqrt({bound["shoulder_to_wrist_max_mm"]:.3f}² - (z_base + {bound["vertical_offset_mm"]:.3f})²)`（mm）。',
                '\n'.join(['| TCP 基座 Z（mm） | 水平半径外边界（mm） |','|---:|---:|']+
                    [f'| {row["tcp_base_z_mm"]:.0f} | {row["max_horizontal_radius_mm"]:.1f} |' for row in bound['table'] if row['max_horizontal_radius_mm'] is not None])]
            lines += ['这是外边界，不是保证可达的内工作区；以内仍受关节限位、IK 分支和碰撞影响。接近边界时优先选更近的真实衣物点，避免在边缘高抬；不要机械地添加固定 120 mm 接近点。']
        else:
            lines += ['本臂不套用 arm6 的竖直抓取半径公式；尤其 arm7 有额外关节且基座方向不同，需要用自身模型求 IK。']
    lines += ['## 给动作生成的建议',
        '- 先依据衣物选择动作，再比较两臂到候选抓点的基座距离；不要默认某个画面方向一定由某一臂负责。按任务需要选择单臂或双臂，不强制两臂共同抓取。',
        '- GPT 只声明真实抓点和持布动作。接近、闭合、释放、撤离和 Home 由宿主生成，不输出空手辅助点。',
        '- 持布过程的每个动作目标也需要可达；不要认为抓点与终点可达，就可以任意抬高中间点。',
        '- IK 失败不是碰撞证明，也不等于 yaw 转不过去。先区分真实抓点、中间持布目标、宿主辅助点。',
        '- 代码修正保留任务意图和有效动作；真实抓点失败时，可以在同一保存观测上局部调整失败抓点、可接受姿态或双臂分配，不能凭空把坐标截到边界当作衣物点。无需重做整套任务规划。',
        '如需核实具体候选，可只读运行（参数使用 World mm/deg）：',
        '`/home/sja/miniconda3/envs/cali/bin/python scripts/robot_capabilities.py --arm left --pose-mm-deg X Y Z R P YAW`',
        '上述命令直接检查输入的 TCP 目标。仅当本次动作契约规定宿主额外下压时，使用 `--grasp --grasp-offset-mm N` 明确传入该偏移；已是实际 TCP 目标时不添加这些参数。工具不连接机械臂、不做运动，成功仅代表 IK 通过，不代表路径无碰撞。']
    return '\n\n'.join(lines)+'\n'
