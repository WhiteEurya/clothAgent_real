#!/usr/bin/env python3
"""Viser dual-arm URDF and Home-route viewer. Saved inputs only; no hardware API."""
from __future__ import annotations
import argparse
from functools import partial
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloth_agent.dual_arm.geometry import pose_matrix, segment_distance
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.setup import mesh_capsules

STRATEGIES = {'同步回 Home': [('left', 'right')],
              'arm6 先回': [('left',), ('right',)],
              'arm7 先回': [('right',), ('left',)]}


def route_joints(start, home, strategy, progress):
    """Dimensionless progress, exact saved joint endpoints, stationary other arm."""
    groups = STRATEGIES[strategy]
    phase = min(int(progress * len(groups)), len(groups)-1)
    u = np.clip(progress * len(groups)-phase, 0, 1)
    s = 10*u**3 - 15*u**4 + 6*u**5
    q = {k: np.asarray(v, dtype=float).copy() for k,v in start.items()}
    for group in groups[:phase]:
        for k in group:
            q[k] = np.asarray(home[k], dtype=float).copy()
    for k in groups[phase]:
        q[k] = q[k] + s*(np.asarray(home[k])-q[k])
    return q


def run(directory, host, port, runtime_envelopes=False, config_path=None):
    import trimesh
    import viser
    import viser.transforms as tf
    import yourdfpy
    from viser.extras import ViserUrdf

    snapshot = json.loads((directory/'snapshot.json').read_text())
    raw = json.loads((config_path or directory/'installation_config.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    runtime_config = None
    if runtime_envelopes:
        from cloth_agent.dual_arm.config import DualConfig
        from cloth_agent.dual_arm.safety import padded_capsules
        # Validate numeric/schema structure for visualization of saved drafts.
        # Only this local copy uses synthetic status; never rewrite raw or files.
        from copy import deepcopy
        preview_structure = deepcopy(raw)
        preview_structure['calibration_status'] = 'synthetic'
        preview_structure['safety']['status'] = 'synthetic'
        runtime_config = DualConfig.parse(preview_structure, ROOT, homing_only=True)
        runtime_config.execution_mode = 'controller_sequential'
    start = {k: np.array(v['joints_deg']) for k,v in snapshot['arms'].items()}
    home = {k: np.array(a['home_joints_deg']) for k,a in raw['arms'].items()}
    colors = {'left': (65,155,245), 'right': (245,155,55)}
    models, visuals, handles, tcps, accessories = {}, {}, {}, {}, {}
    from cloth_agent.dual_arm.viewer_accessories import load_hand_eye, add_accessories
    server = viser.ViserServer(host=host, port=port, label='双臂 Home · 只读快照预览')
    server.gui.configure_theme(dark_mode=True,show_share_button=False)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/world/grid', plane='xy', width=1.6, height=1.4, cell_size=.1,
                          section_size=.5, plane_opacity=0, position=(.225,0,0))
    server.scene.add_label('/world/note', 'Z=0 为底座参考平面，非已测量桌面', position=(.225,-.4,0))
    server.scene.add_frame('/world/origin', axes_length=.12, axes_radius=.003)
    for k, arm in raw['arms'].items():
        world = np.asarray(arm['world_from_base_mm'])
        root = f'/robots/{k}'
        server.scene.add_frame(root, position=world[:3,3]/1000,
                               wxyz=tf.SO3.from_matrix(world[:3,:3]).wxyz,
                               axes_length=.1, axes_radius=.003)
        server.scene.add_label(root+'/label', f"arm{arm['axis']} · {arm['ip']}",position=(0,0,-.045))
        visual_path = ROOT/arm['urdf']
        visual_urdf = yourdfpy.URDF.load(visual_path,load_meshes=True,load_collision_meshes=False,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=visual_path.parent))
        if not visual_urdf.scene.geometry:
            raise RuntimeError(f'No URDF visual meshes loaded: {visual_path}')
        visual = ViserUrdf(server, visual_urdf, root_node_name=root,
                           mesh_color_override=tuple(c/255 for c in colors[k]))
        visuals[k] = (visual, list(visual_urdf.actuated_joint_names))
        caps = (arm['collision_capsules'] if runtime_envelopes else
                mesh_capsules(ROOT/arm['urdf'], arm['axis'],
                              tool_radius_mm=report['tool_placeholder_radius_mm'],tcp_offset=arm['tcp_offset_mm_deg']))
        cfg = SimpleNamespace(arm_id=k,axis=arm['axis'],urdf=ROOT/arm['urdf'],raw=arm,
                              capsules=caps,tcp_offset=np.array(arm['tcp_offset_mm_deg']),world_from_base=world)
        models[k] = ArmModel(cfg)
        handles[k] = {}
        layers = {'外形包络': models[k].capsules(start[k])}
        if runtime_envelopes:
            layers['运行检查包络'] = padded_capsules(runtime_config, {k: models[k]}, start)[k]
        for layer, geometry in layers.items():
            for cap in geometry:
                length = np.linalg.norm(cap.end-cap.start)/1000
                mesh = trimesh.creation.capsule(height=length,radius=cap.radius/1000,count=[12,12])
                handles[k][layer, cap.name] = server.scene.add_mesh_simple(f'/envelopes/{k}/{layer}/{cap.name}',
                    mesh.vertices, mesh.faces, color=colors[k],opacity=.2 if layer=='外形包络' else .08,visible=False)
        tcps[k] = server.scene.add_frame(f'/tcp/{k}',axes_length=.085,axes_radius=.003,origin_radius=.009,origin_color=(255,30,160))
        server.scene.add_label(f'/tcp/{k}/label',f"arm{arm['axis']} TCP · 配置工具点",position=(.035,0,.045),depth_test=False,font_screen_scale=1.3)
        hand_eye_path = ROOT/('config/extrinsics_A.yaml' if k=='left' else 'config/calibration/dual_arm_working_20261008/camB_extrinsics.yaml')
        accessories[k]=add_accessories(server,f'/accessories/{k}',k,load_hand_eye(hand_eye_path),ROOT/'assets/robots/xarm7/xarm7.urdf',k=='left')

    server.gui.add_markdown('## 双臂 Home 路径\n**只读已保存快照；没有机器人连接和执行按钮。**\n\n'
                            f'快照目录：`{directory.name}`\n\n蓝色 arm6 / 橙色 arm7；底座原点间距 {np.linalg.norm(np.asarray(raw["arms"]["right"]["world_from_base_mm"])[:3,3]-np.asarray(raw["arms"]["left"]["world_from_base_mm"])[:3,3]):.2f} mm。')
    if config_path:
        server.gui.add_markdown(f'参数来自：`{config_path}`；姿态仍取自上述快照。')
    with server.gui.add_folder('诊断与限制',expand_by_default=False):
        server.gui.add_markdown('\n\n'.join('- '+v for v in report['warnings'])+
            '\n\nURDF 外形仅代表模型：arm6 已复用 xArm 标准夹爪模型；两台开合角度未采集，使用默认值。相机光学坐标按手眼标定放置（B 按 link_eef 挂载解释）；90×25×25 mm 外壳与光心偏置、打印件为示意，未加入碰撞检查。')
        server.gui.add_markdown('\n\n'.join(f"**{k}**：模型/控制器 TCP 误差 {v['model_vs_controller_tcp_mm_deg'][0]:.2f} mm / {v['model_vs_controller_tcp_mm_deg'][1]:.2f}°" for k,v in report['checks'].items()))
    strategy = server.gui.add_dropdown('回位顺序',options=list(STRATEGIES),initial_value='arm6 先回')
    slider = server.gui.add_slider('路径进度 %（不是时间）',min=0,max=100,step=1,initial_value=0)
    playing = server.gui.add_checkbox('播放路径',initial_value=False)
    show_caps = server.gui.add_checkbox('显示保守胶囊包络',initial_value=runtime_envelopes)
    envelope_layer = server.gui.add_dropdown('包络显示',
        options=['外形包络', '运行检查包络', '两层对比'] if runtime_envelopes else ['外形包络'],
        initial_value='外形包络')
    if runtime_envelopes:
        server.gui.add_markdown('外形包络仅展示配置中的几何模型；运行检查包络另含误差及停车余量。切换显示不会更改运动检查。\n\n'
                                '进度条仅展示直接关节插值，不代表已通过规划，也不展示回退路径。')
    show_accessories=server.gui.add_checkbox('显示夹爪补全、相机和打印件',initial_value=True)
    current_button=server.gui.add_button('显示采集时的位置')
    home_button=server.gui.add_button('显示保存的 Home')
    info=server.gui.add_markdown('')
    lock=threading.RLock()
    editor_controls = []
    mesh_radii = {}
    if runtime_envelopes:
        from cloth_agent.dual_arm.envelope_editor import (
            REACTION_TERMS, STEP_LABELS, envelope_breakdown, save_draft)
        from cloth_agent.dual_arm.geometry import Capsule
        envelope_layer.value = '两层对比'
        with server.gui.add_folder('逐项余量与保存', expand_by_default=True):
            profile = server.gui.add_dropdown('执行模型',
                options=['Home：控制器分段顺序运动', '旧模型：主机逐点伺服'],
                initial_value='Home：控制器分段顺序运动')
            stop_model = server.gui.add_dropdown('停止空间算法',
                options=['新：关节停止范围细分', '旧：统一增加半径'],
                initial_value='新：关节停止范围细分')
            stop_arm = server.gui.add_dropdown('制动示意运动臂',options=['left','right'],initial_value='left')
            show_stop = server.gui.add_checkbox('显示制动姿态示意',initial_value=True)
            proof_button = server.gui.add_button('检查当前示例段（不运动）')
            proof_status = server.gui.add_markdown('新算法不把停止位移加进整圈半径；检查各关节独立停止的完整范围。'
                '彩色虚影只展示部分制动姿态，不是完整范围的证明。')
            server.gui.add_markdown('Home 已改为一臂完成后再动另一臂。此模型不叠加主机逐点发送延迟；'
                                    '停止、跟踪和几何误差仍保留。旧模型可切换对比。')
            server.gui.add_markdown('调整后实时预览。保存按钮写入 `config/dual_arm.envelope_draft.json`，'
                                    '再次保存会备份旧版本。调整值标为待验证，不覆盖实机配置。')
            stage = server.gui.add_dropdown('显示叠加到哪一步',
                options=['全部叠加', '仅外形', *STEP_LABELS.values()], initial_value='全部叠加')
            selected = server.gui.add_dropdown('逐项明细部件',
                options=[f'{k}/{c["name"]}' for k,a in raw['arms'].items() for c in a['collision_capsules']],
                initial_value='right/tool')
            detail = server.gui.add_markdown('')
            radius_table = server.gui.add_markdown('')
            save_button = server.gui.add_button('保存当前调整')
            save_status = server.gui.add_markdown('尚未保存。')

        def number_control(label, container, key, low, high, step):
            control = server.gui.add_number(label, initial_value=container[key],
                                            min=low, max=high, step=step)
            def changed(_event):
                with lock:
                    container[key] = float(control.value)
                    save_status.content = '有未保存的调整。'
                    update()
            control.on_update(changed)
            editor_controls.append(control)

        with server.gui.add_folder('共用跟踪与延迟参数', expand_by_default=False):
            number_control('位置跟踪误差 mm', raw['limits'], 'tracking_error_mm', 0, 100, .5)
            number_control('关节跟踪误差 °', raw['limits'], 'tracking_error_deg', 0, 30, .05)
            number_control('控制频率 Hz', raw['limits'], 'rate_hz', 1, 200, 1)
            for name, section, field in REACTION_TERMS:
                number_control(STEP_LABELS[name]+' s', raw[section], field, 0, .5, .005)
            number_control('额外碰撞净空 mm（不计入半径）', raw['limits'], 'clearance_mm', 0, 100, 1)
        for k,a in raw['arms'].items():
            with server.gui.add_folder(f'{k} 误差与停止参数', expand_by_default=False):
                row = raw['safety']['arms'][k]
                number_control('基座误差 mm', row, 'base_error_mm', 0, 100, .5)
                number_control('几何误差 mm', row, 'geometry_error_mm', 0, 100, .5)
                number_control('延迟计算速度上界 °/s', row, 'max_joint_speed_deg_s', 0, 100, .5)
                for j in range(a['axis']):
                    number_control(f'J{j+1} 停止角位移 °', row['stop_excursion_deg'], j, 0, 30, .05)
            with server.gui.add_folder(f'{k} 外形半径', expand_by_default=False):
                for cap in a['collision_capsules']:
                    number_control(cap['name']+' mm', cap, 'radius_mm', 1, 500, 1)

        @save_button.on_click
        def _save(_event):
            with lock:
                try:
                    mode = 'controller_sequential' if profile.value.startswith('Home') else 'servo_stream'
                    path = save_draft(raw, envelope_breakdown(raw, models, execution_mode=mode),
                                      ROOT/'config/dual_arm.envelope_draft.json', execution_mode=mode)
                    save_status.content = f'已保存：`{path}`（待验证草稿）。'
                except (OSError, ValueError) as exc:
                    save_status.content = f'保存失败：{exc}'

    def update():
        with lock, server.atomic():
            q=route_joints(start,home,strategy.value,slider.value/100)
            geometry={}
            if runtime_envelopes:
                mode = 'controller_sequential' if profile.value.startswith('Home') else 'servo_stream'
                accounting = envelope_breakdown(raw, models, execution_mode=mode)
                arm_key, cap_name = selected.value.split('/')
                row = accounting[arm_key][cap_name]
                cumulative = row['nominal_mm']
                lines = ['| 项目 | 增加半径 mm | 累计半径 mm |', '|---|---:|---:|',
                         f'| 外形 | {cumulative:.2f} | {cumulative:.2f} |']
                for key, label in STEP_LABELS.items():
                    amount = row['increments_mm'][key]
                    cumulative += amount
                    lines.append(f'| {label} | {amount:.2f} | {cumulative:.2f} |')
                use_sweep = mode == 'controller_sequential' and stop_model.value.startswith('新')
                detail.content = ('**旧半径叠加的分项对照（新算法将停止/关节误差改为关节范围检查）**\n\n' if use_sweep else '') + '\n'.join(lines) + f'\n\n额外碰撞净空：{raw["limits"]["clearance_mm"]:.1f} mm（不重复加到两臂半径）。'
                radius_table.content = '| 部件 | 外形半径 | 旧叠加半径 mm |\n|---|---:|---:|\n' + '\n'.join(
                    f'| {k}/{name} | {r["nominal_mm"]:.1f} | {r["total_mm"]:.1f} |'
                    for k, rows in accounting.items() for name,r in rows.items())
            for k,model in models.items():
                visual,names=visuals[k]
                angles={f'joint{i+1}':v for i,v in enumerate(np.radians(q[k]))}
                visual.update_cfg(np.array([angles.get(name,0.) for name in names]))
                flange=model.config.world_from_base@model.frame_at(q[k],'link_eef')
                accessories[k].position=flange[:3,3]/1000
                accessories[k].wxyz=tf.SO3.from_matrix(flange[:3,:3]).wxyz
                accessories[k].visible=show_accessories.value
                tcp=model.config.world_from_base@pose_matrix(model.forward(q[k]))
                tcps[k].position=tcp[:3,3]/1000
                tcps[k].wxyz=tf.SO3.from_matrix(tcp[:3,:3]).wxyz
                geometry[k]=model.capsules(q[k])
                if runtime_envelopes:
                    displayed = []
                    for cap in geometry[k]:
                        radius = cap.radius
                        if stage.value != '仅外形':
                            for term,label in STEP_LABELS.items():
                                if not use_sweep or term not in ('stop','tracking_deg'):
                                    radius += accounting[k][cap.name]['increments_mm'][term]
                                if stage.value == label:
                                    break
                        displayed.append(Capsule(cap.name,cap.start,cap.end,radius))
                    geometry[k] = displayed
                layers = {'外形包络': model.capsules(q[k])}
                if runtime_envelopes:
                    layers['运行检查包络'] = geometry[k]
                for layer, capsules in layers.items():
                    for cap in capsules:
                        handle=handles[k][layer,cap.name]
                        mesh_key = (k, layer, cap.name)
                        if mesh_radii.get(mesh_key) != cap.radius:
                            mesh = trimesh.creation.capsule(height=np.linalg.norm(cap.end-cap.start)/1000,
                                                           radius=cap.radius/1000,count=[12,12])
                            handle.vertices = mesh.vertices
                            handle.faces = mesh.faces
                            mesh_radii[mesh_key] = cap.radius
                        handle.position=(cap.start+cap.end)/2000
                        direction=cap.end-cap.start
                        rotation=trimesh.geometry.align_vectors([0,0,1],direction) if np.linalg.norm(direction)>1e-9 else np.eye(4)
                        handle.wxyz=tf.SO3.from_matrix(rotation[:3,:3]).wxyz
                        handle.visible=show_caps.value and envelope_layer.value in (layer,'两层对比')
            if runtime_envelopes:
                update_stop_ghosts(q, use_sweep)
            nearest=min((float(segment_distance(a.start,a.end,b.start,b.end)-a.radius-b.radius),a.name,b.name)
                        for a in geometry['left'] for b in geometry['right'])
            info.content=(f"### 当前预览层间距：{nearest[0]:.1f} mm\n最近：arm6 `{nearest[1]}` / arm7 `{nearest[2]}`\n\n"
                          '此处为可编辑预览，不代表实机配置或整条路径验证结果。'
                          + (' 新算法的此数值仅为当前位置几何间距，停止空间须用范围检查。'
                             if runtime_envelopes and use_sweep else ''))

    if runtime_envelopes:
        ghost_handles = {}

        def example_segment(q):
            # One arm, up to one degree per joint toward its taught Home.
            key=stop_arm.value
            end={k:v.copy() for k,v in q.items()}
            end[key] += np.clip(home[key]-q[key],-1.,1.)
            return end

        def update_stop_ghosts(q, enabled):
            for handle in ghost_handles.values():
                handle.visible=False
            if not enabled or not show_stop.value:
                return
            key=stop_arm.value
            end=example_segment(q)
            direction=np.sign(end[key]-q[key])
            if not np.any(direction):
                return
            stop=np.asarray(raw['safety']['arms'][key]['stop_excursion_deg'])*direction
            if not np.any(stop):
                return
            row=raw['safety']['arms'][key]
            error=row['base_error_mm']+row['geometry_error_mm']+raw['limits']['tracking_error_mm']
            for sample,fraction in enumerate((-1.,-.5,.5,1.)):
                try:
                    capsules=models[key].capsules(q[key]+fraction*stop)
                except ValueError:
                    continue
                for cap in capsules:
                    identity=(key,sample,cap.name)
                    radius=cap.radius+error
                    handle=ghost_handles.get(identity)
                    mesh_key=('stop',*identity)
                    if handle is None or mesh_radii.get(mesh_key)!=radius:
                        mesh=trimesh.creation.capsule(height=np.linalg.norm(cap.end-cap.start)/1000,
                                                     radius=radius/1000,count=[12,12])
                        if handle is None:
                            handle=server.scene.add_mesh_simple(f'/stop_samples/{key}/{sample}/{cap.name}',
                                mesh.vertices,mesh.faces,color=(220,90,190),opacity=.12,visible=False)
                            ghost_handles[identity]=handle
                        else:
                            handle.vertices=mesh.vertices;handle.faces=mesh.faces
                        mesh_radii[mesh_key]=radius
                    handle.position=(cap.start+cap.end)/2000
                    delta=cap.end-cap.start
                    rotation=trimesh.geometry.align_vectors([0,0,1],delta) if np.linalg.norm(delta)>1e-9 else np.eye(4)
                    handle.wxyz=tf.SO3.from_matrix(rotation[:3,:3]).wxyz
                    handle.visible=True

        @proof_button.on_click
        def _prove(_event):
            from cloth_agent.dual_arm.stop_sweep import validate_native_sweep
            from copy import deepcopy
            with lock:
                # This is a local, explicitly unvalidated preview. Preserve the
                # real/draft flags in raw and never call execution validation.
                preview_raw=deepcopy(raw)
                preview_raw['safety']['status']='measured'
                preview=SimpleNamespace(raw=preview_raw,arms=runtime_config.arms,
                    limits=preview_raw['limits'],obstacles=preview_raw['obstacles'])
                q=route_joints(start,home,strategy.value,slider.value/100)
                proof_status.content='正在检查示例段及独立停止范围……'
                try:
                    result=validate_native_sweep(preview,models,q,example_segment(q))
                    proof_status.content=f'当前示例段可覆盖：检查 {result["nodes"]} 个区间，接受 {result["accepted_cells"]} 个子区间。仅预览，不是整条 Home 的执行授权。'
                except ValueError as exc:
                    proof_status.content=f'当前示例段未通过：{exc}'
    for control in (strategy,slider,show_caps,show_accessories,envelope_layer):
        control.on_update(lambda _: update())
    if runtime_envelopes:
        for control in (stage, selected, profile, stop_model, stop_arm, show_stop):
            control.on_update(lambda _: update())
    @current_button.on_click
    def _(_event):
        playing.value=False;slider.value=0;update()
    @home_button.on_click
    def _(_event):
        playing.value=False;slider.value=100;update()
    @server.on_client_connect
    def _(client):
        client.camera.position=(1.4,-1.1,1.05)
        client.camera.look_at=(float(np.asarray(raw['arms']['right']['world_from_base_mm'])[0,3])/2000,0,.35)
        client.camera.up_direction=(0,0,1)
    update()
    print(f'READ_ONLY_VISER_READY http://{host}:{server.get_port()}',flush=True)
    try:
        while True:
            time.sleep(.12)
            if playing.value:
                slider.value=(slider.value+1)%101
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8087)
    p.add_argument('--runtime-envelopes',action='store_true',
                   help='Show configured collision envelopes including runtime safety padding')
    p.add_argument('--config',type=Path,help='Optional saved configuration/draft for offline preview')
    args=p.parse_args()
    run(args.directory.resolve(),args.host,args.port,args.runtime_envelopes,args.config)
