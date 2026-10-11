"""Saved, read-only trajectory diagnostics and Viser scene (millimetres -> metres)."""
from __future__ import annotations
import argparse
from functools import partial
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .dual_arm.geometry import pose_matrix
from .dual_arm.stop_sweep import position_error_mm


def arrays(q):
    return {k:np.asarray(v,dtype=float) for k,v in q.items()}


def observation_files(folder):
    before=Path(folder)/'before'
    if all((before/n).is_file() for n in ('world_xyz_mm.npy','rgb.png')):
        return dict(xyz=str((before/'world_xyz_mm.npy').resolve()),rgb=str((before/'rgb.png').resolve()))
    return None


def load_preview_cloud(files,max_points=120000):
    """Display-only subsampling; saved world points already contain calibration."""
    from PIL import Image
    xyz=np.load(files['xyz'],allow_pickle=False)
    with Image.open(files['rgb']) as image:rgb=np.asarray(image.convert('RGB'))
    if xyz.shape!=rgb.shape or xyz.ndim!=3 or xyz.shape[-1]!=3:
        raise ValueError('RGB 与 World 点云尺寸不匹配')
    valid=np.isfinite(xyz).all(axis=-1)
    points=xyz[valid];colors=rgb[valid]
    stride=max(1,int(np.ceil(len(points)/max_points)))
    return (points[::stride]/1000).astype(np.float32),colors[::stride]


def target_configuration(record,target,models,config):
    """Read-only pose illustration; never a collision-certified route."""
    from .dual_arm.geometry import matrix_pose,pose_error
    q=arrays(record['initial']);arm=target['arm'];model=models[arm]
    world=pose_matrix(target['world_pose_mm_deg'])
    base=matrix_pose(np.linalg.inv(config.arms[arm].world_from_base)@world)
    error=None
    try:
        joints=target.get('joints_deg')
        if joints is None and record.get('source'):
            from .codex_motion import parse_code
            name,args=parse_code(record['source'],allow_grasp=True)[target['command_index']-1]
            if name=='move_joints':joints=np.degrees(args[arm])
        if joints is not None:q[arm]=np.asarray(joints)
        else:q[arm]=model.inverse(base,q[arm])
        status='IK_OK'
    except Exception as exc:
        candidate=getattr(exc,'ik_candidate',None)
        if candidate is None:return dict(status='UNAVAILABLE',error=str(exc),joints=None)
        q[arm]=np.asarray(candidate['joints_deg']);status='APPROXIMATE';error=str(exc)
    actual=config.arms[arm].world_from_base@pose_matrix(model.forward(q[arm]))
    distance,angle=pose_error(matrix_pose(actual),target['world_pose_mm_deg'])
    return dict(status=status,error=error,joints=q,actual_world=actual,
                distance_mm=distance,angle_deg=angle)


def record_preview(runtime, source, initial, operations, error, folder):
    """Persist diagnostics before any execution; never authorizes a trajectory."""
    from .codex_motion import parse_code
    targets=[]
    for index,(name,kw) in enumerate(parse_code(source,allow_grasp=True),1):
        if name not in ('grasp','move_tcp','move_transfer','move_joints'):continue
        for arm,value in kw.items():
            if name in ('move_tcp','move_transfer','grasp'):pose=np.r_[np.array(value[:3])*1000,np.degrees(value[3:])]
            else:
                world=runtime.config.arms[arm].world_from_base@pose_matrix(runtime.models[arm].forward(np.degrees(value)))
                from .dual_arm.geometry import matrix_pose
                pose=matrix_pose(world)
            entry=dict(command_index=index,arm=arm,world_pose_mm_deg=pose.tolist())
            if name=='move_joints':entry['joints_deg']=np.degrees(value).tolist()
            targets.append(entry)
    record=dict(schema_version=1,source=source,initial={k:np.asarray(v).tolist() for k,v in initial.items()},
        targets=targets,status='VERIFIED' if error is None else 'REJECTED',error=error,
        failure_context=dict(getattr(runtime,'planning_context',{})),
        operations=[dict(kind=kind,data=item.to_dict() if kind=='move' else item)
                    for kind,data in (operations or [])
                    for item in (data.segments if kind=='move' and hasattr(data,'segments') else (data,))],
        candidates=getattr(runtime,'preview_candidates',[]),failures=getattr(runtime,'preview_failures',[]),
        configuration=runtime.config.raw,observation=observation_files(folder))
    path=Path(folder)/'trajectory_preview.json'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
    preview=getattr(runtime,'preview',None)
    if preview is not None:preview.show(record,str(path))
    return path


def load_saved_round(folder,config):
    """Older runs lack traces: show only saved endpoints, target and exact witness.

    Never reconstruct an unrecorded route and call it the actual planned path.
    """
    folder=Path(folder)
    path=folder/'trajectory_preview.json'
    if path.exists():
        record=json.loads(path.read_text())
        record['observation']=observation_files(folder)
        return record
    from types import SimpleNamespace
    from .dual_arm.kinematics import ArmModel
    observation=json.loads((folder/'before/observation.json').read_text())
    outcome=json.loads((folder/'outcome.json').read_text())
    context=outcome.get('failure_context',{})
    failures=[]
    for message in (outcome.get('error',''),context.get('direct_failure','')):
        begin=message.find('{')
        if begin>=0:
            try:
                report=json.loads(message[begin:])
                if 'segment' in report:failures.append(dict(context=context,report=report))
            except ValueError:pass
    runtime=SimpleNamespace(config=config,models={k:ArmModel(a) for k,a in config.arms.items()},
        planning_context=context,preview_candidates=[],preview_failures=failures)
    operations=[]
    plan=folder/'native_plan.json'
    if plan.exists():
        from .dual_arm.sequential_home import PositionSegment
        for row in json.loads(plan.read_text()):
            d=row['data']
            operations.append((row['kind'],PositionSegment(d['arm'],arrays(d['start']),arrays(d['end']),
                d['speed_deg_s'],d['acceleration_deg_s2'],d['timeout_s']) if row['kind']=='move' else d))
    # Build via the same serializer in a temporary directory, leaving old runs intact.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p=record_preview(runtime,(folder/'motion.py').read_text(),arrays(observation['joints_deg']),operations,
            outcome.get('error'),tmp)
        record=json.loads(p.read_text())
    record['legacy_note']='旧 run 未保存完整候选路径；仅展示已记录路径段。配置来自当前本机配置，未核实是否与当时完全相同。'
    record['observation']=observation_files(folder)
    return record


class TrajectoryPreview:
    def __init__(self,config,*,host='127.0.0.1',port=8088,wait=False):
        import viser
        self.config=config;self.wait=wait;self.lock=threading.RLock()
        self.ready=threading.Event();self.record=None;self.handles=[];self.models={};self.visuals={}
        self.visual_paths={};self.robot_roots={}
        self.goal_visuals={};self.goal_roots={};self.goal_cache={}
        self.closed=threading.Event()
        self.cloud=None;self.cloud_note=''
        self.server=viser.ViserServer(host=host,port=port,label='Astra · 轨迹与碰撞预览')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/grid',plane='xy',width=2,height=2,cell_size=.1,section_size=.5)
        self.info=self.server.gui.add_markdown('等待规划结果。预览不会连接机械臂。')
        self.view_mode=self.server.gui.add_dropdown('模型预览',options=['当前位置 → 所选目标（示意）','规划路径 / 碰撞见证'],initial_value='当前位置 → 所选目标（示意）')
        self.target_choice=self.server.gui.add_dropdown('选择目标',options=['无'],initial_value='无')
        self.show_goal=self.server.gui.add_checkbox('显示目标 URDF（半透明）',initial_value=True)
        self.show_cloud=self.server.gui.add_checkbox('显示本轮彩色点云',initial_value=True)
        self.point_size=self.server.gui.add_slider('点云粒径（mm）',min=1,max=8,step=.5,initial_value=2.5)
        self.all_targets=self.server.gui.add_checkbox('显示其他目标点',initial_value=True)
        self.play=self.server.gui.add_checkbox('播放姿态转换（仅可视化）',initial_value=False)
        self.path_kind=self.server.gui.add_dropdown('路径',options=['通过验证的路径','候选路径（未通过整轮验证）'],initial_value='通过验证的路径')
        self.progress=self.server.gui.add_slider('路径进度（非执行时间）',min=0,max=100,step=.2,initial_value=0)
        self.collision=self.server.gui.add_dropdown('碰撞/未决记录',options=['无'],initial_value='无')
        self.witness=self.server.gui.add_checkbox('显示所选碰撞见证姿态',initial_value=False)
        self.capsules=self.server.gui.add_checkbox('显示 capsule（含位置误差余量）',initial_value=True)
        self.execute=self.server.gui.add_button('执行本次已验证规划',disabled=True,visible=wait)
        self.execute.on_click(self._approve)
        for control in (self.show_cloud,self.point_size):
            control.on_update(lambda _:self._update_cloud())
        @self.play.on_update
        def restart(_):
            with self.lock:
                if self.play.value and self.progress.value>=100:self.progress.value=0
        for control in (self.view_mode,self.target_choice,self.show_goal,self.all_targets,self.path_kind,self.progress,self.collision,self.witness,self.capsules):
            control.on_update(lambda _:self.render())
        @self.server.on_client_connect
        def connect(client):
            client.camera.position=(1.7,-1.3,1.1);client.camera.look_at=(.5,0,.3)
        print(f'规划预览：http://{host}:{self.server.get_port()}',flush=True)
        self.player=threading.Thread(target=self._animate,daemon=True)
        self.player.start()

    def _update_cloud(self):
        with self.lock:
            if self.cloud is not None:
                self.cloud.visible=self.show_cloud.value
                self.cloud.point_size=self.point_size.value/1000

    def _select_target(self,index):
        with self.lock:
            self.play.value=False
            self.target_choice.value=self.target_choice.options[index+1]
            self.progress.value=0
            self.render()

    def _animate(self):
        while not self.closed.wait(.15):
            with self.lock:
                if not self.play.value or self.record is None:continue
                if self.progress.value>=100:
                    self.play.value=False
                    continue
                self.progress.value=min(100,self.progress.value+3)
                self.render()

    def _approve(self,_):
        with self.lock:
            if _.target is self.execute and self.record is not None and self.record['status']=='VERIFIED' and not self.execute.disabled:
                self.execute.disabled=True;self.ready.set()

    def await_execution(self,cancel):
        if not self.wait:return
        print('规划已显示；在 viser 点击“执行本次已验证规划”继续。',flush=True)
        while not self.ready.wait(.1):
            if cancel.is_set():raise KeyboardInterrupt('preview cancelled')
        if cancel.is_set():raise KeyboardInterrupt('preview cancelled')

    def show(self,record,path=''):
        import yourdfpy
        from viser.extras import ViserUrdf
        from .dual_arm.kinematics import ArmModel
        from .dual_arm.config import DualConfig
        with self.lock:
            self.ready.clear();self.execute.disabled=True
            self.execute.remove()
            self.execute=self.server.gui.add_button('执行本次已验证规划',disabled=True,visible=self.wait)
            self.execute.on_click(self._approve)
            self.play.value=False
            self.record=record;self.path=path;self.goal_cache={}
            if self.cloud is not None:self.cloud.remove();self.cloud=None
            self.cloud_note='本记录没有配套观测点云。'
            if record.get('observation'):
                try:
                    points,colors=load_preview_cloud(record['observation'])
                    self.cloud=self.server.scene.add_point_cloud('/observation/cloud',points,colors,
                        point_size=self.point_size.value/1000,point_shape='circle',precision='float32',
                        point_shading='flat',visible=self.show_cloud.value)
                    self.cloud_note=f'本轮 RGB-D 点云：显示 {len(points):,} 点（World 坐标，仅显示抽样）。'
                except Exception as exc:self.cloud_note=f'点云加载失败：{exc}'
            self.path_lines={}
            # Separate models: GUI callbacks must never mutate planner FK caches.
            self.config=DualConfig.parse(record['configuration'],self.config.root,homing_only=True)
            self.models={k:ArmModel(a) for k,a in self.config.arms.items()}
            for k,arm in self.config.arms.items():
                root=f'/robots/{k}'
                if k not in self.robot_roots:
                    self.robot_roots[k]=self.server.scene.add_frame(root,show_axes=False)
                self.robot_roots[k].position=arm.world_from_base[:3,3]/1000
                self.robot_roots[k].wxyz=self.quaternion(arm.world_from_base[:3,:3])
                if self.visual_paths.get(k)==arm.urdf:continue
                if k in self.visuals:self.visuals[k].remove()
                urdf=yourdfpy.URDF.load(arm.urdf,load_meshes=True,load_collision_meshes=False,
                    filename_handler=partial(yourdfpy.filename_handler_magic,dir=arm.urdf.parent))
                self.visuals[k]=ViserUrdf(self.server,urdf,root_node_name=root)
                goal_root=f'/goal_robots/{k}'
                if k not in self.goal_roots:
                    self.goal_roots[k]=self.server.scene.add_frame(goal_root,show_axes=False)
                if k in self.goal_visuals:self.goal_visuals[k].remove()
                goal_urdf=yourdfpy.URDF.load(arm.urdf,load_meshes=True,load_collision_meshes=False,
                    filename_handler=partial(yourdfpy.filename_handler_magic,dir=arm.urdf.parent))
                self.goal_visuals[k]=ViserUrdf(self.server,goal_urdf,root_node_name=goal_root,
                    mesh_color_override=(1.,.55,.15,.28))
                self.visual_paths[k]=arm.urdf
            for k,arm in self.config.arms.items():
                self.goal_roots[k].position=arm.world_from_base[:3,3]/1000
                self.goal_roots[k].wxyz=self.quaternion(arm.world_from_base[:3,:3])
            self.target_choice.options=['无']+[f'{i+1}: 指令 {t["command_index"]} · {t["arm"]}' for i,t in enumerate(record['targets'])]
            failed=record.get('failure_context',{})
            selected=next((i+1 for i,t in enumerate(record['targets']) if
                t['command_index']==failed.get('command_index') and t['arm']==failed.get('arm')),1 if record['targets'] else 0)
            self.target_choice.value=self.target_choice.options[selected]
            failures=record.get('failures',[])
            self.collision.options=['无']+[f'{i+1}: 指令 {r["context"].get("command_index","?")} · {r["report"].get("status","UNKNOWN")}' for i,r in enumerate(failures)]
            self.collision.value=self.collision.options[-1]
            self.witness.value=bool(failures and failures[-1]['report'].get('witness'))
            self.path_kind.value='通过验证的路径' if record['status']=='VERIFIED' else '候选路径（未通过整轮验证）'
            self.progress.value=0
            self.render()
            self.server.flush()
            self.execute.disabled=record['status']!='VERIFIED'

    @staticmethod
    def quaternion(matrix):return Rotation.from_matrix(matrix).as_quat()[[3,0,1,2]]

    def edges(self):
        r=self.record
        if self.path_kind.value=='通过验证的路径':
            return [row['data'] for row in r['operations'] if row['kind']=='move']
        edges=list(r.get('candidates',[]))
        if not edges:
            for failure in r.get('failures',[]):
                segment=failure['report'].get('segment',{})
                if 'start_deg' in segment:edges.append(dict(start=segment['start_deg'],end=segment['end_deg']))
        return edges

    def render(self):
        import trimesh
        with self.lock:
            if self.record is None:return
            with self.server.atomic():
                for handle in reversed(self.handles):handle.remove()
                self.handles=[]
                def keep(handle):self.handles.append(handle);return handle
                r=self.record;edges=self.edges();q=arrays(r['initial']);selected=None
                target=None;goal=None
                if self.target_choice.value!='无':
                    target_index=int(self.target_choice.value.split(':')[0])-1
                    target=r['targets'][target_index]
                    if target_index not in self.goal_cache:
                        self.goal_cache[target_index]=target_configuration(r,target,self.models,self.config)
                    goal=self.goal_cache[target_index]
                if self.collision.value!='无':selected=r['failures'][int(self.collision.value.split(':')[0])-1]
                if edges:
                    pos=min(self.progress.value/100*len(edges),len(edges)-1e-9);index=int(pos);u=pos-index
                    edge=edges[index];q={k:(1-u)*np.array(edge['start'][k])+u*np.array(edge['end'][k]) for k in q}
                witness=selected['report'].get('witness') if selected else None
                transition=self.view_mode.value=='当前位置 → 所选目标（示意）'
                if transition:
                    q=arrays(r['initial'])
                    if goal and goal['joints'] is not None:
                        u=self.progress.value/100
                        q={k:(1-u)*q[k]+u*goal['joints'][k] for k in q}
                elif witness and self.witness.value:q=arrays(witness['joints_deg'])
                objects=witness.get('objects',[]) if witness and self.witness.value and not transition else []
                colors={'left':(60,150,245),'right':(245,160,50)}
                for k,model in self.models.items():
                    angles=dict(zip(model.names,np.radians(q[k])))
                    self.visuals[k].update_cfg(angles)
                    self.goal_roots[k].visible=bool(self.show_goal.value and target and target['arm']==k and goal and goal['joints'] is not None)
                    if self.goal_roots[k].visible:
                        self.goal_visuals[k].update_cfg(dict(zip(model.names,np.radians(goal['joints'][k]))))
                    caps=model.capsules(q[k])
                    for c in caps:
                        radius=c.radius+position_error_mm(self.config,k)
                        mesh=trimesh.creation.capsule(height=np.linalg.norm(c.end-c.start)/1000,radius=radius/1000,count=[8,12])
                        d=c.end-c.start
                        rot=trimesh.geometry.align_vectors([0,0,1],d)[:3,:3] if np.linalg.norm(d)>1e-9 else np.eye(3)
                        hit=f'{k}/{c.name}' in objects
                        keep(self.server.scene.add_mesh_simple(f'/caps/{k}/{c.name}',mesh.vertices,mesh.faces,
                            color=(255,30,30) if hit else colors[k],opacity=.55 if hit else .15,
                            position=(c.start+c.end)/2000,wxyz=self.quaternion(rot),visible=self.capsules.value))
                        if hit:keep(self.server.scene.add_label(f'/hits/{k}/{c.name}',f'{k}/{c.name} 碰撞见证',position=(c.start+c.end)/2000))
                    world=model.config.world_from_base@pose_matrix(model.forward(q[k]))
                    keep(self.server.scene.add_frame(f'/tcp/{k}',position=world[:3,3]/1000,wxyz=self.quaternion(world[:3,:3]),axes_length=.05,axes_radius=.002))
                    initial=model.config.world_from_base@pose_matrix(model.forward(np.array(r['initial'][k])))
                    keep(self.server.scene.add_icosphere(f'/start/{k}',radius=.009,color=(50,220,100),position=initial[:3,3]/1000))
                    keep(self.server.scene.add_label(f'/start/{k}/label',f'{k} 起点',position=(0,0,.02)))
                    cache_key=(self.path_kind.value,k)
                    if cache_key not in self.path_lines:
                        lines=[]
                        for edge in edges:
                            a=np.array(edge['start'][k]);b=np.array(edge['end'][k])
                            points=[(model.config.world_from_base@pose_matrix(model.forward(a+(b-a)*u)))[:3,3]/1000 for u in np.linspace(0,1,17)]
                            lines.extend(zip(points[:-1],points[1:]))
                        self.path_lines[cache_key]=lines
                    lines=self.path_lines[cache_key]
                    if lines:keep(self.server.scene.add_line_segments(f'/paths/{k}',np.asarray(lines),colors[k],thickness=2,thickness_units='screen'))
                    segment=selected['report'].get('segment',{}) if selected else {}
                    if 'start_deg' in segment:
                        a=np.array(segment['start_deg'][k]);b=np.array(segment['end_deg'][k])
                        if not np.allclose(a,b):
                            points=[(model.config.world_from_base@pose_matrix(model.forward(a+(b-a)*u)))[:3,3]/1000 for u in np.linspace(0,1,17)]
                            color=(255,40,40) if selected['report'].get('status')=='UNSAFE' else (255,210,40)
                            keep(self.server.scene.add_line_segments(f'/failed_segment/{k}',np.array(list(zip(points[:-1],points[1:]))),color,thickness=6,thickness_units='screen'))
                            keep(self.server.scene.add_icosphere(f'/failed_segment/{k}/marker',radius=.003,color=color,position=points[0]))
                for i,t in enumerate(r['targets']):
                    active=target is t
                    if not active and not self.all_targets.value:continue
                    frame=pose_matrix(t['world_pose_mm_deg'])
                    root=f'/targets/{i}'
                    color=(245,60,185) if active else colors[t['arm']]
                    keep(self.server.scene.add_frame(root,position=frame[:3,3]/1000,wxyz=self.quaternion(frame[:3,:3]),
                        show_axes=active,axes_length=.04,axes_radius=.0012))
                    marker=keep(self.server.scene.add_icosphere(root+'/point',radius=.003 if active else .002,color=color))
                    marker.on_click(lambda _,index=i:self._select_target(index))
                    if active:
                        keep(self.server.scene.add_icosphere(root+'/halo',radius=.005,color=color,opacity=.12))
                        theta=np.linspace(0,2*np.pi,65)
                        ring=np.column_stack([.009*np.cos(theta),.009*np.sin(theta),np.zeros_like(theta)])
                        keep(self.server.scene.add_line_segments(root+'/ring',np.stack([ring[:-1],ring[1:]],axis=1),color,thickness=2.5))
                        # Place the callout above the target in world Z, even
                        # when the tool's local Z points down toward the table.
                        offset=frame[:3,:3].T@np.array([0,0,.075])
                        keep(self.server.scene.add_line_segments(root+'/leader',np.array([[np.zeros(3),offset]]),color,thickness=1.5))
                        arm_name='arm6' if t['arm']=='left' else 'arm7'
                        xyz=t['world_pose_mm_deg'][:3]
                        label=f'◎ {arm_name} · 目标 #{t["command_index"]} | X {xyz[0]:.1f}  Y {xyz[1]:.1f}  Z {xyz[2]:.1f} mm'
                        keep(self.server.scene.add_label(root+'/label',label,position=offset))
                details=''
                if goal:
                    if goal['joints'] is None:details+='\n\n目标 URDF 无可用解：'+goal['error']
                    else:
                        label='近似姿态，IK 未通过' if goal['status']=='APPROXIMATE' else '目标姿态，IK 通过（未验证这条示意路径）'
                        details+=f'\n\n**{label}** · TCP 误差 {goal["distance_mm"]:.3f} mm / {goal["angle_deg"]:.3f}°。'
                        actual=goal['actual_world'][:3,3]/1000
                        wanted=np.array(target['world_pose_mm_deg'][:3])/1000
                        keep(self.server.scene.add_icosphere('/goal_actual',radius=.003,color=(255,130,20),position=actual))
                        keep(self.server.scene.add_label('/goal_actual/label',label,position=(0,0,.03)))
                        keep(self.server.scene.add_line_segments('/goal_error',np.array([[actual,wanted]]),(255,40,40),thickness=2))
                if selected:
                    report=selected['report'];ctx=selected['context']
                    details+=f'\n\n所选记录：指令 {ctx.get("command_index")}，路线 {ctx.get("route","未知")}；{report.get("status")}。'
                    if 'cartesian_fraction' in ctx:details+=f' 所在 TCP 插值进度：{100*ctx["cartesian_fraction"]:.2f}%。'
                    if witness:details+=f'\n\n对象：{witness["objects"]}；带余量间隙 **{witness["gap_mm"]:.4f} mm**。红色为误差/停止范围内的碰撞见证，不代表名义路径已经撞上。'
                    else:details+=' 未得到碰撞见证，不能把 UNKNOWN 当成已碰撞。'
                error=(r.get('error') or '完整规划已通过验证').split('{')[0]
                self.info.content=(f'**{r["status"]}** · `{self.path}`\n\n{error}'+details+
                    '\n\n'+self.cloud_note+' 洋红色光环为所选目标，蓝/橙小球为其他目标；点击小球可切换。'+
                    '\n\n蓝色 arm6，橙色 arm7，绿色为起点。路径线为关节插值的 TCP 几何采样，不是控制器时间曲线。'+
                    '\n\n半透明 URDF 为所选目标（或明确标注的近似解）。当前位置→目标播放仅作姿态示意，不是规划器验证过的路径，不会驱动机械臂。'+
                    '\n\nCapsule 含位置误差；停止/关节误差通过所选见证姿态显示，不另画成统一膨胀半径。'+
                    ('\n\n'+r['legacy_note'] if r.get('legacy_note') else ''))

    def close(self):
        self.closed.set()
        self.player.join(timeout=1)
        self.server.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('config/dual_arm.local.json'))
    parser.add_argument('--port',type=int,default=8088)
    args=parser.parse_args()
    from .dual_arm.config import DualConfig
    from .codex_astra_pipeline import ROOT
    config=DualConfig.load(args.config,root=ROOT,homing_only=True)
    preview=TrajectoryPreview(config,port=args.port)
    try:
        preview.show(load_saved_round(args.round,config),str(args.round))
        while True:time.sleep(.5)
    except KeyboardInterrupt:pass
    finally:preview.close()


if __name__=='__main__':main()
