"""Robot-only adapter to existing native planning, capture and execution; no Harness policy."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid
from contextlib import contextmanager
from .backend import ARM_MAP, ClothBackend, tool, vector
from .session import write_json


def real_tools():
    definitions=ClothBackend.tools(None)
    for d in definitions:
        name=d['name'];schema=d['inputSchema']
        d['description']={'observe':'Stationary live RGB from selected calibrated camera; NEVER moves to a viewing pose.',
                          'robot_state':'Fresh dual-arm measured state; no motion.',
                          'stop':'Stop both arms and latch this operator service; restart required.',
                          'pixel_to_world':'Measured calibrated pixel query for an exact live observation.'}.get(name,
                          d['description'].replace('Offline: NOT_EXECUTED.','Requires onsite deadman and complete existing safety proof.'))
        if 'arm' in schema['properties']:schema['properties']['arm']['enum']=['arm6']
        if name=='observe':schema['properties']={'camera':{'type':'string','enum':['camA']}};schema['required']=['camera']
        if name in ClothBackend.mutating or name=='pixel_to_world':
            schema['properties']['observation_id']={'type':'string','minLength':1,'maxLength':100}
            schema['required'].append('observation_id')
    definitions.append(tool('reset','Environment reset using the existing arm6 reset routine. Not a learned flattening skill. Requires operator authorization; mock/replay never execute.',
        {'seed':{'type':'integer','minimum':0,'maximum':4294967295}},('seed',)))
    return definitions


def validate_request(name,args):
    definitions={d['name']:d for d in real_tools()}
    if name not in definitions or not isinstance(args,dict):raise ValueError('unknown primitive')
    schema=definitions[name]['inputSchema']
    if set(args)-set(schema['properties']) or set(schema['required'])-set(args):raise ValueError('invalid primitive keys')
    for k,v in args.items():
        s=schema['properties'][k]
        if 'enum' in s and v not in s['enum']:raise ValueError('invalid '+k)
        if s['type']=='string' and (not isinstance(v,str) or not 1<=len(v)<=100):raise ValueError('invalid '+k)
        if s['type']=='integer' and (type(v) is not int or not s.get('minimum',0)<=v<=s.get('maximum',4294967295)):raise ValueError('invalid '+k)
        if s['type']=='array':
            n=s.get('minItems',ARM_MAP[args['arm']]['dof'] if k=='joints_deg' else 0)
            if not isinstance(v,list) or len(v)!=n or any(type(x) not in (int,float) or not math.isfinite(x) for x in v):raise ValueError('invalid '+k)
    if name=='pixel_to_world' and any(int(x)!=x for x in args['pixel']):raise ValueError('integer pixels required')


def motion_source(name,args):
    """Only numeric literals/basic calls, never executable text supplied by agent."""
    validate_request(name,args)
    arm=ARM_MAP[args['arm']]['internal']
    if name=='move_tcp':
        p=args['pose_mm_deg'];v=[x/1000 for x in p[:3]]+[math.radians(x) for x in p[3:]]
        return f'motion.move_tcp({arm}={v!r})\n'
    if name=='move_joints':return f'motion.move_joints({arm}={[math.radians(x) for x in args["joints_deg"]]!r})\n'
    if name=='gripper':return f'motion.gripper(arm={arm!r}, action={args["target"]!r})\n'
    raise ValueError('not a motion primitive')


def configure_arm6(runtime):
    """Existing saved inactive-arm geometry; no inactive controller or safety bypass."""
    config=runtime.config
    path=config.root/'data/robot/xarm7_perception_pose.json'
    record=json.loads(path.read_text());arm=config.arms['right']
    if (record.get('robot_ip')!=arm.ip or record.get('axis')!=arm.axis
            or record.get('control_box_sn')!=arm.serial):
        raise ValueError('saved arm7 obstacle identity mismatch')
    q=record['joint_angles_deg']
    if len(q)!=arm.axis or any(type(v) not in (int,float) or not math.isfinite(v) for v in q):
        raise ValueError('invalid saved arm7 obstacle geometry')
    config.static_arm_obstacles={'right':dict(arm='right',joint_angles_deg=q,
        reference_file=str(path),recorded_at=record.get('recorded_at'),
        assumption='saved_perception_pose',live_feedback=False)}
    runtime.active_arms=('left',)


class NativePort:
    """The production planner/executor, instantiated only by the operator service."""
    def __init__(self,config_path,root,*,recover_stop=False):
        import fcntl
        from cloth_agent.codex_native_runtime import NativeRuntime
        from cloth_agent.dual_arm.cli import connect
        from cloth_agent.dual_arm.execution import DualArmCoordinator
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.cameras={};self.runtime=None
        r=NativeRuntime(config_path);self.runtime=r
        try:
            # Only the real RAX startup opts in. Read-only preflight stays passive.
            r.config.home_recover_stopped=recover_stop
            configure_arm6(r)
            if not getattr(r.config,'path_validation_enabled',True):raise PermissionError('path validation disabled')
            if getattr(r.config,'collision_envelopes_disabled_arms',()):raise PermissionError('collision envelopes disabled')
            if {k:a.axis for k,a in r.config.arms.items()}!={'left':6,'right':7}:raise ValueError('arm identity mismatch')
            self.configs={'camA':r.config}
            if self.configs['camA'].raw['camera']['mount_arm']!='left':raise ValueError('camA must belong to arm6')
            self.config_hash=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
            r._lease=open('/tmp/clothagent_codex_dual_arm.lock','a')
            fcntl.flock(r._lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            connections=connect(r.config,r.models,r.cancel,True)
            try:r.coordinator=DualArmCoordinator(r.config,r.models,connections,r.cancel,self.root/'controller',realtime=True)
            except BaseException:
                for c in connections.values():c.disconnect()
                raise
            if recover_stop:self._recover_startup_stop(connections['left'])
            self.state()
        except BaseException:
            self.close();raise

    def _recover_startup_stop(self,connection):
        """One startup recovery via existing guarded controller implementation."""
        state=connection.checked_value(connection.arm.get_state(),'startup controller state')
        if state!=4:
            connection.recover_stopped=False
            return
        path=self.root/'startup_stop_recovery.json'
        record={'arm':'arm6','state_before':state,'status':'STARTED','time_unix_s':time.time(),
                'method':'existing XArmConnection.prepare_position','motion_command_sent':False}
        write_json(path,record)
        try:
            # Existing implementation checks faults/warnings, cancellation,
            # stationary state and empty queue. Never clear errors or Home.
            connection.prepare_position()
            state=connection.checked_value(connection.arm.get_state(),'state after startup recovery')
            if state not in (0,2):raise RuntimeError('STOP recovery did not reach ready state')
            write_json(path,{**record,'status':'RECOVERED','state_after':state})
        except BaseException as exc:
            write_json(path,{**record,'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
            raise

    def state(self):
        r=self.runtime;rows=r.coordinator.snapshots()
        return {'status':'LIVE','live':True,'sampled_unix_s':time.time(),
                'active_arms':['arm6'],'inactive_arm7':'assumed static obstacle; not connected or measured',
                'joints_deg':{'left':[float(x) for x in rows['left']['joints']]},
                'base_tcp_mm_deg':{'left':[float(x) for x in rows['left']['pose']]}}

    def capture(self,camera,folder):
        from cloth_agent.dual_arm.observation import capture,create_camera
        r=self.runtime;cfg=self.configs[camera]
        if camera not in self.cameras:self.cameras[camera]=create_camera(cfg)
        return capture(folder,cfg,r.models,r.coordinator.snapshots,camera=self.cameras[camera])

    def validate_observation(self,obs,camera):
        obs.validate_for(self.configs[camera],live=True,check_age=True)
        self.runtime.arrival(obs.meta['joints_deg'])

    def plan(self,source,obs):
        # Production plan certifies all moves AND gripper stationary envelopes.
        # Do not call execute(): its optional garment/height rewriting is outside this baseline.
        operations=self.runtime.plan(source,obs.meta['joints_deg'],preserve_command_boundaries=True)
        return {'status':'SAFE','operations':operations}

    def execute(self,plan,folder):return self.runtime._execute_operations(plan['operations'],folder)
    def stop(self):
        if self.runtime:return self.runtime.stop()
        return {}
    def close(self):
        for camera in self.cameras.values():camera.stop()
        if self.runtime:self.runtime.close()
