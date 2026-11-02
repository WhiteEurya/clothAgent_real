"""Operator-owned protected primitive service. No agent code is evaluated here."""
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


from .xarm_driver import NativePort, real_tools, validate_request, motion_source


class Supervisor:
    """Single command owner, existing InputGuard, independent lease/deadman watchdog."""
    def __init__(self,port,guard,root,heartbeat_seconds=1.):
        self.port,self.guard,self.root=port,guard,Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.state_lock=threading.RLock()
        self.owner=None;self.ping_at=0.;self.heartbeat_seconds=heartbeat_seconds
        self.active=False;self.generation=None;self.stopped=False;self.stop_reason=None
        self.observations={};self.actions=0;self.attempts=0;self.execution_seconds=0.;self.closed=threading.Event()
        self.worker=threading.Thread(target=self._watch,daemon=True);self.worker.start()

    def audit(self,event,**fields):
        with self.state_lock:
            with (self.root/'service.jsonl').open('a') as f:f.write(json.dumps({'unix_s':time.time(),'event':event,**fields},default=str)+'\n')
        if event in ('request','result','refused','STOP_LATCHED'):
            print('[operator service] '+json.dumps({'event':event,**fields},ensure_ascii=False,default=str),flush=True)

    def heartbeat(self,owner):
        with self.state_lock:
            if self.stopped:raise PermissionError('service stopped; operator restart required')
            if self.owner not in (None,owner):raise PermissionError('another session owns service')
            self.owner=owner;self.ping_at=time.monotonic()
        return self.status()

    def permitted(self):
        return (not self.stopped and self.owner is not None and
                time.monotonic()-self.ping_at<=self.heartbeat_seconds and
                self.guard.permits(generation=self.generation if self.active else None))

    def trip(self,reason):
        with self.state_lock:
            if self.stopped:return
            self.stopped=True;self.stop_reason=reason
        try:result=self.port.stop()
        except Exception as exc:result={'stop_unconfirmed':type(exc).__name__}
        self.audit('STOP_LATCHED',reason=reason,stop_results=result)

    def _watch(self):
        while not self.closed.wait(.02):
            with self.state_lock:
                expired=self.owner is not None and time.monotonic()-self.ping_at>self.heartbeat_seconds
            if expired:
                self.trip('host heartbeat expired');return
            if self.active and not self.permitted():
                self.trip('operator deadman expired/released');return

    def status(self):
        return {'mode':'real_supervised','armed':self.permitted(),'hardware_connected':True,
                'safety':'UNKNOWN','stopped':self.stopped,'stop_reason':self.stop_reason,
                'physical_actions':self.actions,'physical_action_attempts':self.attempts,
                'robot_execution_seconds':self.execution_seconds,
                'execution_rule':'Every action requires a new complete SAFE proof; status is not an action certificate.',
                'task_success':None,'config_sha256':getattr(self.port,'config_hash',None)}

    @contextmanager
    def exclusive(self):
        if not self.lock.acquire(blocking=False):raise PermissionError('service busy; actions are never queued')
        try:yield
        finally:self.lock.release()

    def call(self,owner,name,args):
        validate_request(name,args)
        if name=='stop':self.trip('stop requested');return self.status()
        with self.exclusive():
            if self.stopped or owner!=self.owner or time.monotonic()-self.ping_at>self.heartbeat_seconds:
                raise PermissionError('operator service unavailable or lease expired')
            self.audit('request',tool=name,arguments=args)
            try:
                if name=='robot_state':return self.port.state()
                if name=='coordinates':return {'frame_id':'world_mm','world_base':'arm6/link_base','position_unit':'mm','orientation':'Euler xyz degrees','joint_unit':'degrees','table_z_mm':0,'additional_height_offset_mm':0,'arms':ARM_MAP}
                if name=='observe':
                    folder=self.root/'observations'/uuid.uuid4().hex
                    began=time.monotonic();obs=self.port.capture(args['camera'],folder)
                    self.observations[obs.meta['observation_id']]=(obs,args['camera'])
                    # Bound memory to newest observation per camera; old IDs are invalidated.
                    self.observations={k:v for k,v in self.observations.items() if v[1]!=args['camera'] or k==obs.meta['observation_id']}
                    data=(folder/'rgb.png').read_bytes()
                    return {'observation_id':obs.meta['observation_id'],'camera':args['camera'],
                            'captured_unix_s':obs.meta['captured_unix_s'],'calibration_id':obs.meta['calibration_id'],
                            'image_path':str(folder/'rgb.png'),'sha256':hashlib.sha256(data).hexdigest(),
                            'provenance':'live_real','action_conditioned':True,'observation_seconds':time.monotonic()-began}
                if name in ClothBackend.mutating:
                    if not self.permitted():return {'status':'NOT_EXECUTED','safety':'UNKNOWN','reason':'onsite LB deadman not held','physical_execution':False}
                    self.generation=self.guard.token();self.active=True
                if args['observation_id'] not in self.observations:raise PermissionError('unknown/invalidated observation; observe again')
                obs,camera=self.observations[args['observation_id']]
                self.port.validate_observation(obs,camera)
                if name=='pixel_to_world':
                    return {'world_xyz_mm':obs.sample(args['pixel'],10).tolist(),'frame_id':'world_mm',
                            'calibration_id':obs.meta['calibration_id'],'observation_id':args['observation_id'],'historical':False}
                if not self.permitted():return {'status':'NOT_EXECUTED','safety':'UNKNOWN','reason':'onsite LB deadman not held','physical_execution':False}
                if not self.permitted():raise PermissionError('operator generation changed')
                folder=self.root/'actions'/uuid.uuid4().hex;folder.mkdir(parents=True)
                source=motion_source(name,args);(folder/'motion.py').write_text(source)
                began=time.monotonic();plan=self.port.plan(source,obs);planning=time.monotonic()-began
                # Keep the existing tri-state semantics, including UNKNOWN rejection.
                from cloth_agent.dual_arm.stop_sweep import require_safe
                require_safe(plan)
                if not plan['operations']:return {'status':'NOOP','safety':'SAFE','physical_execution':False}
                self.port.validate_observation(obs,camera)
                if not self.permitted():raise PermissionError('operator/host authorization lost before dispatch')
                self.attempts+=1;began=time.monotonic()
                result=self.port.execute(plan,folder)
                elapsed=time.monotonic()-began
                self.execution_seconds+=elapsed
                self.observations.clear()
                if result.get('status')=='EXECUTED':self.actions+=1
                else:self.trip('controller execution failed')
                result={**result,'safety':'SAFE' if result.get('status')=='EXECUTED' else 'UNKNOWN',
                        'plan_safety':'SAFE','planning_seconds':planning,'robot_execution_seconds':elapsed,
                        'physical_actions':self.actions,'task_success':None}
                write_json(folder/'result.json',result);self.audit('result',tool=name,result=result)
                return result
            except BaseException as exc:
                self.trip('operation failed: '+type(exc).__name__)
                safety=getattr(exc,'result',{}).get('status','UNKNOWN')
                self.audit('refused',tool=name,safety=safety,error=type(exc).__name__)
                if not isinstance(exc,Exception):raise
                return {'status':'REFUSED','safety':safety,'reason':str(exc),'physical_outcome':None,'stopped':True}
            finally:self.active=False;self.generation=None

    def close(self):
        self.trip('operator service closing');self.closed.set();self.worker.join(timeout=2);self.port.close()
