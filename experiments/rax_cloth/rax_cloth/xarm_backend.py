"""Registered RAX Backend Contract; uses upstream arming, no second RPC service."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
from .backend import ClothBackend
from .xarm_driver import NativePort,real_tools,validate_request,motion_source
from .session import write_json

NAMES={d['name']:('xarm_hold' if d['name']=='stop' else 'xarm_'+d['name']) for d in real_tools()}
REVERSE={v:k for k,v in NAMES.items()}
DEFAULTS={'mode':'mock','robot_config':None,'replay_manifest':None,'replay_manifest_sha256':None,'robot_scope':'arm6','manual_reset':False}


class XArmBackend:
    mutating=frozenset(NAMES[k] for k in (*ClothBackend.mutating,'reset'))
    def __init__(self,config,*,port_factory=NativePort):
        unknown=set(config.backend_options)-set(DEFAULTS)
        if unknown:raise ValueError('unknown backend options: '+str(sorted(unknown)))
        self.options={**DEFAULTS,**config.backend_options};self.mode=self.options['mode']
        if self.options['robot_scope']!='arm6':raise ValueError('only arm6 is active')
        if self.mode not in ('mock','replay','real_supervised'):raise ValueError('unsupported environment')
        if self.mode=='real_supervised' and not self.options['robot_config']:raise ValueError('robot_config required')
        from services import arming
        self.arming=arming;self.root=Path(config.root);self.root.mkdir(parents=True,exist_ok=True)
        self.port=None;self.port_factory=port_factory;self.offline=None
        self.frames=self.root/'agent-frames';self.frames.mkdir(exist_ok=True)
        self.observations={};self.last_offline_id=None;self.actions=0;self.attempts=0;self.execution_seconds=0.
        self.unresolved_actions=0
        self.reset_actions=0;self.reset_attempts=0;self.reset_seconds=0.
        if self.mode=='real_supervised':
            # Recover from durable per-action records, never infer physical success
            # from a request or replay an interrupted dispatch automatically.
            for marker in (self.root/'actions').glob('*/dispatch.json'):
                is_reset=json.loads(marker.read_text()).get('arguments',{}).get('purpose')=='environment_reset'
                self.attempts+=1
                self.reset_attempts+=int(is_reset)
                result_path=marker.parent/'result.json'
                if not result_path.exists():
                    self.unresolved_actions+=1
                    continue
                result=json.loads(result_path.read_text())
                if result.get('status')=='NOT_DISPATCHED':
                    self.attempts-=1
                    self.reset_attempts-=int(is_reset)
                    continue
                self.unresolved_actions+=int(result.get('status')=='EXECUTION_OUTCOME_UNKNOWN')
                self.actions+=int(result.get('status')=='EXECUTED')
                self.execution_seconds+=float(result.get('robot_execution_seconds',0))
                if is_reset:
                    self.reset_actions+=int(result.get('status')=='EXECUTED')
                    self.reset_seconds+=float(result.get('robot_execution_seconds',0))
        self.active=False;self.arm_generation=None;self.latched=False;self.closed=False;self.stop_reason=None
        self.lock=threading.RLock();self.done=threading.Event()
        if self.mode=='real_supervised':self.arming.write(self.root,False,reason='new backend requires fresh operator authorization')
        if self.mode!='real_supervised':
            self.offline=ClothBackend(config)
            self.offline.sequence=max((int(p.name) for p in (self.root/'observations').glob('*') if p.name.isdigit()),default=-1)+1
        self.worker=threading.Thread(target=self._watch,daemon=True);self.worker.start()

    def tools(self):
        result=real_tools()
        if self is not None and self.options.get('manual_reset'):
            result=[d for d in result if d['name']!='reset']
        for d in result:
            d['name']=NAMES[d['name']]
            d['description']=d['description'].replace('onsite deadman','upstream operator arming')
        return result

    def _arm_record(self):
        record=self.arming.read(self.root)
        # Explicit finite operator grant; no indefinite authorization in this port.
        return record if record.get('armed') and isinstance(record.get('until'),(int,float)) else {'armed':False}

    @staticmethod
    def _generation(record):return (record.get('since'),record.get('until'),record.get('by'))

    def _authorized(self):
        record=self._arm_record()
        return (not self.latched and record.get('armed',False) and
                (not self.active or self._generation(record)==self.arm_generation))

    def _watch(self):
        while not self.done.wait(.02):
            if self.active and not self._authorized():self.hold('operator disarmed or authorization expired');return

    def hold(self,reason,*,stop_motion=True):
        with self.lock:
            if self.latched:return
            self.latched=True;self.stop_reason=reason
        try:
            if self.port and stop_motion:self.port.stop()
        finally:
            if self.mode=='real_supervised':self.arming.write(self.root,False,reason=reason)

    def _port(self):
        if self.port is None:self.port=self.port_factory(self.options['robot_config'],self.root/'robot',recover_stop=True)
        return self.port

    def validate(self,name,args):
        if name not in REVERSE:raise ValueError('unknown xArm tool')
        if name==NAMES['reset'] and self.options.get('manual_reset'):
            raise PermissionError('this session uses manual environment reset')
        validate_request(REVERSE[name],args)
        if name==NAMES['stop']:return
        if self.closed or self.latched:raise PermissionError('backend stopped: '+str(self.stop_reason))
        if self.mode=='real_supervised' and name in self.mutating and not self._authorized():
            raise PermissionError('not armed by onsite operator (services.arming)')

    def status(self):
        return {'mode':self.mode,'armed':self._authorized() if self.mode=='real_supervised' else False,
                'hardware_connected':self.port is not None and not self.closed,'safety':'UNKNOWN','stopped':self.latched,
                'stop_reason':self.stop_reason,'physical_actions':self.actions,'physical_action_attempts':self.attempts,
                'unresolved_physical_action_attempts':self.unresolved_actions,
                'robot_execution_seconds':self.execution_seconds,'task_success':None,
                'environment_reset_actions':self.reset_actions,'environment_reset_attempts':self.reset_attempts,
                'environment_reset_execution_seconds':self.reset_seconds,
                'task_physical_actions':self.actions-self.reset_actions,
                'connection_state':('closed' if self.closed else 'connected' if self.port is not None else 'not_yet_requested'),
                'safety_scope':'per_proposed_action',
                'safety_explanation':'UNKNOWN here means no proposed action has been certified. It is not a controller fault. Every motion still requires a fresh complete SAFE proof.',
                'execution_accounting':'Completed primitive calls, not task successes. Duration excludes unknown crash intervals.',
                'safety_policy':'Each motion requires a fresh complete SAFE proof from the existing planner.'}

    def _image(self,value):
        source=Path(value['image_path']);data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=value['sha256']:raise ValueError('observation image changed')
        target=self.frames/(uuid.uuid4().hex+'.png');target.write_bytes(data)
        return {**value,'image_path':'/frames/'+target.name},[{'type':'image','mimeType':'image/png','data':base64.b64encode(data).decode()}]

    def call(self,name,args):
        self.validate(name,args);tool=REVERSE[name]
        if tool=='stop':self.hold('hold requested');return self.status(),[]
        if tool=='reset':return self._reset(args),[]
        if self.offline:
            translated={k:v for k,v in args.items() if k not in ('camera','observation_id')}
            if tool=='observe' and self.mode=='replay' and args['camera']!=self.offline.manifest['observations'][0]['camera']:
                raise ValueError('no recorded image for selected camera')
            if 'observation_id' in args and args['observation_id']!=self.last_offline_id:raise ValueError('observe first; observation_id mismatch')
            value,images=self.offline.call(tool,translated)
            if tool=='robot_state':
                value={**value,'active_arms':['arm6'],'inactive_arm7':'not connected or measured'}
                if isinstance(value.get('joints_deg'),dict):
                    value['joints_deg']={k:v for k,v in value['joints_deg'].items() if k in ('left','arm6')}
            if tool=='observe':
                self.last_offline_id=uuid.uuid4().hex
                value={**value,'observation_id':self.last_offline_id}
                return self._image(value)
            return value,images
        try:
            if tool=='coordinates':
                return {'frame_id':'world_mm','world_base':'arm6/link_base','position_unit':'mm',
                        'orientation':'Euler xyz degrees','joint_unit':'degrees','table_z_mm':0,
                        'additional_height_offset_mm':0},[]
            port=self._port()
            if tool=='robot_state':return port.state(),[]
            if tool=='observe':
                folder=self.root/'observations'/uuid.uuid4().hex;began=time.monotonic()
                obs=port.capture(args['camera'],folder)
                self.observations={k:v for k,v in self.observations.items() if v[1]!=args['camera']}
                self.observations[obs.meta['observation_id']]=(obs,args['camera'])
                value={'observation_id':obs.meta['observation_id'],'camera':args['camera'],
                       'image_path':str(folder/'rgb.png'),'sha256':hashlib.sha256((folder/'rgb.png').read_bytes()).hexdigest(),
                       'captured_unix_s':obs.meta['captured_unix_s'],'calibration_id':obs.meta['calibration_id'],
                       'provenance':'live_real','action_conditioned':True,'observation_seconds':time.monotonic()-began}
                return self._image(value)
            if args['observation_id'] not in self.observations:raise ValueError('unknown/invalidated observation_id')
            obs,camera=self.observations[args['observation_id']]
            if name in self.mutating:
                self.arm_generation=self._generation(self._arm_record());self.active=True
                if not self._authorized():raise PermissionError('operator authorization changed')
            port.validate_observation(obs,camera)
            if tool=='pixel_to_world':
                return {'world_xyz_mm':obs.sample(args['pixel'],10).tolist(),'frame_id':'world_mm',
                        'calibration_id':obs.meta['calibration_id'],'historical':False},[]
            folder=self.root/'actions'/uuid.uuid4().hex;folder.mkdir(parents=True)
            source=motion_source(tool,args);(folder/'motion.py').write_text(source)
            began=time.monotonic();plan=port.plan(source,obs);planning=time.monotonic()-began
            from cloth_agent.dual_arm.stop_sweep import require_safe
            require_safe(plan)
            port.validate_observation(obs,camera)
            if not self._authorized():raise PermissionError('operator disarmed before dispatch')
            if not plan['operations']:return {'status':'NOOP','safety':'SAFE','physical_execution':False},[]
            return self._dispatch(port,plan,folder,planning,name,args),[]
        except BaseException:
            self.hold('robot operation failed');raise
        finally:self.active=False;self.arm_generation=None

    def _reset(self,args):
        if self.mode!='real_supervised':
            return {'status':'NOT_EXECUTED','purpose':'environment_reset','physical_execution':False,
                    'physical_outcome':None,'task_success':None}
        from .reset_adapter import run_existing_reset
        folder=self.root/'resets'/uuid.uuid4().hex
        folder.parent.mkdir(exist_ok=True)
        self.arm_generation=self._generation(self._arm_record());self.active=True
        def authorize():
            if not self._authorized():raise PermissionError('reset requires current operator authorization')
        def dispatch(plan,planning,directory):
            authorize()
            action=self.root/'actions'/uuid.uuid4().hex;action.mkdir(parents=True)
            result=self._dispatch(port,plan,action,planning,NAMES['reset'],
                {**args,'purpose':'environment_reset','reset_directory':str(folder),
                 'stage_directory':str(directory)})
            return result
        try:
            authorize();port=self._port()
            result=run_existing_reset(port,folder,args['seed'],authorize,dispatch)
            return {'status':result['status'],'outcome':result.get('outcome'),
                    'purpose':'environment_reset','not_a_demonstration':True,
                    'reset_id':folder.name,'task_success':None,
                    'note':'Reset is not flattening success. Request a new camA observation.'}
        except BaseException:
            self.hold('environment reset failed');raise
        finally:
            self.observations.clear();self.active=False;self.arm_generation=None

    def _dispatch(self,port,plan,folder,planning,name,args):
        from cloth_agent.dual_arm.stop_sweep import require_safe
        require_safe(plan)
        write_json(folder/'dispatch.json',{'tool':name,'arguments':args,'time_unix_s':time.time(),
                                         'safety':'SAFE','planning_seconds':planning,
                                         'meaning':'dispatch intended; physical completion not yet known'})
        # Logging takes time too: authorization must still hold at dispatch.
        if not self._authorized():
            write_json(folder/'result.json',{'status':'NOT_DISPATCHED','robot_execution_seconds':0})
            raise PermissionError('operator disarmed before dispatch')
        self.attempts+=1;began=time.monotonic()
        is_reset=args.get('purpose')=='environment_reset'
        self.reset_attempts+=int(is_reset)
        try:
            result=port.execute(plan,folder)
        except BaseException as exc:
            duration=time.monotonic()-began
            self.execution_seconds+=duration
            if is_reset:self.reset_seconds+=duration
            self.observations.clear()
            write_json(folder/'result.json',{'status':'EXECUTION_OUTCOME_UNKNOWN',
                'error':f'{type(exc).__name__}: {exc}','planning_seconds':planning,
                'robot_execution_seconds':duration,'task_success':None})
            self.unresolved_actions+=1
            raise
        duration=time.monotonic()-began
        self.execution_seconds+=duration;self.observations.clear()
        if is_reset:self.reset_seconds+=duration
        write_json(folder/'result.json',{**result,'planning_seconds':planning,'robot_execution_seconds':duration})
        if result.get('status')!='EXECUTED':raise RuntimeError('existing controller execution failed: '+str(result))
        self.actions+=1
        self.reset_actions+=int(is_reset)
        return {**result,'safety':'SAFE','planning_seconds':planning,'robot_execution_seconds':duration,'task_success':None}

    def close(self):
        if self.closed:return
        self.closed=True
        # A read-only session never started motion. Disconnecting it must not
        # put an otherwise ready controller into STOP for the next session.
        # Any dispatch attempt/active operation still requires an explicit stop.
        try:self.hold('session closing',stop_motion=bool(self.attempts or self.active))
        finally:
            self.done.set();self.worker.join(timeout=2)
            if self.port:self.port.close()
            if self.offline:self.offline.close()
