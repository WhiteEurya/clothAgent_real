"""Official RAX Backend Contract, with explicitly nonphysical offline modes.

The configuration is host-owned. No agent argument selects mode or imports code.
Real service commissioning is intentionally unavailable in this setting release.
"""
from __future__ import annotations
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ARM_MAP = {'arm6': {'internal': 'left', 'dof': 6, 'camera': 'camA'},
           'arm7': {'internal': 'right', 'dof': 7, 'camera': 'camB'}}


def tool(name, description, properties=None, required=()):
    return dict(name=name, description=description, inputSchema=dict(
        type='object', properties=properties or {}, required=list(required), additionalProperties=False))


def vector(n):
    return dict(type='array', minItems=n, maxItems=n, items={'type': 'number'})


class ClothBackend:
    mutating = frozenset({'move_tcp', 'move_joints', 'gripper'})

    def __init__(self, config):
        self.mode = config.backend_options.get('mode', 'mock')
        if self.mode not in {'mock', 'replay'}:
            raise PermissionError('sim requires a validated action-conditioned cloth simulator; '
                                  'real_supervised requires a commissioned protected service and operator watchdog. Both disabled.')
        self.root = Path(config.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.options = dict(config.backend_options)
        self.closed = False
        self.sequence = 0
        self.observation = None
        self.manifest = None
        if self.mode == 'replay':
            manifest_path = Path(self.options['replay_manifest']).resolve()
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest.get('schema_version') != 1 or not self.manifest.get('observations'):
                raise ValueError('replay manifest requires schema_version=1 and observations')
            self.manifest_base = manifest_path.parent
        self._definitions = {t['name']: t for t in self.tools()}

    def tools(self):
        arm = {'type': 'string', 'enum': list(ARM_MAP)}
        return [
            tool('observe', 'Saved RGB observation. Replay is historical, never the consequence of a proposed action.'),
            tool('robot_state', 'Mock UNKNOWN or recorded historical joints; never live feedback.'),
            tool('coordinates', 'Frame and units; table contact baseline Z=0 mm. No added height offset.'),
            tool('pixel_to_world', 'Existing calibrated measured pixel conversion; no cloth heuristic.',
                 {'pixel': vector(2)}, ['pixel']),
            tool('move_tcp', 'Protected Cartesian command, world_mm, Euler xyz degrees. Offline: NOT_EXECUTED.',
                 {'arm': arm, 'pose_mm_deg': vector(6)}, ['arm', 'pose_mm_deg']),
            tool('move_joints', 'Protected joint command in degrees (arm6:6, arm7:7). Offline: NOT_EXECUTED.',
                 {'arm': arm, 'joints_deg': {'type': 'array', 'items': {'type': 'number'}}}, ['arm', 'joints_deg']),
            tool('gripper', 'Basic open/close; no garment strategy. Offline: NOT_EXECUTED.',
                 {'arm': arm, 'target': {'type': 'string', 'enum': ['open', 'close']}}, ['arm', 'target']),
            tool('stop', 'Stop this offline backend; no hardware connection exists.'),
        ]

    def validate(self, name, arguments):
        if self.closed:
            raise PermissionError('backend closed')
        if name not in self._definitions or not isinstance(arguments, dict):
            raise ValueError('unknown tool or arguments')
        schema = self._definitions[name]['inputSchema']
        if set(arguments) - set(schema['properties']) or set(schema['required']) - set(arguments):
            raise ValueError('unknown or missing argument keys')
        for k, value in arguments.items():
            s = schema['properties'][k]
            if 'enum' in s and value not in s['enum']:
                raise ValueError(f'invalid {k}')
            if s['type'] == 'array':
                if not isinstance(value, list) or any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
                    raise ValueError(f'finite numeric array required: {k}')
                n = s.get('minItems', ARM_MAP[arguments['arm']]['dof'] if k == 'joints_deg' else 0)
                if len(value) != n:
                    raise ValueError(f'{k} requires {n} values')
        if name == 'pixel_to_world' and any(x != int(x) for x in arguments['pixel']):
            raise ValueError('pixel coordinates must be integer-valued')

    def status(self):
        return dict(mode=self.mode, armed=False, hardware_connected=False, closed=self.closed,
                    safety='UNKNOWN', physical_actions=0, task_success=None)

    def call(self, name, arguments):
        self.validate(name, arguments)
        if name in self.mutating:
            return dict(status='NOT_EXECUTED', mode=self.mode, safety='UNKNOWN',
                        physical_outcome=None, reason='No action-conditioned physical environment',
                        requested={'tool': name, 'arguments': arguments}), []
        if name == 'stop':
            self.close()
            return {'status': 'STOPPED', 'hardware_connected': False}, []
        if name == 'coordinates':
            return dict(frame_id='world_mm', world_base='arm6/link_base', position_unit='mm',
                        orientation='Euler xyz degrees', joint_unit='degrees', table_z_mm=0,
                        arms=ARM_MAP, additional_height_offset_mm=0), []
        if name == 'robot_state':
            return dict(status='HISTORICAL' if self.observation else 'UNKNOWN',
                        joints_deg=self.observation.meta['joints_deg'] if self.observation else None,
                        captured_unix_s=self.observation.meta['captured_unix_s'] if self.observation else None,
                        live=False), []
        if name == 'pixel_to_world':
            if self.observation is None:
                raise ValueError('A calibrated replay observation is required; mock has no geometry')
            point = self.observation.sample(arguments['pixel'], self.options.get('max_depth_spread_mm', 10))
            return dict(world_xyz_mm=point.tolist(), frame_id='world_mm',
                        calibration_id=self.observation.meta['calibration_id'], historical=True), []
        began = time.monotonic()
        folder = self.root / 'observations' / f'{self.sequence:06d}'
        self.sequence += 1
        folder.mkdir(parents=True, exist_ok=False)
        if self.mode == 'replay':
            # The existing read-only loader verifies RGB/cloud hashes and dimensions.
            from cloth_agent.dual_arm.observation import Observation
            row = self.manifest['observations'][0]
            source = (self.manifest_base / row['directory']).resolve()
            if row.get('camera') not in {'camA', 'camB'}:
                raise ValueError('explicit camA/camB provenance required')
            for file in ('rgb.png', 'observation.json', 'world_xyz_mm.npy'):
                if hashlib.sha256((source / file).read_bytes()).hexdigest() != row['sha256'][file]:
                    raise ValueError('replay source changed since manifest: '+file)
            self.observation = Observation.load(source)
            if self.observation.meta.get('synthetic') is not False:
                raise ValueError('replay requires real recorded observations')
            for file in ('rgb.png', 'observation.json', 'world_xyz_mm.npy'):
                shutil.copyfile(source / file, folder / file)
                if hashlib.sha256((folder / file).read_bytes()).hexdigest() != row['sha256'][file]:
                    raise ValueError('replay source changed during copy: '+file)
            self.observation = Observation.load(folder)
            meta = dict(source_observation_id=self.observation.meta['observation_id'],
                        captured_unix_s=self.observation.meta['captured_unix_s'],
                        calibration_id=self.observation.meta['calibration_id'], provenance='historical_real',
                        camera=row['camera'], action_conditioned=False)
        else:
            from PIL import Image
            Image.new('RGB', (32, 24), (96, 96, 96)).save(folder / 'rgb.png')
            meta = dict(provenance='synthetic_interface_fixture', camera=None, action_conditioned=False)
        data = (folder / 'rgb.png').read_bytes()
        meta.update(image_path=str(folder / 'rgb.png'), sha256=hashlib.sha256(data).hexdigest(),
                    observation_seconds=time.monotonic()-began, task_success=None)
        (folder / 'metadata.json').write_text(json.dumps(meta, indent=2))
        return meta, [dict(type='image', mimeType='image/png', data=base64.b64encode(data).decode())]

    def close(self):
        self.closed = True
