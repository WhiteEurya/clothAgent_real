"""Read-only adapter for existing episodes; no synthetic next states or robot access."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from cloth_agent.dual_arm.observation import Observation as SavedObservation
from .interfaces import Observation,Action

def read(path):return json.loads(Path(path).read_text())

def load_observation(path):
    saved=SavedObservation.load(path);m=saved.meta
    if m.get('synthetic') is not False:raise ValueError('real-data evaluation excludes synthetic observations')
    return Observation(np.asarray(Image.open(saved.image).convert('RGB')),None,float(m['captured_unix_s']),
        str(m.get('camera_serial','unknown'))+':'+m['calibration_id'],world_xyz_mm=saved.xyz,provenance=m)

def discover(source):
    source=Path(source).resolve();episodes=[source] if (source/'goal').is_dir() and source.name.startswith('episode_') else sorted(p for p in source.glob('episode_*') if p.is_dir())
    pairs=[];excluded=[]
    for ep in episodes:
        for folder in sorted(ep.glob('round_*')):
            key=ep.name+'_'+folder.name
            if not (folder/'execution.json').exists():excluded.append({'id':key,'reason':'no execution log'});continue
            log=read(folder/'execution.json')
            if log.get('physical_execution') is not True or log.get('completed_commands',0)==0:
                excluded.append({'id':key,'reason':'no physical action evidence'});continue
            if not all((folder/side/'observation.json').exists() for side in ('before','after')):
                excluded.append({'id':key,'reason':'missing true before/after observation'});continue
            intent=read(folder/'dispatch_intent.json') if (folder/'dispatch_intent.json').exists() else None
            if not intent or intent.get('plan',{}).get('decision')!='ACTION':
                excluded.append({'id':key,'reason':'no dispatched action provenance'});continue
            plan=intent['plan'];meta=read(folder/'before/observation.json')
            if intent.get('observation_sha256')!=meta['sha256']['rgb.png']:
                excluded.append({'id':key,'reason':'dispatch image differs from before; recovery needs separate pair'});continue
            if read(folder/'after/observation.json')['captured_unix_s']<=meta['captured_unix_s']:
                excluded.append({'id':key,'reason':'nonchronological observations'});continue
            pairs.append({'id':key,'episode_id':str(ep),'source':str(folder),'before':str(folder/'before'),'after':str(folder/'after'),
                          'goal':str(ep/'goal/target.png'),'plan':plan,'execution':log,'split':'development'})
    return pairs,excluded

def action_from(pair):
    p=pair['plan'];return Action(tuple(p['grasp_px']),tuple(p['target_px']),p.get('grasp_angle_deg'),p.get('action_purpose'))
