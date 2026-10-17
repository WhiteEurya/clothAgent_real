"""Opt-in two-anchor grounding; deterministic Host template expansion.
No robot execution. Existing replay side/depth/IK/collision checks remain downstream.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import jsonschema
import numpy as np
from PIL import Image
from .discovery_schema import obj

class GroundingUncertain(ValueError):
    """Valid model response lacks confident anchors; recapture may help."""


class NoRobustAction(GroundingUncertain):
    """Explicit grasp-gate rejection, not a request to repeat grounding."""
    def __init__(self, reasons, observation=None):
        self.reasons = list(reasons)
        self.observation = observation
        super().__init__('NO_ROBUST_ACTION: ' + '; '.join(self.reasons))


POINT={'anyOf':[{'type':'array','items':{'type':'integer','minimum':0},'minItems':2,'maxItems':2},{'type':'null'}]}
from cloth_agent.action_robustness import GROUNDING_PROMPT, SCHEMA as ROBUSTNESS_SCHEMA, strip_grounding_evidence
SCHEMA=obj(grasp_px=POINT,target_px=POINT,confidence={'type':'number','minimum':0,'maximum':1})
SCHEMA['properties']['robustness']=ROBUSTNESS_SCHEMA  # Optional for historical response compatibility.
PROMPT='''Locate only these two visible fabric anchors in CURRENT RGB. Left/right mean viewer sides after mentally rotating collar above hem, without mirroring. Coordinates are original-image pixels [x,y]. Return grasp_px, target_px, confidence and compact robustness metadata. Use null and confidence 0 if unresolved. No action redesign or long explanation.''' + GROUNDING_PROMPT
SPEC_E001=dict(edge_id='e001',grasp='exposed outer right cuff / distal sleeve fabric',target='adjacent right upper-torso receiving fabric, inward across the sleeve root')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def edge_hash(edge):
    keys=['edge_id','source_node','action_family','action_template','seed_action']
    return hashlib.sha256(json.dumps({k:edge[k] for k in keys},sort_keys=True).encode()).hexdigest()

def validate(value,size=None):
    jsonschema.validate(value,SCHEMA)
    if size:
        for k in ['grasp_px','target_px']:
            p=value[k]
            if p is not None and not (0<=p[0]<size[0] and 0<=p[1]<size[1]):raise ValueError(k+' outside image')
    if not math.isfinite(value['confidence']):raise ValueError('nonfinite confidence')

def relative_slots(slots):
    a=np.array(slots[0]['pixel_xy'],float);v=np.array(slots[-1]['pixel_xy'],float)-a
    length=float(v@v)
    if length<100:raise ValueError('reference displacement too short')
    n=np.array([-v[1],v[0]])
    return [dict(operation_index=s['operation_index'],along=float((np.array(s['pixel_xy'])-a)@v/length),
                 across=float((np.array(s['pixel_xy'])-a)@n/length),z_above_surface_mm=s['z_above_surface_mm'],rpy_rad=s['rpy_rad']) for s in slots]

def expand_slots(profile,a,b,yaw_delta):
    from scipy.spatial.transform import Rotation
    a=np.asarray(a,float);v=np.asarray(b,float)-a;n=np.array([-v[1],v[0]])
    if float(v@v)<100:raise ValueError('grasp/target displacement too short')
    rz=Rotation.from_euler('z',yaw_delta)
    result=[]
    for s in profile:
        xy=np.rint(a+s['along']*v+s['across']*n).astype(int).tolist()
        rpy=(rz*Rotation.from_euler('xyz',s['rpy_rad'])).as_euler('xyz').tolist()
        result.append(dict(operation_index=s['operation_index'],pixel_xy=xy,z_above_surface_mm=s['z_above_surface_mm'],
                           rpy_rad=rpy,relation_evidence='Host expands frozen relative waypoint profile between current grasp/target; no GPT waypoint generation.'))
    return result

def compile_e001(graph_path,reference_step,output):
    """Offline compile from an executed, expected-outcome reference; never model-generated."""
    from .graph_policy import load_frozen
    from cloth_agent.dual_arm.observation import Observation
    graph=load_frozen(graph_path);edge=next(e for e in graph['edges'] if e['edge_id']=='e001')
    folder=Path(reference_step).resolve();g=read(folder/'grounding.json');dispatch=read(folder/'dispatch_intent.json');execution=read(folder/'execution.json')
    after=read(folder.parent/f'step_{int(folder.name[5:])+1:03d}'/'node_match.json')
    assert dispatch['edge_id']==g['edge_id']=='e001' and execution['status']=='EXECUTED'
    assert after['result']['status']=='MATCH' and after['result']['node_id']==edge['target_node']
    obs=Observation.load(Path(dispatch['before']['path']).parent)
    assert sha(obs.image)==dispatch['before']['sha256']
    methods=[o['method'] for o in edge['seed_action']['exact_command']['operations']]
    assert methods==['grasp','move_tcp','move_tcp','move_tcp','move_tcp'] and len(g['slots'])==5
    # This implementation intentionally supports the archived constant-orientation sleeve template only.
    seedrpy=[o['arguments']['right'][3:] for o in edge['seed_action']['exact_command']['operations']]
    assert np.allclose(seedrpy,np.broadcast_to(seedrpy[0],(5,3)),atol=1e-5)
    assert np.allclose([s['rpy_rad'] for s in g['slots']],seedrpy,atol=1e-5)
    assert g['slots'][0]['z_above_surface_mm']==0
    frame=g['garment_frame'];c=obs.sample(frame['collar_pixel_xy'],20.);h=obs.sample(frame['hem_center_pixel_xy'],20.)
    axis=np.asarray(c)[:2]-np.asarray(h)[:2]
    item=dict(edge_sha256=edge_hash(edge),spec=SPEC_E001,primitive_sequence=methods,profile=relative_slots(g['slots']),
              reference_image=str(obs.image),reference_sha256=sha(obs.image),frame=frame,
              world_axis_yaw=math.atan2(axis[1],axis[0]),image_size=list(Image.open(obs.image).size),
              camera_serial=obs.meta['camera_serial'],reference_grounding=str(folder/'grounding.json'),reference_grounding_sha256=sha(folder/'grounding.json'),
              registration_roi=[500,180,850,450],registration='SIFT torso-print features + partial-affine RANSAC; no mirrors',
              confidence_min=.8,scope='e001 sleeve template only; no unsupported-edge or full-mode automatic fallback')
    bundle=dict(version='minimal-grounding-host-template-v1',edges={'e001':item},source_graph=str(Path(graph_path).resolve()),
                note='Relative pixel waypoint shape, measured-surface heights and constant seed RPY compiled once from an actual successful replay. Original graph unchanged.')
    path=Path(output)
    if path.exists():assert read(path)==bundle,'Do not overwrite frozen template'
    else:write(path,bundle);write(str(path)+'.sha256.json',{'sha256':sha(path)})
    return bundle


def register_frame(template,observation):
    """Current-image CV evidence, no stale collar/hem coordinates copied without registration."""
    import cv2
    if sha(template['reference_image'])!=template['reference_sha256']:raise ValueError('reference RGB changed')
    if observation.meta.get('camera_serial')!=template['camera_serial']:raise ValueError('minimal template camera mismatch')
    size=tuple(template['image_size'])
    if Image.open(observation.image).size!=size:raise ValueError('minimal template resolution mismatch')
    a=cv2.imread(template['reference_image'],cv2.IMREAD_GRAYSCALE);b=cv2.imread(str(observation.image),cv2.IMREAD_GRAYSCALE)
    mask=np.zeros(a.shape,np.uint8);x0,y0,x1,y1=template['registration_roi'];mask[y0:y1,x0:x1]=255
    # Search the table; do not match static robot/background features in the reference.
    qmask=np.zeros(b.shape,np.uint8)
    qx0,qy0,qx1,qy1=template.get('query_registration_roi',[300,0,1100,680])
    qmask[qy0:qy1,qx0:qx1]=255
    sift=cv2.SIFT_create(nfeatures=2000);ka,da=sift.detectAndCompute(a,mask);kb,db=sift.detectAndCompute(b,qmask)
    if da is None or db is None:raise ValueError('Host frame registration missing descriptors')
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2);good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
    # Ensure one-to-one correspondences, not many sources voting for the same query feature.
    good=sorted(good,key=lambda x:x.distance);seen=set();unique=[]
    for m in good:
        if m.trainIdx not in seen:seen.add(m.trainIdx);unique.append(m)
    if len(unique)<8:raise ValueError('Host frame registration too few unique matches')
    src=np.float32([ka[m.queryIdx].pt for m in unique]);dst=np.float32([kb[m.trainIdx].pt for m in unique])
    cv2.setRNGSeed(0);affine,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4.,maxIters=2000,confidence=.99)
    if affine is None or inliers is None:raise ValueError('Host frame registration failed')
    use=inliers.ravel().astype(bool);residual=np.linalg.norm(src@affine[:,:2].T+affine[:,2]-dst,axis=1)
    scale=float(np.linalg.norm(affine[:,0]));angle=math.atan2(affine[1,0],affine[0,0])
    spread=np.ptp(src[use],axis=0)
    if use.sum()<8 or use.mean()<.5 or min(spread)<40 or np.median(residual[use])>3 or not .8<=scale<=1.2 or abs(angle)>math.radians(30):
        raise ValueError('Host frame registration lacks consistent wide image support')
    frame={}
    for key in ['collar_pixel_xy','hem_center_pixel_xy']:
        point=np.rint(affine@np.r_[template['frame'][key],1]).astype(int).tolist()
        if not(0<=point[0]<size[0] and 0<=point[1]<size[1]):raise ValueError('registered garment frame outside image')
        frame[key]=point
    frame['visual_evidence']=f'Host current RGB SIFT registration: {int(use.sum())} inliers across torso print; historical collar/hem identities transferred. Not independent semantic proof.'
    return frame,dict(affine=affine.tolist(),inliers=int(use.sum()),unique_matches=len(unique),median_residual_px=float(np.median(residual[use])),scale=scale,angle_deg=math.degrees(angle),reference_span_px=spread.tolist())


def marked_seed_reference(template, edge, folder):
    """Annotate this edge's verified seed anchors; never alter the source RGB."""
    from PIL import ImageDraw
    if not edge.get('replayable') or edge.get('policy_outcome') not in {'SUCCESS','PROGRESS'}:
        raise ValueError('reference grounding requires a replayable successful/useful seed edge')
    reference=Path(template['reference_image'])
    if sha(reference)!=template['reference_sha256']:
        raise ValueError('reference RGB changed')
    im=Image.open(reference).convert('RGB')
    if list(im.size)!=list(template['image_size']):
        raise ValueError('reference RGB dimensions changed')
    value=dict(grasp_px=template['seed_grasp_px'],target_px=template['seed_target_px'],confidence=1.)
    validate(value,im.size)
    if value['grasp_px'] is None or value['target_px'] is None:
        raise ValueError('reference seed anchors missing')
    draw=ImageDraw.Draw(im)
    for point,color,label in [(value['grasp_px'],'red','Pa grasp'),(value['target_px'],'blue','Pb target')]:
        x,y=point
        draw.ellipse((x-7,y-7,x+7,y+7),fill=color,outline='white',width=2)
        draw.text((max(0,min(x+10,im.width-90)),max(0,y-20)),label,fill=color,stroke_width=1,stroke_fill='white')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    output=folder/'seed_reference_anchors.png';im.save(output)
    write(folder/'seed_reference_anchors.json',dict(edge_id=edge['edge_id'],
        reference_image=str(reference),reference_sha256=sha(reference),
        seed_grasp_px=value['grasp_px'],seed_target_px=value['target_px'],
        annotated_image=str(output),annotated_sha256=sha(output),
        seed_step_id=template.get('source_seed_step'),policy_outcome=edge['policy_outcome']))
    return output


class MinimalGrounder:
    def __init__(self,path):
        self.path=Path(path)
        if read(str(self.path)+'.sha256.json')['sha256']!=sha(self.path):raise ValueError('minimal Host template checksum mismatch')
        self.bundle=read(self.path)
        if self.bundle['version']!='minimal-grounding-host-template-v1':raise ValueError('unsupported minimal template version')
    def template(self,edge):
        t=self.bundle['edges'].get(edge['edge_id'])
        if t is None:raise ValueError('minimal mode has no compiled Host template for '+edge['edge_id'])
        if edge_hash(edge)!=t['edge_sha256']:raise ValueError('selected edge changed from compiled Host template')
        return t
    def ground(self,requests,edge,before,folder):
        t=self.template(edge);folder=Path(folder)
        payload=copy.deepcopy(t['spec']);images=[('current',before.image)]
        verified_reference=getattr(requests,'grounding_reference_overrides',{}).get(edge['edge_id'])
        if edge['edge_id']=='e004' or verified_reference is not None:
            reference=marked_seed_reference(verified_reference or t,edge,folder)
            images.append(('successful_seed_reference',reference))
            payload['reference_instruction']='参考图中的红点/蓝点是该 edge 上次成功或产生有效进展时的 grasp/target；请在当前图中找具有相同 manipulation role 且可靠的对应点，不要复制绝对像素位置。输出当前图的 grasp_px、target_px、confidence 和简短 robustness 标签。'
        value=requests.ask('ground_edge_minimal',payload,images,validator=lambda v:validate(v,t['image_size']))
        reason=value.get('robustness',{}).get('reason') or 'Required reliable grasp/receiving anchor is unavailable'
        value=strip_grounding_evidence(value,folder)
        write(folder/'minimal_anchors.json',value)
        if value['grasp_px'] is None or value['target_px'] is None:
            raise NoRobustAction([reason],before)
        return self.expand(value,edge,before,folder)
    def expand(self,value,edge,before,folder):
        reason=value.get('robustness',{}).get('reason') or 'Required reliable grasp/receiving anchor is unavailable'
        if 'robustness' in value:
            value=strip_grounding_evidence(value,folder)
        t=self.template(edge);folder=Path(folder);validate(value,t['image_size'])
        if value['grasp_px'] is None or value['target_px'] is None:
            raise NoRobustAction([reason],before)
        if value['confidence']<t['confidence_min'] or value['grasp_px'] is None or value['target_px'] is None:raise GroundingUncertain('minimal anchors unresolved/low confidence')
        if t.get('version')=='action-local-host-template-v2':
            from .action_local_frame import expand
            return expand(t,value,edge,before,folder)
        start=time.perf_counter();frame,audit=register_frame(t,before)
        c=np.asarray(before.sample(frame['collar_pixel_xy'],20.));h=np.asarray(before.sample(frame['hem_center_pixel_xy'],20.))
        axis=c[:2]-h[:2]
        if not np.isfinite(axis).all() or np.linalg.norm(axis)<50:raise ValueError('invalid measured current garment axis')
        delta=math.atan2(axis[1],axis[0])-t['world_axis_yaw'];delta=math.atan2(math.sin(delta),math.cos(delta))
        if abs(delta)>math.radians(30):raise ValueError('minimal orientation outside compiled template scope')
        slots=expand_slots(t['profile'],value['grasp_px'],value['target_px'],delta)
        for s in slots:
            x,y=s['pixel_xy']
            if not(0<=x<t['image_size'][0] and 0<=y<t['image_size'][1]):raise ValueError('expanded waypoint outside image')
        write(folder/'minimal_host_expansion.json',dict(template=str(self.path),registration=audit,yaw_delta_rad=delta,elapsed_s=time.perf_counter()-start,confidence=value['confidence']))
        return dict(status='READY',edge_id=edge['edge_id'],action_family=edge['action_family'],slots=slots,garment_frame=frame,
                    explanation='Host expanded frozen edge profile from two current RGB anchors.',visual_violations=[],missing_visual_parameters=[],
                    execution_risk=['Two-anchor mode does not perform the full-mode visual constraint audit. Cloth coupling and hidden layers remain unknown; all Host validation still required.'])
