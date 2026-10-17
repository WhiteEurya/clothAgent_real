"""Two measured anchors + world up; no garment landmarks, image registration or robot calls."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .graph_minimal_grounding import edge_hash,sha,write,validate
from .discovery_data import digest

VERSION='action-local-host-template-v2'


def frame(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all():raise ValueError('invalid measured anchors')
    u=b-a;u[2]=0.;length=float(np.linalg.norm(u))
    if length<10.:raise ValueError('action-local horizontal anchor separation below 10 mm')
    u/=length;z=np.array([0.,0.,1.]);v=np.cross(z,u)
    return np.column_stack([u,v,z]),length


def contact_reference(obs):
    patch=obs.meta.get('support_height_patch',{})
    contact=patch.get('contact_reference',{})
    if not patch.get('enabled') or patch.get('z_policy')!='raw_plane_to_reference_height':
        raise ValueError('seed-contact Z requires recorded contact-height patch')
    if contact.get('semantics')!='empirical_open_gripper_contact_tcp_height' or contact.get('user_confirmed') is not True:
        raise ValueError('missing confirmed open-gripper TCP contact reference')
    height=float(patch['reference_height_mm'])
    if not math.isfinite(height):raise ValueError('invalid contact reference height')
    contract={k:patch.get(k) for k in ('mode','z_policy','reference_height_mm','plane_z_ax_by_c','input_coordinate_policy','contact_reference')}
    return height,digest(contract)


def compile_local(edge,step,obs,record,*,contact_z=False):
    a=np.asarray(obs.sample(record['seed_grasp_px'],20.));b=np.asarray(obs.sample(record['seed_target_px'],20.))
    basis,length=frame(a,b);rows=[]
    contact_height,contact_hash=contact_reference(obs) if contact_z else (None,None)
    for i,op in enumerate(step['commanded_action']['operations']):
        pose=np.asarray(op['arguments']['right'],float);point=pose[:3]*1000
        if i==0:point[2]+=record['actual_grasp_lowering']['offset_mm']
        delta=basis.T@(point-a)
        rows.append(dict(operation_index=i,method=op['method'],along_fraction=float(delta[0]/length),
            across_fraction=float(delta[1]/length),height_from_grasp_mm=float(delta[2]),
            tool_rotation_relative=(basis.T@Rotation.from_euler('xyz',pose[3:]).as_matrix()).tolist()))
        if contact_z:rows[-1]['tcp_height_above_contact_mm']=float(point[2]-contact_height)
    return dict(version=VERSION,z_policy='seed_contact_tcp' if contact_z else 'legacy_visual_pa',
        contact_reference_height_mm=contact_height,contact_reference_sha256=contact_hash,
        edge_sha256=edge_hash(edge),spec=dict(edge_id=edge['edge_id'],
        grasp=record['grasp_semantic_relation'],target=record['target_semantic_relation']),
        image_size=list(obs.xyz.shape[1::-1]),camera_serial=obs.meta['camera_serial'],confidence_min=.8,
        seed_grasp_px=record['seed_grasp_px'],seed_target_px=record['seed_target_px'],
        seed_anchor_world_mm=[a.tolist(),b.tolist()],seed_basis=basis.tolist(),seed_horizontal_length_mm=length,
        profile=rows,primitive_sequence=[r['method'] for r in rows],
        position_rule=('XY follows Pa/Pb; each TCP Z uses confirmed contact baseline plus this edge executed seed offset. Visual depth residual is diagnostic only, never cloth thickness.' if contact_z else 'XY fractions scale with current horizontal Pa-Pb distance; Z offsets are fixed millimetres relative to measured Pa. Pb Z is not used to tilt world-up.'),
        orientation_rule='R_tool_current = B_current @ R_tool_relative_seed',
        source_seed_step=edge['seed_step_id'],reference_image=str(obs.image),reference_sha256=sha(obs.image),
        semantic_limit='No independent collar/hem sleeve-side or fold-direction check. Edge semantics rely on node matching and GPT anchors.',
        safety_limit='No contact/clearance guarantee from two anchors; native workspace/IK/collision checks remain mandatory.')


def expand(template,value,edge,obs,folder):
    validate(value,template['image_size'])
    if list(obs.xyz.shape[1::-1])!=template['image_size'] or obs.meta.get('camera_serial')!=template['camera_serial']:
        raise ValueError('action-local camera/resolution mismatch')
    if value['confidence']<template['confidence_min'] or value['grasp_px'] is None or value['target_px'] is None:
        raise ValueError('minimal anchors unresolved/low confidence')
    a=np.asarray(obs.sample(value['grasp_px'],20.));b=np.asarray(obs.sample(value['target_px'],20.));basis,length=frame(a,b)
    contact_z=template.get('z_policy')=='seed_contact_tcp'
    if contact_z:
        contact_height,contact_hash=contact_reference(obs)
        if contact_hash!=template['contact_reference_sha256']:
            raise ValueError('contact-height calibration changed; recompile/revalidate template')
    poses=[]
    for r in template['profile']:
        rel=np.asarray(r['tool_rotation_relative'],float)
        if rel.shape!=(3,3) or not np.isfinite(rel).all() or not np.allclose(rel.T@rel,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rel),1):
            raise ValueError('invalid relative tool rotation')
        xyz=a+basis@np.array([r['along_fraction']*length,r['across_fraction']*length,r['height_from_grasp_mm']])
        if contact_z:xyz[2]=contact_height+r['tcp_height_above_contact_mm']
        pose=np.r_[xyz/1000,Rotation.from_matrix(basis@rel).as_euler('xyz')].tolist()
        if not np.isfinite(pose).all():raise ValueError('nonfinite local waypoint')
        poses.append(dict(operation_index=r['operation_index'],method=r['method'],world_m_rad=pose))
    result=dict(coordinate_frame='ACTION_LOCAL',status='READY',edge_id=edge['edge_id'],action_family=edge['action_family'],
        anchors_px=dict(grasp_px=value['grasp_px'],target_px=value['target_px']),anchor_world_mm=[a.tolist(),b.tolist()],
        basis=basis.tolist(),poses=poses,confidence=value['confidence'],
        height_semantics=dict(policy=template.get('z_policy','legacy_visual_pa'),
            visual_anchor_z_mm=[float(a[2]),float(b[2])],
            commanded_grasp_tcp_z_mm=poses[0]['world_m_rad'][2]*1000,
            contact_reference_mm=contact_height if contact_z else None,
            note='seed TCP offsets are empirical action evidence, not measured cloth thickness or universal contact calibration'),
        visual_violations=[],missing_visual_parameters=[],explanation='Host Pa-Pb/world-up expansion.',
        execution_risk=[template['semantic_limit'],template['safety_limit']])
    write(folder/'minimal_host_expansion.json',dict(candidate=result,horizontal_length_mm=length,
        scale=length/template['seed_horizontal_length_mm'],garment_registration_used=False))
    return result


def validate_candidate(candidate):
    if candidate.get('coordinate_frame')!='ACTION_LOCAL' or candidate.get('status')!='READY':raise ValueError('invalid action-local candidate')
    if candidate.get('visual_violations') or candidate.get('missing_visual_parameters'):raise ValueError('unresolved local candidate')
    poses=candidate.get('poses',[])
    if len(poses)<2 or [r['operation_index'] for r in poses]!=list(range(len(poses))):raise ValueError('invalid local operation indices')
    for r in poses:
        if np.asarray(r['world_m_rad']).shape!=(6,) or not np.isfinite(r['world_m_rad']).all():raise ValueError('invalid local pose')
    basis=np.asarray(candidate['basis'])
    if not np.allclose(basis.T@basis,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(basis),1):raise ValueError('nonorthogonal local basis')


def instantiate_local(candidate,edge,obs):
    from .graph_policy import command
    validate_candidate(candidate)
    if candidate['edge_id']!=edge['edge_id'] or candidate['action_family']!=edge['action_family']:raise ValueError('local edge identity mismatch')
    ops=edge['seed_action']['exact_command']['operations'];poses=candidate['poses']
    if len(ops)!=len(poses) or ops[0]['method']!='grasp' or any(o['method'] not in ('move_tcp','move_transfer') for o in ops[1:]):
        raise ValueError('unsupported local primitive skeleton')
    if any(o['method']!=p['method'] or set(o['arguments'])!={'right'} for o,p in zip(ops,poses)):raise ValueError('local primitive skeleton changed')
    lines=[f"motion.{p['method']}(right={p['world_m_rad']!r})" for p in poses]
    source='\n'.join(lines)+'\n';command(source)
    measurements=[dict(operation_index=p['operation_index'],commanded_world_m_rad=p['world_m_rad'],
        evidence=candidate.get('height_semantics',{})) for p in poses]
    return edge,source,measurements
