"""Offline frozen-template reconstruction and historical-anchor audit; no robot/model runtime."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .graph_policy import load_frozen, checked_file, command, instantiate, file_hash
from .graph_minimal_grounding import MinimalGrounder, expand_slots, write
from .graph_policy_replay import primitive_candidate, validate_sleeve_side, validate_hem_direction, instantiate_grounded
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.dual_arm.config import DualConfig


def pose_errors(actual,expected):
    if len(actual)!=len(expected):raise ValueError('operation count changed')
    rows=[]
    for i,(a,b) in enumerate(zip(actual,expected)):
        if a['method']!=b['method']:raise ValueError('primitive method changed')
        p=np.asarray(a['arguments']['right'],float);q=np.asarray(b['arguments']['right'],float)
        rows.append(dict(operation_index=i,xyz_error_mm=float(np.linalg.norm(p[:3]-q[:3])*1000),
            xy_error_mm=float(np.linalg.norm(p[:2]-q[:2])*1000),z_error_mm=float(abs(p[2]-q[2])*1000),
            rotation_error_deg=float(np.degrees((Rotation.from_euler('xyz',p[3:]).inv()*Rotation.from_euler('xyz',q[3:])).magnitude()))))
    return rows


def geometry_checks(candidate,edge,obs,arm,measurements):
    if candidate.get('coordinate_frame')=='ACTION_LOCAL':
        from .action_local_frame import validate_candidate
        validate_candidate(candidate)
        for m in measurements:
            pose=np.array(m['commanded_world_m_rad']);pose[:3]*=1000;pose[3:]=np.degrees(pose[3:])
            arm.validate_point(arm.base_pose(pose)[:3])
        return dict(depth='ANCHORS_PASS',workspace='PASS',waypoint_geometry='PASS',waypoint_count=len(measurements),
            semantics='NOT_INDEPENDENTLY_VERIFIED',target_height_delta_mm=measurements[-1]['commanded_world_m_rad'][2]*1000-candidate['anchor_world_mm'][1][2],
            height_policy=candidate.get('height_semantics',{}).get('policy','legacy_visual_pa'),
            note='No collar/hem validation or per-waypoint contact measurement. No IK/collision/contact proof.')
    slots=candidate['slots'];size=obs.xyz.shape[1::-1]
    for slot,m in zip(slots,measurements):
        x,y=slot['pixel_xy']
        if not (0<=x<size[0] and 0<=y<size[1]):raise ValueError('waypoint outside image')
        if not 0<=slot['z_above_surface_mm']<=300:raise ValueError('height outside template contract')
        pose=np.array(m['commanded_world_m_rad']);pose[:3]*=1000;pose[3:]=np.degrees(pose[3:])
        arm.validate_point(arm.base_pose(pose)[:3])
    if slots[0]['z_above_surface_mm']!=0:raise ValueError('grasp lowering duplicated')
    if math.dist(slots[0]['pixel_xy'],slots[-1]['pixel_xy'])<10:raise ValueError('degenerate grasp-target geometry')
    if 'garment_frame' in candidate:
        validate_sleeve_side(candidate,edge,obs)
        if edge['action_family']=='bottom_hem_toward_collar_fold':validate_hem_direction(candidate,candidate['garment_frame'])
    return dict(depth='PASS',workspace='PASS',waypoint_geometry='PASS',
        waypoint_count=len(slots),max_surface_relative_height_mm=max(s['z_above_surface_mm'] for s in slots),
        note='Endpoint workspace box only; Cartesian linear segments stay in this convex box. No IK, link collision, approach/return or joint-transfer-path verification.')


def instantiate_local(edge,obs,slots):
    value=dict(status='READY',edge_id=edge['edge_id'],action_family=edge['action_family'],slots=slots,
        explanation='Offline frozen Pa/Pb profile reconstruction',unresolved_conditions=[])
    _,source,measurements=instantiate(value,[edge],obs)
    return source,measurements


def overlay(obs,slots,path):
    image=Image.open(obs.image).convert('RGB');d=ImageDraw.Draw(image)
    points=[tuple(s['pixel_xy']) for s in slots]
    d.line(points,fill='orange',width=2)
    for i,(x,y) in enumerate(points):
        color='red' if i==0 else 'blue' if i==len(points)-1 else 'orange'
        d.ellipse((x-6,y-6,x+6,y+6),outline=color,width=2)
        d.text((x+8,y+8),str(i),fill=color)
    image.save(path)


def historical_candidates(root):
    """Explicit semantic mapping; matching edge numbers alone is insufficient."""
    selected=[];excluded=[]
    for p in sorted(root.glob('policy_replay_*/step_*/grounding.json')):
        data=json.loads(p.read_text())
        if data.get('status')!='READY' or data['edge_id'] not in ('e000','e001','e003'):continue
        if data['edge_id']=='e003':
            excluded.append(dict(path=str(p.resolve()),reason='Historical e003 grasps mid-torso side edge and explicitly excludes hem corner; new e003 requires hem-side inset. Do not transfer solely by edge ID.'))
            continue
        try:
            dispatch=json.loads((p.parent/'dispatch_intent.json').read_text())
            match=json.loads((p.parent/'node_match.json').read_text())['result']
            required='S0' if data['edge_id']=='e000' else 'S1'
            if match['status']!='MATCH' or match['node_id']!=required or match['confidence']<.8 or match['violated_conditions'] or match['unresolved_conditions']:
                raise ValueError('unreliable/incompatible archived source-state match')
            image=checked_file(dispatch['before'])
            selected.append(dict(edge_id=data['edge_id'],grounding_path=str(p.resolve()),grounding_sha256=file_hash(p),
                observation_directory=str(image.parent),source_node=required,source_match=match,
                grasp_px=data['slots'][0]['pixel_xy'],target_px=data['slots'][-1]['pixel_xy'],
                anchor_source='Archived full GPT grounding, noisy reference; no new model call',
                compatibility='Same sleeve side and exposed sleeve to torso relation. S0=both sleeves extended; S1=left inward/right exposed. Historical node match is not independent ground truth.'))
        except (ValueError,KeyError,OSError) as exc:excluded.append(dict(path=str(p.resolve()),reason=str(exc)))
    return selected,excluded


def audit(graph_path,output,history):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    graph_path=Path(graph_path).resolve();graph=load_frozen(graph_path)
    trajectory=load_frozen(graph['seed_trajectory_path'])
    records=load_frozen(graph_path.parent/'edge_experiences.json')
    grounder=MinimalGrounder(graph['minimal_templates_path'])
    edges={e['edge_id']:e for e in graph['edges']};steps={s['step_id']:s for s in trajectory['steps']}
    config=DualConfig.load('config/dual_arm.local.json',root=Path.cwd(),homing_only=True);arm=config.arms['right']
    paths=[graph_path,Path(graph['seed_trajectory_path']),grounder.path,graph_path.parent/'edge_experiences.json']
    hashes={str(p):file_hash(p) for p in paths}
    protocol=dict(xyz_reconstruction_tolerance_mm=2.,rotation_reconstruction_tolerance_deg=.1,
        historical_comparison='Depth/workspace/waypoint checks, not equality to old GPT motion',
        source_artifact_hashes=hashes,workspace_base_mm={k:[float(x) if np.isfinite(x) else None for x in v] for k,v in arm.workspace.items()},
        unbounded_workspace_axes='null bound means unbounded; X both bounds and Z upper are unbounded in existing config',
        depth_policy='Use each archived observation as recorded, with its own compensation metadata. Do not rewrite historical depth.',
        no_robot=True,no_model_calls=True,ik_collision_tested=False)
    write(output/'protocol.json',protocol)
    rows=[];full_seed=[]
    # Gate: all stored local profiles must reconstruct their own seed before historical tests.
    for record in records:
        eid=record['edge_id'];edge=edges[eid];step=steps[edge['seed_step_id']]
        folder=output/'seed'/eid;folder.mkdir(parents=True)
        row=dict(edge_id=eid,replayable=edge['replayable'],status='FAIL',stage='local_reconstruction')
        try:
            obs=Observation.load(checked_file(trajectory['observations'][step['before_observation']]['observation']).parent)
            slots=expand_slots(record['normalized_waypoint_geometry'],record['seed_grasp_px'],record['seed_target_px'],0.)
            if grounder.bundle.get('frame_mode') in {'action_local','action_local_contact'}:
                from .action_local_frame import compile_local,expand
                template=grounder.bundle['edges'].get(eid) or compile_local(edge,step,obs,record,contact_z=grounder.bundle.get('frame_mode')=='action_local_contact')
                local=expand(template,dict(grasp_px=record['seed_grasp_px'],target_px=record['seed_target_px'],confidence=1.),edge,obs,folder)
                _,source,measurements=instantiate_grounded(local,edge,obs)
            else:
                source,measurements=instantiate_local(edge,obs,slots)
            actual=command(source)['operations']
            offset=record['actual_grasp_lowering']['offset_mm']
            actual[0]['arguments']['right'][2]-=offset/1000  # same recorded host offset, once
            errors=pose_errors(actual,step['commanded_action']['operations'])
            xyz=max(e['xyz_error_mm'] for e in errors);rot=max(e['rotation_error_deg'] for e in errors)
            if xyz>2 or rot>.1:raise ValueError(f'reconstruction exceeds tolerance: {xyz} mm, {rot} deg')
            row.update(status='PASS',max_xyz_error_mm=xyz,max_xy_error_mm=max(e['xy_error_mm'] for e in errors),
                max_z_error_mm=max(e['z_error_mm'] for e in errors),max_rotation_error_deg=rot)
            write(folder/'pose_errors.json',errors);write(folder/'reconstructed_command.json',actual)
            write(folder/'measurements.json',measurements);overlay(obs,slots,folder/'overlay.png')
            if eid in grounder.bundle['edges']:
                candidate=grounder.expand(dict(grasp_px=record['seed_grasp_px'],target_px=record['seed_target_px'],confidence=1.),edge,obs,folder/'runtime')
                _,motion,m=instantiate_grounded(candidate,edge,obs)
                actual_full=command(motion)['operations'];actual_full[0]['arguments']['right'][2]-=offset/1000
                er=pose_errors(actual_full,step['commanded_action']['operations'])
                if max(x['xyz_error_mm'] for x in er)>2 or max(x['rotation_error_deg'] for x in er)>.1:raise ValueError('full runtime seed roundtrip failed')
                full_seed.append(dict(edge_id=eid,status='PASS',checks=geometry_checks(candidate,edge,obs,arm,m)))
        except (ValueError,KeyError,TypeError,OSError) as exc:row.update(status='FAIL',error=str(exc))
        rows.append(row)
    write(output/'seed_reconstruction.json',rows);write(output/'seed_runtime_checks.json',full_seed)
    if any(r['status']!='PASS' for r in rows):
        write(output/'summary.json',dict(status='SEED_RECONSTRUCTION_FAILED',historical='NOT_RUN',seed=rows))
        return
    candidates,excluded=historical_candidates(Path(history));write(output/'historical_manifest.json',dict(samples=candidates,excluded=excluded))
    historical=[]
    for i,sample in enumerate(candidates):
        folder=output/'historical'/f'{i:03d}_{sample["edge_id"]}';folder.mkdir(parents=True)
        row=dict(sample_index=i,**sample,status='FAIL',stage='frame_registration_and_expansion')
        try:
            obs=Observation.load(sample['observation_directory']);edge=edges[sample['edge_id']]
            row['historical_support_patch_present']='support_height_patch' in obs.meta
            candidate=grounder.expand(dict(grasp_px=sample['grasp_px'],target_px=sample['target_px'],confidence=1.),edge,obs,folder)
            row['stage']='depth_instantiation'
            _,motion,measurements=instantiate_grounded(candidate,edge,obs)
            write(folder/'candidate.json',candidate);write(folder/'measurements.json',measurements)
            (folder/'motion_offline_only.py').write_text(motion)
            if candidate.get('coordinate_frame')=='ACTION_LOCAL':
                # Overlay XY footprints only; not used for motion reconstruction.
                plot_slots=[]
                for m in measurements:
                    xy=np.array(m['commanded_world_m_rad'][:2])*1000
                    distance=np.linalg.norm(obs.xyz[:,:,:2]-xy,axis=2);distance[~np.isfinite(distance)]=np.inf
                    y,x=np.unravel_index(distance.argmin(),distance.shape);plot_slots.append(dict(pixel_xy=[int(x),int(y)]))
                overlay(obs,plot_slots,folder/'xy_footprint_overlay.png')
                row['known_semantic_issue']='Prior independent audit found this e000 grasp on wrong sleeve' if '20261012T135253418197Z' in sample['grounding_path'] else None
            else:overlay(obs,candidate['slots'],folder/'overlay.png')
            row['stage']='workspace_and_waypoint_geometry'
            row['checks']=geometry_checks(candidate,edge,obs,arm,measurements)
            row.update(status='PASS',stage='completed')
        except (ValueError,KeyError,TypeError,OSError) as exc:row['error']=str(exc)
        historical.append(row)
    write(output/'historical_results.json',historical)
    coverage=[dict(edge_id=r['edge_id'],action_family=r['action_family'],replayable=edges[r['edge_id']]['replayable'],
        local_profile_compiled=r['normalized_waypoint_geometry'] is not None,
        minimal_runtime_supported=r['status']=='MINIMAL_SUPPORTED',unsupported_reason=r['reason'],
        historical_samples=sum(h['edge_id']==r['edge_id'] for h in historical),
        historical_pass=sum(h['edge_id']==r['edge_id'] and h['status']=='PASS' for h in historical)) for r in records]
    write(output/'coverage.json',coverage)
    summary=dict(status='COMPLETE',actions=len(edges),replayable_edges=sum(e['replayable'] for e in edges.values()),
        local_geometry_profiles=sum(r['local_profile_compiled'] for r in coverage),
        replayable_local_geometry_profiles=sum(r['local_profile_compiled'] and r['replayable'] for r in coverage),
        executable_minimal_templates=len(grounder.bundle['edges']),minimal_unsupported=sum(not r['minimal_runtime_supported'] for r in coverage),
        seed_reconstruction_pass=sum(r['status']=='PASS' for r in rows),seed_max_xyz_error_mm=max(r['max_xyz_error_mm'] for r in rows),
        seed_max_rotation_error_deg=max(r['max_rotation_error_deg'] for r in rows),full_runtime_seed_pass=len(full_seed),
        historical_samples=len(historical),historical_pass=sum(r['status']=='PASS' for r in historical),
        historical_fail=sum(r['status']!='PASS' for r in historical),historical_covered_edges=sorted(set(r['edge_id'] for r in historical)),
        known_semantic_mismatches=sum(bool(r.get('known_semantic_issue')) for r in historical),
        source_artifacts_unchanged=all(file_hash(p)==h for p,h in hashes.items()),robot_calls=0,model_calls=0)
    write(output/'summary.json',summary)
    for name,data in [('seed_reconstruction',rows),('coverage',coverage),('historical',historical)]:
        columns=sorted(set(k for r in data for k in r))
        with (output/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            for r in data:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    (output/'report.md').write_text('# Frozen seed template offline audit\n\n'+
        '```json\n'+json.dumps(summary,indent=2)+'\n```\n\n'+
        'Local geometry and full runtime checks are distinct. Action-local mode needs no garment registration; legacy mode still does. Numerical passes do not verify anchor semantics.\n\n'+
        '| Edge | Replayable | Local Pa/Pb geometry | Minimal runtime | Historical pass/total | Unsupported reason |\n|---|---|---|---|---|---|\n'+
        '\n'.join(f"| {r['edge_id']} | {r['replayable']} | {r['local_profile_compiled']} | {r['minimal_runtime_supported']} | {r['historical_pass']}/{r['historical_samples']} | {r['unsupported_reason'] or ''} |" for r in coverage)+
        '\n\nHistorical e003 was excluded: its mid-torso grasp does not match the new hem-side grasp relation. No compatible independent archived grounding was tested for the new e003; zero samples is NOT a pass. Remaining unsupported edges were not forced through runtime.\n\n'+
        'Historical samples are selected from archived full grounding and source matches; these are noisy model references. Archived depth compensation is preserved. Current workspace is only a partially bounded box (X unbounded, Z upper unbounded); a pass is NOT IK/collision/contact/physical success. No robot or model was called.\n\n'+
        'Action-local mode cannot independently reject the previously documented wrong-sleeve e000 anchors (sample 0). A numerical pass for that sample is NOT a semantic success.\n\n'+
        '## Historical failures\n\n'+'\n'.join(f"- {r['sample_index']} / {r['edge_id']}: {r['stage']}: {r.get('error')}" for r in historical if r['status']!='PASS')+'\n')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--history',type=Path,default=Path('runs/graph_policy_full02'))
    args=p.parse_args();audit(args.graph,args.output,args.history)

if __name__=='__main__':main()
