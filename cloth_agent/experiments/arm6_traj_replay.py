"""Arm6 demonstration adapter for graph_policy_replay; no hardware on import."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import signal

import numpy as np
from scipy.spatial.transform import Rotation

from .graph_policy import (Requests, checked_file, evidence_file, frozen, load_frozen,
                           graph_validation, reliable_match)
from .graph_policy_schema import GRAPH, EDGE, MATCH
from .graph_policy_matching import topology_match
from .graph_minimal_grounding import SCHEMA as ANCHORS, validate as validate_anchors
from .discovery_data import save, file_hash
from .replay_failures import capture_cli_failure, stop_after_failure, close_cli_runtime

PROTOCOL = 'arm6-traj-experience-graph-v1'


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_demonstration_authorization(action, manifest):
    if manifest.get('status') == 'COMPLETE' and manifest.get('operator_accepted') is True:
        return
    authorization = action.get('replay_authorization', {})
    if (manifest.get('status') == 'REJECTED' and manifest.get('operator_accepted') is False
            and authorization.get('kind') == 'USER_REQUESTED_REJECTED_DEMONSTRATION'
            and authorization.get('manifest') == action.get('manifest')
            and action.get('manifest', {}).get('sha256')
            and isinstance(authorization.get('reason'), str) and authorization['reason'].strip()):
        return
    raise ValueError('source demonstration is not operator accepted or explicitly authorized for replay')


def compile_templates(graph_path):
    graph = load_frozen(graph_path)
    if graph.get('protocol') != PROTOCOL or graph.get('robot') != 'Arm6' or graph.get('camera') != 'A':
        raise ValueError('requires Arm6 Camera A experience graph')
    trajectory = load_frozen(checked_file(graph['source_trajectory']))
    core = {k:copy.deepcopy(graph[k]) for k in GRAPH['properties']}
    core['edges'] = [{k:e[k] for k in EDGE['required']} for e in core['edges']]
    graph_validation(core, trajectory)
    for terminal in graph.get('terminal_node_ids', []):
        if terminal not in {n['node_id'] for n in graph['nodes']} or any(
                e['source_node'] == terminal for e in graph['edges']):
            raise ValueError('terminal node must exist and have no outgoing actions')
    for observation in trajectory['observations'].values():
        for key in ('rgb','surface','observation'):
            checked_file(observation[key])
    templates = {}
    for edge in graph['edges']:
        action = edge['seed_action']['exact_command']
        if action.get('arm') != 'left' or action.get('axis') != 6:
            raise ValueError('non-Arm6 source action')
        manifest = json.loads(checked_file(action['manifest']).read_text())
        validate_demonstration_authorization(action, manifest)
        states = jsonl(checked_file(action['measured_trajectory']))
        events = jsonl(checked_file(action['manual_input_trace']))
        checked_file(action['controller_events'])
        closes = [e for e in events if e.get('intent',{}).get('gripper') == 'close']
        if len(closes) != 1:
            raise ValueError(f'{edge["edge_id"]}: automatic extraction requires one close event')
        close = closes[0]
        lift = next((e for e in events if e['t_s'] > close['t_s']
                     and e.get('intent',{}).get('up',0) > 0), None)
        if lift is None:
            raise ValueError('no measured lift onset')
        if any(e.get('intent',{}).get('gripper') == 'open' for e in events
               if close['t_s'] < e['t_s'] < lift['t_s']):
            raise ValueError('gripper reopened before lift')
        held = [s for s in states if s['phase']=='manual' and close['t_s'] <= s['t_s'] <= lift['t_s']]
        if len(held) < 3 or lift['t_s']-held[-1]['t_s'] > .2:
            raise ValueError('insufficient feedback near lift onset')
        settled = held[-3:]
        poses = np.asarray([s['world_tcp_mm_deg'] for s in settled],float)
        if (not np.isfinite(poses).all() or np.max(np.ptp(poses[:,:3],axis=0)) > 1.
                or np.ptp([s['gripper_pulse'] for s in settled]) > 15):
            raise ValueError('grasp not stationary before lift')
        grasp = np.asarray(held[-1]['world_tcp_mm_deg'],float)
        handoff = np.asarray(action['handoff']['world_tcp_mm_deg'],float)
        points = {p['name']:p for p in action['selected_points']}
        start, target = [np.asarray(points[k]['world_xyz_mm'],float) for k in ('start','target')]
        vector = target[:2]-start[:2]
        if np.linalg.norm(vector) < 10 or handoff[2] <= grasp[2]:
            raise ValueError('invalid demonstrated transfer')
        native = next(p for p in action['automated_phases'] if p['phase']=='transfer_place')
        plan = json.loads(checked_file(native['actual_native_plan']).read_text())
        for phase in action['automated_phases']:
            for field in ('actual_native_plan','motion_source'):
                if phase.get(field):checked_file(phase[field])
        # Verify release exists in actual dispatch, not merely its source code.
        release = [op for op in plan if op['kind']=='gripper' and op['data']['action']=='release_partial']
        if len(release) != 1:
            raise ValueError('demonstration lacks actual partial release evidence')
        templates[edge['edge_id']] = dict(arm='left',axis=6,
            seed_start_mm=start.tolist(),seed_target_mm=target.tolist(),
            measured_grasp_mm_deg=grasp.tolist(),grasp_sample_time_s=held[-1]['t_s'],
            close_input_time_s=close['t_s'],lift_input_time_s=lift['t_s'],
            grasp_offset_mm=(grasp[:3]-start).tolist(),
            lift_above_surfaces_mm=float(handoff[2]-max(start[2],target[2])),
            approach_height_mm=manifest['parameters']['approach_height_mm'],
            place_offset_mm=manifest['parameters']['place_offset_mm'],
            clearance_mm=manifest['parameters']['clearance_mm'],
            release_delta_pulse=180.,release_settle_s=.5,
            reference_start_px=points['start']['pixel_xy'],reference_target_px=points['target']['pixel_xy'],
            provenance=action, authorization='operator requested replay of this experience graph',
            validation_status='EXPERIMENTAL_DEMONSTRATION_TEMPLATE_NOT_VALIDATED_POLICY')
    return graph, trajectory, templates


def instantiate_poses(template, start, target):
    start, target = np.asarray(start,float), np.asarray(target,float)
    seed_vector = np.asarray(template['seed_target_mm'])[:2]-np.asarray(template['seed_start_mm'])[:2]
    vector = target[:2]-start[:2]
    if not np.isfinite(np.r_[start,target]).all() or np.linalg.norm(vector)<10:
        raise ValueError('invalid current anchors')
    yaw = np.arctan2(vector[1],vector[0])-np.arctan2(seed_vector[1],seed_vector[0])
    rotate = Rotation.from_euler('z',yaw)
    offset = np.asarray(template['grasp_offset_mm'],float)
    xy = (rotate.as_matrix()[:2,:2] @ offset[:2]) * (np.linalg.norm(vector)/np.linalg.norm(seed_vector))
    grasp = np.r_[start[:2]+xy,start[2]+offset[2],
        (rotate * Rotation.from_euler('xyz',template['measured_grasp_mm_deg'][3:],degrees=True)).as_euler('xyz',degrees=True)]
    approach = grasp.copy(); approach[2]=start[2]+template['approach_height_mm']
    lifted = grasp.copy(); lifted[2]=max(start[2],target[2])+template['lift_above_surfaces_mm']
    placed = lifted.copy(); placed[:2]=target[:2]; placed[2]=target[2]+template['place_offset_mm']
    if lifted[2] < placed[2] or lifted[2] <= grasp[2]:
        raise ValueError('transferred lift height is insufficient')
    across=lifted.copy();across[:2]=target[:2]
    retreat=placed.copy();retreat[2]+=template['clearance_mm']
    return dict(approach=approach,grasp=grasp,lifted=lifted,across=across,placed=placed,retreat=retreat)


def prepare_motion(runtime, observation, template, anchors, folder):
    from cloth_agent.gamepad_collect import tcp_command, approach_program
    observation.validate_for(runtime.config,live=True,check_age=False)
    spread=runtime.config.limits['max_surface_spread_mm']
    start,target=[observation.sample(anchors[k],spread) for k in ('grasp_px','target_px')]
    poses=instantiate_poses(template,start,target)
    source=(approach_program(runtime,poses['approach'])+tcp_command(poses['grasp'])
            +'motion.gripper(arm="left", action="close")\n'
            +''.join(tcp_command(poses[k]) for k in ('lifted','across','placed'))
            +'motion.gripper(arm="left", action="open")\n'+tcp_command(poses['retreat']))
    initial={k:np.asarray(s['joints']) for k,s in runtime.coordinator.snapshots().items()}
    operations=runtime.plan(source,initial,preserve_command_boundaries=True)
    openings=[i for i,(kind,data) in enumerate(operations) if kind=='gripper' and data['action']=='open']
    if len(openings)!=2:raise ValueError('unexpected release sequence')
    i=openings[-1]
    operations[i]=('gripper',dict(operations[i][1],action='release_partial',
        opening_delta_pulse=template['release_delta_pulse'],settle_s=template['release_settle_s']))
    save(folder/'grounded_poses.json',{k:v.tolist() for k,v in poses.items()})
    (folder/'motion.py').write_text(source)
    return operations,initial


def run_cli(args, parser):
    unsupported=(args.online_growth or args.collect_only or args.recover_no_progress or args.growth_state
                 or args.max_fallbacks or args.repair_self_loop or args.retry_missed_grasp or args.debug_grasp_retry
                 or args.initial_node or args.grounding_templates)
    if unsupported:parser.error('Arm6 demonstration replay supports frozen graph matching and two-anchor grounding only')
    if args.max_steps<1 or not 0<args.match_threshold<=1:parser.error('invalid replay limits')
    output=args.output or args.graph.parent/('arm6_replay_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    runtime=None;live=None;result=dict(status='PREPARING',physical_actions=0,steps=[],robot='Arm6')
    previous=signal.getsignal(signal.SIGTERM)
    def terminate(*_):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,terminate)
    try:
        graph,trajectory,templates=compile_templates(args.graph)
        frozen(output/'graph_used.json',graph)
        frozen(output/'host_templates.json',dict(protocol='arm6-demonstration-template-v1',edges=templates))
        if not args.execute or args.dry_run:
            result.update(status='DRY_RUN_OK',compiled_edges=list(templates),robot_calls=0,model_calls=0)
            return result
        from .discovery_model import CodexMultimodal
        from .graph_policy_replay import PROMPTS
        from cloth_agent.gamepad_collect_runtime import CollectionRuntime
        from cloth_agent.gamepad_collect import TrajectoryRecorder
        requests=Requests(output/'model',CodexMultimodal(args.model,args.timeout_s),
            schemas={'node_match':MATCH,'anchors':ANCHORS},max_calls=args.max_calls,no_cache=args.no_cache,
            common='You match cloth observations and ground an existing Arm6 demonstration. Images are attached in image_ids order. Return only schema-valid JSON. Do not generate robot code.',
            prompts={'node_match':PROMPTS['node_match'],'anchors':
                'Identify the CURRENT image equivalents of the two reference clicked anchors. Reference grasp/target pixels and relative action semantics are supplied. Return only current grasp_px,target_px and confidence. Do not copy old pixels without visual correspondence. Null points/confidence 0 if ambiguous, hidden or unsupported. Do not redesign the action.'})
        match_graph=dict(graph,match_nodes=graph['nodes'],goal=dict(subgoal='demonstrated_cloth_manipulation'))
        from .replay_live_graph import create_live_graph
        live=create_live_graph(output,None if args.no_live_graph else args.graph_port)
        live.update(graph,phase='INITIAL_OBSERVATION',observations=trajectory['observations'])
        runtime=CollectionRuntime(args.config)
        graph_hash=file_hash(args.graph);pending=None;attempted=set();metrics={}
        before=runtime.observe(output/'initial_observation')
        for index in range(args.max_steps+1):
            if file_hash(args.graph)!=graph_hash:raise ValueError('input graph changed during replay')
            folder=output/f'step_{index:04d}';folder.mkdir()
            save(output/'status.json',dict(result,status='MATCHING',step=index))
            live.update(graph,phase='MATCHING')
            match=topology_match(match_graph,trajectory,before,requests,folder/'matching',metrics,
                                 pending=pending,threshold=args.match_threshold)
            save(folder/'node_match.json',match)
            result.update(current_node=match.get('node_id'),last_observation=str(before.image))
            live.update(graph,current=match.get('node_id'),phase='MATCHED',observation=str(before.image))
            print('ARM6 MATCH '+json.dumps(match,ensure_ascii=False),flush=True)
            if pending:
                save(output/f'step_{index-1:04d}'/'outcome.json',dict(
                    expected_node=pending['target_node'],matching=match,after=evidence_file(before.image)))
                if not reliable_match(match,graph,args.match_threshold) or match['node_id']!=pending['target_node']:
                    result['status']='OUTCOME_MISMATCH';break
            if not reliable_match(match,graph,args.match_threshold):
                result['status']=match['status'] if match['status']!='MATCH' else 'UNCERTAIN_MATCH';break
            node=match['node_id']
            if node in graph.get('terminal_node_ids', []):
                result['status']='GOAL_REACHED';break
            choices=[e for e in graph['edges'] if e['source_node']==node]
            if not choices:
                result['status']='NO_OUTGOING_EDGE';break
            if len(choices)!=1:result['status']='AMBIGUOUS_ACTION';break
            if index==args.max_steps:result['status']='STEP_LIMIT';break
            edge=choices[0];eid=edge['edge_id']
            if eid in attempted:result['status']='REPEATED_EDGE_BLOCKED';break
            template=templates[eid]
            live.update(graph,current=node,action=eid,target=edge['target_node'],phase='GROUNDING')
            save(output/'status.json',dict(result,status='GROUNDING',edge_id=eid))
            reference=trajectory['observations'][edge['seed_action']['source_observation']]
            anchors=requests.ask('anchors',dict(edge_id=eid,action_family=edge['action_family'],
                action_template=edge['action_template'],reference_start_px=template['reference_start_px'],
                reference_target_px=template['reference_target_px']),
                [('reference',checked_file(reference['rgb'])),('current',before.image)])
            from PIL import Image
            with Image.open(before.image) as im:validate_anchors(anchors,im.size)
            save(folder/'grounding.json',anchors)
            if anchors['confidence']<args.match_threshold or anchors['grasp_px'] is None or anchors['target_px'] is None:
                result['status']='GROUNDING_UNCERTAIN';break
            operations,initial=prepare_motion(runtime,before,template,anchors,folder)
            save(folder/'dispatch_intent.json',dict(edge_id=eid,source_node=node,target_node=edge['target_node'],
                policy_status='USER_AUTHORIZED_EXPERIMENTAL_REPLAY',before=evidence_file(before.image),
                compiled_template=evidence_file(output/'host_templates.json')))
            if file_hash(args.graph)!=graph_hash:raise ValueError('input graph changed before dispatch')
            live.update(graph,phase='EXECUTING')
            save(output/'status.json',dict(result,status='EXECUTING',edge_id=eid))
            recorder=TrajectoryRecorder(runtime,folder)
            try:
                recorder.start();recorder.stage('graph_replay');recorder.check()
                runtime.arrival(initial)
                execution=runtime._execute_operations(operations,folder)
                save(folder/'execution.json',execution)
                result['steps'].append(dict(edge_id=eid,execution=execution))
                if execution['status']!='EXECUTED':raise RuntimeError(f'edge {eid} failed: {execution}')
                recorder.check();result['physical_actions']+=1;attempted.add(eid)
                recorder.stage('return_perception');runtime.open_at_home_on_return=True
                try:runtime.move_to_perception(folder/'return_perception')
                finally:runtime.open_at_home_on_return=False
                recorder.check()
            finally:
                recorder.close()
            recorder.check()
            before=runtime.observe(folder/'after')
            pending=edge
            save(output/'status.json',dict(result,status='MATCHING_OUTCOME'))
        result['model_metrics']=requests.metrics()
        result['matching_metrics']=metrics
    except (Exception,KeyboardInterrupt) as exc:
        failure=capture_cli_failure(output,exc)
        result.update(failure)
        if runtime is not None:stop_after_failure(runtime,result)
    finally:
        signal.signal(signal.SIGTERM,previous)
        close_cli_runtime(runtime,result)
        if live is not None:
            live.update(graph,phase=result['status'],action=None)
            live.close()
        save(output/'result.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    return result
