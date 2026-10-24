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
from .discovery_schema import obj, S, N
from .graph_policy_matching import topology_match
from .graph_minimal_grounding import SCHEMA as ANCHORS, validate as validate_anchors
from .discovery_data import save, file_hash
from .replay_failures import capture_cli_failure, stop_after_failure, close_cli_runtime

PROTOCOL = 'arm6-traj-experience-graph-v1'
SUPPORTED_PROTOCOLS = (PROTOCOL, 'arm6-archive-graph-v1')
EDGE_CHOICE = obj(edge_id=S, confidence=N, evidence=S)

GARMENT_FRAME = '''Use a garment-relative frame consistently for recognition and anchor transfer.
Up points toward the neckline; down points toward the bottom hem. Left/right mean
the observer's left/right when viewing the garment front with neckline up, NOT the wearer's sides.
Infer this frame independently in current and reference RGBs from visible garment landmarks.
Account for in-plane rotation before comparing fold boundaries, overlap, or hem width.
Legacy descriptions saying image-left/right, vertical, upper/lower describe the reference image:
transfer the corresponding garment region to current; never interpret them as fixed current-image axes.
Rotation alone is not a state change. Compare actual cloth coverage and layer boundaries;
do not equate states just because both have a similar silhouette. Do not infer a desired action outcome.
If landmarks are occluded, use visible corresponding boundaries only when supported;
do not invent a neckline, hidden hem, or handedness. Report ambiguity when it affects the decision.
All returned grasp_px/target_px remain in ORIGINAL CURRENT image pixel coordinates.
No image rotation, mirroring, or coordinate transform is performed by the host.'''


def matching_graph(graph):
    return dict(graph, match_nodes=[n for n in graph['nodes'] if n['representative_observations']], goal=dict(
        subgoal='demonstrated_cloth_manipulation', side_convention=GARMENT_FRAME))


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


def compile_templates(graph_path, *, allow_partial=False):
    graph = load_frozen(graph_path)
    if graph.get('protocol') not in SUPPORTED_PROTOCOLS or graph.get('robot') != 'Arm6' or graph.get('camera') != 'A':
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
    templates = {}; blocked = {}
    for edge in graph['edges']:
        action = edge['seed_action']['exact_command']
        if action.get('arm') != 'left' or action.get('axis') != 6:
            raise ValueError('non-Arm6 source action')
        manifest = json.loads(checked_file(action['manifest']).read_text())
        validate_demonstration_authorization(action, manifest)
        states = jsonl(checked_file(action['measured_trajectory']))
        events = jsonl(checked_file(action['manual_input_trace']))
        checked_file(action['controller_events'])
        # Verify every evidence reference even when its motion format is unsupported.
        verify_action_evidence(action)
        reason = None
        if action.get('type') == 'ARM6_FREE_MANUAL_DEMONSTRATION':
            reason = 'free manual trajectory compiler is not implemented'
        target = next(n for n in graph['nodes'] if n['node_id'] == edge['target_node'])
        if not target['representative_observations']:
            reason = 'target has no after RGB; audit-only edge'
        if reason is None:
            try:
                templates[edge['edge_id']] = compile_demonstration_template(edge, action, manifest, states, events)
            except ValueError as exc:
                reason = str(exc)
        if reason is not None:
            if not allow_partial:
                raise ValueError(f'{edge["edge_id"]}: {reason}')
            blocked[edge['edge_id']] = reason
    if allow_partial:
        graph['replay_unavailable_edges'] = blocked
    return graph, trajectory, templates


def compile_demonstration_template(edge, action, manifest, states, events):
    if action.get('type') == 'ARM6_AUTOMATIC_PICK_PLACE_DEMONSTRATION':
        return compile_automatic_template(action, manifest)
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
    return dict(arm='left',axis=6,
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


def compile_automatic_template(action, manifest):
    """Extract the dispatched Cartesian sequence, not gamepad events or guessed poses."""
    phases=[p for p in action['automated_phases'] if p['phase']=='automatic_pick_place']
    if len(phases)!=1:raise ValueError('requires one automatic pick/place phase')
    phase=phases[0]
    if phase['result'].get('status')!='EXECUTED':raise ValueError('automatic phase not executed')
    plan=json.loads(checked_file(phase['actual_native_plan']).read_text())
    if len(plan)!=9 or [p['kind'] for p in plan]!=['gripper','move','move','gripper','move','move','move','gripper','move']:
        raise ValueError('unsupported automatic primitive sequence')
    if [plan[i]['data'].get('action') for i in (0,3,7)]!=['open','close','release_partial']:
        raise ValueError('automatic gripper sequence is not open/close/partial-release')
    if any(p['data'].get('arm')!='left' for p in plan):raise ValueError('non-Arm6 automatic operation')
    poses=np.asarray([plan[i]['data'].get('target_pose_mm_deg',[]) for i in (2,4,5,6,8)],float)
    if poses.shape!=(5,6) or not np.isfinite(poses).all():raise ValueError('missing dispatched Cartesian poses')
    grasp,lifted,across,placed,retreat=poses
    points={p['name']:p for p in action['selected_points']}
    start,target=[np.asarray(points[k]['world_xyz_mm'],float) for k in ('start','target')]
    if not np.isfinite(np.r_[start,target]).all() or np.linalg.norm(target[:2]-start[:2])<10:
        raise ValueError('invalid recorded anchors')
    if (not np.allclose(grasp[:2],lifted[:2],atol=.01) or
            not np.allclose(across[:2],target[:2],atol=.01) or
            not np.allclose(placed[:2],target[:2],atol=.01) or
            not np.allclose(retreat[:2],placed[:2],atol=.01) or
            abs(across[2]-lifted[2])>.01 or lifted[2]<=max(grasp[2],placed[2]) or retreat[2]<=placed[2]):
        raise ValueError('automatic path is not a vertical lift and horizontal transfer')
    orientations=Rotation.from_euler('xyz',poses[:,3:],degrees=True)
    if np.max((orientations[0].inv()*orientations).magnitude())>1e-5:
        raise ValueError('automatic orientation changes require a different template')
    release=plan[7]['data'];delta=float(release['opening_delta_pulse']);settle=float(release['settle_s'])
    if not np.isfinite([delta,settle]).all() or delta<=0 or settle<0:raise ValueError('invalid partial release')
    return dict(arm='left',axis=6,seed_start_mm=start.tolist(),seed_target_mm=target.tolist(),
        measured_grasp_mm_deg=grasp.tolist(),grasp_pose_source='executed_native_plan_target',
        grasp_offset_mm=(grasp[:3]-start).tolist(),
        lift_above_surfaces_mm=float(lifted[2]-max(start[2],target[2])),
        approach_height_mm=manifest['parameters']['approach_height_mm'],
        place_offset_mm=float(placed[2]-target[2]),clearance_mm=float(retreat[2]-placed[2]),
        release_delta_pulse=delta,release_settle_s=settle,
        reference_start_px=points['start']['pixel_xy'],reference_target_px=points['target']['pixel_xy'],
        provenance=action,authorization='operator requested replay of this experience graph',
        validation_status='EXPERIMENTAL_DEMONSTRATION_TEMPLATE_NOT_VALIDATED_POLICY')


def verify_action_evidence(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            checked_file(value)
        else:
            for item in value.values():verify_action_evidence(item)
    elif isinstance(value, list):
        for item in value:verify_action_evidence(item)


def executable_choices(graph, templates, node, attempted):
    outgoing = [e for e in graph['edges'] if e['source_node'] == node]
    supported = [e for e in outgoing if e['edge_id'] in templates]
    untried = [e for e in supported if e['edge_id'] not in attempted]
    # An observed no-progress self-loop is not a preferred route when a
    # demonstrated state-changing action is available.
    progress = [e for e in untried if e['target_node'] != node]
    return outgoing, supported, progress or untried


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
    expanded = getattr(args, 'arm6_expanded_graph', False)
    route_arg = getattr(args, 'arm6_route', None)
    review_enabled = getattr(args, 'arm6_review', False)
    live_start = getattr(args, 'arm6_live_start', False)
    if review_enabled and not route_arg:parser.error('--arm6-review requires --arm6-route')
    if live_start and not review_enabled:parser.error('--arm6-live-start requires --arm6-review and --arm6-route')
    unsupported=(args.online_growth or args.collect_only or args.recover_no_progress or args.growth_state
                 or args.max_fallbacks or args.repair_self_loop or args.retry_missed_grasp or args.debug_grasp_retry
                 or args.initial_node or args.grounding_templates)
    if unsupported:parser.error('Arm6 demonstration replay supports frozen graph matching and two-anchor grounding only')
    if args.max_steps<1 or not 0<args.match_threshold<=1:parser.error('invalid replay limits')
    output=args.output or args.graph.parent/('arm6_replay_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    runtime=None;live=None;retrieval=None;requests=None;metrics={}
    matcher_name=getattr(args,'matcher','current')
    result=dict(status='PREPARING',physical_actions=0,steps=[],robot='Arm6',matcher=matcher_name)
    from .arm6_replay_media import ReplayMedia
    media=ReplayMedia(output)
    previous=signal.getsignal(signal.SIGTERM)
    def terminate(*_):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,terminate)
    try:
        graph,trajectory,templates=compile_templates(args.graph, allow_partial=expanded)
        from .arm6_trial_review import fixed_route, review_step
        route=fixed_route(graph,templates,route_arg,args.max_steps)
        if route:
            media.sequence_video=True
            result.update(experiment='fixed_graph_route',planned_edges=[e['edge_id'] for e in route],
                          expected_start_node=route[0]['source_node'],goal_node=route[-1]['target_node'],
                          operator_assisted=review_enabled,operator_reviews=[])
        if review_enabled:media.data['title']='Arm6 operator-assisted graph route'
        if expanded:
            result['unavailable_edges'] = graph['replay_unavailable_edges']
        frozen(output/'graph_used.json',graph)
        frozen(output/'host_templates.json',dict(protocol='arm6-demonstration-template-v1',edges=templates))
        if matcher_name=='retrieval_gpt_rerank':
            from .retrieval_matcher import RetrievalMatcher,worker_python
            from .visual_retrieval_index import validate_memory
            memory=validate_memory(args.retrieval_memory,args.graph,graph,trajectory)
            frozen(output/'matcher_config.json',dict(mode=matcher_name,memory=evidence_file(args.retrieval_memory/'memory.json'),
                graph_sha256=memory['graph_sha256'],references=len(memory['images']),top_k=args.retrieval_top_k,
                worker_python=worker_python(args.retrieval_python),runtime_reference_encoding=False,
                candidate_policy='Top-K images grouped by actual graph node; one GPT invocation; no global fallback'))
        if not args.execute or args.dry_run:
            result.update(status='DRY_RUN_OK',compiled_edges=list(templates),
                          matchable_nodes=[n['node_id'] for n in matching_graph(graph)['match_nodes']],
                          robot_calls=0,model_calls=0)
            return result
        from .discovery_model import CodexMultimodal
        from .graph_policy_replay import PROMPTS
        from cloth_agent.gamepad_collect_runtime import CollectionRuntime
        from cloth_agent.gamepad_collect import TrajectoryRecorder
        requests=Requests(output/'model',CodexMultimodal(args.model,args.timeout_s),
            schemas={'node_match':MATCH,'anchors':ANCHORS,'edge_choice':EDGE_CHOICE},max_calls=args.max_calls,no_cache=args.no_cache,
            common='You match cloth observations and ground an existing Arm6 demonstration. Images are attached in image_ids order. Return only schema-valid JSON. Do not generate robot code.\n'+GARMENT_FRAME,
            prompts={'node_match':PROMPTS['node_match'],'edge_choice':
                'Choose ONE supplied supported outgoing demonstration for the current RGB. Compare current with each candidate source RGB and its grasp/target relations. Prefer compatible visible cloth regions and the demonstrated state change. Do not invent actions, infer success from graph order, or select an unsupported edge. If no single choice is supported, return empty edge_id with low confidence. Explain concrete visual evidence.',
                'anchors':
                'Identify the CURRENT image equivalents of the two reference clicked anchors. Reference grasp/target pixels and relative action semantics are supplied. Return only current grasp_px,target_px and confidence. Do not copy old pixels without visual correspondence. Null points/confidence 0 if ambiguous, hidden or unsupported. Do not redesign the action.'})
        match_graph=matching_graph(graph)
        if matcher_name=='retrieval_gpt_rerank':
            # Fail before any camera/controller connection if memory or ML worker is unavailable.
            retrieval=RetrievalMatcher(args.retrieval_memory,args.graph,graph,trajectory,output,
                python=args.retrieval_python,top_k=args.retrieval_top_k,timeout=args.retrieval_timeout_s)
        match_observation=retrieval.match if retrieval is not None else topology_match
        from .replay_live_graph import create_live_graph
        live=create_live_graph(output,None if args.no_live_graph else args.graph_port)
        live.update(graph,phase='INITIAL_OBSERVATION',observations=trajectory['observations'])
        runtime=CollectionRuntime(args.config)
        graph_hash=file_hash(args.graph);pending=None;attempted=set()
        before=runtime.observe(output/'initial_observation')
        initial_match=None
        if live_start:
            from .arm6_live_start import prepare_live_start
            reference=trajectory['observations'][route[0]['seed_action']['source_observation']]
            save(output/'status.json',dict(result,status='LIVE_START_PLACEMENT'))
            live.update(graph,phase='LIVE_START_PLACEMENT')
            before,initial_match=prepare_live_start(runtime,checked_file(reference['rgb']),
                lambda obs,folder:match_observation(match_graph,trajectory,obs,requests,folder,metrics,
                                                    threshold=args.match_threshold),
                graph,route[0]['source_node'],args.match_threshold,output/'start_placement')
            result['start_placement']=str(output/'start_placement/accepted.json')
        for index in range(args.max_steps+1):
            if file_hash(args.graph)!=graph_hash:raise ValueError('input graph changed during replay')
            folder=output/f'step_{index:04d}';folder.mkdir()
            media.begin(index, before.image)
            save(output/'status.json',dict(result,status='MATCHING',step=index))
            live.update(graph,phase='MATCHING')
            if index==0 and initial_match is not None:
                match=initial_match
            else:
                match=match_observation(match_graph,trajectory,before,requests,folder/'matching',metrics,
                                     pending=None if expanded else pending,threshold=args.match_threshold)
            media.phase('selection')
            save(folder/'node_match.json',match)
            result.update(current_node=match.get('node_id'),last_observation=str(before.image))
            live.update(graph,current=match.get('node_id'),phase='MATCHED',observation=str(before.image))
            print('ARM6 MATCH '+json.dumps(match,ensure_ascii=False),flush=True)
            expected=(route[index]['source_node'] if index<len(route) else route[-1]['target_node']) if route else None
            if pending:
                save(output/f'step_{index-1:04d}'/'outcome.json',dict(
                    expected_node=pending['target_node'],matching=match,after=evidence_file(before.image)))
                if not review_enabled and (not reliable_match(match,graph,args.match_threshold)
                        or (not expanded and match['node_id']!=pending['target_node'])):
                    result['status']='OUTCOME_MISMATCH';break
            if not review_enabled and not reliable_match(match,graph,args.match_threshold):
                result['status']=match['status'] if match['status']!='MATCH' else 'UNCERTAIN_MATCH';break
            if route and not review_enabled and match['node_id']!=expected:
                result.update(status='ROUTE_STATE_MISMATCH',expected_node=expected);break
            node=expected if route else match['node_id']
            if route and index==len(route):
                if review_enabled:
                    goal=next(n for n in graph['nodes'] if n['node_id']==node)
                    media.phase('operator_review')
                    review_step(before.image,checked_file(trajectory['observations'][goal['representative_observations'][-1]]['rgb']),
                                None,None,node,match,folder/'goal_review',cancel=runtime.cancel)
                    result['operator_reviews'].append(str(folder/'goal_review/operator_review.json'))
                result.update(status='GOAL_REACHED',current_node=node,
                              completion_basis='operator_confirmed' if review_enabled else 'model_match');break
            if not route and node in graph.get('terminal_node_ids', []):
                result['status']='GOAL_REACHED';break
            if route:
                choices=[route[index]]
            elif expanded:
                outgoing,supported,choices=executable_choices(graph,templates,node,attempted)
                if not supported and outgoing:
                    result.update(status='ACTION_TEMPLATE_UNAVAILABLE',
                                  blocked_outgoing_edges={e['edge_id']:graph['replay_unavailable_edges'][e['edge_id']] for e in outgoing})
                    break
                if supported and not choices:
                    result['status']='REPEATED_EDGE_BLOCKED';break
            else:
                choices=[e for e in graph['edges'] if e['source_node']==node]
            if not choices:
                result['status']='NO_OUTGOING_EDGE';break
            if index==args.max_steps:result['status']='STEP_LIMIT';break
            if len(choices)>1:
                if not expanded:result['status']='AMBIGUOUS_ACTION';break
                candidates=[];images=[('current',before.image)]
                for candidate in choices:
                    oid=candidate['seed_action']['source_observation']
                    candidates.append(dict(edge_id=candidate['edge_id'],reference_image=candidate['edge_id'],
                        action_family=candidate['action_family'],action_template=candidate['action_template']))
                    images.append((candidate['edge_id'],checked_file(trajectory['observations'][oid]['rgb'])))
                choice=requests.ask('edge_choice',dict(candidates=candidates),images)
                save(folder/'edge_choice.json',choice)
                choices=[e for e in choices if e['edge_id']==choice['edge_id']]
                if choice['confidence']<args.match_threshold or len(choices)!=1:
                    result['status']='AMBIGUOUS_ACTION';break
            edge=choices[0];eid=edge['edge_id']
            if eid in attempted:result['status']='REPEATED_EDGE_BLOCKED';break
            media.edge(eid)
            media.phase('grounding')
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
            if review_enabled:
                media.phase('operator_review')
                anchors=review_step(before.image,checked_file(reference['rgb']),anchors,template,node,match,
                                    folder/'anchor_review',cancel=runtime.cancel)
                result['operator_reviews'].append(str(folder/'anchor_review/operator_review.json'))
                result['current_node']=node
                save(folder/'grounding_operator_selected.json',anchors)
            elif anchors['confidence']<args.match_threshold or anchors['grasp_px'] is None or anchors['target_px'] is None:
                result['status']='GROUNDING_UNCERTAIN';break
            if file_hash(args.graph)!=graph_hash:raise ValueError('input graph changed before depth recovery')
            def reground_view(fresh,attempt):
                # Changing camera pose invalidates both old pixel anchors. Recheck
                # recognition before reusing the selected action on the new view.
                rematch=match_observation(match_graph,trajectory,fresh,requests,attempt/'matching',metrics,
                                       threshold=args.match_threshold)
                save(attempt/'node_match.json',rematch)
                if not review_enabled and (not reliable_match(rematch,graph,args.match_threshold) or rematch['node_id']!=node):
                    raise ValueError('depth recovery view does not reliably match the original source state')
                value=requests.ask('anchors',dict(edge_id=eid,action_family=edge['action_family'],
                    action_template=edge['action_template'],reference_start_px=template['reference_start_px'],
                    reference_target_px=template['reference_target_px']),
                    [('reference',checked_file(reference['rgb'])),('current',fresh.image)])
                with Image.open(fresh.image) as im:validate_anchors(value,im.size)
                save(attempt/'grounding.json',value)
                if review_enabled:
                    value=review_step(fresh.image,checked_file(reference['rgb']),value,template,node,rematch,
                                      attempt/'anchor_review',cancel=runtime.cancel)
                    result['operator_reviews'].append(str(attempt/'anchor_review/operator_review.json'))
                    save(attempt/'grounding_operator_selected.json',value)
                elif value['confidence']<args.match_threshold or value['grasp_px'] is None or value['target_px'] is None:
                    raise ValueError('depth recovery grounding is uncertain')
                if file_hash(args.graph)!=graph_hash:raise ValueError('input graph changed during depth recovery')
                return value
            from .arm6_depth_recovery import prepare_with_view_retry
            media.phase('planning_and_depth_recovery')
            original_before=before
            before,new_anchors,(operations,initial)=prepare_with_view_retry(
                runtime,before,anchors,folder/'depth_recovery',
                lambda observation,points:prepare_motion(runtime,observation,template,points,folder),reground_view)
            if before is not original_before:
                save(folder/'grounding_before_view_retry.json',anchors)
                anchors=new_anchors;save(folder/'grounding.json',anchors)
                result['last_observation']=str(before.image)
                result.setdefault('depth_recoveries',[]).append(str(folder/'depth_recovery/result.json'))
            media.snapshot('action_before', before.image)
            save(folder/'dispatch_intent.json',dict(edge_id=eid,source_node=node,target_node=edge['target_node'],
                policy_status='USER_AUTHORIZED_EXPERIMENTAL_REPLAY',before=evidence_file(before.image),
                compiled_template=evidence_file(output/'host_templates.json'),operator_assisted=review_enabled))
            if file_hash(args.graph)!=graph_hash:raise ValueError('input graph changed before dispatch')
            live.update(graph,phase='EXECUTING')
            save(output/'status.json',dict(result,status='EXECUTING',edge_id=eid))
            media.phase('execution')
            recorder=TrajectoryRecorder(runtime,folder)
            try:
                recorder.start();recorder.stage('graph_replay');recorder.check()
                runtime.arrival(initial)
                execution=runtime._execute_operations(operations,folder)
                save(folder/'execution.json',execution)
                result['steps'].append(dict(edge_id=eid,execution=execution))
                if execution['status']!='EXECUTED':raise RuntimeError(f'edge {eid} failed: {execution}')
                recorder.check();result['physical_actions']+=1;attempted.add(eid)
                media.phase('return_perception')
                recorder.stage('return_perception');runtime.open_at_home_on_return=True
                try:runtime.move_to_perception(folder/'return_perception')
                finally:runtime.open_at_home_on_return=False
                recorder.check()
            finally:
                recorder.close()
            recorder.check()
            media.phase('end_observation')
            before=runtime.observe(folder/'after')
            media.snapshot('after', before.image)
            media.phase(None)
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
        if retrieval is not None:
            try:retrieval.close()
            except Exception as exc:result.setdefault('cleanup_errors',[]).append(dict(resource='retrieval_worker',error=str(exc)))
        if requests is not None:
            result['model_metrics']=requests.metrics()
            result['matching_metrics']=metrics
        if live is not None:
            live.update(graph,phase=result['status'],action=None)
            live.close()
        try:
            video=media.finish(result['status'])
            result['timing_file']=str(output/'timing.json')
            if video:result['video']=video
            if media.data.get('sequence_video'):result['sequence_video']=media.data['sequence_video']
            if media.data.get('sequence_video_error'):result['sequence_video_error']=media.data['sequence_video_error']
        except Exception as exc:
            result['media_error']=f'{type(exc).__name__}: {exc}'
        save(output/'result.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    return result
