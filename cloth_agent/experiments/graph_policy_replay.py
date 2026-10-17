"""Frozen Graph-as-Policy. Default dry-run is offline; hardware needs --execute.

Normal graph mode only recognizes nodes and grounds a host-selected edge.
CLI defaults to one grasp-handle recovery on NO_ROBUST_ACTION, without graph extension.
Legacy extending fallback is opt-in and cannot be combined with grasp-handle recovery.
--online-growth enables persistent source/action -> multiple outcomes, with free planning
only at provisional frontiers and reusable templates after visually supported success.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal

import jsonschema

from .discovery_data import digest, file_hash, save
from .replay_failures import (record_controller_failure,stop_after_failure,
                              capture_cli_failure,close_cli_runtime,cli_failed)
from .graph_policy import (ROOT, Requests, checked_file, command, evidence_file,
                           execute_recorded, frozen, instantiate, load_frozen,
                           observation_geometry, reliable_match, validate_replay_graph)
from .graph_policy_schema import MATCH, SLOT, VECTOR2, NODE, TEMPLATE
from .graph_policy_recovery import Recovery, DESCRIBE_PROMPT, free_plan_one, validate_extension
from .graph_policy_matching import signatures, topology_match
from .discovery_schema import S, IDS, obj, array, enum

from . import missed_grasp_retry

PROTOCOL = 'frozen-graph-as-policy-v1'
EMPTY_PROTOCOL = 'empty-online-folding-base-v1'


def create_empty_base(path):
    """Task specification only: no prior observations, nodes, actions or outcomes."""
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    base = dict(protocol=EMPTY_PROTOCOL, initialization='FROM_SCRATCH',
        nodes=[], match_nodes=[], edges=[], terminal_node_ids=[], goal_references=[],
        grasp_offset_mm=Arm7Runtime.grasp_offset_mm,
        goal=dict(subgoal='full_fold',
            objective='Fold the T-shirt: sleeves and both torso sides inward, then hem toward collar.',
            side_convention='Mentally rotate collar above hem without mirroring; left/right are viewer sides.',
            completion_criterion='Both sleeves and torso sides must be contained and the hem folded toward the collar into a compact folded garment. An intermediate sleeve or torso fold is not completion. These are task goals, not predefined nodes.'))
    frozen(Path(path),base)
    return Path(path)


def validate_growth_base(graph):
    if graph.get('protocol') != EMPTY_PROTOCOL:
        return validate_policy(graph)
    if (graph.get('initialization') != 'FROM_SCRATCH'
            or any(graph.get(k) != [] for k in ('nodes','match_nodes','edges','terminal_node_ids','goal_references'))
            or not isinstance(graph.get('goal'),dict)
            or not graph['goal'].get('completion_criterion')
            or not isinstance(graph.get('grasp_offset_mm'),(float,int))
            or not 0 <= graph['grasp_offset_mm'] <= 30):
        raise ValueError('invalid empty online graph base')
    return dict(observations={},steps=[])

# Negative extra lowering raises the grasp; shared Camera B correction is separate.
RIGHT_SIDE_EXTRA_MM = 0.0
RIGHT_SLEEVE_EXTRA_MM = 0.0
from cloth_agent.action_robustness import GROUNDING_PROMPT, SCHEMA as ROBUSTNESS_SCHEMA, strip_grounding_evidence
GROUNDING = obj(status=enum('READY', 'BLOCKED', 'NEED_MORE_INFORMATION'),
    edge_id=S, action_family=S, slots=array(SLOT), explanation=S,
    visual_violations=array(obj(constraint=S, current_visual_evidence=S)),
    execution_risk=IDS, missing_visual_parameters=IDS,
    garment_frame=obj(collar_pixel_xy=VECTOR2, hem_center_pixel_xy=VECTOR2, visual_evidence=S))
GROUNDING['properties']['robustness'] = ROBUSTNESS_SCHEMA
HEM_FRAME = obj(status=enum('RESOLVED', 'UNKNOWN'),
    collar_pixel_xy=VECTOR2, hem_center_pixel_xy=VECTOR2,
    collar_evidence=S, hem_evidence=S, correspondence_evidence=S)

SCHEMAS = {'missed_grasp_review': missed_grasp_retry.SCHEMA, 'node_match': MATCH, 'ground_edge': GROUNDING, 'recovery_node': NODE, 'hem_frame': HEM_FRAME}
OWNERSHIP = dict(version='constraint-ownership-v2',
    gpt='Visible edge semantics and current image anchors; fixed action family and primitive sequence.',
    host=['RGB-D sampling', 'IK', 'workspace', 'robot/camera collision', 'trajectory clearance'],
    execution_risk=['cloth coupling', 'torso displacement', 'hidden layer relations'],
    rule='Unproven physical effects are risks, not visual violations. READY is not safety approval.')
PROMPTS = {
    'missed_grasp_review': missed_grasp_retry.PROMPT,
    'hem_frame': """Identify collar and hem in CURRENT RGB only. This is orientation verification,
not manipulation planning. Earlier RGBs are visual context, not coordinate references.
Use the earliest unfolded seed image and available current-run earlier views to distinguish the
neck opening, shoulder/sleeve attachments and hem seam. Track correspondence by visible garment
structure; never assume the garment kept the same pixel orientation between runs or images.
Image-top is NOT necessarily the collar. A white label is NOT proof of a hem; it may be a neck label.
Return current-image coordinates and separate evidence for each end, plus the structural
correspondence to earlier views. If either identity or correspondence cannot be established,
return UNKNOWN; do not guess. No action parameters or desired motion are supplied.""",
    'recovery_node': DESCRIBE_PROMPT,
    'node_match': '''Directly compare the current RGB with the attached representative RGBs for EVERY candidate node.
Use representative_observations to map image IDs to nodes. Images are primary evidence; text only
clarifies the decision conditions. Cite visible layer/boundary correspondences or differences
between current and reference images, rather than merely repeating node descriptions.
Recognize ONLY the current RGB against the supplied fixed visual node conditions
and short visual signatures. Every candidate has attached representative RGBs.
The shortlist is not proof of membership: return NO_MATCH
if none fits. No transition or action information is provided. Do not infer an expected
node from numbering, chronology, or task order. Do not answer what to do next. MATCH requires a
single supported node with positive visible evidence and no violated or unresolved critical condition.
Match decision conditions for action reuse, not merely similar color, silhouette or texture.
For status MATCH, violated_conditions and unresolved_conditions refer ONLY to the selected node_id.
Do NOT put reasons for rejecting OTHER candidate nodes in either field. Such reference comparisons
belong in matched_evidence. If a critical condition of the selected node is actually violated or
unresolved, return NO_MATCH or AMBIGUOUS instead of MATCH. A valid MATCH has both arrays empty.
Generic hidden-layer caveats are not evidence of a mismatch; report any uncertainty that actually
prevents recognition. NO_MATCH and AMBIGUOUS must use empty node_id. Never force classification.
Return only the supplied recognition schema. Do not add nodes or alter conditions.''',
    'ground_edge': '''The host has already selected EXACTLY ONE immutable edge. Ground only that
edge in current RGB: no edge choice, no next-step strategy, no free planning.
Constraint ownership applies even when the frozen template mixes these responsibilities:
- You verify only visually observable semantic conditions and anchor correspondence.
- Existing HOST validators own IK, workspace, robot/camera collision and trajectory clearance.
  Do not claim those checks passed, or BLOCK merely because RGB cannot prove them.
- Cloth coupling, whether the torso will be dragged, hidden layer relations and other unobservable
  future effects belong in execution_risk. Uncertainty alone is NOT a visual violation and must
  NOT cause BLOCKED. Do not assert absence of coupling or invent hidden-layer evidence.
- BLOCKED requires current visible evidence of an actual edge-condition violation. Cite the exact
  constraint and current_visual_evidence in visual_violations. READY requires no visual violations.
- If a required visible anchor/parameter cannot be localized at all, use NEED_MORE_INFORMATION
  with missing_visual_parameters, without fabricating coordinates. This is not a prediction of
  physical failure. Do not put host checks or unobservable future effects in missing_visual_parameters.
READY means a semantic candidate may proceed to Host validation, never execution authorization.
Use the supplied task.side_convention for ALL left/right relations, never raw image-left/right.
Localize the collar center and hem center in garment_frame with visible evidence. Mentally rotate
collar above hem WITHOUT mirroring: if collar is below hem in the image, garment-frame viewer-left
is image-right. The sleeve grasp must lie on the specified side of this collar-to-hem axis.
Missing reliable garment orientation belongs in missing_visual_parameters; do not guess it.
For bottom_hem_toward_collar_fold, use the separately verified hem_orientation anchors.
Do not reverse collar and hem or infer them from image-top/bottom or a tag alone. If the
verified identities conflict with current visual evidence, return NEED_MORE_INFORMATION.
Copy the selected edge_id and action_family exactly. Supply
one slot for every supplied primitive, in identical order, without adding or removing primitives.
reference_orientation supplies the recorded world RPY and, when available, the original
pre-action RGB and camera geometry. Use these to preserve gripper-to-garment orientation in
the CURRENT image; reference pixels/XYZ are not current action targets and do not prove success.
Use current-image pixel anchors (x,y), surface-relative heights in mm and world RPY in radians.
The host measures RGB-D at the anchors. Grasp height MUST be zero: the existing arm7 lowering is
applied ONCE by the host. The supplied grasp_offset_mm is the authoritative current Host offset,
including any explicit per-edge user override of the historical template offset. Never subtract it
from your grasp height or add it again. Propose heights, yaw and anchors consistent with the supplied template;
Host must validate their geometry before dispatch. Explain each visible correspondence concisely;
record nonobservable effects as execution_risk rather than
redesigning the action. The host may apply the SAME parallel-jaw local-Z half-turn to all held poses,
preserving relative world rotation. Never add a half-turn or copy old absolute coordinates blindly.''',
}
COMMON = ('Return only JSON in the supplied schema. Images are actual multimodal attachments in '
          'image_ids order. Give concise visible evidence and uncertainty, not private reasoning.')
PROMPTS['ground_edge'] += GROUNDING_PROMPT


GROWTH_PROTOCOL = 'online-multi-outcome-v1'
SCHEMAS['growth_effect'] = obj(progress=enum('PROGRESS', 'NO_PROGRESS', 'REGRESSION', 'UNKNOWN', 'COMPLETE'),
                               task_progress=S, evidence=IDS)
SCHEMAS['growth_action'] = obj(action_family=S, action_template=TEMPLATE)
PROMPTS['growth_effect'] = """Compare before/after cloth observations. Report decision-relevant folding
progress and concise visible evidence. EXECUTED is not progress. Translation/wrinkles alone are
NO_PROGRESS. Unknown visibility is UNKNOWN. COMPLETE requires the full supplied task, not one fold.
Do not create states, select actions or predict success."""
PROMPTS['growth_action'] = """Describe a reusable RELATIVE action template for the exact already executed
frontier command and its before/after RGBs. Do not plan a new action, change the primitive sequence,
copy absolute coordinates, or split action identity by outcome. Supply garment-relative grasp/target
relations, the executed grasp/transport height strategy, orientation rule, and constraints.
Preserve the command's chosen TCP heights as evidence; do not replace its grasp strategy with a mandatory surface-height rule.
The template is a one-shot hypothesis, not a validated policy. Only describe the supplied action."""

def policy_requests(output, backend, *, grounding_mode='full', grounding_templates=None, **kwargs):
    from .graph_minimal_grounding import SCHEMA, PROMPT, MinimalGrounder
    if grounding_mode not in ('full','minimal'):raise ValueError('unknown grounding mode')
    from .edge_applicability import SCHEMAS as AS, PROMPTS as AP
    from .recapture_consistency import SCHEMA as PAIR_SCHEMA, PROMPT as PAIR_PROMPT
    requests = Requests(output, backend, schemas={**SCHEMAS,**AS,'node_match_pair':PAIR_SCHEMA,'ground_edge_minimal':SCHEMA},
                    prompts={**PROMPTS,**AP,'node_match_pair':PAIR_PROMPT,'ground_edge_minimal':PROMPT},
                    common=COMMON, protocol=PROTOCOL, **kwargs)
    requests.grounding_mode=grounding_mode
    requests.minimal_grounder=None
    if grounding_mode=='minimal':
        if grounding_templates is None:raise ValueError('minimal mode requires compiled Host templates')
        requests.minimal_grounder=MinimalGrounder(grounding_templates)
    return requests


def instantiate_grounded(candidate,edge,before):
    if candidate.get('coordinate_frame')=='ACTION_LOCAL':
        from .action_local_frame import instantiate_local
        return instantiate_local(candidate,edge,before)
    return instantiate(primitive_candidate(candidate),[edge],before)


def grounded_side_check(candidate,edge,before):
    if candidate.get('coordinate_frame')=='ACTION_LOCAL':
        return dict(status='NOT_INDEPENDENTLY_EVALUATED',reason='Action-local mode has no collar/hem frame; semantic side/direction is entrusted to selected-edge grounding.')
    return validate_sleeve_side(candidate,edge,before)


def selected_edge_grounding(requests,graph,edge,before,folder,hem_frame=None):
    """Both replay runners share opt-in grounding; downstream validators are unchanged."""
    minimal=getattr(requests,'minimal_grounder',None)
    if minimal is not None:return minimal.ground(requests,edge,before,folder)
    payload=grounding_payload(graph,edge,before)
    if hem_frame is not None:payload['hem_orientation']=hem_frame
    images=[('current',before.image)]
    reference=grounding_reference_image(graph,edge)
    if reference is not None:images.append(('action_reference',reference))
    value=requests.ask('ground_edge',payload,images,validator=validate_grounding)
    value=strip_grounding_evidence(value,folder)
    from .graph_minimal_grounding import NoRobustAction
    reasons=[value.get('explanation',''),*value.get('missing_visual_parameters',[]),
             *[v['current_visual_evidence'] for v in value.get('visual_violations',[])]]
    if value['status']!='READY' and any('NO_ROBUST_ACTION' in reason for reason in reasons):
        save(Path(folder)/'grounding.json',value)
        raise NoRobustAction([r for r in reasons if r],before)
    return value


def grounding_with_recapture(runtime, requests, graph, trajectory, edge, before, folder, metrics, threshold, hem_frame=None):
    """One fresh-image retry only for uncertain minimal anchors, never safety failures."""
    from .graph_minimal_grounding import GroundingUncertain, NoRobustAction
    try:
        return selected_edge_grounding(requests,graph,edge,before,folder,hem_frame),before
    except NoRobustAction:
        raise  # Never recapture/re-ground a grasp-gate rejection before recovery.
    except GroundingUncertain as exc:
        retry=Path(folder)/'grounding_retry_001';retry.mkdir()
        save(retry/'trigger.json',dict(reason=str(exc),edge_id=edge['edge_id'],
            original_rgb=evidence_file(before.image),max_retries=1,confidence_threshold_unchanged=True))
        print('Grounding 置信度不足：重新拍图、核对当前节点，再尝试一次同一 edge。',flush=True)
        fresh=runtime.observe(retry/'observation')
        old_no_cache=getattr(requests,'no_cache',False)
        requests.no_cache=True
        try:
            prior_path=Path(folder)/'matching_attempts.json'
            prior=json.loads(prior_path.read_text()) if prior_path.exists() else {}
            prior_attempts=prior.get('attempts',[])
            candidates=prior_attempts[-1]['candidate_nodes'] if prior_attempts else list(dict.fromkeys([edge['source_node'],edge['target_node']]))
            match=topology_match(graph,trajectory,fresh,requests,retry,metrics,threshold=threshold,
                                 candidate_override=candidates)
            save(retry/'node_match.json',dict(result=match,current_rgb=evidence_file(fresh.image)))
            if not reliable_match(match,graph,threshold) or match['node_id']!=edge['source_node']:
                from .recapture_consistency import review_pair,ConfirmedSourceChange,RecaptureSourceUnmatched
                review=review_pair(graph,trajectory,before,fresh,requests,retry,metrics,candidates,threshold)
                confirmed=review['current_match']
                if confirmed.get('status')=='NO_MATCH':
                    raise RecaptureSourceUnmatched(confirmed,fresh,review,candidates)
                if review['observation_relation']!='UNKNOWN' and reliable_match(confirmed,graph,threshold):
                    if confirmed['node_id']!=edge['source_node']:
                        raise ConfirmedSourceChange(confirmed,fresh,review,candidates)
                    match=confirmed
                    save(retry/'confirmed_node_match.json',dict(result=match,review='joint_image_review.json'))
                else:
                    error=ValueError('joint image review could not resolve the source-node conflict')
                    error.recapture_match=confirmed
                    raise error
            candidate=selected_edge_grounding(requests,graph,edge,fresh,retry,hem_frame)
            save(retry/'result.json',dict(status='READY',edge_id=edge['edge_id'],current_rgb=evidence_file(fresh.image)))
            metrics['grounding_recapture_successes']=metrics.get('grounding_recapture_successes',0)+1
            return candidate,fresh
        except Exception as error:
            if isinstance(error,GroundingUncertain):error.observation=fresh
            save(retry/'result.json',dict(status='STOP',error=f'{type(error).__name__}: {error}'))
            raise
        finally:
            requests.no_cache=old_no_cache
            metrics['grounding_recapture_attempts']=metrics.get('grounding_recapture_attempts',0)+1


def consolidate(source_path, output):
    """Explicit user review of this seed's existing merged states; no model/taxonomy discovery."""
    source_path = Path(source_path).resolve()
    source = load_frozen(source_path)
    trajectory = load_frozen(source['seed_trajectory_path'])
    validate_replay_graph(source, trajectory)
    expected = {f'e{i:03d}' for i in range(14)}
    if {e['edge_id'] for e in source['edges']} != expected:
        raise ValueError('this explicit review requires the complete e000..e013 experience graph')
    mapping = source['node_mapping']
    if not (mapping['n006'] == mapping['n007'] == mapping['n008'] == 'S6'
            and mapping['n009'] == mapping['n010'] == 'S7'):
        raise ValueError('expected previously reviewed visual consolidation is absent')
    graph = copy.deepcopy(source)
    graph.update(protocol=PROTOCOL, graph_id='policy_graph_v2', terminal_node_ids=['S10'],
                 policy_provenance=dict(authority='USER', source_graph=str(source_path),
                    source_sha256=file_hash(source_path),
                    review='e001 inward progress; S6/S7 failed self-loops excluded; '
                           'e008/e010 preferred; user authorizes e012 to S10 terminal; e013 excluded',
                    terminal_review='S10_ENDPOINT_E012_AUTHORIZED'))
    # Record precisely the original text that is omitted from the visual-only view.
    omitted = 'Current exposed edges must be localized separately for each outgoing action.'
    graph['matcher_projection_omissions'] = [omitted]
    graph['match_nodes'] = []
    for node in graph['nodes']:
        if not node['representative_observations']:
            continue
        n = copy.deepcopy(node)
        n['observation_condition']['uncertainty_conditions'] = [
            s for s in n['observation_condition']['uncertainty_conditions'] if s != omitted]
        graph['match_nodes'].append(n)
    for edge in graph['edges']:
        eid = edge['edge_id']
        edge.update(replayable=True, preference='PREFERRED', policy_outcome=edge['seed_outcome'],
                    evidence_status='ONE_SHOT_HYPOTHESIS', stop_reason=None)
        if eid == 'e001':
            edge.update(seed_outcome='PROGRESS', policy_outcome='PROGRESS',
                        generalization_status='ONE_SHOT_HYPOTHESIS')
            constraints = edge['action_template']['other_constraints']
            edge['action_template']['other_constraints'] = [
                'Curled inward sleeve is accepted intermediate progress S2, not final completion.'
                if s.startswith('Retain this negative result:') else s for s in constraints]
        if eid in {'e006', 'e007', 'e009'}:
            edge.update(replayable=False, preference='EXCLUDED', policy_outcome='NO_CHANGE',
                        evidence_status='FAILED_EVIDENCE')
        if eid == 'e012':
            # Execution is authorized, but the historical uncertain outcome remains intact.
            edge.update(policy_outcome='PROGRESS', evidence_status='ONE_SHOT_HYPOTHESIS')
        if eid == 'e013':
            edge.update(replayable=False, preference='EXCLUDED',
                        evidence_status='UNVERIFIED_OUTCOME' if eid == 'e012' else 'MISSING_AFTER',
                        stop_reason='REQUIRES_FRESH_VALIDATION')
    # S10 is the reviewed endpoint; S11 remains historical, never replayable.
    graph['replay_policy'] = dict(frozen=True, host_unique_preferred=True,
                                  expected_target_requires_fresh_match=True,
                                  fallback_allowed=False, edge_reuse='ONCE_PER_RUN')
    validate_policy(graph)
    frozen(Path(output), graph)
    return graph


def validate_policy(graph):
    from .graph_template_compiler import PROTOCOL as COMPILED_PROTOCOL, validate_compiled_policy
    if graph.get('protocol') == COMPILED_PROTOCOL:
        return validate_compiled_policy(graph)
    if 'recovery_extension' in graph:
        return validate_extension(graph, validate_policy)
    if graph.get('protocol') != PROTOCOL:
        raise ValueError('not a frozen Graph-as-Policy artifact')
    provenance = graph['policy_provenance']
    source = load_frozen(provenance['source_graph'])
    if provenance.get('authority') != 'USER' or file_hash(provenance['source_graph']) != provenance['source_sha256']:
        raise ValueError('policy provenance mismatch')
    trajectory = load_frozen(graph['seed_trajectory_path'])
    if digest(trajectory) != graph['seed_trajectory_hash'] or graph['seed_trajectory_hash'] != source['seed_trajectory_hash']:
        raise ValueError('seed provenance mismatch')
    if graph['nodes'] != source['nodes'] or graph['node_mapping'] != source['node_mapping']:
        raise ValueError('policy cannot redefine experience states')
    original = {e['edge_id']: e for e in source['edges']}
    if len(graph['edges']) != len(original) or {e['edge_id'] for e in graph['edges']} != set(original):
        raise ValueError('policy must preserve every experience edge')
    observed = {n['node_id'] for n in graph['nodes'] if n['representative_observations']}
    for edge in graph['edges']:
        old = original[edge['edge_id']]
        for key in ('source_node', 'target_node', 'action_family', 'seed_action', 'seed_step_id'):
            if edge[key] != old[key]:
                raise ValueError('policy changed original action or endpoints')
        if not isinstance(edge.get('replayable'), bool) or edge.get('preference') not in {'PREFERRED', 'EXCLUDED'}:
            raise ValueError('invalid policy eligibility')
        if edge['replayable']:
            if (edge['preference'] != 'PREFERRED' or edge['source_node'] not in observed
                    or edge['target_node'] not in observed or not (edge['seed_outcome'] in {'SUCCESS', 'PROGRESS'}
                        or (edge['edge_id'] == 'e012' and edge['source_node'] == 'S9'
                            and edge['target_node'] == 'S10' and graph.get('terminal_node_ids') == ['S10']
                            and provenance.get('terminal_review') == 'S10_ENDPOINT_E012_AUTHORIZED'
                            and edge['policy_outcome'] == 'PROGRESS'))
                    or edge['evidence_status'] != 'ONE_SHOT_HYPOTHESIS'):
                raise ValueError('unverified/failed/unobserved edge cannot replay')
            step = next(s for s in trajectory['steps'] if s['step_id'] == edge['seed_step_id'])
            if step['actual_execution'].get('status') != 'EXECUTED' or not step.get('after_observation'):
                raise ValueError('edge requires completed execution and after observation')
            ops = edge['seed_action']['exact_command']['operations']
            if (len(ops) < 2 or ops[0]['method'] != 'grasp'
                    or any(o['method'] not in {'move_tcp', 'move_transfer'} for o in ops[1:])
                    or any(set(o['arguments']) != {'right'} for o in ops)):
                raise ValueError('unsupported primitive sequence')
    # This first policy intentionally forbids these edges even if a file is re-signed accidentally.
    for edge in graph['edges']:
        if edge['edge_id'] in {'e006', 'e007', 'e009', 'e013'} and edge['replayable']:
            raise ValueError('explicitly excluded review edge')
    terminals = graph.get('terminal_node_ids', [])
    if any(n not in observed or preferred(graph, n) for n in terminals):
        raise ValueError('terminal must be observed and have no replayable outgoing edge')
    expected_nodes = []
    for node in graph['nodes']:
        if node['node_id'] not in observed:
            continue
        n = copy.deepcopy(node)
        n['observation_condition']['uncertainty_conditions'] = [s for s in
            n['observation_condition']['uncertainty_conditions'] if s not in graph['matcher_projection_omissions']]
        expected_nodes.append(n)
    if graph['match_nodes'] != expected_nodes:
        raise ValueError('matcher projection changed visual states')
    for node in graph['nodes']:
        if len(preferred(graph, node['node_id'])) > 1:
            raise ValueError('multiple PREFERRED replayable edges; host must not guess')
    return trajectory


def preferred(graph, node):
    return [e for e in graph['edges'] if e['source_node'] == node
            and e['replayable'] and e['preference'] == 'PREFERRED']


def choose(graph, node):
    if node in graph.get('terminal_node_ids', []):
        return None, 'COMPLETED'
    edges = preferred(graph, node)
    if len(edges) == 1:
        return edges[0], None
    if len(edges) > 1:
        return None, 'NO_REPLAYABLE_EDGE'
    outgoing = [e for e in graph['edges'] if e['source_node'] == node]
    return None, ('REQUIRES_FRESH_VALIDATION' if any(
        e['stop_reason'] == 'REQUIRES_FRESH_VALIDATION' for e in outgoing) else 'NO_REPLAYABLE_EDGE')


def matcher_payload(graph):
    return signatures(graph, [n['node_id'] for n in graph['match_nodes']])


def matcher_images(graph, trajectory):
    ids = dict.fromkeys(o for n in graph['match_nodes'] for o in n['representative_observations'])
    return [(oid, checked_file(trajectory['observations'][oid]['rgb'])) for oid in ids]


def right_side_extra(edge):
    if edge['action_family'].startswith('right_sleeve_'):
        return RIGHT_SLEEVE_EXTRA_MM
    return RIGHT_SIDE_EXTRA_MM if edge['action_family'].startswith(
        ('right_sleeve_', 'right_torso_', 'right_upper_torso_')) else 0.


def policy_grasp_offset(graph, edge):
    return graph['grasp_offset_mm'] + right_side_extra(edge)


def execute_policy_edge(runtime, source, before, folder, graph, edge):
    original = runtime.grasp_offset_mm
    selected = policy_grasp_offset(graph, edge)
    save(Path(folder)/'grasp_offset_override.json', dict(edge_id=edge['edge_id'],
        action_family=edge['action_family'], base_offset_mm=original,
        extra_offset_mm=selected-original, applied_offset_mm=selected,
        policy='seed-recorded base lowering only; no side-specific delta'))
    try:
        runtime.grasp_offset_mm = selected
        return execute_recorded(runtime, source, before, folder)
    finally:
        runtime.grasp_offset_mm = original


def grounding_reference_image(graph, edge):
    """Resolve growth image paths or frozen seed observation IDs, never a target image."""
    source=edge['seed_action'].get('source_observation')
    if not source:return None
    trajectory_path=graph.get('seed_trajectory_path')
    if trajectory_path:
        trajectory=load_frozen(trajectory_path)
        if source in trajectory['observations']:
            return checked_file(trajectory['observations'][source]['rgb'])
    image=Path(source)
    if not image.is_absolute():image=ROOT/image
    if not image.is_file():return None
    meta_path=image.parent/'observation.json'
    if meta_path.is_file():
        meta=json.loads(meta_path.read_text())
        checksum=meta.get('sha256',{}).get(image.name)
        if checksum and file_hash(image)!=checksum:
            raise ValueError('reference observation RGB checksum mismatch')
    return image


def grounding_payload(graph, edge, observation):
    reference=grounding_reference_image(graph,edge)
    geometry={}
    if reference is not None:
        meta_path=reference.parent/'observation.json'
        if meta_path.is_file():
            meta=json.loads(meta_path.read_text())
            geometry={k:meta[k] for k in ('frame_id','calibration_id','intrinsics',
                'world_from_camera_mm','corrected_world_from_camera_affine_mm') if k in meta}
    orientations=[dict(operation_index=i,world_rpy_rad=copy.deepcopy(op['arguments']['right'][3:]))
                  for i,op in enumerate(edge['seed_action']['exact_command']['operations'])]
    return dict(task=dict(subgoal=graph['goal']['subgoal'],
                          side_convention=graph['goal'].get('side_convention', '')),
                constraint_ownership=copy.deepcopy(OWNERSHIP), selected_edge={k: copy.deepcopy(edge[k]) for k in
                ('edge_id', 'action_family', 'action_template')},
                primitive_sequence=[dict(operation_index=i, method=op['method']) for i, op in
                    enumerate(edge['seed_action']['exact_command']['operations'])],
                reference_orientation=dict(operation_orientations=orientations,
                    reference_image_id='action_reference' if reference is not None else None,
                    reference_geometry=geometry),
                grasp_offset_mm=policy_grasp_offset(graph, edge),
                grasp_offset_override=dict(extra_mm=right_side_extra(edge),
                    scope='no side-specific lowering; shared support-height patch remains active'),
                image_size=list(observation.xyz.shape[1::-1]),
                current_geometry=observation_geometry(observation))


def validate_grounding(value):
    if value.get('coordinate_frame')=='ACTION_LOCAL':
        from .action_local_frame import validate_candidate
        return validate_candidate(value)
    jsonschema.validate(value, GROUNDING)
    violations = value['visual_violations']
    missing = value['missing_visual_parameters']
    if any(not v['constraint'].strip() or not v['current_visual_evidence'].strip() for v in violations):
        raise ValueError('visual violation requires a constraint and current visible evidence')
    if value['status'] == 'BLOCKED' and not violations:
        raise ValueError('BLOCKED requires positive visual violation evidence; unknown effects are execution_risk')
    if value['status'] == 'READY' and (violations or missing):
        raise ValueError('READY cannot contain visual violations or missing visual parameters')
    if value['status'] == 'NEED_MORE_INFORMATION' and (not missing or violations):
        raise ValueError('NEED_MORE_INFORMATION requires missing visual parameters and no proven violation')


def verify_hem_frame(requests, trajectory, before, initial_image, previous_image, folder):
    images = [('current', before.image)]
    seed_image = None
    if trajectory['steps']:
        first = trajectory['steps'][0]['before_observation']
        seed_image = checked_file(trajectory['observations'][first]['rgb'])
    seen = {file_hash(before.image)}
    for label, path in [('seed_initial', seed_image), ('run_initial', initial_image),
                        ('previous_observation', previous_image)]:
        if path is not None and file_hash(path) not in seen:
            seen.add(file_hash(path)); images.append((label, path))
    def validate(value):
        jsonschema.validate(value, HEM_FRAME)
        if value['status'] == 'RESOLVED':
            h, w = before.xyz.shape[:2]
            points = [value['collar_pixel_xy'], value['hem_center_pixel_xy']]
            if any(not (0 <= x < w and 0 <= y < h) for x, y in points):
                raise ValueError('hem orientation anchors outside current image')
            if math.dist(*points) < 10 or any(not value[k].strip() for k in
                    ['collar_evidence', 'hem_evidence', 'correspondence_evidence']):
                raise ValueError('hem orientation lacks distinct anchors or evidence')
    frame = requests.ask('hem_frame', dict(image_size=list(before.xyz.shape[1::-1]),
        image_roles={label: ('current coordinates required' if label == 'current'
                           else 'earlier visual context only') for label, _ in images}),
        images, validator=validate)
    validate(frame)
    save(folder / 'hem_orientation.json', dict(result=frame,
        images=[dict(id=label, **evidence_file(path)) for label, path in images]))
    if frame['status'] != 'RESOLVED':
        raise ValueError('hem/collar identity unresolved; no grasp dispatched')
    return frame


def validate_hem_direction(candidate, frame):
    collar, hem = frame['collar_pixel_xy'], frame['hem_center_pixel_xy']
    length = math.dist(collar, hem)
    proposed = candidate['garment_frame']
    if (math.dist(proposed['collar_pixel_xy'], collar) > .2 * length or
            math.dist(proposed['hem_center_pixel_xy'], hem) > .2 * length):
        raise ValueError('grounding contradicts independently verified collar/hem')
    grasp = candidate['slots'][0]['pixel_xy']
    target = candidate['slots'][-1]['pixel_xy']
    if math.dist(grasp, hem) > .3 * length:
        raise ValueError('hem fold grasp is not near verified hem end')
    axis = [collar[i] - hem[i] for i in range(2)]
    progress = sum((target[i] - grasp[i]) * axis[i] for i in range(2)) / length
    if progress <= 0 or math.dist(target, collar) >= math.dist(grasp, collar):
        raise ValueError('hem fold displacement does not approach verified collar')
    return dict(status='CONSISTENT', collar_pixel_xy=collar, hem_center_pixel_xy=hem,
                grasp_pixel_xy=grasp, target_pixel_xy=target, collarward_pixels=progress,
                limitation='Model-based structural identity plus host consistency; not physical proof.')


def validate_sleeve_side(candidate, edge, observation):
    """Check current pixel grasp against the declared garment frame, never image-left/right.

    Frame landmarks remain model estimates; this is a consistency guard, not an independent
    guarantee that collar/hem detection or grasp acquisition is correct.
    """
    family = edge['action_family']
    side = 'left' if family.startswith('left_sleeve_') else 'right' if family.startswith('right_sleeve_') else None
    if side is None:
        return None
    frame = candidate['garment_frame']
    collar, hem = frame['collar_pixel_xy'], frame['hem_center_pixel_xy']
    h, w = observation.xyz.shape[:2]
    if not frame['visual_evidence'].strip() or any(not (0 <= p[0] < w and 0 <= p[1] < h) for p in (collar, hem)):
        raise ValueError('missing/out-of-image garment orientation evidence')
    dx, dy = collar[0]-hem[0], collar[1]-hem[1]
    length = math.hypot(dx, dy)
    if length < 1:
        raise ValueError('degenerate collar-to-hem axis')
    grasp = candidate['slots'][0]['pixel_xy']
    # In top-left-origin pixels, viewer-left of a collar-up garment is (dy, -dx).
    signed_left = ((grasp[0]-collar[0])*dy - (grasp[1]-collar[1])*dx)/length
    if (signed_left if side == 'left' else -signed_left) <= 0:
        raise ValueError(f'{side} sleeve grasp lies on opposite/ambiguous garment side: signed_left_px={signed_left:.2f}')
    return dict(side=side, signed_left_px=signed_left, garment_frame=frame)


def primitive_candidate(candidate):
    # Shared legacy primitive generator receives only an already checked READY candidate.
    # Unknown physical risks remain in the original audit, never in its legacy blocking field.
    validate_grounding(candidate)
    if candidate['status'] != 'READY':
        raise ValueError('grounding not ready')
    return {**{k: candidate[k] for k in ('status', 'edge_id', 'action_family', 'slots', 'explanation')},
            'unresolved_conditions': []}


def summary(graph):
    return dict(nodes=len(graph['nodes']), matchable_nodes=len(graph['match_nodes']),
                replayable_edges=sum(e['replayable'] for e in graph['edges']),
                failed_edges=sum(e['evidence_status'] == 'FAILED_EVIDENCE' for e in graph['edges']),
                validation_blocked_edges=sum(e['evidence_status'] in {'UNVERIFIED_OUTCOME', 'MISSING_AFTER'}
                                             for e in graph['edges']))


def dry_run(graph_path, output):
    """Static offline preflight. Does NOT claim image recognition, IK or physical success."""
    graph = load_frozen(graph_path)
    trajectory = validate_policy(graph)
    images = matcher_images(graph, trajectory)  # verify all real RGB hashes
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save(output / 'matcher_payload.json', matcher_payload(graph))
    save(output / 'matcher_images.json', [dict(id=k, **evidence_file(p)) for k, p in images])
    selections = []
    for node in graph['nodes']:
        edge, stop = choose(graph, node['node_id'])
        selections.append(dict(node_id=node['node_id'], edge_id=edge['edge_id'] if edge else None,
                               stop_reason=stop))
    result = dict(status='DRY_RUN_OK', mode='OFFLINE_STATIC_PREFLIGHT', **summary(graph),
        graph_sha256=file_hash(graph_path), selections=selections, model_calls=0, robot_calls=0,
        fallback_used=False, limitations=['No new RGB captured; no actual GPT matching or grounding.',
            'Live RGB-D, IK, collision/workspace validation and physical transitions require --execute.',
            'Exclusions inherit the existing arm7 collision configuration; this runner changes none.'])
    frozen(output / 'dry_run.json', result)
    return result


def replay(graph_path, output, runtime, requests, *, max_steps=20, threshold=.8,
           max_fallbacks=0, fallback_timeout_s=900, fallback_fn=None, initial_node=None, debug_grasp_retry=False, retry_missed_grasp=False, repair_self_loop=False, live_graph_port=None, applicability_recovery=False,
           max_recovery_attempts=None):
    """Only the --execute branch constructs a physical runtime; tests inject doubles."""
    graph_path = Path(graph_path); graph = load_frozen(graph_path)
    trajectory = validate_policy(graph)
    input_graph = copy.deepcopy(graph)
    # Preserve explicit legacy Python callers; normal/CLI replay selects the new bounded mode.
    legacy_recovery=bool(max_fallbacks or applicability_recovery or repair_self_loop or debug_grasp_retry or retry_missed_grasp)
    if max_recovery_attempts is None:max_recovery_attempts=0 if legacy_recovery else 1
    if max_recovery_attempts not in (0,1):raise ValueError('max_recovery_attempts must be 0 or 1')
    if max_recovery_attempts and legacy_recovery:
        raise ValueError('grasp-handle recovery cannot combine with legacy extending/repair/retry modes')
    if max_fallbacks < 0: raise ValueError('max_fallbacks must be nonnegative')
    if retry_missed_grasp and max_fallbacks:
        raise ValueError("missed-grasp retry requires strict replay")
    if debug_grasp_retry and max_fallbacks:
        raise ValueError('manual grasp retry requires strict replay without free planning')
    if initial_node is not None and initial_node not in {n['node_id'] for n in graph['match_nodes']}:
        raise ValueError('initial node must be observed in graph')
    if runtime.grasp_offset_mm != graph['grasp_offset_mm']:
        raise ValueError('runtime lowering differs from seed')
    if not 0 < threshold <= 1 or max_steps < 1:
        raise ValueError('invalid confidence/step limit')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    frozen(output / 'policy_used.json', graph)
    frozen(output / 'constraint_ownership.json', OWNERSHIP)
    frozen(output / 'runner_config.json', dict(max_steps=max_steps, match_threshold=threshold,
        max_recovery_attempts=max_recovery_attempts,grasp_handle_recovery_graph_mutation=False,
        max_fallbacks=max_fallbacks, fallback_timeout_s=fallback_timeout_s, initial_node=initial_node,
        applicability_recovery=applicability_recovery,applicability_recovery_budget='ONCE_PER_EDGE_PER_RUN',
        repair_self_loop=repair_self_loop,debug_grasp_retry=debug_grasp_retry,retry_missed_grasp=retry_missed_grasp,missed_grasp_retry_lowering_mm=2,grounding_recapture_retries=1,
        grounding_mode=getattr(requests,'grounding_mode','full'),
        grounding_templates=str(requests.minimal_grounder.path) if getattr(requests,'minimal_grounder',None) else None,
        matching_policy='topology-reference-images-v2',
        recovery_trigger='NO_ROBUST_ACTION' if max_recovery_attempts else 'NO_MATCH_OR_SOURCE_NODE_UNCHANGED_OR_RECOVERY_NODE_WITHOUT_POLICY',
        original_policy_immutable=True, recovery_edges_replayable=False, right_side_extra_mm=RIGHT_SIDE_EXTRA_MM,
        right_sleeve_extra_mm=RIGHT_SLEEVE_EXTRA_MM))
    graph_hash = file_hash(graph_path)
    metrics = dict(status='STARTED', graph_sha256=graph_hash, fallback_used=False,
                   graph_only_success=False, transitions=[], executed_edges=[], graph_reused=False)
    runtime.auto_grasp_debug = bool(debug_grasp_retry) and not retry_missed_grasp and not repair_self_loop
    from .edge_self_loop_repair import EdgeRepairs
    repairs=EdgeRepairs(output,requests,graph_path) if repair_self_loop else None
    repaired_edges=set()
    metrics['edge_replans']=0
    missed_retry_edges = set()
    metrics["missed_grasp_retries"] = 0
    runtime.debug_unstick_attempts = 0
    from .replay_live_graph import create_live_graph
    live=create_live_graph(output,live_graph_port)
    live.update(graph,phase='MATCHING',observations=trajectory['observations'])
    from .edge_applicability import ApplicabilityRecovery
    applicability=ApplicabilityRecovery(output,requests,live,graph) if applicability_recovery else None
    pending = None
    confirmed_match=None;source_reentries_since_action=0
    recovery = Recovery(graph_path, graph, trajectory, output, requests) if max_fallbacks else None
    if recovery is not None:
        graph, trajectory = recovery.graph, recovery.trajectory
    metrics.update(local_match_time=0., expanded_match_time=0., global_match_time=0.,
                   representative_image_fallback_count=0)
    metrics.update(fallback_steps=0, recovery_enabled=bool(recovery) or bool(max_recovery_attempts), fallback_transitions=[])
    from .robustness_recovery import HandleRecovery, RecoveryUnresolved
    from .graph_minimal_grounding import NoRobustAction
    handle_recovery=HandleRecovery(output,max_recovery_attempts,
        on_update=lambda events: live.update(graph,recoveries=events))
    metrics.update(recovery_attempts=0,grasp_handle_recovery_enabled=bool(max_recovery_attempts))
    try:
        before = runtime.observe(output / 'initial_observation')
        initial_image = before.image
        previous_image = None
        # An additional matching turn is mandatory after the last permitted dispatch.
        for index in range(max_steps + 1):
            if file_hash(graph_path) != graph_hash or load_frozen(graph_path) != input_graph:
                raise ValueError('policy graph changed during replay')
            folder = output / f'step_{index:03d}'; folder.mkdir()
            recovery_ids=[n['node_id'] for n in reversed(graph['match_nodes'])
                          if n['node_id'].startswith('R_')]
            live.update(graph,phase='AFTER_MATCHING' if pending else 'MATCHING')
            if confirmed_match is not None:
                match,candidates=confirmed_match;confirmed_match=None
                save(folder/'matching_attempts.json',dict(attempts=[dict(stage='joint_review_reentry',
                    candidate_nodes=candidates,representative_images=True,response=match)]))
            else:
                match = topology_match(graph, trajectory, before, requests, folder, metrics,
                    pending=pending, threshold=threshold, initial_node=initial_node if index==0 else None,
                    recovery_nodes=recovery_ids)
            save(folder / 'node_match.json', dict(result=match, current_rgb=evidence_file(before.image)))
            matched = reliable_match(match, graph, threshold)
            handle_recovery.matched(match,matched,before)
            if pending is not None:
                from .node_match_progress import node_match_progress
                pending['progress_evaluation']=node_match_progress(pending['source_node'],pending['target_node'],
                    match.get('node_id'),reliable=matched)
                save(folder/'progress_evaluation.json',pending['progress_evaluation'])
            if applicability is not None and applicability.active and pending:
                active_edge=next(e for e in graph['edges'] if e['edge_id']==applicability.active['edge_id'])
                applicability.finish(active_edge,match['node_id'] if matched else None,before)
                if pending.get('mode')=='APPLICABILITY_FRONTIER':
                    pending.update(after=evidence_file(before.image),actual_node=match['node_id'] if matched else None)
            if not matched:
                if match['status'] != 'NO_MATCH' or recovery is None:
                    if repairs is not None and pending and pending.get('mode')=='EDGE_REPAIR':
                        repairs.finish(pending['edge_id'],None,before)
                    metrics['status'] = 'NO_MATCH' if match['status'] == 'NO_MATCH' else 'AMBIGUOUS'
                    break
                node = recovery.observe_node(before, folder)
            else:
                node = match['node_id']
            live.update(graph,current=node,action=None,target=None,phase='MATCHED',observation=str(before.image))
            fallback_reason = None
            recovery_context = None
            if pending is not None:
                pending.update(after=evidence_file(before.image), actual_node=node,
                               reached_expected_target_node=node == pending['target_node'])
                if repairs is not None and pending.get('mode')=='EDGE_REPAIR':
                    repairs.finish(pending['edge_id'],node,before)
                # Frontier planning may change the original edge's action role.
                # Keep its actual outcome in recovery, without treating it as an
                # execution of the frozen edge or dropping its self-loop evidence.
                is_free = pending.get('mode') in ('FALLBACK_FREE_PLAN','APPLICABILITY_FRONTIER')
                unchanged = node == pending['source_node']
                alternative = matched and not is_free and not unchanged and node != pending['target_node']
                if recovery is not None and (is_free or not matched or unchanged or alternative):
                    recovery.transition(pending['source_node'], node,
                        pending['before']['path'], before.image, source=pending['command_source'],
                        outcome=pending['execution'], mode=pending['mode'], edge_id=pending.get('edge_id'))
                if unchanged:
                    if repairs is not None and not is_free and pending['edge_id'] in repaired_edges and recovery is None:
                        metrics['status']='EDGE_REPAIR_SELF_LOOP'; break
                    if repairs is not None and not is_free and pending['edge_id'] not in repaired_edges:
                        if index==max_steps:
                            metrics['status']='STEP_LIMIT'; break
                        edge=next(e for e in graph['edges'] if e['edge_id']==pending['edge_id'])
                        repair_folder=folder/'edge_repair'
                        review=missed_grasp_retry.review(requests,pending,before,repair_folder/'review')
                        live.update(graph,phase='REPAIR',action=edge['edge_id'],target=edge['target_node'])
                        context=repairs.begin(edge,pending,review,before,repair_folder/'planner')
                        repaired_edges.add(edge['edge_id'])
                        metrics['edge_replans']+=1
                        result=(fallback_fn or free_plan_one)(runtime,before,repair_folder/'planner',graph,
                            timeout_s=fallback_timeout_s,budget=requests,context=context)
                        repairs.executed(edge['edge_id'],result)
                        outcome=result['outcome']
                        if outcome.get('status') in {'PLANNING_FAILED','PLANNING_INCOMPLETE'}:
                            metrics.update(status='SAFETY_REJECT',execution=outcome); break
                        if outcome.get('status')!='EXECUTED' or outcome.get('physical_execution') is not True:
                            metrics.update(status='EDGE_REPAIR_FAILED',execution=outcome); break
                        source_reentries_since_action=0
                        pending=dict(mode='EDGE_REPAIR',command_source=result['source'],execution=outcome,
                            edge_id=edge['edge_id'],source_node=node,target_node=edge['target_node'],
                            before=evidence_file(before.image),reached_expected_target_node=False)
                        metrics['transitions'].append(pending)
                        metrics['executed_edges'].append(edge['edge_id'])
                        metrics['fallback_used']=True
                        previous_image=before.image
                        before=runtime.observe(repair_folder/'after_observation')
                        pending['after']=evidence_file(before.image)
                        continue
                    if (retry_missed_grasp and not is_free and index < max_steps
                            and pending['edge_id'] not in missed_retry_edges):
                        retry_folder=folder/'missed_grasp_retry'
                        retry_folder.mkdir()
                        review=missed_grasp_retry.review(requests,pending,before,retry_folder)
                        if review['result']=='UNCHANGED' and review['confidence']>=.8:
                            edge=next(e for e in graph['edges'] if e['edge_id']==pending['edge_id'])
                            try:
                                retry_source=missed_grasp_retry.lowered_source(pending['command_source'])
                            except ValueError as exc:
                                metrics.update(status='MISSED_RETRY_UNSUPPORTED',error=str(exc)); break
                            (retry_folder/'motion.py').write_text(retry_source)
                            save(retry_folder/'intent.json',dict(edge_id=edge['edge_id'],lowering_mm=2,
                                retry_limit_per_edge=1,reason=review))
                            missed_retry_edges.add(edge['edge_id'])
                            outcome=execute_policy_edge(runtime,retry_source,before,retry_folder,graph,edge)
                            save(retry_folder/'execution.json',outcome)
                            if outcome.get('status') in {'PLANNING_FAILED','PLANNING_INCOMPLETE'}:
                                metrics.update(status='SAFETY_REJECT',execution=outcome); break
                            if outcome.get('status')!='EXECUTED' or outcome.get('physical_execution') is not True:
                                metrics.update(status='ACTION_EXECUTION_FAILURE',execution=outcome); break
                            metrics['missed_grasp_retries']+=1
                            metrics['executed_edges'].append(edge['edge_id'])
                            pending=dict(mode='GRAPH_REUSE',command_source=retry_source,execution=outcome,
                                edge_id=edge['edge_id'],source_node=node,target_node=edge['target_node'],
                                before=evidence_file(before.image),reached_expected_target_node=False,
                                retry_reason='NO_VISIBLE_GARMENT_CHANGE',extra_lowering_mm=2)
                            metrics['transitions'].append(pending)
                            previous_image=before.image
                            before=runtime.observe(retry_folder/'after_observation')
                            pending['after']=evidence_file(before.image)
                            continue
                    if recovery is None:
                        metrics['status'] = 'SOURCE_NODE_UNCHANGED'; break
                    fallback_reason = 'SOURCE_NODE_UNCHANGED'
                    recovery_context = dict(reason=fallback_reason, previous_transition=copy.deepcopy(pending),
                        current_match=match, interpretation='No node-level progress; not proof of empty grasp.')
                if alternative:
                    if recovery is None:
                        metrics['status'] = 'UNEXPECTED_NODE'; break
                    # A reliable fresh match can re-enter an existing policy state.
                    # Preserve the observed branch without changing the seed edge,
                    # promoting its template, or reusing the previous edge's command.
                    event = dict(event='GLOBAL_REENTRY', source_node=pending['source_node'],
                        edge_id=pending['edge_id'], expected_target=pending['target_node'],
                        node_id=node, confidence=match['confidence'],
                        current_rgb=evidence_file(before.image))
                    save(folder/'global_reentry.json', event)
                    metrics.setdefault('graph_reentries', []).append(event)
                if is_free and matched and not unchanged and preferred(graph, node):
                    event = dict(event='GRAPH_RESUME', node_id=node, current_rgb=evidence_file(before.image))
                    save(folder / 'graph_resume.json', event)
                    metrics.setdefault('graph_resumptions', []).append(event)
                pending = None
            if matched and node in graph.get('terminal_node_ids', []):
                metrics.update(status='COMPLETED', terminal_node=node,
                    completion_basis='FRESH_TERMINAL_NODE_MATCH',
                    graph_only_success=bool(metrics['graph_reused'] and not metrics['fallback_used']))
                break
            recovery_node = node not in {n['node_id'] for n in input_graph['nodes']
                                         if any(e['source_node'] == n['node_id'] for e in input_graph['edges'])}
            if recovery is not None and (fallback_reason is not None or not matched or
                    (recovery_node and node.startswith('R_'))):
                if index == max_steps:
                    metrics['status'] = 'STEP_LIMIT'; break
                if metrics['fallback_steps'] >= max_fallbacks:
                    metrics['status'] = 'FALLBACK_LIMIT'; break
                metrics['fallback_used'] = True
                metrics['fallback_steps'] += 1
                save(folder / 'fallback_intent.json', dict(event='FALLBACK_FREE_PLAN',
                    node_id=node, current_rgb=evidence_file(before.image),
                    reason=fallback_reason or ('NO_MATCH' if not matched else 'RECOVERY_NODE_WITHOUT_POLICY')))
                if recovery_context is not None:
                    save(folder / 'recovery_context.json', recovery_context)
                if file_hash(graph_path) != graph_hash:
                    raise ValueError('input policy changed before fallback')
                live.update(graph,current=node,action=None,target=None,phase='FRONTIER')
                runner = fallback_fn or free_plan_one
                fallback = runner(runtime, before, folder/'fallback', graph,
                                  timeout_s=fallback_timeout_s, budget=requests, context=recovery_context)
                save(folder/'fallback_result.json', fallback)
                outcome = fallback['outcome']
                if outcome.get('status') in {'PLANNING_FAILED','PLANNING_INCOMPLETE'}:
                    metrics.update(status='SAFETY_REJECT', execution=outcome); break
                if outcome.get('status') == 'MODEL_REPORTED_COMPLETE':
                    metrics.update(status='FALLBACK_REPORTED_COMPLETE', completion_basis='FREE_PLANNER_REPORT'); break
                if outcome.get('status') != 'EXECUTED' or outcome.get('physical_execution') is not True:
                    metrics.update(status='FALLBACK_FAILED', fallback_result=fallback,execution=outcome); break
                source_reentries_since_action=0
                pending = dict(mode='FALLBACK_FREE_PLAN', edge_id=None, source_node=node, target_node=None,
                    before=evidence_file(before.image), reached_expected_target_node=False,
                    command_source=fallback['source'], execution=outcome)
                metrics['fallback_transitions'].append(pending)
                previous_image = before.image
                before = runtime.observe(folder/'fallback_after')
                pending['after'] = evidence_file(before.image)
                continue
            edge, stop = choose(graph, node)
            if stop:
                metrics['status'] = stop; break
            if index == max_steps:
                metrics['status'] = 'STEP_LIMIT'; break
            if edge['edge_id'] in metrics['executed_edges']:
                metrics['status'] = 'NO_REPLAYABLE_EDGE'; break
            live.update(graph,current=node,action=edge['edge_id'],target=edge['target_node'],phase='GROUNDING')
            save(folder / 'host_selection.json', dict(node_id=node, edge_id=edge['edge_id'],
                                                     rule='UNIQUE_PREFERRED_AND_REPLAYABLE'))
            try:
                hem_frame = None
                if getattr(requests,'minimal_grounder',None) is not None:
                    requests.minimal_grounder.template(edge)  # reject unsupported edge before extra calls
                elif edge['action_family'] == 'bottom_hem_toward_collar_fold':
                    hem_frame = verify_hem_frame(requests, trajectory, before, initial_image,
                                                 previous_image, folder)
                from .graph_minimal_grounding import GroundingUncertain
                try:
                    handle_recovery.check_grounding(edge,before)
                    candidate, before = grounding_with_recapture(runtime,requests,graph,trajectory,edge,before,folder,metrics,threshold,hem_frame)
                except NoRobustAction:
                    raise
                except GroundingUncertain as uncertain:
                    if applicability is None:raise
                    before=getattr(uncertain,'observation',before)
                    live.update(graph,phase='APPLICABILITY')
                    candidate=applicability.prepare(edge,before,folder)
                    metrics['applicability_recoveries']=len(applicability.rows)
                    if candidate is None:
                        live.update(graph,phase='FRONTIER',target=None)
                        result=(fallback_fn or free_plan_one)(runtime,before,folder/'applicability_frontier',graph,
                            timeout_s=fallback_timeout_s,budget=requests,context=applicability.context(edge))
                        outcome=result['outcome'];source=result.get('source')
                        applicability.executed(source,outcome)
                        metrics['fallback_used']=True
                        if outcome.get('status')!='EXECUTED' or outcome.get('physical_execution') is not True:
                            metrics.update(status='APPLICABILITY_FRONTIER_FAILED',execution=outcome);break
                        source_reentries_since_action=0
                        pending=dict(mode='APPLICABILITY_FRONTIER',edge_id=edge['edge_id'],source_node=node,
                            target_node=edge['target_node'],before=evidence_file(before.image),command_source=source,execution=outcome)
                        metrics['transitions'].append(pending)
                        previous_image=before.image
                        live.update(graph,phase='AFTER_MATCHING')
                        before=runtime.observe(folder/'applicability_after')
                        continue
                    live.update(graph,phase='GROUNDING')
                validate_grounding(candidate)
                save(folder / 'grounding.json', candidate)
                if candidate['status'] != 'READY':
                    reasons = [v['current_visual_evidence'] for v in candidate['visual_violations']]
                    reasons += candidate['missing_visual_parameters']
                    metrics.update(status='EDGE_INSTANTIATION_FAILURE',
                        error='Grounding ' + candidate['status'] + ': ' + '; '.join(reasons),
                        instantiation_failure=dict(edge_id=edge['edge_id'],
                            model_status=candidate['status'], explanation=candidate['explanation'],
                            visual_violations=candidate['visual_violations'],
                            missing_visual_parameters=candidate['missing_visual_parameters'],
                            execution_risk=candidate['execution_risk'],
                            safety_validation_attempted=False, dispatch_attempted=False))
                    break
                if hem_frame is None and getattr(requests,'minimal_grounder',None) is not None and edge['action_family']=='bottom_hem_toward_collar_fold' and candidate.get('coordinate_frame')!='ACTION_LOCAL':
                    hem_frame=candidate['garment_frame']
                if hem_frame is not None:
                    save(folder / 'hem_direction_check.json', validate_hem_direction(candidate, hem_frame))
                _, source, measurements = instantiate_grounded(candidate,edge,before)
                side_check = grounded_side_check(candidate,edge,before)
                save(folder / 'garment_side_check.json', side_check)
            except NoRobustAction as rejected:
                before=rejected.observation or before
                save(folder/'gate_rejection.json',dict(node_id=node,edge_id=edge['edge_id'],
                    reasons=rejected.reasons,current_rgb=evidence_file(before.image),dispatch_attempted=False))
                if file_hash(graph_path)!=graph_hash or load_frozen(graph_path)!=input_graph:
                    raise ValueError('policy graph changed before recovery')
                live.update(graph,phase='HANDLE_RECOVERY',action=None,target=None)
                previous_image=before.image
                try:
                    before=handle_recovery.recover(runtime,requests,graph,edge,node,before,rejected,
                        fallback_fn or free_plan_one,fallback_timeout_s)
                except RecoveryUnresolved as exc:
                    metrics.update(status='RECOVERY_UNRESOLVED',error=str(exc))
                    stop_after_failure(runtime,metrics)
                    break
                # No expected node/edge bias or fabricated transition after a recovery manipulation.
                pending=None;confirmed_match=None;source_reentries_since_action=0
                metrics['fallback_used']=True
                live.update(graph,phase='MATCHING',current=None,action=None,target=None)
                continue
            except (ValueError, RuntimeError, jsonschema.ValidationError, IndexError) as exc:
                if record_controller_failure(metrics,exc):
                    stop_after_failure(runtime,metrics)
                    break
                recapture_match=getattr(exc,'recapture_match',None)
                if recapture_match is not None and metrics['transitions']:
                    from .node_match_progress import node_match_progress
                    last=metrics['transitions'][-1]
                    conflict=dict(previous=copy.deepcopy(last.get('progress_evaluation')),
                        recapture_match=recapture_match,reason='No intervening garment action; source recheck inconsistent or unreliable.')
                    last['progress_evaluation']=node_match_progress(last['source_node'],last['target_node'],
                        recapture_match.get('node_id'),conflict=True)
                    last['match_conflict']=conflict
                    save(folder/'progress_conflict.json',dict(**conflict,progress_evaluation=last['progress_evaluation']))
                    if applicability is not None:applicability.invalidate_match(last['edge_id'],conflict)
                from .recapture_consistency import ConfirmedSourceChange,RecaptureSourceUnmatched
                if isinstance(exc,(ConfirmedSourceChange,RecaptureSourceUnmatched)):
                    if source_reentries_since_action>=2:
                        metrics.update(status='SOURCE_REENTRY_LIMIT',error='Repeated source corrections without an intervening garment action');break
                    source_reentries_since_action+=1
                    metrics['source_reentries']=metrics.get('source_reentries',0)+1
                    global_required=isinstance(exc,RecaptureSourceUnmatched)
                    save(folder/'source_reentry.json',dict(old_source=node,cancelled_edge=edge['edge_id'],
                        confirmed_source=None if global_required else exc.recapture_match['node_id'],
                        requires_global_match=global_required,basis='JOINT_REFERENCE_IMAGE_REVIEW',
                        old_edge_dispatched=False,review=exc.review))
                    before=exc.observation;pending=None
                    confirmed_match=None if global_required else (exc.recapture_match,exc.candidates)
                    live.update(graph,current=None if global_required else exc.recapture_match['node_id'],
                                action=None,target=None,phase='MATCHING' if global_required else 'MATCHED')
                    continue
                metrics.update(status='EDGE_INSTANTIATION_FAILURE', error=str(exc)); break
            save(folder / 'surface_measurements.json', measurements)
            (folder / 'motion.py').write_text(source)
            if file_hash(graph_path) != graph_hash or load_frozen(graph_path) != input_graph:
                raise ValueError('policy graph changed before dispatch')
            save(folder / 'dispatch_intent.json', dict(event='GRAPH_REUSE', edge_id=edge['edge_id'],
                graph_sha256=graph_hash, before=evidence_file(before.image), command=command(source),
                execution_risk=candidate['execution_risk'], constraint_ownership=OWNERSHIP['version']))
            # Existing Arm7Runtime.execute validates observation, full IK/path/collision/workspace,
            # arrival and controller state BEFORE dispatch. No alternate planning on rejection.
            live.update(graph,phase='EXECUTING')
            outcome = execute_policy_edge(runtime, source, before, folder, graph, edge)
            save(folder / 'execution.json', outcome)
            handle_recovery.graph_outcome(edge,source,before,outcome)
            if applicability is not None and applicability.active:applicability.executed(source,outcome)
            if outcome.get('status') in {'PLANNING_FAILED', 'PLANNING_INCOMPLETE'}:
                from .edge_self_loop_repair import repairable_ik_failure
                if repairs is not None and repairable_ik_failure(outcome) and edge['edge_id'] not in repaired_edges:
                    repair_folder=folder/'ik_repair'
                    rejected=dict(mode='GRAPH_REUSE',edge_id=edge['edge_id'],source_node=node,
                        actual_node=None,target_node=edge['target_node'],command_source=source,
                        before=evidence_file(before.image),execution=outcome)
                    # No physical dispatch occurred: retain the source RGB rather than inventing a self-loop.
                    live.update(graph,phase='REPAIR')
                    context=repairs.begin(edge,rejected,dict(result='UNKNOWN',confidence=0.,
                        evidence='No execution; structured IK failure, not a cloth outcome'),
                        before,repair_folder/'planner',trigger='IK_FAILED')
                    repaired_edges.add(edge['edge_id']);metrics['edge_replans']+=1
                    result=(fallback_fn or free_plan_one)(runtime,before,repair_folder/'planner',graph,
                        timeout_s=fallback_timeout_s,budget=requests,context=context)
                    repairs.executed(edge['edge_id'],result)
                    outcome=result['outcome']
                    save(repair_folder/'execution.json',outcome)
                    if outcome.get('status') in {'PLANNING_FAILED','PLANNING_INCOMPLETE'}:
                        metrics.update(status='SAFETY_REJECT',execution=outcome); break
                    if outcome.get('status')!='EXECUTED' or outcome.get('physical_execution') is not True:
                        metrics.update(status='EDGE_REPAIR_FAILED',execution=outcome); break
                    pending=dict(mode='EDGE_REPAIR',command_source=result['source'],execution=outcome,
                        edge_id=edge['edge_id'],source_node=node,target_node=edge['target_node'],
                        before=evidence_file(before.image),reached_expected_target_node=False)
                    metrics['transitions'].append(pending);metrics['executed_edges'].append(edge['edge_id'])
                    metrics['fallback_used']=True
                    previous_image=before.image
                    before=runtime.observe(repair_folder/'after_observation')
                    pending['after']=evidence_file(before.image)
                    continue
                metrics.update(status='SAFETY_REJECT', execution=outcome); break
            if outcome.get('status') != 'EXECUTED' or outcome.get('physical_execution') is not True:
                metrics.update(status='ACTION_EXECUTION_FAILURE', execution=outcome); break
            source_reentries_since_action=0
            metrics['executed_edges'].append(edge['edge_id']); metrics['graph_reused'] = True
            pending = dict(mode='GRAPH_REUSE', command_source=source, execution=outcome,
                           edge_id=edge['edge_id'], source_node=node, target_node=edge['target_node'],
                           before=evidence_file(before.image), execution_risk=candidate['execution_risk'],
                           reached_expected_target_node=False)
            metrics['transitions'].append(pending)
            previous_image = before.image
            live.update(graph,phase='AFTER_MATCHING')
            before = runtime.observe(folder / 'after_observation')
            pending['after'] = evidence_file(before.image)
    except RecoveryUnresolved as exc:
        metrics.update(status='RECOVERY_UNRESOLVED',error=str(exc))
        stop_after_failure(runtime,metrics)
    except BaseException as exc:
        if record_controller_failure(metrics,exc):
            stop_after_failure(runtime,metrics)
        else:
            metrics.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'ERROR',
                           error=f'{type(exc).__name__}: {exc}')
            stop_after_failure(runtime,metrics)
            raise
    finally:
        metrics['recovery_attempts']=len(handle_recovery.attempts)
        if handle_recovery.attempts:metrics['fallback_used']=True
        if metrics.get('execution',{}).get('error'):
            if record_controller_failure(metrics,metrics['execution']['error']):
                stop_after_failure(runtime,metrics)
        if applicability is not None:
            if applicability.active:
                applicability.active.update(status='STOPPED_BEFORE_VERIFIED_OUTCOME',stop_reason=metrics['status'])
            applicability.snapshot()
            metrics['applicability_experience']=str((applicability.folder/'experience.json').resolve())
        live.update(graph,phase='STOPPED',result=metrics['status'])
        live.close()
        if repairs is not None:
            metrics['edge_experience']=str((repairs.output/'edge_experience.json').resolve())
            repairs.snapshot()
        metrics['debug_unstick_attempts'] = getattr(runtime,'debug_unstick_attempts',0)
        runtime.auto_grasp_debug = False
        try: runtime.close()
        except Exception as exc: metrics['cleanup_error'] = str(exc)
        metrics.update(requests.metrics())
        if recovery is not None and recovery.latest_path is not None:
            metrics['extended_graph'] = str(recovery.latest_path.resolve())
        if handle_recovery.rejections:
            try:graph_unchanged=file_hash(graph_path)==graph_hash and load_frozen(graph_path)==input_graph
            except (OSError,ValueError):graph_unchanged=False
            handle_recovery.finish(metrics['status'],graph_unchanged)
            metrics['robustness_recovery_audit']=str((handle_recovery.output/'audit.json').resolve())
        frozen(output / 'final_metrics.json', metrics)
        (output / 'report.md').write_text('# Graph replay and recovery audit\n\n```json\n'
            + json.dumps(metrics, ensure_ascii=False, indent=2) + '\n```\n')
    return metrics


class OnlineGrowth:
    """Persistent evidence graph: one source/action identity, many observed targets."""
    def __init__(self, graph_path, output, *, state_path=None):
        self.base_path = Path(graph_path).resolve()
        self.base = load_frozen(self.base_path)
        self.trajectory = validate_growth_base(self.base)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        if state_path:
            candidate = json.loads(Path(state_path).read_text())
            if 'snapshot' in candidate:
                if file_hash(candidate['snapshot']) != candidate['sha256']:
                    raise ValueError('latest growth pointer hash mismatch')
                state_path = candidate['snapshot']
            self.state = load_frozen(state_path)
            self.validate_state()
        else:
            nodes = {}
            projected = {n['node_id']:n for n in self.base['match_nodes']}
            for original in self.base['nodes']:
                node = projected.get(original['node_id'], original)
                nodes[node['node_id']] = dict(node=copy.deepcopy(node), status='EXISTING',
                    observation_count=0, incoming_transitions=[], outgoing_actions=[], outcome_history=[])
            actions = {}
            for edge in self.base['edges']:
                if edge['source_node'] not in nodes:
                    continue
                aid = edge['edge_id']
                actions[aid] = dict(action_id=aid, source_node=edge['source_node'],
                    edge=copy.deepcopy(edge), origin='BASE_POLICY',
                    preferred=edge['replayable'] and edge['preference']=='PREFERRED',
                    replayable=edge['replayable'], nominal_target=edge['target_node'],
                    outcomes={edge['target_node']: dict(seed_evidence=True, observed_count=0, history=[])}
                        if edge['target_node'] in nodes else {}, successful_observations=0)
                nodes[edge['source_node']]['outgoing_actions'].append(aid)
            self.state = dict(protocol=GROWTH_PROTOCOL, base_graph=str(self.base_path),
                base_sha256=file_hash(self.base_path), version=0, nodes=nodes, actions=actions,
                observations={}, attempts=[], transitions=[], episodes=[], merge_candidates=[], metrics_history=[])
        self.trajectory = copy.deepcopy(self.trajectory)
        self.trajectory['observations'].update(self.state['observations'])
        self.latest_path = None
        self.snapshot()

    def validate_state(self):
        s = self.state
        if s.get('protocol') != GROWTH_PROTOCOL or s['base_sha256'] != file_hash(self.base_path):
            raise ValueError('growth state/base provenance mismatch')
        originals = {n['node_id']: n for n in self.base['nodes']}
        originals.update({n['node_id']:n for n in self.base['match_nodes']})
        for nid, row in s['nodes'].items():
            jsonschema.validate(row['node'], NODE)
            if nid != row['node']['node_id'] or row['status'] not in ('EXISTING', 'PROVISIONAL'):
                raise ValueError('invalid growth node identity/status')
            if nid in originals and row['node'] != originals[nid]:
                raise ValueError('growth cannot rewrite existing decision conditions')
            if nid not in originals and row['status'] != 'PROVISIONAL':
                raise ValueError('new nodes must remain provisional')
            for obsid in row['node']['representative_observations']:
                observation = s['observations'].get(obsid) or self.trajectory['observations'][obsid]
                checked_file(observation['rgb'])
        if not set(originals) <= set(s['nodes']):
            raise ValueError('growth removed original nodes')
        original_edges = {e['edge_id']: e for e in self.base['edges'] if e['source_node'] in originals}
        if not set(original_edges) <= set(s['actions']):
            raise ValueError('growth removed original actions')
        transitions = {t['transition_id']: t for t in s['transitions']}
        if len(transitions) != len(s['transitions']):
            raise ValueError('duplicate observed transitions')
        for aid, action in s['actions'].items():
            from .growth_edge_recovery import original_edge
            seed_edge=original_edge(action,s)
            if aid != action['action_id'] or action['source_node'] not in s['nodes']:
                raise ValueError('invalid source/action identity')
            if aid not in s['nodes'][action['source_node']]['outgoing_actions']:
                raise ValueError('missing outgoing action index')
            if aid in original_edges:
                edge = original_edges[aid]
                if (seed_edge != edge or action['replayable'] != edge['replayable']
                        or action['preferred'] != (edge['replayable'] and edge['preference']=='PREFERRED')
                        or action['nominal_target'] != edge['target_node']):
                    raise ValueError('growth cannot change frozen policy eligibility/template')
            elif action['replayable']:
                edge=seed_edge
                if (not edge or edge['edge_id']!=aid or edge['source_node']!=action['source_node']
                        or not any(a['selected_action']==aid and a['planner_source']=='GPT_FRONTIER'
                            and a.get('command_source') and command(a['command_source'])==edge['seed_action']['exact_command']
                            and a['execution'].get('status')=='EXECUTED' for a in s['attempts'])):
                    raise ValueError('learned template has no matching executed frontier command')
                jsonschema.validate(edge['action_template'],TEMPLATE)
                if s['nodes'][action['source_node']]['status'] != 'PROVISIONAL' or not action['edge']:
                    raise ValueError('only provisional frontier actions can gain replay eligibility')
                success = [t for t in s['transitions'] if t['selected_action']==aid
                    and t['result_node'] != action['source_node']
                    and t['effect']['progress'] in ('PROGRESS', 'COMPLETE')
                    and t['execution']['status']=='EXECUTED' and t['execution']['physical_execution'] is True]
                if not success or action['successful_observations'] != len(success):
                    raise ValueError('frontier reuse requires observed visual success')
            for target, outcome in action['outcomes'].items():
                if target not in s['nodes'] or outcome['observed_count'] != len(outcome['history']):
                    raise ValueError('invalid action outcome index')
                for tid in outcome['history']:
                    t = transitions[tid]
                    if t['selected_action'] != aid or t['source_node'] != action['source_node'] or t['result_node'] != target:
                        raise ValueError('outcome changed source/action identity')
        for t in s['transitions']:
            if t['selected_action'] not in s['actions'] or t['result_node'] not in s['nodes']:
                raise ValueError('dangling transition')
            checked_file(t['before']); checked_file(t['after'])

    def view(self):
        graph = copy.deepcopy(self.base)
        graph['nodes'] = [r['node'] for r in self.state['nodes'].values()]
        graph['match_nodes'] = [n for n in graph['nodes'] if n['representative_observations']]
        return graph

    def metrics(self):
        s = self.state
        physical = [a for a in s['attempts'] if a.get('execution', {}).get('physical_execution') is True]
        n = len(physical); ts = s['transitions']
        reused = sum(a['planner_source'] in ('GRAPH','GRAPH_REVISED') for a in physical)
        distribution = {aid:{target:o['observed_count'] for target,o in a['outcomes'].items()
                             if o['observed_count']} for aid,a in s['actions'].items()}
        return dict(total_physical_actions=n, graph_policy_actions=reused,
            gpt_frontier_actions=sum(a['planner_source']=='GPT_FRONTIER' for a in physical),
            recovery_actions=sum(a['planner_source']=='GPT_RECOVERY' for a in physical),
            local_outcome_matches=sum(t['local_match']['result']=='MATCH' for t in ts),
            global_reentries=sum(t['transition_type']=='GLOBAL_REENTRY' for t in ts),
            new_provisional_nodes=sum(r['status']=='PROVISIONAL' for r in s['nodes'].values()),
            no_progress_self_loops=sum(t['transition_type']=='NO_PROGRESS' for t in ts),
            multi_outcome_actions=sum(len(outcomes)>1 for outcomes in distribution.values()),
            additional_observed_outcomes=sum(max(0,len(outcomes)-1) for outcomes in distribution.values()),
            current_nodes=len(s['nodes']), current_transitions=len(ts),
            graph_action_ratio=reused/n if n else None,
            new_node_rate=sum(t['new_node_created'] for t in ts)/n if n else None,
            new_node_rate_definition='post-action new nodes / all physical attempts; initial node discovery excluded',
            observed_outcome_counts=distribution,
            unmatched_physical_attempts=n-len(ts))

    def snapshot(self, *, render=False):
        self.state['version'] += 1
        self.latest_path = self.output / f'growth_{self.state["version"]:06d}.json'
        frozen(self.latest_path, self.state)
        save(self.output/'latest.json', dict(snapshot=str(self.latest_path.resolve()),
                                           sha256=file_hash(self.latest_path)))
        save(self.output/'growth_metrics.json', self.metrics())
        import csv
        columns = ['after_attempt', 'total_physical_actions', 'graph_policy_actions', 'gpt_frontier_actions',
                   'local_outcome_matches', 'global_reentries', 'new_provisional_nodes', 'no_progress_self_loops',
                   'multi_outcome_actions', 'additional_observed_outcomes', 'current_nodes', 'current_transitions',
                   'graph_action_ratio', 'new_node_rate']
        with (self.output/'metrics_history.csv').open('w', newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
            writer.writeheader(); writer.writerows(self.state['metrics_history'])
        if render:
            try: self.render()
            except Exception as exc:
                save(self.output/'render_error.json',dict(error=f'{type(exc).__name__}: {exc}'))
        return self.latest_path

    def render(self):
        import shutil
        import subprocess
        nodes = self.state['nodes']; actions = self.state['actions']
        ids = {k:f's{i}' for i,k in enumerate(nodes)}
        aids = {k:f'a{i}' for i,k in enumerate(actions)}
        lines = ['digraph Experience {', 'rankdir=LR;']
        mermaid = ['flowchart LR']
        for nid,row in nodes.items():
            provisional = row['status']=='PROVISIONAL'
            label = nid + (' [PROVISIONAL]' if provisional else '')
            lines.append(f'{ids[nid]} [label={json.dumps(label)}, shape=ellipse, style=filled, fillcolor="{ "#ffe0a3" if provisional else "#e4eef7" }"];')
            mermaid.append(f'{ids[nid]}["{label}"]')
        for aid,a in actions.items():
            label = aid + (' [preferred]' if a['preferred'] else ' [audit]')
            lines.append(f'{aids[aid]} [label={json.dumps(label)}, shape=box];')
            lines.append(f'{ids[a["source_node"]]} -> {aids[aid]};')
            mermaid.append(f'{ids[a["source_node"]]} --> {aids[aid]}["{label}"]')
            for target,o in a['outcomes'].items():
                types = sorted({t['transition_type'] for t in self.state['transitions'] if t['transition_id'] in o['history']})
                nominal = target==a['nominal_target']
                label = ('nominal; ' if nominal else '') + '/'.join(types) + f' n={o["observed_count"]}'
                color = '#cf3333' if target==a['source_node'] and o['observed_count'] else '#7c3eb5' if 'GLOBAL_REENTRY' in types else '#258a45' if nominal and a['preferred'] else '#999999' if not a['replayable'] and not o['observed_count'] else '#d68a00'
                lines.append(f'{aids[aid]} -> {ids[target]} [label={json.dumps(label)}, color="{color}", style={"solid" if nominal else "dashed"}];')
                mermaid.append(f'{aids[aid]} -->|"{label}"| {ids[target]}')
        lines.append('}')
        dot = self.latest_path.with_suffix('.dot'); dot.write_text('\n'.join(lines)+'\n')
        md = self.latest_path.with_suffix('.md')
        md.write_text('# Online experience graph\n\nState → Action → Outcome State. Orange states remain provisional.\n'
                      'Green: preferred nominal; orange: alternative; red: no progress; purple: re-entry; gray: excluded seed evidence. Counts exclude seed priors.\n\n```mermaid\n'
                      +'\n'.join(mermaid)+'\n```\n\n```json\n'+json.dumps(self.metrics(),indent=2)+'\n```\n')
        if shutil.which('dot'):
            subprocess.run(['dot','-Tsvg',str(dot),'-o',str(self.latest_path.with_suffix('.svg'))],check=True,timeout=60)

    def observe_node(self, observation, *, source=None, action=None, task_progress='initial observation'):
        nid = f'P{len(self.state["nodes"]):06d}'
        while nid in self.state['nodes']: nid += '_new'
        oid = f'growth_{digest(str(self.output.resolve()))[:12]}_{nid}'
        def validate(node):
            jsonschema.validate(node, NODE)
            if node['node_id'] != nid or node['representative_observations'] != [oid] or not node['observation_condition']['positive_evidence']:
                raise ValueError('provisional node needs current representative and decision evidence')
        node = self.requests.ask('recovery_node', dict(task=self.base['goal'], node_id=nid,
            observation_id=oid, task_progress=task_progress), [(oid,observation.image)], validator=validate)
        validate(node)
        rgb = dict(rgb=evidence_file(observation.image))
        self.state['observations'][oid] = rgb; self.trajectory['observations'][oid] = rgb
        self.state['nodes'][nid] = dict(node=node,status='PROVISIONAL',observation_count=0,
            incoming_transitions=[],outgoing_actions=[],outcome_history=[],created_from=source,
            created_by_action=action,task_progress=task_progress,representative_observation=rgb)
        return nid

    def select(self, node):
        choices = [self.state['actions'][aid] for aid in self.state['nodes'][node]['outgoing_actions']
                   if self.state['actions'][aid]['preferred'] and self.state['actions'][aid]['replayable']]
        if len(choices)>1: raise ValueError('ambiguous preferred growth action')
        return choices[0] if choices else None

    def match(self, observation, folder, *, source=None, action=None, threshold=.8):
        graph = self.view(); all_ids = [n['node_id'] for n in graph['match_nodes']]
        local_ids = list(dict.fromkeys(n for n in [source, *self.state['actions'][action]['outcomes']] if n in all_ids)) if source else []
        audit = dict(local_match=dict(candidates=local_ids,result='NOT_USED',confidence=None),
                     global_reentry=dict(used=False,result='NOT_USED',confidence=None),new_node_created=False)
        stages = [('local',local_ids),('global',all_ids)] if source else [('global',all_ids)]
        for stage, candidates in stages:
            if not candidates: continue
            from .graph_policy_matching import reference_images,validate_match_response
            payload = signatures(graph,candidates,with_representatives=True)
            images = reference_images(graph,self.trajectory,observation,candidates)
            subset = dict(nodes=[self.state['nodes'][n]['node'] for n in candidates])
            def validate(value):
                validate_match_response(value,subset,threshold)
            value = self.requests.ask('node_match',payload,images,validator=validate); validate(value)
            ok = reliable_match(value,subset,threshold)
            row = dict(candidates=candidates,result='MATCH' if ok else value['status'] if value['status']!='MATCH' else 'AMBIGUOUS',
                       confidence=value['confidence'],response=value,representative_images=True,image_ids=[k for k,_ in images])
            if stage=='local': audit['local_match']=row
            else: audit['global_reentry']=dict(used=True,**row)
            save(Path(folder)/'outcome_match.json',audit)
            if ok: return value['node_id'], audit
            if value['status']!='NO_MATCH':
                ids=value['candidate_nodes']
                if any(self.state['nodes'][n]['status']=='PROVISIONAL' for n in ids) and len(ids)>1:
                    self.state['merge_candidates'].append(dict(status='MERGE_CANDIDATE',nodes=ids,
                        observation=evidence_file(observation.image),evidence=value,
                        decision_equivalence='UNCONFIRMED_REQUIRES_REVIEW',permanent_merge=False))
                return None,audit
        audit['global_reentry']['result']='NO_MATCH'
        return None,audit

    def transition(self, attempt, node, audit, effect, *, promote=True):
        s=self.state; source=attempt['source_node']; aid=attempt['selected_action']; a=s['actions'][aid]
        kind = ('NO_PROGRESS' if node==source else 'NEW_NODE' if audit['new_node_created'] else
                'GLOBAL_REENTRY' if audit['global_reentry']['used'] else
                'EXPECTED' if node==a['nominal_target'] else 'ALTERNATIVE_OUTCOME')
        tid=f'T{len(s["transitions"]):07d}'
        row=dict(transition_id=tid,attempt_id=attempt['attempt_id'],source_node=source,selected_action=aid,
            result_node=node,transition_type=kind,planner_source=attempt['planner_source'],effect=effect,
            before=attempt['before'],after=attempt['after'],execution=attempt['execution'],**audit)
        s['transitions'].append(row)
        attempt.update(result_node=node,transition_type=kind,**audit)
        o=a['outcomes'].setdefault(node,dict(seed_evidence=False,observed_count=0,history=[]))
        o['observed_count']+=1; o['history'].append(tid)
        s['nodes'][node]['incoming_transitions'].append(tid)
        s['nodes'][source]['outcome_history'].append(tid)
        if a['origin']=='GPT_FRONTIER' and node!=source and effect['progress'] in ('PROGRESS','COMPLETE'):
            a['successful_observations']+=1
            if promote and a['edge'] is not None:
                a.update(preferred=True,replayable=True)
                if a['nominal_target'] is None: a['nominal_target']=node
        s['metrics_history'].append(dict(after_attempt=attempt['attempt_id'],**self.metrics()))
        return row


def growth_ground(runtime, requests, graph, trajectory, action, before, folder, initial_image, previous_image):
    """Use the SAME frozen-edge grounding, side/hem checks and native dispatcher."""
    edge=action['edge']
    hem=None
    if getattr(requests,'minimal_grounder',None) is not None:
        requests.minimal_grounder.template(edge)
    elif edge['action_family']=='bottom_hem_toward_collar_fold':
        hem=verify_hem_frame(requests,trajectory,before,initial_image,previous_image,folder)
    candidate=selected_edge_grounding(requests,graph,edge,before,folder,hem)
    validate_grounding(candidate); save(folder/'grounding.json',candidate)
    if candidate['status']!='READY': raise ValueError('selected graph action cannot be grounded: '+candidate['status'])
    if hem is None and getattr(requests,'minimal_grounder',None) is not None and edge['action_family']=='bottom_hem_toward_collar_fold' and candidate.get('coordinate_frame')!='ACTION_LOCAL':
        hem=candidate['garment_frame']
    if hem is not None: save(folder/'hem_direction_check.json',validate_hem_direction(candidate,hem))
    _,source,measurements=instantiate_grounded(candidate,edge,before)
    save(folder/'garment_side_check.json',grounded_side_check(candidate,edge,before))
    save(folder/'surface_measurements.json',measurements)
    (folder/'motion.py').write_text(source)
    return source


def online_replay(graph_path, output, runtime, requests, *, max_steps=100, threshold=.8,
                  state_path=None, frontier_fn=None, fallback_timeout_s=900, live_graph_port=None,
                  collect_only=False, recover_no_progress=False):
    """One physical episode; reload a frozen growth snapshot for the next reset."""
    if max_steps < 1 or not 0 < threshold <= 1:
        raise ValueError('invalid growth step/confidence limit')
    if collect_only and getattr(requests,'grounding_mode','full')!='full':
        raise ValueError('collect-only requires the full planner, not minimal grounding')
    if recover_no_progress and (collect_only or getattr(requests,'grounding_mode','full')!='full'):
        raise ValueError('no-progress recovery requires Graph replay with full grounding')
    growth=OnlineGrowth(graph_path,output,state_path=state_path); growth.requests=requests
    graph=growth.base; s=growth.state; output=Path(output)
    if runtime.grasp_offset_mm != graph['grasp_offset_mm']:
        raise ValueError('runtime lowering differs from seed')
    episode=dict(episode_id=len(s['episodes'])+1,status='STARTED',steps=[],initial_node=None,recoveries=[],
                 runner_config=dict(max_steps=max_steps,match_threshold=threshold,matching='SOURCE_PLUS_ALL_ACTION_OUTCOMES_THEN_GLOBAL',
                                    collect_only=collect_only,
                                    recover_no_progress=recover_no_progress,
                                    action_selection='FULL_GPT_EVERY_STEP' if collect_only else 'RECORDED_GRAPH_WITH_RECOVERY' if recover_no_progress else 'GRAPH_THEN_FRONTIER',
                                    grounding_mode=getattr(requests,'grounding_mode','full'),
                                    grounding_templates=str(requests.minimal_grounder.path) if getattr(requests,'minimal_grounder',None) else None,
                                    frontier='PROVISIONAL_WITHOUT_REPLAYABLE_ACTION',model_family='GPT-6'))
    s['episodes'].append(episode)
    from .replay_live_graph import create_live_graph
    live=create_live_graph(output,live_graph_port)
    live.update(s,phase='MATCHING',observations=growth.trajectory['observations'])
    try:
        before=runtime.observe(output/'initial_observation'); initial_image=before.image; previous_image=None
        node,audit=growth.match(before,output,threshold=threshold)
        if node is None:
            if audit['global_reentry']['result']!='NO_MATCH':
                episode['status']='AMBIGUOUS'; return episode
            node=growth.observe_node(before)
        live.update(s,current=node,phase='MATCHED',observation=str(before.image))
        episode['initial_node']=node; s['nodes'][node]['observation_count']+=1
        growth.snapshot()
        for index in range(max_steps):
            if file_hash(graph_path)!=s['base_sha256']: raise ValueError('base policy changed during episode')
            if not collect_only and node in graph.get('terminal_node_ids',[]): episode['status']='COMPLETED'; break
            folder=output/f'step_{index:04d}'; folder.mkdir()
            if recover_no_progress:
                from .growth_edge_recovery import select_recorded_action
                action=select_recorded_action(growth,node)
            else:
                action=None if collect_only else growth.select(node)
            if not collect_only and action is None and s['nodes'][node]['status']!='PROVISIONAL':
                episode['status']='NO_REPLAYABLE_ACTION'; break
            mode='GRAPH' if action else 'GPT_FRONTIER'
            if action and action.get('revisions'):mode='GRAPH_REVISED'
            if action is None:
                aid=f'F{len(s["actions"]):07d}'
                action=dict(action_id=aid,source_node=node,edge=None,origin='GPT_FRONTIER',
                    preferred=False,replayable=False,nominal_target=None,outcomes={},successful_observations=0)
                s['actions'][aid]=action; s['nodes'][node]['outgoing_actions'].append(aid)
            aid=action['action_id']
            live.update(s,current=node,action=aid,target=action.get('nominal_target'),phase='GROUNDING' if mode=='GRAPH' else 'FRONTIER')
            attempt=dict(attempt_id=f'A{len(s["attempts"]):07d}',episode_id=episode['episode_id'],
                source_node=node,selected_action=aid,planner_source=mode,before=evidence_file(before.image),
                execution=dict(status='NOT_DISPATCHED',physical_execution=False),command_source=None,
                local_match=dict(candidates=[],result='NOT_OBSERVED',confidence=None),
                global_reentry=dict(used=False,result='NOT_OBSERVED'),new_node_created=False,
                result_node=None,transition_type='UNKNOWN')
            s['attempts'].append(attempt); episode['steps'].append(attempt['attempt_id'])
            growth.snapshot(); save(folder/'dispatch_intent.json',attempt)
            if mode=='GRAPH':
                try:
                    source=growth_ground(runtime,requests,graph,growth.trajectory,action,before,folder,initial_image,previous_image)
                except (ValueError,RuntimeError,jsonschema.ValidationError,IndexError) as exc:
                    episode.update(status='EDGE_INSTANTIATION_FAILURE',error=str(exc))
                    if record_controller_failure(episode,exc):stop_after_failure(runtime,episode)
                    break
                attempt['command_source']=source; growth.snapshot()
                if file_hash(graph_path)!=s['base_sha256']: raise ValueError('base policy changed before dispatch')
                live.update(s,phase='EXECUTING')
                outcome=execute_policy_edge(runtime,source,before,folder,graph,action['edge'])
            else:
                class FrontierRecorder:
                    def __getattr__(self, name): return getattr(runtime, name)
                    def execute(self, code, observation, dispatch_folder):
                        attempt['command_source']=code; growth.snapshot()
                        live.update(s,phase='EXECUTING')
                        execution=runtime.execute(code,observation,dispatch_folder)
                        attempt['execution']=execution; growth.snapshot()
                        return execution
                # The collector's action planner sees only task references/current
                # observation, never Graph nodes, suggested edges or templates.
                planning_graph=({k:copy.deepcopy(graph[k]) for k in ('goal','goal_references') if k in graph}
                                if collect_only else growth.view())
                context=(dict(reason='REPLAY_REVISED_EDGE',selected_edge=action['edge'],source_node=node)
                         if mode=='GRAPH_REVISED' else dict(reason='COLLECT_ONLY_FULL_PLANNER') if collect_only else
                         dict(reason='PROVISIONAL_FRONTIER_WITHOUT_SUCCESSFUL_ACTION',source_node=node))
                result=(frontier_fn or free_plan_one)(FrontierRecorder(),before,folder/'frontier',planning_graph,
                    timeout_s=fallback_timeout_s,budget=requests,
                    context=context)
                outcome=result['outcome']; source=result.get('source'); attempt['command_source']=source
            attempt['execution']=outcome; save(folder/'execution.json',outcome); growth.snapshot()
            if outcome.get('status')!='EXECUTED' or outcome.get('physical_execution') is not True:
                episode['status']=('SAFETY_REJECT' if outcome.get('status') in ('PLANNING_FAILED','PLANNING_INCOMPLETE')
                                   else 'FRONTIER_REPORTED_COMPLETE' if outcome.get('status')=='MODEL_REPORTED_COMPLETE'
                                   else 'ACTION_EXECUTION_FAILURE')
                episode['execution']=outcome
                if record_controller_failure(episode,outcome.get('error','')):stop_after_failure(runtime,episode)
                break
            live.update(s,phase='AFTER_MATCHING')
            after=runtime.observe(folder/'after_observation'); attempt['after']=evidence_file(after.image); growth.snapshot()
            result_node,audit=growth.match(after,folder,source=node,action=aid,threshold=threshold)
            attempt['matching']=audit; attempt['result_node']=result_node; growth.snapshot()
            try:
                effect=requests.ask('growth_effect',dict(task=graph['goal']),[('before',before.image),('after',after.image)])
                jsonschema.validate(effect,SCHEMAS['growth_effect'])
                if effect['progress'] in ('PROGRESS','COMPLETE') and not effect['evidence']:
                    raise ValueError('positive progress requires visual evidence')
            except Exception as exc:
                effect=dict(progress='UNKNOWN',task_progress='Progress assessment unavailable',evidence=[])
                attempt['effect_error']=f'{type(exc).__name__}: {exc}'
            attempt['effect']=effect; attempt['matching']=audit
            if result_node is None:
                if audit['global_reentry']['result']!='NO_MATCH':
                    episode['status']='AMBIGUOUS'; growth.snapshot(); break
                result_node=growth.observe_node(after,source=node,action=aid,task_progress=effect['task_progress'])
                audit['new_node_created']=True
            live.update(s,current=result_node,action=None,target=None,phase='MATCHED',observation=str(after.image))
            s['nodes'][result_node]['observation_count']+=1
            if mode=='GPT_FRONTIER':
                exact=command(source); ops=exact['operations']
                if (len(ops)>=2 and ops[0]['method']=='grasp' and all(o['method'] in ('move_tcp','move_transfer') for o in ops[1:])
                        and all(set(o['arguments'])=={'right'} for o in ops)):
                    # This request describes an executed action, never dispatches or changes it.
                    try:
                        template=requests.ask('growth_action',dict(task=graph['goal'],exact_command=exact),
                                              [('before',before.image),('after',after.image)])
                        jsonschema.validate(template,SCHEMAS['growth_action'])
                        action['edge']=dict(edge_id=aid,source_node=node,target_node=result_node,
                            seed_action=dict(exact_command=exact,source_observation=attempt['before']['path']),**template)
                    except Exception as exc:
                        action['template_error']=f'{type(exc).__name__}: {exc}'
            transition=growth.transition(attempt,result_node,audit,effect,promote=not collect_only)
            save(folder/'transition.json',transition); growth.snapshot()
            live.update(s,current=result_node,action=None,target=None,phase='MATCHED',observation=str(after.image))
            print('ONLINE_GROWTH '+json.dumps(dict(source_node=node, selected_action=aid,
                result_node=result_node, transition_type=transition['transition_type'],
                planner_source=mode, metrics=growth.metrics()),ensure_ascii=False),flush=True)
            previous_image=before.image; before=after; node=result_node
            if recover_no_progress and transition['transition_type']=='NO_PROGRESS':
                if episode['recoveries']:
                    episode.update(status='RECOVERY_UNRESOLVED',error='One recovery action per episode exhausted')
                    break
                from .growth_edge_recovery import recover_self_loop
                recovered=recover_self_loop(growth,runtime,requests,live,episode,action,transition,
                    before,folder/'recovery',frontier_fn or free_plan_one,threshold=threshold,timeout_s=fallback_timeout_s)
                if recovered is None:
                    episode['status']='RECOVERY_UNRESOLVED'
                    failure=episode['recoveries'][-1].get('execution',{})
                    if record_controller_failure(episode,failure.get('error') or episode['recoveries'][-1].get('reason','')):
                        stop_after_failure(runtime,episode)
                    break
                previous_image=before.image;before,node=recovered
                live.update(s,current=node,action=None,target=None,phase='MATCHED',observation=str(before.image))
            if not collect_only and node in graph.get('terminal_node_ids',[]): episode['status']='COMPLETED'; break
        else: episode['status']='STEP_LIMIT'
    except BaseException as exc:
        episode.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',error=f'{type(exc).__name__}: {exc}')
        record_controller_failure(episode,exc)
        stop_after_failure(runtime,episode)
    finally:
        live.update(s,phase='STOPPED',result=episode['status'])
        live.close()
        try: runtime.close()
        except Exception as exc: episode['cleanup_error']=str(exc)
        episode['metrics']=growth.metrics(); episode['request_metrics']=requests.metrics()
        growth.snapshot(render=True)
        episode['growth_state']=str(growth.latest_path.resolve())
        save(output/'episode.json',episode)
    return episode


def run_growth_episodes(args, output):
    """Persist between episodes; the user resets cloth, never the robot blindly."""
    from .discovery_model import CodexMultimodal
    from .graph_policy import make_runtime
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    if getattr(args,'from_scratch',False):
        args.graph=create_empty_base(output/'empty_policy.json')
    state_path=args.growth_state; results=[]; index=0
    while args.episodes==0 or index<args.episodes:
        if index:
            try:
                answer=input('本 episode 已结束。重新摆放衣服、移开手后按 Enter 开始下一 episode；输入 q 结束：')
            except (EOFError,KeyboardInterrupt): break
            if answer.strip().lower() in ('q','quit','exit'): break
        index+=1; folder=output/f'episode_{index:04d}'
        requests=policy_requests(folder/'model',CodexMultimodal(args.model,args.timeout_s),
                                 no_cache=args.no_cache,max_calls=args.max_calls,
                                 grounding_mode=getattr(args,'grounding_mode','full'),
                                 grounding_templates=getattr(args,'grounding_templates',None))
        runtime=None; result={}; old=signal.getsignal(signal.SIGTERM)
        def terminate(*_): raise KeyboardInterrupt('SIGTERM')
        signal.signal(signal.SIGTERM,terminate)
        try:
            runtime=make_runtime(args.config,args.viser_port)
            result=online_replay(args.graph,folder,runtime,requests,max_steps=args.max_steps,
                threshold=args.match_threshold,state_path=state_path,fallback_timeout_s=args.timeout_s,
                collect_only=getattr(args,'collect_only',False),
                recover_no_progress=getattr(args,'recover_no_progress',False),
                live_graph_port=None if getattr(args,'no_live_graph',False) else getattr(args,'graph_port',8090))
        except (Exception,KeyboardInterrupt) as exc:
            result=capture_cli_failure(folder,exc,record_name='episode.json')
            if runtime is not None:stop_after_failure(runtime,result)
        finally:
            signal.signal(signal.SIGTERM,old)
            close_cli_runtime(runtime,result)
        save(folder/'cli_result.json',result)
        if result.get('growth_state'):
            state_path=Path(result['growth_state'])
            save(output/'latest.json',dict(snapshot=str(state_path.resolve()),sha256=file_hash(state_path)))
        results.append(result)
        save(output/'episodes.json',results)
        print(json.dumps(dict(episode=index,status=result['status'],metrics=result.get('metrics',{})),ensure_ascii=False),flush=True)
        from .replay_stop_summary import print_stop_summary
        print_stop_summary(folder,result)
        if cli_failed(result):
            break
    return dict(status=results[-1]['status'] if results else 'STOPPED',episodes=len(results),
                growth_state=str(state_path) if state_path else None,
                cli_failed=cli_failed(results[-1]) if results else False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument('--graph', type=Path)
    start.add_argument('--from-scratch', action='store_true', help='start with zero nodes/actions; online growth only')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--online-growth', action='store_true', help='collect multi-outcome experience and reuse successful frontier actions')
    parser.add_argument('--collect-only', action='store_true',
                        help='online growth with a fresh full GPT action every step; Graph records experience but never selects actions')
    parser.add_argument('--recover-no-progress', action='store_true',
                        help='replay recorded growth actions; repair a self-loop once per episode and replace its edge only after verified progress')
    parser.add_argument('--growth-state', type=Path, help='frozen growth snapshot or latest.json pointer; always observe anew')
    parser.add_argument('--episodes', type=int, default=1, help='growth episodes; 0 continues until stopped, with manual cloth reset between episodes')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/dual_arm.local.json')
    parser.add_argument('--viser-port', type=int, default=8088)
    parser.add_argument('--graph-port', type=int, default=8090, help='read-only live graph on localhost; 0 selects an available port')
    parser.add_argument('--no-applicability-recovery', action='store_true', help='disable bounded applicability escalation after minimal grounding recapture fails')
    parser.add_argument('--no-live-graph', action='store_true', help='disable live graph telemetry')
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--repair-self-loop', action='store_true', help='repair self-loop or pre-dispatch IK failure once per edge; automatically reuse validated template revisions')
    parser.add_argument('--retry-missed-grasp', action='store_true', help='before/after unchanged garment review; one retry per edge at original XY and grasp Z minus 2mm; disables obstruction debug')
    parser.add_argument('--debug-grasp-retry', action='store_true', help='obstruction debug only: wide stalled close +2mm prevalidated lift; no empty-grasp detection/retry')
    parser.add_argument('--initial-node', choices=['S0'],
                        help='explicit standard initial placement: verify S0 first, broaden on failure')
    parser.add_argument('--max-fallbacks', type=int, default=0,
                        help='legacy graph-extending free planning; requires --max-recovery-attempts 0')
    parser.add_argument('--max-recovery-attempts',type=int,choices=[0,1],default=1,
                        help='one grasp-handle recovery on NO_ROBUST_ACTION, without adding Graph nodes (default: 1)')
    parser.add_argument('--match-threshold', type=float, default=.8)
    parser.add_argument('--max-calls', type=int, default=64)
    parser.add_argument('--timeout-s', type=float, default=900)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--grounding-mode',choices=['full','minimal'],default='full',
                        help='minimal: two visual anchors; requires a compiled Host template for every selected edge')
    parser.add_argument('--grounding-templates',type=Path,help='frozen minimal-mode Host template bundle')
    args = parser.parse_args(argv)
    if args.collect_only and (not args.online_growth or args.grounding_mode!='full'):
        parser.error('--collect-only requires --online-growth and --grounding-mode full')
    if args.recover_no_progress and (not args.online_growth or args.collect_only or args.grounding_mode!='full'):
        parser.error('--recover-no-progress requires --online-growth, full grounding, and no --collect-only')
    if not args.online_growth and args.max_recovery_attempts and (
            args.max_fallbacks or args.repair_self_loop or args.retry_missed_grasp or args.debug_grasp_retry):
        parser.error('grasp-handle recovery requires --max-fallbacks 0 and no legacy repair/retry modes; use --max-recovery-attempts 0 for legacy behavior')
    if args.grounding_mode=='minimal':
        if args.grounding_templates is None and args.graph is not None:
            compiled_graph=load_frozen(args.graph)
            if args.repair_self_loop:
                from .edge_self_loop_repair import latest_repair_templates
                args.grounding_templates=latest_repair_templates(args.graph)
                if args.grounding_templates is not None:
                    print('Reusing validated edge repair templates: '+str(args.grounding_templates))
            if args.grounding_templates is None and compiled_graph.get('minimal_templates_path'):
                args.grounding_templates=Path(compiled_graph['minimal_templates_path'])
        if args.grounding_templates is None:parser.error('--grounding-mode minimal requires compiled graph or --grounding-templates')
        from .graph_minimal_grounding import MinimalGrounder
        MinimalGrounder(args.grounding_templates)
    if args.repair_self_loop and (args.online_growth or args.grounding_mode != "minimal"):
        parser.error("--repair-self-loop requires minimal replay (not online growth)")
    if args.retry_missed_grasp and (args.online_growth or args.grounding_mode != "minimal" or args.max_fallbacks != 0):
        parser.error("--retry-missed-grasp requires minimal strict replay with --max-fallbacks 0")
    if args.debug_grasp_retry and (args.online_growth or args.grounding_mode != 'minimal' or args.max_fallbacks != 0):
        parser.error('--debug-grasp-retry requires minimal strict replay with --max-fallbacks 0')
    if args.max_fallbacks < 0 or args.max_steps < 1 or args.max_calls < 1 or args.timeout_s <= 0 or not 0 < args.match_threshold <= 1:
        parser.error('invalid step/call/time/confidence limit')
    if args.episodes < 0 or ((args.growth_state or args.episodes != 1) and not args.online_growth):
        parser.error('growth-state/episodes require --online-growth; episodes must be nonnegative')
    if args.from_scratch and (not args.online_growth or args.growth_state):
        parser.error('--from-scratch requires --online-growth and cannot load --growth-state')
    if not args.from_scratch:
        graph = load_frozen(args.graph)
        (validate_growth_base if args.online_growth else validate_policy)(graph)
    output = args.output or (ROOT/'runs' if args.from_scratch else args.graph.parent) / (('policy_replay_' if args.execute else 'policy_dry_run_')
        + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if output.exists(): parser.error('output already exists; never resume a physical dispatch')
    if not args.execute:
        if args.online_growth:
            growth_output=output
            if args.from_scratch:
                output.mkdir(parents=True,exist_ok=False)
                args.graph=create_empty_base(output/'empty_policy.json')
                growth_output=output/'initial_graph'
            growth=OnlineGrowth(args.graph,growth_output,state_path=args.growth_state)
            growth.snapshot(render=True)
            if args.from_scratch:
                save(output/'latest.json',dict(snapshot=str(growth.latest_path.resolve()),sha256=file_hash(growth.latest_path)))
            result=dict(status='DRY_RUN_OK', mode='ONLINE_GROWTH_STATIC_ONLY',
                        collect_only=args.collect_only,
                        recover_no_progress=args.recover_no_progress,
                        action_selection='FULL_GPT_EVERY_STEP' if args.collect_only else 'RECORDED_GRAPH_WITH_RECOVERY' if args.recover_no_progress else 'GRAPH_THEN_FRONTIER',
                        growth_state=str(growth.latest_path), metrics=growth.metrics(),robot_calls=0,model_calls=0)
        else:
            result = dry_run(args.graph, output)
        result['grounding_mode']=args.grounding_mode
        if args.grounding_mode=='minimal':
            minimal=MinimalGrounder(args.grounding_templates)
            result['minimal_compiled_edges']=sorted(minimal.bundle['edges'])
            result['minimal_scope']='Only compiled edges supported; this static check does not run RGB grounding, depth, IK or robot execution.'
    elif args.online_growth:
        result = run_growth_episodes(args, output)
    else:
        from .discovery_model import CodexMultimodal
        from .graph_policy import make_runtime
        requests = policy_requests(output / 'model', CodexMultimodal(args.model, args.timeout_s),
                                   no_cache=args.no_cache, max_calls=args.max_calls,
                                   grounding_mode=args.grounding_mode,grounding_templates=args.grounding_templates)
        runtime = None; result={}
        previous = signal.getsignal(signal.SIGTERM)
        def terminate(*_): raise KeyboardInterrupt('SIGTERM')
        signal.signal(signal.SIGTERM, terminate)
        try:
            runtime = make_runtime(args.config, args.viser_port)
            result = replay(args.graph, output, runtime, requests,
                            max_steps=args.max_steps, threshold=args.match_threshold,
                            max_fallbacks=args.max_fallbacks, fallback_timeout_s=args.timeout_s, initial_node=args.initial_node,
                            debug_grasp_retry=args.debug_grasp_retry,retry_missed_grasp=args.retry_missed_grasp,repair_self_loop=args.repair_self_loop,
                            live_graph_port=None if args.no_live_graph else args.graph_port,
                            applicability_recovery=args.max_recovery_attempts==0 and args.grounding_mode=='minimal' and not args.no_applicability_recovery,
                            max_recovery_attempts=args.max_recovery_attempts)
        except (Exception,KeyboardInterrupt) as exc:
            result=capture_cli_failure(output,exc)
            if runtime is not None:stop_after_failure(runtime,result)
        finally:
            signal.signal(signal.SIGTERM, previous)
            close_cli_runtime(runtime,result)
        save(output/'cli_result.json',result)
    print(json.dumps(dict(output=str(output), **result), ensure_ascii=False, indent=2))
    from .replay_stop_summary import print_stop_summary
    print_stop_summary(output,result)
    return 1 if cli_failed(result) or result.get('cli_failed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
