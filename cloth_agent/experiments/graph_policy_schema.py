"""Structural contracts only; all cloth semantics are supplied by runtime GPT."""
from .discovery_schema import S, N, IDS, array, obj, enum

B = {'type': 'boolean'}
I = {'type': 'integer', 'minimum': 0}
CONDITION = obj(positive_evidence=IDS, negative_evidence=IDS, uncertainty_conditions=IDS)
NODE = obj(node_id=S, task_context=S, observation_condition=CONDITION,
           representative_observations=IDS, confidence=N)
TEMPLATE = obj(grasp_relation=S, target_relation=S, grasp_z_rule=S, yaw_rule=S,
               other_constraints=IDS)
EDGE = obj(edge_id=S, source_node=S, target_node=S, action_family=S,
           action_template=TEMPLATE, seed_step_id=S,
           seed_action=obj(exact_command={'type': 'object'}, source_observation=S),
           observed_outcome=S, outcome_evidence=IDS,
           seed_outcome=enum('SUCCESS', 'PROGRESS', 'NO_PROGRESS', 'FAILURE', 'UNKNOWN'),
           generalization_status=enum('ONE_SHOT_HYPOTHESIS', 'OBSERVED_FAILURE'))
GRAPH = obj(graph_id=S, nodes=array(NODE), edges=array(EDGE), limitations=IDS)
MATCH = obj(status=enum('MATCH', 'NO_MATCH', 'AMBIGUOUS'), node_id=S,
            candidate_nodes=IDS, matched_evidence=IDS, violated_conditions=IDS,
            unresolved_conditions=IDS, confidence=N)
MATCH['properties']['violated_conditions'] = {**MATCH['properties']['violated_conditions'], 'description': 'For MATCH, violations of the selected node ONLY, never reasons for rejecting other candidates. Valid MATCH requires an empty array.'}
MATCH['properties']['unresolved_conditions'] = {**MATCH['properties']['unresolved_conditions'], 'description': 'For MATCH, unresolved critical conditions of the selected node ONLY. Valid MATCH requires an empty array.'}
VECTOR2 = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 2, 'maxItems': 2}
VECTOR3 = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
SLOT = obj(operation_index=I, pixel_xy=VECTOR2,
           z_above_surface_mm={'type': 'number', 'minimum': 0, 'maximum': 300},
           rpy_rad=VECTOR3, relation_evidence=S)
# Optional for legacy artifacts; new graph compilation explicitly requests this seed-only evidence.
EDGE['properties']['minimal_visual_evidence'] = obj(
    status=enum('RESOLVED','UNKNOWN'), confidence=N,
    garment_frame=obj(collar_pixel_xy=VECTOR2,hem_center_pixel_xy=VECTOR2,visual_evidence=S),
    registration_roi={'type':'array','items':I,'minItems':4,'maxItems':4})
INSTANTIATE = obj(status=enum('READY', 'BLOCKED'), edge_id=S, action_family=S,
                  slots=array(SLOT), explanation=S, unresolved_conditions=IDS)
CONFORMANCE = obj(status=enum('CONFORMS', 'VIOLATES', 'UNKNOWN'),
                  evidence=IDS, violated_constraints=IDS, unresolved_conditions=IDS, confidence=N)
ASSESS = obj(expected_target=enum('REACHED', 'NOT_REACHED', 'UNKNOWN'),
             matched_node_id=S, task_goal=enum('COMPLETE', 'INCOMPLETE', 'UNKNOWN'),
             acquisition=enum('HELD', 'MISS', 'UNKNOWN'), evidence=IDS,
             failure_category=enum('NONE', 'NODE_MISMATCH', 'EDGE_GENERALIZATION_FAILURE',
                                   'ACTION_EXECUTION_FAILURE', 'UNKNOWN'),
             limitations=IDS, confidence=N)

COMMON = '''You are the runtime GPT-6 for a one-shot cloth graph-policy experiment.
Return only JSON matching the supplied schema. Images are attached in image_ids order.
Keep observations, controller facts and one-shot hypotheses separate. Controller EXECUTED
or completed gripper motion does not prove cloth acquisition or task completion.
Arm7 applies parallel-jaw symmetry: it may postmultiply ALL grasp/held-transport
orientations by the SAME tool-local Z half-turn to select the closer initial orientation.
This preserves XYZ and relative world rotation, not a new action family. The effective
command and grasp_orientation_policy.json record the host adjustment.
No private chain-of-thought; only concise evidence and uncertainty. Never infer empty jaws
from unchanged cloth, or assume deeper grasp is better. Preserve negative results.'''
PROMPTS = {
    'compile': '''Compile ALL supplied seed steps into a connected observation-action graph covering the supplied task scope.
For full_fold, preserve the full sequence across sleeve, torso and hem transitions actually observed.
Do not collapse the trajectory into a single sleeve task. Consecutive observed transitions share a node.
Goal reference images describe desired stages, not observations or evidence of execution.
If coverage is PARTIAL_TRAJECTORY, retain that limitation and never invent missing stages.
No previous graph is supplied. Include all physical attempts with before/after evidence;
identify missing evidence honestly. Each edge cites exactly one seed_step_id and copies its
commanded_action into seed_action.exact_command verbatim, with source_observation equal to
its before_observation. Do not fabricate commands or observed transitions. Every seed
physical attempt must appear in an edge, even failures. Nodes specify decision-relevant
conditions for reusing outgoing edges, not absolute pixels or universal claims from one image.
All representative_observations and outcome_evidence must use supplied evidence IDs.
A failed action must have seed_outcome FAILURE and generalization_status OBSERVED_FAILURE;
otherwise use ONE_SHOT_HYPOTHESIS, never VALIDATED. Missing or interrupted execution
without a known result is UNKNOWN, not proof of physical failure. For missing after RGB,
you may create a zero-confidence unknown sink with no representatives, uncertainty conditions
and no outgoing edges. Such sinks and their incoming edges are audit-only, never replayable. UNKNOWN is appropriate when acquisition
or task effect cannot be established. No edge is established as a general policy by one success.
Describe reusable relative grasp/target relations, necessary transfer constraints, Z and yaw rules.
The default Host template uses only measured Pa/Pb and world up. Do not require collar,
hem or registration texture for action reuse; minimal_visual_evidence is legacy-only metadata
and can be omitted. Host derives position and relative tool rotations from the executed command.
Use NO_PROGRESS when only translation/wrinkles changed; do not label that useful PROGRESS.
The current arm7 executor applies its recorded grasp offset ONCE; templates must not double it.
Replay preserves the exact primitive sequence (grasp then transfer/TCP slots), allowing only
new image-local anchors, nonnegative surface-relative heights, and orientations. If that
cannot express a safe generalization, explicitly state the uncertainty, do not invent a strategy.''',
    'match': '''NODE MATCHING ONLY. Never match or nominate nodes without representative observations;
these are unobserved audit-only sinks. Match the current RGB to graph nodes for reusing outgoing
edges in this task. Do not propose, choose, instantiate or freely plan any action.
MATCH requires one reliable node, positive evidence, no violated or unresolved decision-critical
condition. Return AMBIGUOUS with candidate nodes when multiple nodes remain reasonable.
NO_MATCH uses empty node_id. Report uncertainty rather than force a match.''',
    'instantiate': '''Instantiate ONLY one of the supplied eligible existing outgoing edges.
Do not change action_family or strategy; do not add/remove/reorder primitive slots. For each
operation in that edge's seed command choose a current RGB pixel anchor (x,y, top-left origin),
height above its measured surface in mm, and world roll/pitch/yaw radians. The host samples
actual RGB-D at each anchor. The grasp slot MUST have z_above_surface_mm=0: the unchanged
arm7 host lowering is applied once. Heights for held transfers must implement the edge template.
Do not copy old absolute XYZ blindly. Retain edge relations/constraints and explain the visible
anchor correspondence. If a template needs a different primitive sequence or cannot be
instantiated from current evidence, return BLOCKED. Never invent a new edge or free plan.''',
    'conformance': '''Independently audit this candidate instantiation against the existing edge,
current image, surface measurements and task. This is NOT planning: do not repair or propose
new actions. CONFORMS only if grasp, target, heights, orientation and every primitive preserve
the edge's action family and ALL template constraints. Primitive equality alone is insufficient.
Reject uncertain semantic correspondence. Account for the host grasp offset applied once.''',
    'assess': '''Audit the executed existing edge using before/after RGB and actual execution
evidence. Did after reach its expected target node? Does it meet the ENTIRE supplied task goal, not just an intermediate fold?
Do not equate controller success with task success. Cite supplied evidence IDs; distinguish
observable effects from inferred acquisition. Diagnose NODE_MISMATCH only with evidence the
original match was wrong, otherwise EDGE_GENERALIZATION_FAILURE if a conforming execution
fails the expected transition; use UNKNOWN for insufficient evidence. Do not modify the graph.
If graph.terminal_node_ids is present, those user-designated nodes define the accepted task endpoint.
Judge their visible node conditions; do not require further actions beyond that accepted endpoint.
If no edge was executed (terminal node), evaluate current image and goal without claiming reuse.''',
}
SCHEMAS = dict(compile=GRAPH, match=MATCH, instantiate=INSTANTIATE,
               conformance=CONFORMANCE, assess=ASSESS)

MERGE = obj(groups=array(obj(node=NODE, source_node_ids=IDS, visual_evidence=IDS,
                            action_compatibility=S, limitations=IDS)),
            rejected_merges=array(obj(node_ids=IDS, reason=S, evidence_observations=IDS)), limitations=IDS)
SCHEMAS['merge'] = MERGE
PROMPTS['merge'] = """Inspect ALL attached seed RGB images and consolidate the existing trajectory nodes
into visually equivalent cloth states usable for graph replay. The user explicitly requests merging
similar observations into the same state, not retaining an iteration-per-node chain. Use meaningful
cloth configuration, not small wrinkle, translation or numeric action differences. However do not merge
a meaningful fold-stage/layer/boundary change merely to make the graph connected. Nodes in the same
state may have different outgoing actions; action identity is NOT the state identity. Explain shared
positive and negative visual evidence and uncertainty. No predetermined number of states.
Return a partition: every old node exactly once. Each new node's representatives must be the exact
union of its members' observations. Use simple IDs S0,S1,...; provide short descriptive task_context.
UNKNOWN sink remains separate with no images, zero confidence, no invented visual condition.
The host rewires every original directed action edge without changing its action, outcome or identity.
A failed action can become a self-loop; never upgrade it to success or claim an unobserved edge.
Preserve evidence against merges and necessary distinctions. All future matching is against current
RGB and these states; node conditions must describe appearance rather than iteration history.
Goal reference images are not seed observations. Do not infer success just from controller execution.
"""


def seed_compiler_schemas():
    """Action-local compilation does not require legacy garment-frame annotations."""
    import copy
    schemas = copy.deepcopy(SCHEMAS)
    return schemas
