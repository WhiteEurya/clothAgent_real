"""Evidence-cited outcome judgments; no taxonomy or policy output."""
from .discovery_schema import S,N,IDS,obj,array,enum

EVIDENCE={'type':'array','minItems':1,'items':obj(source_id=S,observation=S)}
def judgment(*statuses):
    return obj(status=enum(*statuses),evidence=EVIDENCE,confidence=N,limitations=IDS)

SAMPLE=obj(covered_sample_ids=IDS,sample_id=S,
    acquisition=judgment('HELD','MISS','UNKNOWN'),
    task_progress=judgment('IMPROVED','UNCHANGED','WORSE','UNKNOWN'),
    completion=judgment('COMPLETE','INCOMPLETE','UNKNOWN'),
    side_effect=judgment('LOCAL','WHOLE_GARMENT_COUPLING','SLIP','OTHER','UNKNOWN'),
    execution_reach=judgment('TARGET_REACHED','TRANSPORT_NOT_REACHED','UNKNOWN'),
    evaluator_agreement=S,conflicting_evidence=IDS)
PAIR=obj(covered_sample_ids=IDS,pair_id=S,
    pair_type=enum('GRASP_Z_PAIR','TARGET_PAIR','GRASP_SUBREGION_PAIR'),
    before_comparability=judgment('REASONABLE','PARTIAL','NOT_COMPARABLE','UNKNOWN'),
    changed_action_variables=array(obj(variable=S,sample_a_value=S,sample_b_value=S,evidence=EVIDENCE)),
    acquisition_difference=judgment('DIFFERENT','SAME','UNKNOWN'),
    task_progress_difference=judgment('DIFFERENT','SAME','UNKNOWN'),
    parameter_effect=judgment('SUPPORTED_ASSOCIATION','NOT_SUPPORTED','UNRESOLVED'),
    visible_state_difference=judgment('PRESENT','NOT_ESTABLISHED','UNKNOWN'),
    classification=enum('ACTION_PARAMETER_EFFECT','VISUAL_STATE_DIFFERENCE','BOTH','NO_OUTCOME_DIFFERENCE','UNRESOLVED'),
    classification_evidence=EVIDENCE,
    evidence_strength=enum('STRONG','MODERATE','WEAK','NONE'),
    multiple_actions_feasible=judgment('SUPPORTED','NOT_ESTABLISHED','UNKNOWN'),
    rgb_state_aliasing=judgment('POSSIBLE','NOT_ESTABLISHED','UNKNOWN'),
    confounders=IDS,conclusion=S,limitations=IDS)

COMMON='''This is Stage 3A controlled outcome pair audit of historical records, NOT a randomized experiment. Never modify taxonomy, invent states, train a model, produce a policy or recommend robot execution. Distinguish acquisition, task progress, completion and side effects. Historical evaluator and host policy labels are fallible interpretations, never physical ground truth. UNCHANGED does NOT imply empty grasp. Deeper commanded Z is NOT automatically better. Different commanded actions do NOT imply one is superior. A single failure cannot establish causation. Controller motion success and gripper contact flags do not prove held cloth. Simulated telemetry is excluded from physical evidence. Compare commanded programs with actual action events: requested targets may have been withheld or replaced. If acquisition failed or transport was not reached, target distance was not physically tested. Before/after are fixed perception views where available; grasp/lift images come from a moving wrist camera, so framing change alone is not cloth motion. Treat retry captures as separate phases, not independent trials. Distinguish observed facts, evaluator assertions and hypotheses. Cite source_id for EVERY judgment, including UNKNOWN (cite availability or conflicting sources). Keep observations concise and testable, without private chain-of-thought. Only supplied evidence is available; missing images/records must stay missing. Inference of an action parameter effect is at most a retrospective association with limitations, never a causal proof.'''

def instruction_for(kind):
    if kind=='sample':
        return COMMON+''' Independently assess this ONE sample without seeing its pairing or other samples' judgments. Derive HELD/MISS/UNKNOWN from direct acquisition/lift evidence where possible, not unchanged final images alone. A pre-close image cannot show post-close acquisition. Use UNKNOWN when ambiguity or timing prevents a reliable distinction. Derive task_progress and completion from the actual visible change toward the fixed sleeve goal, and distinguish local effect, whole-garment coupling, slip and unknown. Check whether the recorded target was actually reached. Explicitly note disagreements with the historical evaluator; do not copy its next-experiment suggestions. State limitations of still-image evidence.'''
    return COMMON+''' Audit exactly this pair, with its two previously independent sample assessments and original multimodal evidence. First assess whether before appearances form a reasonable comparison, independent of outcomes. Then enumerate ALL changed commanded/actual action variables, including XYZ, yaw, lifted heights and execution sequence, not just the nominal pair type. Check acquisition and task-progress differences separately. Identify residual visible-state differences, different runs/calibrations, retries, sensing changes and action confounders. Test whether the observed outcome difference supports the named action parameter association. Do not turn two failures into proof of no effect; do not call two UNKNOWN outcomes equivalent. ACTION_PARAMETER_EFFECT or BOTH requires an observed outcome contrast plus a defensible parameter association; NO_OUTCOME_DIFFERENCE requires evidenced sameness of both acquisition and progress. Multiple actions both feasible requires positive evidence of acquisition and useful task progress for both, not merely that both commands ran. RGB aliasing is only a possibility when similar before images and sufficiently controlled actions leave unexplained physical differences, not a synonym for generic uncertainty. If actual transport was withheld, explicitly state that the proposed target effect cannot be tested. Use the requested five-way classification and an evidence strength rating for the named parameter effect (NONE if no effect is identified). Preserve null, negative and unresolved findings.'''


def contract(kind):
    if kind=='sample':return SAMPLE
    if kind=='pair':return PAIR
    raise ValueError('unsupported Stage3A operation')


def validate_evidence(value,kind,expected_ids,evidence_ids):
    allowed=set(evidence_ids or [])
    def visit(x):
        if isinstance(x,dict):
            if 'source_id' in x and x['source_id'] not in allowed:raise ValueError('unknown evidence source_id')
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(value)
    if kind=='sample' and (len(expected_ids)!=1 or value['sample_id']!=expected_ids[0]):
        raise ValueError('independent sample assessment must match exactly one sample')
    if kind=='pair':
        if len(set(expected_ids))!=2:raise ValueError('outcome audit requires two samples')
        if value['classification']=='NO_OUTCOME_DIFFERENCE' and any(value[k]['status']!='SAME' for k in ('acquisition_difference','task_progress_difference')):
            raise ValueError('NO_OUTCOME_DIFFERENCE requires evidenced sameness, not UNKNOWN')
        if value['classification'] in ('ACTION_PARAMETER_EFFECT','BOTH') and value['parameter_effect']['status']!='SUPPORTED_ASSOCIATION':
            raise ValueError('parameter effect classification requires supported association')
