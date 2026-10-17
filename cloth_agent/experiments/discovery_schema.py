"""Output contracts constrain structure, never cloth factor/class vocabulary."""
import jsonschema

S={'type':'string'}
N={'type':'number','minimum':0,'maximum':1}
def array(x):return {'type':'array','items':x}
def obj(**fields):return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
def enum(*x):return {'type':'string','enum':list(x)}
IDS=array(S)
FACTOR=obj(id=S,name=S,description=S,possible_values=IDS,decision_relevance=S,visual_evidence=S,uncertainty_condition=S)
CLASS=obj(id=S,description=S,decision_rule=S,representative_sample_ids=IDS)
TAXONOMY=obj(factors=array(FACTOR),classes=array(CLASS),hypothesis=S,
    representation=enum('DISCRETE','FACTORIZED','HISTORY_CONDITIONED','HYBRID','INSUFFICIENT'),limitations=IDS)
ASSIGNMENT=obj(sample_id=S,factor_values={'type':'object','additionalProperties':obj(value=S,confidence=N,evidence=S)},
    provisional_state_class=S,next_action_family_without_historical_action=S,critical_uncertainty=IDS,confidence=N)
REVISED_ASSIGNMENT=obj(sample_id=S,factor_values={'type':'object','additionalProperties':obj(value=S,confidence=N,evidence=S)},
    state_class=S,action_family=S,critical_uncertainty=IDS,confidence=N)
PAIR=obj(samples=IDS,evidence=S,uncertainty=S)
ACTION_EVIDENCE=obj(samples=IDS,evidence=S,uncertainty=S)
FROZEN_ACTION=obj(covered_sample_ids=IDS,
    sample_assessments=array(obj(sample_id=S,grasp_explanation=S,target_explanation=S,
        displacement_explanation=S,geometry_explanatory_power=enum('HIGH','MEDIUM','LOW','UNKNOWN'),
        action_family=S,unexplained_action_variation=IDS,confidence=N)),
    common_action_families=array(obj(name=S,definition=S,samples=IDS,evidence=S,limitations=IDS)),
    supporting_evidence=array(ACTION_EVIDENCE),
    appearance_similar_action_different=array(ACTION_EVIDENCE),
    appearance_different_action_equivalent=array(ACTION_EVIDENCE),
    proposed_additional_factors=array(obj(name=S,reason=S,visible_evidence=S,samples=IDS,
        status=enum('SUPPORTED_CANDIDATE','HYPOTHESIS_ONLY','NOT_VISUALLY_OBSERVABLE'))),
    needs_additional_factor=enum('YES','NO','UNRESOLVED'),
    conclusion=S,limitations=IDS)
OP=obj(operation=enum('KEEP','SPLIT','MERGE','REDEFINE'),classes=IDS,affected_samples=IDS,reason=S)
ACTION_REVIEW=obj(class_id=S,action_consistency=enum('HIGH','MEDIUM','LOW'),common_action_relations=IDS,
    visual_factors_that_explain_action=IDS,visual_factors_that_do_not_explain_action=IDS,
    suggested_operation=enum('KEEP','SPLIT','MERGE','REDEFINE'),reason=S,affected_samples=IDS)
TRANSITION=obj(class_id=S,action_family=S,sample_count={'type':'integer','minimum':0},
    outcome_groups=array(obj(outcome=S,samples=IDS)),transition_consistency=enum('HIGH','MEDIUM','LOW'),
    requires_split={'type':'boolean'},contrastive_visual_difference=IDS,unresolved_hidden_difference=IDS,
    aliasing=enum('POSSIBLE_STATE_ALIASING','NOT_ESTABLISHED','VISUAL_DIFFERENCE_SUPPORTED'))


def schema(stage, kind):
    if stage=='outcome_pair':
        from .outcome_pair_schema import contract
        return contract(kind)
    if stage=='action' and kind.startswith('frozen_'):return FROZEN_ACTION
    if stage=='image_only' and kind=='classify':return obj(covered_sample_ids=IDS,assignments=array(ASSIGNMENT))
    if stage=='image_only':return obj(covered_sample_ids=IDS,taxonomy=TAXONOMY,
        ambiguous_samples=IDS,irrelevant_visual_variation=IDS,unresolved_disagreements=IDS)
    common=dict(covered_sample_ids=IDS,taxonomy=TAXONOMY,assignments=array(REVISED_ASSIGNMENT),
        operations=array(OP),unresolved_disagreements=IDS)
    if stage=='action':return obj(**common,class_review=array(ACTION_REVIEW),
        appearance_different_action_equivalent=array(PAIR),appearance_similar_action_different=array(PAIR))
    return obj(**common,transition_review=array(TRANSITION),
        research_answers=obj(**{f'Q{i}':obj(answer=S,evidence_sample_ids=IDS,confidence=N,limitations=IDS) for i in range(1,7)}),
        best_supported_representation=S,negative_results=IDS)


def validate(value, stage, kind, expected_ids, taxonomy=None, evidence_ids=None):
    jsonschema.validate(value,schema(stage,kind))
    expected=set(expected_ids)
    if len(value['covered_sample_ids'])!=len(expected) or set(value['covered_sample_ids'])!=expected:
        raise ValueError('covered_sample_ids must exactly cover this request, without duplicates')
    def check_refs(x,key=''):
        if isinstance(x,dict):
            for k,v in x.items():check_refs(v,k)
        elif isinstance(x,list):
            if key in ('covered_sample_ids','ambiguous_samples','representative_sample_ids','affected_samples','samples','evidence_sample_ids') and any(s not in expected for s in x):
                raise ValueError(f'unknown sample ID in {key}')
            for v in x:check_refs(v)
    check_refs(value)
    if stage=='outcome_pair':
        from .outcome_pair_schema import validate_evidence
        validate_evidence(value,kind,expected_ids,evidence_ids)
        return
    if stage=='action' and kind.startswith('frozen_'):
        ids=[a['sample_id'] for a in value['sample_assessments']]
        if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('sample_assessments must cover all expected samples once')
        for field in ('appearance_similar_action_different','appearance_different_action_equivalent'):
            if any(len(set(p['samples']))!=2 for p in value[field]):raise ValueError('counterexamples must identify two distinct samples')
        return
    tax=value.get('taxonomy',taxonomy)
    if tax:
        for field in ('factors','classes'):
            ids=[x['id'] for x in tax[field]]
            if len(set(ids))!=len(ids):raise ValueError(f'duplicate {field} IDs')
    if 'assignments' in value:
        assignments=value['assignments']
        ids=[a['sample_id'] for a in assignments]
        if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('assignments must cover all expected samples once')
        allowed={x['id'] for x in tax['classes']}|{'UNKNOWN','NEED_MORE_INFORMATION','NEW_CLASS_CANDIDATE'}
        factors={x['id'] for x in tax['factors']}
        for a in assignments:
            c=a.get('state_class',a.get('provisional_state_class'))
            if c not in allowed:raise ValueError('assignment class absent from taxonomy')
            if set(a['factor_values'])!=factors:raise ValueError('factor_values must cover taxonomy factors; use UNKNOWN when needed')
    if 'transition_review' in value:
        for r in value['transition_review']:
            ids=[s for group in r['outcome_groups'] for s in group['samples']]
            if r['sample_count']!=len(set(ids)):raise ValueError('transition sample_count inconsistent with outcome groups')
