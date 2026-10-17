"""Joint image-reference adjudication for contradictory no-action recaptures."""
import time
from .graph_policy_schema import MATCH
from .graph_policy_matching import signatures, reference_images, validate_match_response
from .graph_policy import reliable_match, save, evidence_file

SCHEMA=dict(type='object',properties=dict(
    previous_match=MATCH,current_match=MATCH,
    observation_relation=dict(type='string',enum=['SAME_DECISION_CONDITION','CHANGED_DECISION_CONDITION','UNKNOWN']),
    comparison_evidence=dict(type='array',items=dict(type='string',minLength=1),minItems=1)),
    required=['previous_match','current_match','observation_relation','comparison_evidence'],additionalProperties=False)
PROMPT='''Compare previous_current and current RGB SIDE BY SIDE against the same attached candidate reference images. No garment manipulation was commanded between these two observations; this does not prove the cloth is unchanged. Inspect actual layer boundaries, overlays and accessible regions. Changes in people/background, lighting, small wrinkles, or pixels alone do not establish a different decision state. Compare garment configuration and manipulation-relevant conditions.
Return a match for EACH observation, and SAME_DECISION_CONDITION / CHANGED_DECISION_CONDITION / UNKNOWN with concise visible comparison evidence. If both views show the same decision condition, their reliable matches must agree. Do not infer a desired node from ordering and do not plan an action. References are primary evidence. MATCH violations/unresolved conditions refer ONLY to the selected node; rejecting another candidate is matched_evidence, not a violation. Do not force a match when occlusion prevents comparison.'''


class ConfirmedSourceChange(ValueError):
    def __init__(self,match,observation,review,candidates):
        super().__init__('joint image review confirmed a different source; cancel old edge and reselect')
        self.recapture_match=match;self.observation=observation;self.review=review;self.candidates=candidates


class RecaptureSourceUnmatched(ValueError):
    """Joint review rejects the shortlist; global recognition must precede recovery."""
    def __init__(self,match,observation,review,candidates):
        super().__init__('joint image review found no source match; cancel old edge and globally re-enter')
        self.recapture_match=match;self.observation=observation;self.review=review;self.candidates=candidates


def review_pair(graph,trajectory,previous,current,requests,folder,metrics,candidates,threshold):
    import jsonschema
    subset=dict(nodes=[n for n in graph['match_nodes'] if n['node_id'] in candidates])
    def validate(value):
        jsonschema.validate(value,SCHEMA)
        for key in ('previous_match','current_match'):validate_match_response(value[key],subset,threshold)
        a,b=value['previous_match'],value['current_match']
        if value['observation_relation']=='SAME_DECISION_CONDITION' and reliable_match(a,subset,threshold) and reliable_match(b,subset,threshold) and a['node_id']!=b['node_id']:
            raise ValueError('Same decision condition cannot have two different reliable node matches. Compare references again; otherwise return UNKNOWN.')
    images=reference_images(graph,trajectory,current,candidates)
    images.insert(1,('previous_current',previous.image))
    started=time.monotonic()
    try:
        result=requests.ask('node_match_pair',signatures(graph,candidates,with_representatives=True),images,validator=validate)
        validate(result)
        save(folder/'joint_image_review.json',dict(result=result,candidate_nodes=candidates,
            image_ids=[k for k,_ in images],previous=evidence_file(previous.image),current=evidence_file(current.image)))
        return result
    finally:
        metrics['joint_match_time']=metrics.get('joint_match_time',0.)+time.monotonic()-started
        metrics['joint_match_reviews']=metrics.get('joint_match_reviews',0)+1
