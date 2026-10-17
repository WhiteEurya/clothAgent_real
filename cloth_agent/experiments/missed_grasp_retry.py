"""Bounded before/after garment-change review; no hardware dispatch here."""
from pathlib import Path
import math
from .discovery_schema import obj, enum, S, N

SCHEMA = obj(result=enum('UNCHANGED','MOVED','UNKNOWN'), confidence=N, evidence=S)
PROMPT = 'Compare ONLY the before and after RGB observations of the garment.\nReturn UNCHANGED when garment position, boundary, folds and the grasp-area fabric remain\nbasically unchanged. Ignore small wrinkles, lighting differences and camera/image noise.\nReturn MOVED when there is clear garment displacement, boundary displacement, or a meaningful\nchange of folds or fabric placement, including whole-garment translation without task progress.\nCompare garment geometry relative to the table, allowing for viewpoint differences; identical\nnode labels do not establish unchanged fabric. UNKNOWN is only for before/after images that\ncannot be compared reliably, such as garment occlusion or an unresolved large viewpoint change.\nThis is a no-visible-change retry heuristic, NOT proof that the gripper missed the cloth.\nDo not require execution video, visible finger contact, or evidence of lift during execution.\nNo planning, obstruction diagnosis, action parameters, or chain of thought.\nReturn the schema with confidence and brief visible before/after evidence.'

def lowered_source(source):
    from cloth_agent.codex_motion import parse_code
    calls = parse_code(source, allow_grasp=True)
    if not calls or calls[0][0] != 'grasp' or set(calls[0][1]) != {'right'}:
        raise ValueError('missed-grasp retry supports one initial right-arm grasp only')
    if any(name not in {'move_tcp','move_transfer'} for name, _ in calls[1:]):
        raise ValueError('unsupported missed-grasp retry skeleton')
    pose = list(calls[0][1]['right'])
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise ValueError('invalid grasp pose')
    pose[2] -= .002
    calls[0] = ('grasp', {'right':pose})
    return '\n'.join('motion.'+name+'('+','.join(k+'='+repr(v) for k,v in kw.items())+')' for name,kw in calls)+'\n'


def review(requests, pending, current, folder):
    import jsonschema
    from .discovery_data import save
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    images = [('before',Path(pending['before']['path'])),('after',current.image)]
    value=requests.ask('missed_grasp_review',
        dict(criterion='BEFORE_AFTER_GARMENT_CHANGE',
             unchanged_means='No meaningful visible garment displacement or fold change'),images)
    jsonschema.validate(value,SCHEMA)
    save(folder/'review.json',dict(**value,assessment_basis='BEFORE_AFTER_RGB_ONLY',
        images={label:str(path) for label,path in images},
        interpretation='UNCHANGED is a suspected missed-grasp heuristic, not proof of missed contact'))
    return value
