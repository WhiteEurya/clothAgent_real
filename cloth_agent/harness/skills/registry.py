"""Semantic specification + executable implementation, independent of robot code."""
from __future__ import annotations

import copy
from dataclasses import dataclass

from ..common import digest
from ..executors.restricted import RestrictedProgram
from ..policy import PolicyError, obj, validate_schema

NAME = {'type': 'string', 'pattern': '^[a-z][a-z0-9_]{0,47}$'}
TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 2400}
SPEC_SCHEMA = obj({'id': NAME, 'version': {'type': 'integer', 'minimum': 1, 'maximum': 100000},
    'information_need': TEXT, 'applicable_when': TEXT, 'success_check': TEXT, 'on_insufficient': TEXT,
    'limitations': TEXT, 'method': TEXT})
OP_SCHEMA = {'oneOf': [obj({'op': {'const': 'crop'}, 'roi': {'type': 'array', 'minItems': 4, 'maxItems': 4,
    'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}}),
    obj({'op': {'const': 'rotate'}, 'degrees': {'enum': [0, 90, 180, 270]}}),
    obj({'op': {'const': 'resize'}, 'scale': {'type': 'number', 'minimum': .25, 'maximum': 4}})]}
RECIPE_SCHEMA = obj({'roles': {'type': 'array', 'minItems': 1, 'maxItems': 2, 'uniqueItems': True,
                              'items': {'enum': ['clean', 'overlay']}},
    'operations': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': OP_SCHEMA},
    'reuse_existing': {'type': 'boolean'}})


def validate_recipe(recipe):
    validate_schema(recipe, RECIPE_SCHEMA)
    if 'clean' not in recipe['roles']:
        raise PolicyError('Observation requires clean evidence, not overlay alone')
    for op in recipe['operations']:
        if op['op'] == 'crop' and not (op['roi'][0] < op['roi'][2] and op['roi'][1] < op['roi'][3]):
            raise PolicyError('Crop needs an ordered ROI')
    return recipe


@dataclass(frozen=True)
class ObservationSkill:
    specification: dict
    source: str

    @property
    def key(self): return f"{self.specification['id']}@v{self.specification['version']}"

    @property
    def content_hash(self): return digest({'specification': self.specification, 'implementation': self.source})

    def applicable(self, request, source):
        # Semantic applicability is model-judged; host checks only input capabilities.
        return request['skill_id'] == self.specification['id']

    def prepare(self, request, source, available):
        recipe = RestrictedProgram(self.source).run(copy.deepcopy(request), copy.deepcopy(source), copy.deepcopy(available))
        if recipe is None:
            raise PolicyError(f"Skill {self.key} cannot prepare the requested operation combination: roi={request.get('roi')}, degrees_clockwise={request.get('degrees_clockwise')}, enlarge={request.get('enlarge')}")
        return validate_recipe(recipe)

    def execute(self, host, prepared):
        return host.execute_prepared(prepared)

    def validate_output(self, host, delivered):
        host.validate_delivered(delivered)


class SkillRegistry:
    def __init__(self):
        self._skills = {}
        self._active = {}
        self._hashes = {}

    def register(self, skill, *, activate=True):
        validate_schema(skill.specification, SPEC_SCHEMA)
        RestrictedProgram(skill.source)
        if skill.key in self._skills:
            raise PolicyError('Cannot overwrite a registered skill version')
        self._skills[skill.key] = copy.deepcopy(skill)
        self._hashes[skill.key] = skill.content_hash
        if activate: self._active[skill.specification['id']] = skill.key

    def get(self, name):
        key = self._active.get(name, name)
        if key not in self._skills: raise PolicyError(f'Unknown registered skill: {name}')
        skill = self._skills[key]
        if skill.content_hash != self._hashes[key]: raise PolicyError('Registered skill mutated')
        return copy.deepcopy(skill)

    def snapshot(self):
        return {name: {'key': key, 'hash': self.get(key).content_hash} for name, key in sorted(self._active.items())}

    def catalog(self):
        return [copy.deepcopy(self.get(name).specification) for name in sorted(self._active)]

    def clone(self): return copy.deepcopy(self)


ORIENTATION = '''def prepare(request, source, available):
    roles = ["clean", "overlay"] if source["role"] in ["clean", "overlay"] else ["clean"]
    roi = request["roi"]
    w = source["size"][0]
    h = source["size"][1]
    if roi != None:
        w = ceil(roi[2] * w) - floor(roi[0] * w)
        h = ceil(roi[3] * h) - floor(roi[1] * h)
    scale = max(1, min(3, 768 / max(w, h)))
    if roi != None:
        if request["enlarge"] and scale > 1:
            return {"roles": roles, "operations": [{"op": "crop", "roi": roi}, {"op": "rotate", "degrees": request["degrees_clockwise"]}, {"op": "resize", "scale": scale}], "reuse_existing": False}
        return {"roles": roles, "operations": [{"op": "crop", "roi": roi}, {"op": "rotate", "degrees": request["degrees_clockwise"]}], "reuse_existing": False}
    else:
        if request["enlarge"] and scale > 1:
            return {"roles": roles, "operations": [{"op": "rotate", "degrees": request["degrees_clockwise"]}, {"op": "resize", "scale": scale}], "reuse_existing": False}
        return {"roles": roles, "operations": [{"op": "rotate", "degrees": request["degrees_clockwise"]}], "reuse_existing": False}
'''
CROP = '''def prepare(request, source, available):
    if request["degrees_clockwise"] != 0:
        return None
    roi = request["roi"]
    w = ceil(roi[2] * source["size"][0]) - floor(roi[0] * source["size"][0])
    h = ceil(roi[3] * source["size"][1]) - floor(roi[1] * source["size"][1])
    if request["enlarge"]:
        scale = max(1, min(3, 768 / max(w, h)))
        if scale > 1:
            return {"roles": ROLES, "operations": [{"op": "crop", "roi": roi}, {"op": "resize", "scale": scale}], "reuse_existing": False}
    return {"roles": ROLES, "operations": [{"op": "crop", "roi": roi}], "reuse_existing": False}
'''


def builtin_registry():
    registry = SkillRegistry()
    for name, source, method in [
        ('orientation', ORIENTATION, 'Optionally crop the selected ROI, rotate clockwise, then optionally enlarge; preserve aligned clean/overlay views. ROI null rotates the full selected view.'),
        ('local_boundary', CROP.replace('ROLES', '["clean"]'), 'Crop the selected view and optionally enlarge.'),
        ('overlay_occlusion', CROP.replace('ROLES', '["clean", "overlay"]'), 'Crop the aligned pair and optionally enlarge both.')]:
        registry.register(ObservationSkill({'id': name, 'version': 1,
            'information_need': 'Resolve a current visible information gap',
            'applicable_when': 'The requested region and image are available',
            'method': method, 'success_check': 'Interpret visible results; execution alone does not certify sufficiency',
            'on_insufficient': 'Return UNKNOWN', 'limitations': 'RGB visibility only; not physical verification'}, source))
    return registry
