"""Isolated patch artifacts and mandatory deterministic admission gates."""
from __future__ import annotations

import copy
import ast
import difflib
import json
import math
import re
import traceback
from pathlib import Path

from PIL import Image

from ..image_tools_mcp import pixel_hash, transform_point
from .common import canonical, digest, now, read_json, write_json
from .executors.observation import RegisteredObservationHost
from .executors.restricted import RestrictedProgram
from .policy import PolicyError, obj, validate_schema
from .reasoning_contract import HARNESS_SCHEMA, baseline_harness, validate_harness
from .skills import ObservationSkill, builtin_registry
from .skills.registry import NAME, TEXT, SPEC_SCHEMA, RECIPE_SCHEMA, validate_recipe

CONFIG_SCHEMA = obj({'schema_version': {'const': 1}, 'observation_instruction': TEXT,
    'enabled_skills': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'uniqueItems': True, 'items': NAME},
    'reasoning_harness': HARNESS_SCHEMA})
PROPOSAL_SCHEMA = obj({'level': {'enum': ['PROMPT','HARNESS','SKILL_CODE','NEW_TOOL']}, 'target': NAME,
    'problem': TEXT, 'proposed_change': TEXT, 'expected_effect': TEXT,
    'must_preserve': {'type': 'array', 'minItems': 1, 'maxItems': 12, 'items': TEXT},
    'evidence_rollouts': {'type': 'array', 'minItems': 1, 'maxItems': 20, 'uniqueItems': True, 'items': TEXT},
    'unverified': TEXT})
PROPOSAL_SCHEMA['properties']['evidence_explanation'] = {
    'type': 'string', 'description': 'Explanation of evidence; keep evidence_rollouts limited to exact allowed rollout IDs.'}


def proposal_schema(rollout_ids):
    schema = copy.deepcopy(PROPOSAL_SCHEMA)
    schema['properties']['evidence_rollouts']['items'] = {'type': 'string', 'enum': sorted(set(rollout_ids))}
    return schema


def proposal_transport_schema(rollout_ids):
    """Use one quoted prose field rather than a fragile tool-call prose array."""
    schema=proposal_schema(rollout_ids)
    schema['properties']['must_preserve']={'type':'string','minLength':1,'maxLength':12000,
        'description':'One JSON string describing all invariants, separated by semicolons or newlines. Do not emit repeated must_preserve keys.'}
    return schema


def normalize_proposal_transport(proposal):
    proposal=copy.deepcopy(proposal)
    if isinstance(proposal.get('must_preserve'),str):
        # Preserve text exactly. Each chunk obeys the canonical per-item bound;
        # this is representation conversion, not rewriting constraints.
        text=proposal['must_preserve']
        proposal['must_preserve']=[text[i:i+2400] for i in range(0,len(text),2400)]
    return proposal


def normalize_citations(proposal, rollout_ids):
    """Extract only unambiguous explicit IDs, preserving every original citation."""
    validate_schema(proposal, PROPOSAL_SCHEMA)
    known = set(rollout_ids)
    resolved, entries = [], []
    for original in proposal['evidence_rollouts']:
        matches = sorted(k for k in known if re.search(r'(?<![\w-])'+re.escape(k)+r'(?![\w-])', original))
        exact = original.strip() in known
        status = 'EXACT' if exact else 'EXTRACTED' if len(matches) == 1 else 'UNRESOLVED'
        ids = [original.strip()] if exact else matches if len(matches) == 1 else []
        for key in ids:
            if key not in resolved:
                resolved.append(key)
        entries.append({'original': original, 'status': status, 'resolved_ids': ids})
    normalized = copy.deepcopy(proposal)
    normalized['evidence_rollouts'] = resolved
    audit = {'original': proposal, 'normalized': normalized, 'entries': entries,
             'status': 'ACCEPTED' if resolved else 'NEEDS_CITATION_REPAIR',
             'meaning': 'Trace identity only; no claim that the cited explanation is true.'}
    return normalized, audit
TEST_SCHEMA = obj({'request': {'type': 'object'}, 'source_size': {'type': 'array', 'minItems': 2, 'maxItems': 2,
    'items': {'type': 'integer', 'minimum': 16, 'maximum': 256}}, 'expected_recipe': RECIPE_SCHEMA})
IMPLEMENTATION_SCHEMA = obj({'config': CONFIG_SCHEMA,
    'skill': {'anyOf': [{'type': 'null'}, obj({'specification': SPEC_SCHEMA,
        'source': {'type': 'string', 'minLength': 1, 'maxLength': 16000},
        'tests': {'type': 'array', 'minItems': 2, 'maxItems': 8, 'items': TEST_SCHEMA}})]}})


def baseline_config():
    return {'schema_version': 1,
        'observation_instruction': 'Identify selection-critical information gaps on current images. Use a registered skill only when necessary. Bind requests to current clean root geometry. If all evidence is available return no requests. Do not select grasp or target yet.',
        'enabled_skills': ['orientation','local_boundary','overlay_occlusion'], 'reasoning_harness': baseline_harness()}


def validate_config(config, registry):
    validate_schema(config, CONFIG_SCHEMA)
    validate_harness(config['reasoning_harness'])
    for name in config['enabled_skills']: registry.get(name)
    return copy.deepcopy(config)


def behavior_hash(config, registry):
    # Names alone cannot buy an extra consensus vote.
    from .reasoning_learning import execution_signature
    return digest({'instruction': config['observation_instruction'], 'enabled': sorted(config['enabled_skills']),
        'reasoning': execution_signature(config['reasoning_harness']),
        'implementations': {name: ast.dump(ast.parse(registry.get(name).source), include_attributes=False)
                            for name in sorted(config['enabled_skills'])}})


def baseline_bundle():
    registry = builtin_registry()
    return {'schema_version': 1, 'config': baseline_config(), 'skills': [
        {'specification': registry.get(s['id']).specification, 'source': registry.get(s['id']).source}
        for s in registry.catalog()]}


def registry_from_bundle(bundle):
    from .skills import SkillRegistry
    registry = SkillRegistry()
    for row in bundle['skills']: registry.register(ObservationSkill(row['specification'], row['source']))
    validate_config(bundle['config'], registry)
    return registry


def bundle_hash(bundle): return digest(bundle)


def validate_implementation(proposal, implementation, baseline):
    """Validate semantics without writing files; errors can be repaired locally."""
    validate_schema(proposal, PROPOSAL_SCHEMA)
    validate_schema(implementation, IMPLEMENTATION_SCHEMA)
    registry = registry_from_bundle(baseline)
    config, generated = implementation['config'], implementation['skill']
    level, target = proposal['level'], proposal['target']
    if level in ('PROMPT','HARNESS'):
        if generated is not None: raise PolicyError('Configuration patch cannot contain source code')
        if level == 'PROMPT':
            if config['enabled_skills'] != baseline['config']['enabled_skills']:
                raise PolicyError('PROMPT cannot change enabled capabilities')
            def structure(h):
                return {**h, 'applicability': '', 'name': '', 'stages': [{**s, 'instruction': ''} for s in h['stages']]}
            if structure(config['reasoning_harness']) != structure(baseline['config']['reasoning_harness']):
                raise PolicyError('PROMPT cannot change stage topology or host operations')
    else:
        if generated is None: raise PolicyError('Code/tool proposal needs an implementation and tests')
        spec = generated['specification']
        if spec['id'] != target: raise PolicyError('Implementation target mismatch')
        existing = {s['id'] for s in registry.catalog()}
        if level == 'SKILL_CODE':
            if target not in existing or spec['version'] != registry.get(target).specification['version']+1:
                raise PolicyError('SKILL_CODE needs the next version of an existing skill')
        elif target in existing or spec['version'] != 1:
            raise PolicyError('NEW_TOOL needs a new ID and version one')
        # Joint implementation/prompt changes are permitted. Their measured
        # benefit belongs to the whole patch, not an isolated code ablation.
        if target not in config['enabled_skills']: raise PolicyError('Generated capability is not enabled')
        if level == 'SKILL_CODE' and config['enabled_skills'] != baseline['config']['enabled_skills']:
            raise PolicyError('SKILL_CODE cannot change the other enabled capabilities')
        if level == 'NEW_TOOL' and set(config['enabled_skills']) != set(baseline['config']['enabled_skills']) | {target}:
            raise PolicyError('NEW_TOOL may only add its target capability')
        registry.register(ObservationSkill(spec, generated['source']))
    validate_config(config, registry)
    if behavior_hash(config, registry) == behavior_hash(baseline['config'], registry_from_bundle(baseline)):
        raise PolicyError('No executable change; renaming is not a patch')
    bundle = {'schema_version': 1, 'config': config, 'skills': [
        {'specification': registry.get(s['id']).specification, 'source': registry.get(s['id']).source}
        for s in registry.catalog()]}
    return bundle


def materialize_candidate(directory, proposal, implementation, baseline):
    """The model cannot supply paths or edit production; host owns every target."""
    directory = Path(directory)
    bundle = validate_implementation(proposal, implementation, baseline)
    config, generated, target = implementation['config'], implementation['skill'], proposal['target']
    files = {'config.json': config}
    if generated:
        base = f'cloth_agent/harness/skills/{target}'
        files[f'{base}/skill.json'] = generated['specification']
        files[f'{base}/tests/cases.json'] = generated['tests']
        files[f'{base}/evidence.json'] = {'status': 'UNVERIFIED', 'proposal': proposal, 'physical_outcomes': []}
        source_path = directory / base / 'implementation.py'
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(generated['source'])
        # Host-owned reproducible pytest entry; model test inputs remain data.
        test_path = directory / base / 'tests/test_skill.py'
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text('''"""Generated host test wrapper; synthetic candidate cases, not experiment evidence."""
from pathlib import Path
import json
import pytest
from cloth_agent.harness.executors.restricted import RestrictedProgram
from cloth_agent.harness.skills.registry import validate_recipe

ROOT = Path(__file__).resolve().parent
@pytest.mark.parametrize("case", json.loads((ROOT / "cases.json").read_text()))
def test_candidate_prepare(case):
    program = RestrictedProgram((ROOT.parent / "implementation.py").read_text())
    source = {"image_id": "image_0", "role": "clean", "size": case["source_size"]}
    result = program.run(case["request"], source, {"cached_views": 0, "remaining_ops": 24})
    assert validate_recipe(result) == case["expected_recipe"]
''')
    for path, value in files.items(): write_json(directory/path, value, exclusive=True)
    before_sources = {s['specification']['id']: s['source'] for s in baseline['skills']}
    diff = list(difflib.unified_diff(json_lines(baseline['config']), json_lines(config), fromfile='base/config.json', tofile='candidate/config.json'))
    if generated:
        name = f'cloth_agent/harness/skills/{target}/implementation.py'
        diff += list(difflib.unified_diff(before_sources.get(target,'').splitlines(True), generated['source'].splitlines(True), fromfile='base/'+name, tofile='candidate/'+name))
    (directory/'patch.diff').write_text(''.join(diff))
    write_json(directory/'bundle.json', bundle, exclusive=True)
    write_json(directory/'base_version.json', {'bundle_hash': digest(baseline), 'registry': registry_from_bundle(baseline).snapshot()}, exclusive=True)
    # Include exact implementation/spec/tests/config bytes, not volatile debug.
    paths = ['bundle.json', *files]
    if generated: paths.extend([str(source_path.relative_to(directory)),str(test_path.relative_to(directory))])
    seal = {p: digest((directory/p).read_text()) for p in sorted(paths)}
    write_json(directory/'seal.json', {'files': seal, 'bundle_hash': digest(bundle)}, exclusive=True)
    return bundle


def json_lines(value): return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').splitlines(True)


def verify_candidate(directory, *, require_pass=True):
    directory = Path(directory)
    seal = read_json(directory/'seal.json')
    for path, expected in seal['files'].items():
        target = directory/path
        if Path(path).is_absolute() or '..' in Path(path).parts or target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
            raise PolicyError('Candidate path escaped workspace')
        if digest(target.read_text()) != expected: raise PolicyError('Candidate changed after materialization/testing')
    bundle = read_json(directory/'bundle.json')
    if digest(bundle) != seal['bundle_hash']: raise PolicyError('Candidate bundle hash mismatch')
    if require_pass:
        gates = read_json(directory/'gates.json')
        if gates['status'] != 'PASSED' or gates['seal_hash'] != digest(seal): raise PolicyError('Candidate has no passing tests for this version')
    registry_from_bundle(bundle)
    return bundle


def _request(name, *, roi=None, angle=0, enlarge=False):
    return {'gap_id': 'geometry', 'skill_id': name, 'source_image_id': 'image_0', 'roi': roi,
            'degrees_clockwise': angle, 'enlarge': enlarge, 'expected_information_gain': 'Synthetic gate visibility'}


def _fixtures(directory, size):
    directory.mkdir(parents=True, exist_ok=False)
    im = Image.new('RGB', size)
    im.putdata([(x % 251,y % 251,(x+y)%251) for y in range(size[1]) for x in range(size[0])])
    images = []
    for i in range(2):
        path = directory/f'root_{i}.png'; im.save(path); images.append(path)
    return {'images': [{'image_id': f'image_{i}', 'role': role, 'size': list(size), 'rgb_sha256': pixel_hash(im)}
                       for i, role in enumerate(('clean','overlay'))]}, images


def _oracle(root, recipe):
    """Independent Pillow pixel and pixel-center affine oracle for mandatory gates."""
    from ..image_tools_mcp import compose
    im = root.copy(); transform = [1,0,0,0,1,0]
    for op in recipe['operations']:
        w,h = im.size
        if op['op'] == 'crop':
            r=op['roi']; box=[math.floor(r[0]*w),math.floor(r[1]*h),math.ceil(r[2]*w),math.ceil(r[3]*h)]
            im=im.crop(box); local=[1,0,box[0],0,1,box[1]]
        elif op['op'] == 'rotate':
            degrees=op['degrees']; im=im.rotate(-degrees,expand=True)
            local={0:[1,0,0,0,1,0],90:[0,1,0,-1,0,h-1],180:[-1,0,w-1,0,-1,h-1],270:[0,-1,w-1,1,0,0]}[degrees]
        else:
            nw,nh=max(1,round(w*op['scale'])),max(1,round(h*op['scale']))
            im=im.resize((nw,nh),Image.Resampling.LANCZOS)
            local=[w/nw,0,(w/nw-1)/2,0,h/nh,(h/nh-1)/2]
        transform=compose(transform,local)
    return im,transform


def run_gates(directory, *, baseline=None):
    """No model calls. Candidate tests never replace non-overridable host gates."""
    directory=Path(directory)
    result={'status':'REJECTED','checks':[], 'seal_hash':None, 'synthetic_tests_only':True}
    def passed(name, **detail): result['checks'].append({'name':name,'passed':True,**detail})
    try:
        bundle=verify_candidate(directory,require_pass=False)
        result['seal_hash']=digest(read_json(directory/'seal.json'))
        registry=registry_from_bundle(bundle)
        passed('syntax_and_capability_allowlist', registry=registry.snapshot())
        passed('schema_and_bindings')
        test_files=list(directory.glob('cloth_agent/harness/skills/*/tests/cases.json'))
        cases=[]
        for file in test_files:
            skill_id=file.parents[1].name
            for index,test in enumerate(read_json(file)):
                validate_schema(test,TEST_SCHEMA)
                request=test['request']
                if request.get('skill_id')!=skill_id: raise PolicyError('Unit test must exercise patched skill')
                source={'image_id':'image_0','role':'clean','size':test['source_size']}
                actual=registry.get(skill_id).prepare(request,source,{'cached_views':0,'remaining_ops':24})
                if actual!=test['expected_recipe']: raise PolicyError('Generated unit test expectation failed')
                cases.append((request,test['source_size']))
            passed('generated_unit_tests',skill_id=skill_id,count=len(read_json(file)))
        # Baseline capabilities and all active overrides must remain usable.
        cases += [(_request('orientation',angle=angle),[40,60]) for angle in (90,180,270)]
        cases += [(_request(name,roi=[.13,.21,.84,.89],enlarge=enlarge),[41,63])
                  for name in ('local_boundary','overlay_occlusion') for enlarge in (False,True)]
        for index,(request,size) in enumerate(cases):
            root=directory/'tests'/f'case_{index:02d}'
            obs,images=_fixtures(root/'input',size)
            artifact={'skills':registry.catalog()}
            host=RegisteredObservationHost(obs,images,artifact,root/'host',24,registry=registry)
            info=[{'id':request['gap_id'],'status':'UNKNOWN'}]
            plan=host.prepare(request,info)
            # Admission limits prevent candidate tests allocating huge images.
            dimensions=size[0]*size[1]
            for op in plan['recipe']['operations']:
                if op['op']=='resize': dimensions*=op['scale']**2
            if dimensions>1_000_000: raise PolicyError('Gate fixture output exceeds test pixel budget')
            stop=host.execute([request],info)
            if stop: raise PolicyError(f'Valid gate fixture failed: {stop}')
            delivered=[next(i for i in host.catalog if i['image_id']==identity) for identity in host.history[-1]['delivered_image_ids']]
            with Image.open(images[0]) as image: expected,matrix=_oracle(image,plan['recipe'])
            for view in delivered:
                with Image.open(view['path']) as image:
                    if pixel_hash(image)!=pixel_hash(expected): raise PolicyError('Pixel oracle mismatch')
                if any(abs(a-b)>1e-8 for a,b in zip(matrix,view['to_original'])): raise PolicyError('Coordinate invariant failed')
            if host.execute([request],info)!='REPEATED_OBSERVATION': raise PolicyError('Repeated observation must stop')
            no_budget=RegisteredObservationHost(obs,images,artifact,root/'zero_budget',0,registry=registry)
            if no_budget.execute([request],info)!='OBSERVATION_BUDGET_EXHAUSTED' or no_budget.ops:
                raise PolicyError('Budget gate failed')
            # Malformed second request must never cause partial batch execution.
            fresh=RegisteredObservationHost(obs,images,artifact,root/'atomic',24,registry=registry)
            bad={**request,'source_image_id':'missing'}
            try: fresh.execute([request,bad],info)
            except PolicyError: pass
            else: raise PolicyError('Invalid image accepted')
            if fresh.ops: raise PolicyError('Partial batch executed before validation')
            passed('coordinate_pixels_alignment_budget_input_immutability',case=index,skill=request['skill_id'],operations=host.ops)
        verify_candidate(directory,require_pass=False)
        passed('no_robot_import_io_or_unbounded_execution', enforcement='Restricted AST interpreter; only JSON data and finite image recipes')
        result['status']='PASSED'
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        (directory/'gate_exception.txt').write_text(traceback.format_exc())
    write_json(directory/'gates.json',result)
    return result
