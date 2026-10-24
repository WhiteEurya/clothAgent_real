import copy
import json

import pytest

from cloth_agent.experiments.local_structure_matcher import (
    eligible_pattern,match_subgraphs,point_edge_distance,validate_graph,validate_ground,
)


def entity(i,kind,points=None):
    if points is None:points=[[10,10],[30,10],[30,40]] if kind in ('surface','table_region') else [[10,10],[30,10]]
    return dict(id=i,kind=kind,points=points,evidence='visible boundary',confidence=.9)


def relation(s,t,kind):
    return dict(source=s,target=t,kind=kind,evidence='visible connection',confidence=.9)


def pattern():
    return dict(entities=[entity('edge','free_edge'),entity('panel','surface'),entity('fold','fold_line')],
        relations=[relation('edge','panel','bounds'),relation('panel','fold','folds_at'),relation('fold','panel','bounds')],uncertainties=[])


def renamed(graph,prefix):
    graph=copy.deepcopy(graph)
    for e in graph['entities']:e['id']=prefix+e['id']
    for r in graph['relations']:r['source']=prefix+r['source'];r['target']=prefix+r['target']
    return graph


def test_relation_match_is_invariant_to_position_and_rotation():
    p=pattern();q=renamed(p,'current_')
    for e in q['entities']:e['points']=[[100-y,50+x] for x,y in e['points']]
    validate_graph(p,(200,200));validate_graph(q,(200,200))
    matches=match_subgraphs(p,q)
    assert len(matches)==1
    assert matches[0]['mapping']['edge']=='current_edge'


def test_same_geometry_different_boundary_or_missing_fold_does_not_match():
    p=pattern();q=renamed(p,'q_')
    q['entities'][0]['kind']='opening_rim'
    assert not match_subgraphs(p,q)
    q=renamed(p,'q_');q['relations']=[r for r in q['relations'] if r['kind']!='folds_at']
    assert not match_subgraphs(p,q)
    q=renamed(p,'q_');q['relations'][1]['confidence']=.2
    assert not match_subgraphs(p,q)


def test_mapping_is_injective_and_preserves_layer_direction():
    p=dict(entities=[entity('a','surface'),entity('b','surface')],relations=[relation('a','b','covers')],uncertainties=[])
    assert not match_subgraphs(p,dict(entities=[entity('c','surface')],relations=[],uncertainties=[]))
    q=dict(entities=[entity('q','surface'),entity('r','surface')],relations=[relation('r','q','covers')],uncertainties=[])
    match=match_subgraphs(p,q)
    assert match[0]['mapping']=={'a':'r','b':'q'}
    q['relations'].append(relation('q','r','covers'))
    assert not match_subgraphs(p,q)  # contradictory extra reverse coverage


def test_simple_free_edge_does_not_establish_unfolding_motif():
    value=dict(before_structure=pattern(),grasp_entity_id='edge',moving_surface_id='panel',
               operation='open_fold',effect='coverage_reduced',effect_evidence=['panel opens'])
    assert not eligible_pattern(value,[20,10],(200,200))
    p=copy.deepcopy(value);p['before_structure']['entities']=p['before_structure']['entities'][:2]
    p['before_structure']['relations']=p['before_structure']['relations'][:1]
    assert eligible_pattern(p,[20,10],(200,200))
    p=copy.deepcopy(value);p['effect']='unclear'
    assert eligible_pattern(p,[20,10],(200,200))
    p=copy.deepcopy(value);p['before_structure']['entities'].append(entity('unrelated','surface'))
    assert eligible_pattern(p,[20,10],(200,200))


def test_invalid_structure_not_silently_fixed():
    p=pattern();p['relations'][0]['target']='missing'
    with pytest.raises(ValueError):validate_graph(p,(200,200))
    p=pattern();p['entities'][0]['points'][0]=[201,10]
    with pytest.raises(ValueError):validate_graph(p,(200,200))
    p=pattern();p['relations'][0]['kind']='covers'
    with pytest.raises(ValueError):validate_graph(p,(200,200))


def test_grounding_must_use_matched_boundary():
    scene=pattern();candidates=[dict(candidate_id='c000',current_grasp_entity='edge')]
    value=dict(decision='PROPOSE',candidate_id='c000',grasp_px=[20,12],target_px=[80,80],
               correspondence_evidence=['mapped edge'],expected_change='unfold')
    validate_ground(value,candidates,scene,(200,200))
    with pytest.raises(ValueError):validate_ground(dict(value,grasp_px=[100,100]),candidates,scene,(200,200))
    with pytest.raises(ValueError):validate_ground(dict(value,candidate_id='other'),candidates,scene,(200,200))
    assert point_edge_distance([20,15],[[10,10],[30,10]])==pytest.approx(5)


def test_cached_scene_with_no_eligible_reference_never_calls_model(tmp_path,monkeypatch):
    from PIL import Image
    from cloth_agent.experiments import local_structure_matcher as module
    image=tmp_path/'current.png';Image.new('RGB',(200,200)).save(image)
    scene_dir=tmp_path/'scene';scene_dir.mkdir()
    scene_file=scene_dir/'scene.json'
    scene_file.write_text(json.dumps(dict(structure=pattern(),completion='not_unfolded',completion_evidence=['fold'])))
    source=dict(protocol=module.PROTOCOL,image=module.evidence_file(image),scene=module.evidence_file(scene_file))
    (scene_dir/'scene_source.json').write_text(json.dumps(source))
    library=tmp_path/'library.json';library.write_text('{}')
    monkeypatch.setattr(module,'load_patterns',lambda path:dict(records={
        'e0':dict(eligible=False,rejection_reasons=['unresolved fold'])},min_evidence_confidence=.65))
    class NoCalls:
        def ask(self,*args,**kwargs):pytest.fail('ineligible references must not trigger transfer')
        def metrics(self):return dict(model_calls=0)
    monkeypatch.setattr(module,'requests_at',lambda *args:NoCalls())
    result=module.query(library,image,tmp_path/'result',scene_dir=scene_dir)
    assert result['status']=='NO_STRUCTURAL_MATCH'
    assert result['robot_calls']==result['model_metrics']['model_calls']==0
    Image.new('RGB',(200,200),'white').save(image)
    with pytest.raises(ValueError,match='different query image'):
        module.query(library,image,tmp_path/'wrong_image',scene_dir=scene_dir)


def test_comparison_renders_explicit_correspondences(tmp_path):
    from PIL import Image
    from cloth_agent.experiments.local_structure_matcher import comparison_image
    path=tmp_path/'image.png';Image.new('RGB',(200,200)).save(path)
    p=pattern();q=renamed(p,'current_');candidate=match_subgraphs(p,q)[0]
    candidate.update(candidate_id='c000',edge_id='e0')
    comparison_image(path,path,p,q,candidate,tmp_path/'comparison')
    with Image.open(tmp_path/'comparison'/'comparison.png') as im:
        assert im.size==(1600,780)
