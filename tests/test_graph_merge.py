from copy import deepcopy
import pytest
from .test_graph_policy import fixture_graph
from cloth_agent.experiments.graph_merge import merge_graph, connectivity
from cloth_agent.experiments.graph_policy import replay_choices


def proposal(graph):
    return dict(groups=[dict(node=deepcopy(n),source_node_ids=[n['node_id']],visual_evidence=['image evidence'],
        action_compatibility='existing edge',limitations=[]) for n in graph['nodes']],rejected_merges=[],limitations=[])


def test_partition_preserves_every_edge_and_self_loop(tmp_path):
    g,t=fixture_graph(tmp_path); p=proposal(g)
    p['groups'][0]['source_node_ids']=['N0','N1']
    p['groups'][0]['node']['representative_observations']=['o0','o1']
    p['groups']=p['groups'][:1]
    out=merge_graph(g,p,t)
    assert len(out['nodes'])==1 and out['edges'][0]['source_node']==out['edges'][0]['target_node']
    before=deepcopy(g['edges'][0]); after=deepcopy(out['edges'][0])
    for x in (before,after): x.pop('source_node');x.pop('target_node')
    assert before==after
    assert len(replay_choices(out,'N0',[]))==1
    assert replay_choices(out,'N0',['E0'])==[]
    assert g['edges'][0]['target_node']=='N1'


@pytest.mark.parametrize('kind',['omitted','duplicated','lost_image','unknown_member'])
def test_invalid_partition_rejected(tmp_path,kind):
    g,t=fixture_graph(tmp_path);p=proposal(g)
    if kind=='omitted':p['groups'].pop()
    if kind=='duplicated':p['groups'].append(deepcopy(p['groups'][0]))
    if kind=='lost_image':p['groups'][0]['node']['representative_observations']=[]
    if kind=='unknown_member':p['groups'][0]['source_node_ids']=['fake']
    with pytest.raises(ValueError):merge_graph(g,p,t)


def test_failure_edges_not_promoted_by_merge(tmp_path):
    g,t=fixture_graph(tmp_path);g['edges'][0].update(seed_outcome='FAILURE',generalization_status='OBSERVED_FAILURE')
    out=merge_graph(g,proposal(g),t)
    report=connectivity(out,g)
    assert report['route_to_last_observation'] is None
    assert replay_choices(out,'N0',[])==[] and report['excluded_edges']==['E0']


def test_consolidate_renders_replay_prototypes_and_preserves_v1(tmp_path):
    import shutil
    if not shutil.which('dot'):pytest.skip('Graphviz required for graph visualization')
    from .test_graph_policy import write_graph
    from cloth_agent.experiments.graph_merge import consolidate
    from cloth_agent.experiments.graph_policy import load_frozen,file_hash
    g,t=fixture_graph(tmp_path);path=write_graph(tmp_path,g,t);hash_before=file_hash(path)
    class Fake:
        def ask(self,kind,payload,images,validator):
            assert kind=='merge' and len(images)==2
            p=proposal(payload['source_graph']);validator(p);return p
        def metrics(self):return {'model_calls':0}
    out=tmp_path/'graph_v2.json';merged=consolidate(path,out,Fake())
    assert file_hash(path)==hash_before and load_frozen(out)==merged
    assert len(merged['state_prototypes'])==2
    assert (tmp_path/'graph_v2.png').stat().st_size>100
    svg=(tmp_path/'graph_v2.svg').read_text()
    assert 'data:image/png;base64,' in svg
    assert merged['connectivity']['route_to_last_observation']==['E0']
