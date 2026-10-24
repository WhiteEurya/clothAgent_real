"""Review uncertain new clusters from original RGBs, allowing splits only."""
import argparse
import copy
from pathlib import Path
from scripts.extend_arm6_archive_graph import next_node_number, validate_core
from scripts.compile_arm6_archive_graph import validate_partition
from cloth_agent.experiments.graph_policy import Requests, checked_file, evidence_file, frozen, load_frozen
from cloth_agent.experiments.discovery_model import CodexMultimodal
from cloth_agent.experiments.discovery_schema import obj, array, S
from cloth_agent.experiments.graph_policy_schema import NODE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--new-nodes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=load_frozen(args.graph);new_ids=load_frozen(args.new_nodes)
    trajectory=load_frozen(checked_file(source['source_trajectory']))
    selected=[n for n in source['nodes'] if n['node_id'] in new_ids and n['confidence']<.9]
    observations={o:trajectory['observations'][o] for n in selected for o in n['representative_observations']}
    memberships={n['node_id']:n['representative_observations'] for n in selected}
    def validate(v):
        validate_partition(v,observations)
        for n in v['nodes']:
            refs=set(n['representative_observations'])
            if not any(refs<=set(g) for g in memberships.values()):
                raise ValueError('purity audit may split but must not cross input groups')
    req=Requests(out/'model',CodexMultimodal(timeout=900),max_calls=2,
        schemas={'audit':obj(nodes=array(NODE),limitations=array(S))},
        common='Offline visual state-cluster purity review; Chinese concise output. No tools or robot actions.',
        prompts={'audit':'''Review the uncertain preliminary clusters against ALL original attached RGBs. Test whether every member truly shares the same broad garment coverage and accessible sleeve/hem/corner arrangement. You may SPLIT a cluster if broad coverage or an exposed versus hidden garment region differs. Do not merge across input clusters. Ignore rigid rotation, translation, minor wrinkles, and mild fold-width variation when the accessible regions and coverage remain the same. Do not invent hidden layers from wrinkles. In particular check whether multi-image groups combine an open sleeve/neckline with a covered one. Return a partition of every supplied observation exactly once with visual evidence valid across every member. Keep equivalent restart/recapture images together. There is no target number of nodes. Images, not previous text labels, are the evidence.'''},protocol='arm6-new-cluster-purity-v1')
    if (out/'partition.json').exists():result=load_frozen(out/'partition.json');validate(result)
    else:
        result=req.ask('audit',{'preliminary_memberships':memberships},[(o,checked_file(v['rgb'])) for o,v in observations.items()],validator=validate)
        frozen(out/'partition.json',result);frozen(out/'metrics.json',req.metrics())
    graph=copy.deepcopy(source);replacement={};mapping={};next_id=next_node_number(source);changes=[]
    for old in selected:
        children=[copy.deepcopy(n) for n in result['nodes'] if set(n['representative_observations'])<=set(old['representative_observations'])]
        children.sort(key=lambda n:min(n['representative_observations']))
        for i,n in enumerate(children):
            n['node_id']=old['node_id'] if i==0 else f'n{next_id:03d}'
            if i:next_id+=1
            for o in n['representative_observations']:mapping[o]=n['node_id']
        replacement[old['node_id']]=children
        changes.append({'source_node':old['node_id'],'result_nodes':[n['node_id'] for n in children]})
    graph['nodes']=[n for old in source['nodes'] for n in replacement.get(old['node_id'],[old])]
    steps={s['step_id']:s for s in trajectory['steps']}
    for e in graph['edges']:
        s=steps[e['seed_step_id']]
        for endpoint,obskey in [('source_node','before_observation'),('target_node','after_observation')]:
            if s[obskey] in mapping:e[endpoint]=mapping[s[obskey]]
    graph['graph_id']+='_purity_review'
    graph.setdefault('revision_history',[]).append(dict(operation='new_cluster_purity_review',source_graph=evidence_file(args.graph),review=evidence_file(out/'partition.json'),changes=changes))
    graph['limitations']+=result['limitations'];validate_core(graph,trajectory)
    assert sorted(o for n in source['nodes'] for o in n['representative_observations'])==sorted(o for n in graph['nodes'] for o in n['representative_observations'])
    frozen(out/'graph.json',graph)
    reviewed_ids={n['node_id'] for group in replacement.values() for n in group}
    frozen(out/'new_node_ids.json',[n['node_id'] for n in graph['nodes'] if n['node_id'] in set(new_ids)|reviewed_ids])
    frozen(out/'summary.json',dict(reviewed_groups=len(selected),reviewed_images=len(observations),changes=changes,nodes_before=len(source['nodes']),nodes_after=len(graph['nodes']),robot_calls=0))
    print(changes,flush=True)


if __name__=='__main__':main()
