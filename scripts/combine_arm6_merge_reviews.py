"""Combine independent pair reviews of the same frozen graph, then merge once."""
import argparse
import copy
from pathlib import Path
from scripts.review_arm6_graph_merges import validate_review, apply_round
from scripts.extend_arm6_archive_graph import validate_core
from cloth_agent.experiments.graph_policy import checked_file, evidence_file, frozen, load_frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=load_frozen(args.graph)
    pairs=[];decisions=[];seen=set()
    for i,folder in enumerate(args.reviews):
        config=load_frozen(folder/'config.json')
        if config['source_graph']!=evidence_file(args.graph):raise ValueError('reviews must use the same frozen graph')
        batch=load_frozen(folder/'candidates.json');review=load_frozen(folder/'decisions.json')
        validate_review(review,batch)
        lookup={d['pair_id']:d for d in review['decisions']}
        for pair in batch:
            key=tuple(sorted(pair['nodes']))
            if key in seen:raise ValueError('overlapping review requests must be reconciled explicitly')
            seen.add(key);p=copy.deepcopy(pair);d=copy.deepcopy(lookup[p['pair_id']])
            p['pair_id']=d['pair_id']=f'r{i}_'+p['pair_id'];pairs.append(p);decisions.append(d)
    review={'decisions':decisions}
    graph,mapping,applied,deferred=apply_round(source,pairs,review)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    frozen(out/'candidates.json',pairs);frozen(out/'decisions.json',review)
    graph.setdefault('revision_history',[]).append(dict(operation='combined_original_rgb_merge_review',
        source_graph=evidence_file(args.graph),review_sources=[evidence_file(r/'decisions.json') for r in args.reviews],
        merges=applied,deferred=deferred))
    trajectory=load_frozen(checked_file(source['source_trajectory']));validate_core(graph,trajectory)
    assert len(source['edges'])==len(graph['edges'])
    for old,new in zip(source['edges'],graph['edges']):
        assert {k:v for k,v in old.items() if k not in ('source_node','target_node')}=={k:v for k,v in new.items() if k not in ('source_node','target_node')}
    before=sorted(o for n in source['nodes'] for o in n['representative_observations'])
    after=sorted(o for n in graph['nodes'] for o in n['representative_observations'])
    assert before==after and len(after)==len(set(after))
    frozen(out/'graph.json',graph);frozen(out/'node_mapping.json',mapping)
    summary=dict(candidate_pairs=len(pairs),before_nodes=len(source['nodes']),after_nodes=len(graph['nodes']),
        observed_nodes=sum(bool(n['representative_observations']) for n in graph['nodes']),
        preserved_edges=len(graph['edges']),preserved_images=len(after),applied=applied,deferred=deferred,robot_calls=0)
    frozen(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
