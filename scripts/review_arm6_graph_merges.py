"""Offline cross-node RGB review using cached DINO vectors; never connects hardware.

One round applies disjoint reviewed pairs. Overlapping approvals are retained for
future review, not transitively unioned without inspecting the enlarged group.
"""
import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import jsonschema

from cloth_agent.experiments.discovery_model import CodexMultimodal
from cloth_agent.experiments.discovery_schema import obj, array, S, N, enum
from cloth_agent.experiments.graph_policy import (
    Requests, checked_file, evidence_file, frozen, load_frozen, graph_validation,
)
from cloth_agent.experiments.graph_policy_schema import GRAPH, EDGE
from cloth_agent.experiments.visual_retrieval_index import validate_memory

SCHEMA = obj(decisions=array(obj(
    pair_id=S, decision=enum('MERGE', 'KEEP_SEPARATE', 'UNCERTAIN'), confidence=N,
    reason=S, evidence_observations=array(S), task_context=S,
    positive_evidence=array(S), negative_evidence=array(S), uncertainty_conditions=array(S),
)))
PROMPT = '''Compare every candidate pair using ALL its attached ORIGINAL RGBs.
Decide whether the two nodes describe visually equivalent cloth states for an
action graph. Ignore rigid rotation/translation and small wrinkles or fold-width
changes when exposed garment regions, accessible edges and coverage agree.
Use neckline-to-hem garment orientation, not image axes; do not mirror left/right.
Keep separate for a visibly exposed versus hidden sleeve/hem/corner, or a broad
covering layer that changes the accessible region and unfolding step. Similar
silhouettes alone are insufficient. Conversely, do not invent different hidden
layers from ordinary wrinkles. Compare actual RGBs, not descriptions or node IDs.
MERGE must hold for ALL images of both nodes. Report UNCERTAIN when evidence is
insufficient or members conflict. Cite observation IDs and the decisive visible
features. No node connectivity, action history, or desired result is supplied.
For MERGE provide shared task_context and positive/negative/uncertainty conditions
valid across all members. Use concise Chinese. No robot actions or new edges.'''


def candidates(graph, memory, vectors, focus=()):
    refs = memory['images']
    ids = sorted({r['node_id'] for r in refs})
    positions = {n: [i for i, r in enumerate(refs) if r['node_id'] == n] for n in ids}
    similarity = vectors @ vectors.T
    scores = {(a, b): float(similarity[np.ix_(positions[a], positions[b])].max())
              for a in ids for b in ids if a < b}
    sinks = set(ids) - {e['source_node'] for e in graph['edges']}
    selected = {}
    for nid in ids:
        ranked = sorted(((s, p) for p, s in scores.items() if nid in p), reverse=True)
        for score, pair in ranked[:3 if nid in sinks else 2]:
            if score >= (.85 if nid in sinks else .90):
                selected.setdefault(pair, []).append('sink_top3' if nid in sinks else 'node_top2')
    for pair in focus:
        pair = tuple(sorted(pair))
        if pair not in scores:
            raise ValueError('focus must contain two distinct observed nodes')
        selected.setdefault(pair, []).append('explicit_case_review')
    nodes = {n['node_id']: n for n in graph['nodes']}
    ordered = sorted(selected, key=lambda p: (
        'explicit_case_review' not in selected[p], not bool(set(p) & sinks), -scores[p], p))
    return [dict(pair_id=f'p{i:03d}', nodes=list(p), similarity=scores[p],
                 selection_reasons=selected[p], sink_nodes=sorted(set(p) & sinks),
                 observation_members={n: nodes[n]['representative_observations'] for n in p})
            for i, p in enumerate(ordered)]


def validate_review(value, pairs):
    jsonschema.validate(value, SCHEMA)
    expected = {p['pair_id']: p for p in pairs}
    rows = value['decisions']
    if len(rows) != len(expected) or {d['pair_id'] for d in rows} != set(expected):
        raise ValueError('review must cover each candidate exactly once')
    for d in rows:
        pair = expected[d['pair_id']]
        members = pair['observation_members']
        allowed = {o for group in members.values() for o in group}
        cited = set(d['evidence_observations'])
        if cited - allowed or any(not cited.intersection(group) for group in members.values()):
            raise ValueError('evidence must cite known RGBs from both nodes')
        if d['decision'] == 'MERGE' and (not d['positive_evidence'] or not d['task_context']):
            raise ValueError('merge requires shared visible state conditions')


def apply_round(source, pairs, review):
    validate_review(review, pairs)
    graph = copy.deepcopy(source)
    nodes = {n['node_id']: n for n in graph['nodes']}
    lookup = {p['pair_id']: p for p in pairs}
    for p in pairs:
        if len(p['nodes']) != 2 or len(set(p['nodes'])) != 2:
            raise ValueError('candidate is not a pair')
        for nid in p['nodes']:
            if nid not in nodes or not nodes[nid]['representative_observations']:
                raise ValueError('cannot merge unknown or unobserved node')
            if nodes[nid]['representative_observations'] != p['observation_members'][nid]:
                raise ValueError('candidate observation membership changed')
    accepted = [d for d in review['decisions'] if d['decision'] == 'MERGE' and d['confidence'] >= .85]
    accepted.sort(key=lambda d: (not bool(lookup[d['pair_id']]['sink_nodes']), -d['confidence'],
                                 -lookup[d['pair_id']]['similarity'], d['pair_id']))
    mapping = {n: n for n in nodes}
    used = set(); applied = []; deferred = []
    for d in accepted:
        p = lookup[d['pair_id']]; left, right = p['nodes']
        if used.intersection(p['nodes']):
            deferred.append(dict(pair_id=p['pair_id'], reason='overlap_requires_full_group_review'))
            continue
        used.update(p['nodes'])
        nodes[left].update(task_context=d['task_context'], confidence=d['confidence'],
            representative_observations=nodes[left]['representative_observations'] + nodes[right]['representative_observations'],
            observation_condition={k: d[k] for k in ('positive_evidence', 'negative_evidence', 'uncertainty_conditions')})
        del nodes[right]; mapping[right] = left
        applied.append(dict(pair_id=p['pair_id'], source_nodes=p['nodes'], target_node=left,
                            reason=d['reason'], confidence=d['confidence']))
    graph['nodes'] = list(nodes.values())
    for e in graph['edges']:
        e['source_node'] = mapping[e['source_node']]
        e['target_node'] = mapping[e['target_node']]
    graph['terminal_node_ids'] = sorted({mapping[n] for n in source.get('terminal_node_ids', [])})
    # A terminal label must not survive merging with a node that has actions.
    graph['terminal_node_ids'] = [n for n in graph['terminal_node_ids']
                                  if not any(e['source_node'] == n for e in graph['edges'])]
    graph['graph_id'] += '_dino_rgb_review'
    return graph, mapping, applied, deferred


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--memory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--focus', action='append', default=[], help='node1,node2')
    p.add_argument('--focus-only', action='store_true', help='Review only explicitly supplied pairs')
    p.add_argument('--only-nodes-file', type=Path, help='Frozen JSON list; review pairs touching these nodes only')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    args = p.parse_args(argv)
    if args.workers < 1 or args.batch_size < 1:
        p.error('workers and batch-size must be positive')
    source = load_frozen(args.graph)
    memory = validate_memory(args.memory, args.graph)
    trajectory = load_frozen(checked_file(source['source_trajectory']))
    vectors = np.load(args.memory/'reference_embeddings.npy', allow_pickle=False)
    pairs = candidates(source, memory, vectors, [s.split(',') for s in args.focus])
    if args.focus_only:
        pairs = [pair for pair in pairs if 'explicit_case_review' in pair['selection_reasons']]
    only_nodes = None
    if args.only_nodes_file:
        only_nodes = load_frozen(args.only_nodes_file)
        if not set(only_nodes) <= {n['node_id'] for n in source['nodes']}:
            raise ValueError('review scope includes unknown nodes')
        pairs = [pair for pair in pairs if set(pair['nodes']).intersection(only_nodes)]
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    config = dict(source_graph=evidence_file(args.graph), memory=evidence_file(args.memory/'memory.json'),
                  candidate_policy='each observed node top2 >= .90; sinks top3 >= .85; explicit focus pairs',
                  pair_score='maximum normalized DINO dot product across member RGBs',
                  confidence_threshold=.85, images='all original RGB members',
                  grouping='disjoint pairs only; no transitive union', batch_size=args.batch_size,
                  workers=args.workers, robot_calls=0)
    if only_nodes is not None: config['only_nodes'] = only_nodes
    if args.focus_only: config['focus_only'] = True
    for name, value in [('config.json', config), ('candidates.json', pairs)]:
        dest = out/name
        if dest.exists() and load_frozen(dest) != value:
            raise ValueError('resume configuration/candidates changed')
        if not dest.exists(): frozen(dest, value)
    print(json.dumps(dict(status='PREPARED', pairs=len(pairs)), ensure_ascii=False), flush=True)
    if args.prepare_only: return

    def batch(start):
        group = pairs[start:start+args.batch_size]; folder = out/f'batch_{start//args.batch_size:03d}'
        target = folder/'decisions.json'
        if target.exists():
            result = load_frozen(target); validate_review(result, group); return result['decisions']
        refs = sorted({o for pair in group for v in pair['observation_members'].values() for o in v})
        images = [(o, checked_file(trajectory['observations'][o]['rgb'])) for o in refs]
        req = Requests(folder/'model', CodexMultimodal(timeout=600), max_calls=2,
            schemas={'review': SCHEMA}, prompts={'review': PROMPT},
            common='Offline original-RGB cloth state equivalence review. No tools or robot commands.',
            protocol='arm6-dino-original-rgb-review-v1')
        payload = dict(pairs=[{k: v for k, v in pair.items() if k in ('pair_id', 'observation_members')}
                              for pair in group])
        try:
            result = req.ask('review', payload, images, validator=lambda v: validate_review(v, group))
            frozen(target, result)
        finally:
            frozen(folder/'metrics.json', req.metrics())
        print(json.dumps(dict(batch=start//args.batch_size,
            decisions=[(d['pair_id'], d['decision'], d['confidence']) for d in result['decisions']]), ensure_ascii=False), flush=True)
        return result['decisions']

    decisions = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(batch, start) for start in range(0, len(pairs), args.batch_size)]
        for future in as_completed(futures): decisions.extend(future.result())
    review = dict(decisions=sorted(decisions, key=lambda d: d['pair_id']))
    frozen(out/'decisions.json', review)
    graph, mapping, applied, deferred = apply_round(source, pairs, review)
    graph.setdefault('revision_history', []).append(dict(operation='dino_original_rgb_merge_review',
        source_graph=evidence_file(args.graph), candidates=evidence_file(out/'candidates.json'),
        decisions=evidence_file(out/'decisions.json'), merges=applied, deferred=deferred))
    graph['limitations'].append('DINO shortlisted original-RGB pair review; one disjoint merge round, not exhaustive equivalence proof.')
    core = {k: copy.deepcopy(graph[k]) for k in GRAPH['properties']}
    core['edges'] = [{k: e[k] for k in EDGE['required']} for e in core['edges']]
    graph_validation(core, trajectory)
    # Exact preservation of action metadata, evidence and edge IDs.
    assert len(source['edges']) == len(graph['edges'])
    for old, new in zip(source['edges'], graph['edges']):
        assert {k:v for k,v in old.items() if k not in ('source_node','target_node')} == {
            k:v for k,v in new.items() if k not in ('source_node','target_node')}
    before_refs = sorted(o for n in source['nodes'] for o in n['representative_observations'])
    after_refs = sorted(o for n in graph['nodes'] for o in n['representative_observations'])
    assert before_refs == after_refs and len(after_refs) == len(set(after_refs))
    frozen(out/'graph.json', graph); frozen(out/'node_mapping.json', mapping)
    def sinks(g):
        return sorted(n['node_id'] for n in g['nodes'] if n['representative_observations'] and
                      not any(e['source_node'] == n['node_id'] for e in g['edges']))
    summary = dict(status='OFFLINE_REVIEW_COMPLETE', candidate_pairs=len(pairs),
        before_nodes=len(source['nodes']), after_nodes=len(graph['nodes']),
        before_observed_nodes=sum(bool(n['representative_observations']) for n in source['nodes']),
        after_observed_nodes=sum(bool(n['representative_observations']) for n in graph['nodes']),
        preserved_images=len(after_refs), preserved_edges=len(graph['edges']), applied=applied, deferred=deferred,
        sinks_before=sinks(source), sinks_after=sinks(graph), robot_calls=0)
    frozen(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
