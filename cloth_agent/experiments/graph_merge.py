"""Explicit, offline GPT visual node consolidation; no robot execution."""
import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import subprocess
import textwrap

from .graph_policy import (Requests, load_frozen, frozen, checked_file, goal_images,
    graph_validation, eligible_edges, digest, file_hash, save, SCHEMAS)


def merge_graph(source, proposal, trajectory):
    import jsonschema
    jsonschema.validate(proposal, SCHEMAS['merge'])
    old = {n['node_id']: n for n in source['nodes']}
    mapping = {}; nodes = []
    for group in proposal['groups']:
        node = group['node']; members = group['source_node_ids']
        if not members or len(set(members)) != len(members): raise ValueError('empty/duplicate group members')
        if any(k not in old or k in mapping for k in members): raise ValueError('partition must cover each known node once')
        if any(not old[k]['representative_observations'] for k in members) and len(members) != 1:
            raise ValueError('unknown sink cannot merge with observed states')
        reps = set(o for k in members for o in old[k]['representative_observations'])
        if set(node['representative_observations']) != reps or len(node['representative_observations']) != len(reps):
            raise ValueError('merged representatives must exactly preserve source observations')
        if reps and not group['visual_evidence']: raise ValueError('visual merge evidence required')
        for k in members: mapping[k] = node['node_id']
        nodes.append(deepcopy(node))
    if set(mapping) != set(old): raise ValueError('partition omitted nodes')
    for rejected in proposal['rejected_merges']:
        if set(rejected['node_ids']) - set(old): raise ValueError('unknown rejected node')
        if set(rejected['evidence_observations']) - set(trajectory['observations']): raise ValueError('unknown evidence image')
    graph = deepcopy(source)
    graph.update(graph_id=source['graph_id'] + '_visual_merge_v2', nodes=nodes,
                 limitations=source['limitations'] + proposal['limitations'])
    for edge in graph['edges']:
        edge['source_node'] = mapping[edge['source_node']]
        edge['target_node'] = mapping[edge['target_node']]
    graph_validation({k: graph[k] for k in SCHEMAS['compile']['properties']}, trajectory)
    graph['node_mapping'] = mapping
    graph['replay_policy'] = dict(edge_reuse='ONCE_PER_RUN', require_observation_after_each_edge=True)
    return graph


def connectivity(graph, source):
    observed = [n['node_id'] for n in source['nodes'] if n['representative_observations']]
    start = graph['node_mapping'][observed[0]]
    # This is last recorded observation, NOT an inferred successful task terminal.
    last = graph['node_mapping'][observed[-1]]
    visited = {start}; todo = [(start, [])]; path = None
    while todo:
        node, route = todo.pop(0)
        if node == last and path is None: path = route
        for e in eligible_edges(graph, node):
            if e['target_node'] not in visited:
                visited.add(e['target_node']); todo.append((e['target_node'], route+[e['edge_id']]))
    eligible = {e['edge_id'] for n in graph['nodes'] for e in eligible_edges(graph,n['node_id'])}
    return dict(start_node=start, last_observed_node=last,
        reachable_nodes=sorted(visited), route_to_last_observation=path,
        excluded_edges=[e['edge_id'] for e in graph['edges'] if e['edge_id'] not in eligible],
        limitation='Directed reachability only, not physical success or validation of merged-state equivalence.')


def render(output, graph, proposal, trajectory):
    from .decision_state_discovery import montage
    folder = Path(output).parent; stem = Path(output).stem
    assets = folder / (stem + '_images'); assets.mkdir(exist_ok=True)
    pictures = {}; prototypes = []
    for i, node in enumerate(graph['nodes']):
        ids = node['representative_observations']
        if not ids: continue
        path = assets / f'state_{i:02d}.png'
        montage(ids, {k: checked_file(trajectory['observations'][k]['rgb']) for k in ids}, path)
        pictures[node['node_id']] = path
        prototypes.append(dict(id='state_prototype_'+node['node_id'], node_id=node['node_id'],
                               artifact=dict(path=str(path.resolve()),sha256=file_hash(path))))
    aliases = {n['node_id']:f'n{i}' for i,n in enumerate(graph['nodes'])}
    dot = ['digraph G {', 'rankdir=LR; graph [bgcolor="white", nodesep=0.4, ranksep=0.9];',
           'node [shape=box, fontname="DejaVu Sans", fontsize=12]; edge [fontname="DejaVu Sans", fontsize=10];']
    for n in graph['nodes']:
        label=html.escape(n['node_id']) + (' [TERMINAL]' if n['node_id'] in graph.get('terminal_node_ids', []) else ''); text='<BR/>'.join(html.escape(line) for line in textwrap.wrap(n['task_context'],36))
        image = f'<TR><TD><IMG SRC="{html.escape(str(pictures[n["node_id"]].resolve()))}" SCALE="TRUE"/></TD></TR>' if n['node_id'] in pictures else ''
        dot.append(f'{aliases[n["node_id"]]} [label=<<TABLE BORDER="0" CELLBORDER="0"><TR><TD><B>{label}</B></TD></TR>{image}<TR><TD>{text}</TD></TR></TABLE>>];')
    valid = {e['edge_id'] for n in graph['nodes'] for e in eligible_edges(graph,n['node_id'])}
    for e in graph['edges']:
        label=json.dumps(e['edge_id']+'\n'+e['action_family'])
        color = '#167947' if e['edge_id'] in valid else '#b74747'
        style = 'solid' if e['edge_id'] in valid else 'dashed'
        dot.append(f'{aliases[e["source_node"]]} -> {aliases[e["target_node"]]} [label={label}, color="{color}", style={style}];')
    dot += ['}']; dot_path=folder/(stem+'.dot');dot_path.write_text('\n'.join(dot))
    for ext in ('svg','png'):
        subprocess.run(['dot','-T'+ext,str(dot_path),'-o',str(folder/(stem+'.'+ext))],check=True,timeout=60)
    import base64
    svg_path=folder/(stem+'.svg')
    svg=svg_path.read_text()
    for path in pictures.values():
        svg=svg.replace(str(path.resolve()),'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode())
    svg_path.write_text(svg)
    lines=['# Merged observation-action graph','',f'![Directed graph]({stem}.svg)','',
           'Green solid edges are eligible one-shot hypotheses. Red dashed edges are audit-only failures/unknown endpoints.',
           'Self-loops preserve observed actions with no class change. No edges or success evidence were invented.', '',
           '## Reachability', '', '```json',json.dumps(graph['connectivity'],indent=2),'```','',
           '| State | Source nodes | Observations |','|---|---|---|']
    for g in proposal['groups']:
        n=g['node'];lines.append(f"| {n['node_id']} | {', '.join(g['source_node_ids'])} | {', '.join(n['representative_observations'])} |")
    if graph.get('user_review'):
        lines += ['', '## User review', '', '```json', json.dumps(graph['user_review'],ensure_ascii=False,indent=2),'```','']
    for g in proposal['groups']:
        n=g['node'];lines += ['', '## '+n['node_id']+': '+n['task_context'],'']
        if n['node_id'] in pictures:lines += [f"![{n['node_id']}]({pictures[n['node_id']].relative_to(folder)})",'']
        lines += ['```json',json.dumps(g,ensure_ascii=False,indent=2),'```']
    lines += ['', '## Directed edges', '', '| From | Action | To | Recorded outcome |', '|---|---|---|---|']
    for e in graph['edges']:lines.append(f"| {e['source_node']} | {e['edge_id']}: {e['action_family']} | {e['target_node']} | {e['seed_outcome']} |")
    (folder/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    return prototypes


def consolidate(graph_path, output, requests):
    graph_path=Path(graph_path);output=Path(output)
    if output.resolve()==graph_path.resolve() or output.exists():raise ValueError('use a new graph version output')
    source=load_frozen(graph_path);trajectory=load_frozen(source['seed_trajectory_path'])
    if digest(trajectory)!=source['seed_trajectory_hash']:raise ValueError('seed hash mismatch')
    images=[(k,checked_file(v['rgb'])) for k,v in trajectory['observations'].items()]+goal_images(source)
    proposal=requests.ask('merge',dict(goal=source['goal'],source_graph=source),images,
                         validator=lambda p:merge_graph(source,p,trajectory))
    graph=merge_graph(source,proposal,trajectory)
    graph['merge_provenance']=dict(source_graph=str(graph_path.resolve()),source_sha256=file_hash(graph_path),
        proposal_sha256=digest(proposal),model_metrics=requests.metrics())
    graph['connectivity']=connectivity(graph,source)
    frozen(output.with_name(output.stem+'_merge_review.json'),proposal)
    graph['state_prototypes']=render(output,graph,proposal,trajectory)
    frozen(output,graph)
    from .graph_template_compiler import compile_policy
    compile_policy(output, output.parent / (output.stem + '_compiled_policy'))
    return graph


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--max-calls',type=int,default=2)
    p.add_argument('--timeout-s',type=float,default=900)
    a=p.parse_args(argv)
    from .discovery_model import CodexMultimodal
    output=a.output or a.graph.parent/'graph_v2.json'
    req=Requests(output.parent/(output.stem+'_merge'),CodexMultimodal(a.model,a.timeout_s),max_calls=a.max_calls)
    graph=consolidate(a.graph,output,req)
    print(json.dumps(dict(output=str(output),nodes=len(graph['nodes']),edges=len(graph['edges']),
                         connectivity=graph['connectivity'],metrics=req.metrics()),ensure_ascii=False,indent=2))


if __name__=='__main__':raise SystemExit(main())
