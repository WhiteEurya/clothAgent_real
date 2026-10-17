"""Topology selects a shortlist; the model sees only visual signatures, never edges."""
import time
from pathlib import Path
import jsonschema
from .discovery_data import save
from .graph_policy import checked_file, reliable_match
from .graph_policy_schema import MATCH


def validate_match_response(value, subset, threshold):
    jsonschema.validate(value,MATCH)
    reliable_match(value,subset,threshold)
    if value['status']=='MATCH' and (value['violated_conditions'] or value['unresolved_conditions']):
        raise ValueError('Contradictory MATCH: violated_conditions/unresolved_conditions describe ONLY the selected node. If these are reasons for rejecting OTHER nodes, put them in matched_evidence. If they actually contradict the selected node, return AMBIGUOUS or NO_MATCH with empty node_id. Recheck the images; do not erase genuine uncertainty just to pass validation.')


def signatures(graph, candidates, *, with_representatives=False):
    nodes={n['node_id']:n for n in graph['match_nodes']}
    rows=[]
    for key in candidates:
        n=nodes[key]
        # Reuse existing visual text, never ask a model to invent a different taxonomy.
        row=dict(node_id=key, visual_signature=n['task_context'],
                 positive_evidence=n['observation_condition']['positive_evidence'],
                 negative_evidence=n['observation_condition']['negative_evidence'],
                 uncertainty_conditions=n['observation_condition']['uncertainty_conditions'])
        if with_representatives:row['representative_observations']=n['representative_observations']
        rows.append(row)
    return dict(task=dict(subgoal=graph['goal']['subgoal'],
                         side_convention=graph['goal'].get('side_convention','')),nodes=rows)


def reference_images(graph, trajectory, observation, candidates):
    nodes={n['node_id']:n for n in graph['match_nodes']}
    ids=[]
    for key in candidates:
        refs=nodes[key]['representative_observations']
        if not refs:raise ValueError('node lacks representative RGB: '+key)
        ids.extend(refs)
    return [('current',observation.image)]+[(key,checked_file(trajectory['observations'][key]['rgb'])) for key in dict.fromkeys(ids)]


def expanded_candidates(graph, source, target, recovery_nodes=()):
    observed={n['node_id'] for n in graph['match_nodes']}
    result=list(dict.fromkeys(n for n in (source,target) if n in observed))
    neighbors=[]
    for e in graph['edges']:
        if e['source_node']==target:neighbors.append(e['target_node'])
        if e['target_node']==target:neighbors.append(e['source_node'])
    neighbors=list(dict.fromkeys(n for n in neighbors if n in observed and n not in result))
    recovery=list(dict.fromkeys(n for n in recovery_nodes if n in observed and n not in result))
    # Reserve one slot for an explicitly registered recovery node if present.
    for n in neighbors[:max(0,4-len(result)-(1 if recovery else 0))]:result.append(n)
    for n in recovery:
        if len(result)>=4:break
        if n not in result:result.append(n)
    for n in neighbors:
        if len(result)>=4:break
        if n not in result:result.append(n)
    return result[:4]


def topology_match(graph, trajectory, observation, requests, folder, metrics, *, pending=None,
                   threshold=.8, initial_node=None, recovery_nodes=(), candidate_override=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ids=[n['node_id'] for n in graph['match_nodes']]
    for key in ('local_match_time','expanded_match_time','global_match_time'):
        metrics.setdefault(key,0.)
    metrics.setdefault('representative_image_fallback_count',0)
    attempts=[]
    metrics.setdefault('representative_image_match_count',0)
    def ask(stage,candidates):
        subset=dict(nodes=[n for n in graph['match_nodes'] if n['node_id'] in candidates])
        images=reference_images(graph,trajectory,observation,candidates)
        metrics['representative_image_match_count']+=1
        started=time.monotonic()
        row=dict(stage=stage,candidate_nodes=candidates,representative_images=True,
                 image_ids=[k for k,_ in images])
        try:
            def validate(value):
                validate_match_response(value,subset,threshold)
            value=requests.ask('node_match',signatures(graph,candidates,with_representatives=True),
                               images,validator=validate)
            validate(value)
            row['response']=value
            return value,reliable_match(value,subset,threshold)
        except BaseException as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            row['elapsed_s']=time.monotonic()-started
            metrics[stage+'_match_time']+=row['elapsed_s']
            attempts.append(row)
            save(folder/'matching_attempts.json',dict(attempts=attempts,
                time_unit='seconds',timing_scope='host wall time including cache and bounded contract repair'))
    levels=[]
    if candidate_override is not None:
        candidates=list(dict.fromkeys(candidate_override))
        if not candidates or set(candidates)-set(ids):raise ValueError('invalid fixed matching candidates')
        return ask('local',candidates)[0]
    if pending and pending.get('target_node') in ids and pending.get('source_node') in ids:
        local=list(dict.fromkeys([pending['source_node'],pending['target_node']]))
        expanded=expanded_candidates(graph,pending['source_node'],pending['target_node'],recovery_nodes)
        levels=[('local',local)]
        if set(expanded)!=set(local):levels.append(('expanded',expanded))
    elif pending is None and initial_node is not None:
        if initial_node not in ids:raise ValueError('initial node must be observed in graph')
        levels=[('local',[initial_node])]
    # Free-planner recovery has no known target: global recognition remains necessary.
    if not levels or set(levels[-1][1]) != set(ids):levels.append(('global',ids))
    for name,candidates in levels:
        value,ok=ask(name,candidates)
        if ok:return value
    # No forced match. An unresolved final ambiguity still stops the runner.
    return value
