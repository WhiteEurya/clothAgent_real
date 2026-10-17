"""Bounded repair of an observed self-loop, preserving edge identity and evidence."""
import copy

import jsonschema

from .discovery_data import save
from .graph_policy import command, evidence_file
from .graph_policy_schema import TEMPLATE


def select_recorded_action(growth, node):
    """Prefer a verified revision, then success, then an explicit recorded trial."""
    rows=[growth.state['actions'][aid] for aid in growth.state['nodes'][node]['outgoing_actions']]
    for predicate in (lambda a: bool(a.get('revisions')),
                      lambda a: a['successful_observations']>0,
                      lambda a: a['preferred'] and a['replayable'],
                      lambda a: a.get('edge') is not None):
        candidates=[a for a in rows if a.get('edge') is not None and predicate(a)]
        if candidates:return candidates[-1]
    return None


def original_edge(action, state):
    """Validate every committed replacement before allowing a snapshot to reload."""
    revisions=action.get('revisions',[])
    if not revisions:return action['edge']
    previous=revisions[0]['previous_edge']
    attempts={a['attempt_id']:a for a in state['attempts']}
    transitions={t['transition_id']:t for t in state['transitions']}
    for revision in revisions:
        attempt=attempts[revision['attempt_id']]
        transition=transitions[revision['transition_id']]
        edge=revision['edge']
        trigger=transitions[revision['trigger_transition_id']]
        if (revision['previous_edge']!=previous or revision['status']!='COMMITTED'
                or trigger['selected_action']!=action['action_id']
                or trigger['source_node']!=action['source_node']
                or trigger['result_node']!=action['source_node']
                or trigger['execution'].get('status')!='EXECUTED'
                or trigger['execution'].get('physical_execution') is not True
                or attempt['planner_source']!='GPT_RECOVERY'
                or attempt['selected_action']!=action['action_id']
                or attempt['source_node']!=action['source_node']
                or transition['attempt_id']!=attempt['attempt_id']
                or transition['selected_action']!=action['action_id']
                or transition['result_node']==action['source_node']
                or transition['effect']['progress'] not in ('PROGRESS','COMPLETE')
                or transition['execution'].get('status')!='EXECUTED'
                or transition['execution'].get('physical_execution') is not True
                or edge['edge_id']!=action['action_id']
                or edge['source_node']!=action['source_node']
                or edge['target_node']!=transition['result_node']
                or command(attempt['command_source'])!=edge['seed_action']['exact_command']):
            raise ValueError('edge revision lacks verified physical recovery evidence')
        target=action['nominal_target'] or previous.get('target_node')
        if target and target!=action['source_node'] and target!=transition['result_node']:
            raise ValueError('edge revision did not reach the original target')
        jsonschema.validate(edge['action_template'],TEMPLATE)
        previous=edge
    if action['edge']!=previous:raise ValueError('active edge differs from committed revision')
    return revisions[0]['previous_edge']


def recover_self_loop(growth, runtime, requests, live, episode, action, trigger,
                      before, folder, planner, *, threshold, timeout_s):
    """One action, one observation; unknown observations become provisional nodes."""
    from .graph_policy_replay import SCHEMAS
    state=growth.state; node=action['source_node']; aid=action['action_id']
    folder.mkdir()
    row=dict(id=f"recovery_{len(episode['recoveries']):03d}",source_node=node,edge_id=aid,
             status='PLANNING',before=evidence_file(before.image),after=None,
             trigger_transition_id=trigger['transition_id'],matched_node=None)
    episode['recoveries'].append(row)
    attempt=dict(attempt_id=f"A{len(state['attempts']):07d}",episode_id=episode['episode_id'],
        source_node=node,selected_action=aid,planner_source='GPT_RECOVERY',before=row['before'],
        command_source=None,execution=dict(status='NOT_DISPATCHED',physical_execution=False),
        local_match=dict(candidates=[],result='NOT_OBSERVED',confidence=None),
        global_reentry=dict(used=False,result='NOT_OBSERVED'),new_node_created=False,
        result_node=None,transition_type='UNKNOWN')
    state['attempts'].append(attempt);episode['steps'].append(attempt['attempt_id'])
    row['attempt_id']=attempt['attempt_id']
    def snapshot():
        save(folder/'audit.json',row);growth.snapshot()
        live.update(state,current=row.get('matched_node') or node,action=aid,
                    phase='RECOVERY',recoveries=episode['recoveries'])
    snapshot()
    class Recorder:
        def __getattr__(self,name):return getattr(runtime,name)
        def execute(self,source,observation,dispatch_folder):
            attempt['command_source']=source;row['recovery_action']=command(source)
            ops=row['recovery_action']['operations']
            if (len(ops)<2 or ops[0]['method']!='grasp'
                    or any(op['method'] not in ('move_tcp','move_transfer') for op in ops[1:])
                    or any(set(op['arguments'])!={'right'} for op in ops)
                    or ops==command(previous['command_source'])['operations']):
                result=dict(status='PLANNING_FAILED',physical_execution=False,
                            error='Recovery must be one changed single-grasp action')
                attempt['execution']=result;row['execution']=result;snapshot()
                return result
            row['status']='EXECUTING';snapshot()
            result=runtime.execute(source,observation,dispatch_folder)
            attempt['execution']=result;row['execution']=result;snapshot()
            return result
    previous=next(a for a in state['attempts'] if a['attempt_id']==trigger['attempt_id'])
    target=action['nominal_target'] or action['edge'].get('target_node')
    context=dict(reason='REPAIR_EXISTING_EDGE_AFTER_SELF_LOOP',source_node=node,
        selected_edge=copy.deepcopy(action['edge']),target_node=target if target!=node else None,
        failure=dict(transition=dict(trigger,command_source=previous['command_source']),
                     execution=previous['execution']))
    try:
        result=planner(Recorder(),before,folder/'planner',growth.view(),
                       timeout_s=timeout_s,budget=requests,context=context)
        source=result.get('source');outcome=result['outcome']
        if attempt['command_source'] is not None and source!=attempt['command_source']:
            raise ValueError('Recovery result differs from recorded dispatch')
        attempt.update(command_source=source,execution=outcome)
        row.update(recovery_action=command(source) if source else None,execution=outcome)
        if outcome.get('status')!='EXECUTED' or outcome.get('physical_execution') is not True:
            row.update(status='RECOVERY_UNRESOLVED',reason='Recovery was refused or failed preflight/execution')
            snapshot();return None
        after=runtime.observe(folder/'after_observation');row['after']=evidence_file(after.image)
        attempt['after']=row['after'];row['status']='MATCHING';snapshot()
        matched,audit=growth.match(after,folder,source=node,action=aid,threshold=threshold)
        attempt.update(matching=audit,result_node=matched);row.update(matching=audit,matched_node=matched)
        try:
            effect=requests.ask('growth_effect',dict(task=growth.base['goal']),
                                [('before',before.image),('after',after.image)])
            jsonschema.validate(effect,SCHEMAS['growth_effect'])
            if effect['progress'] in ('PROGRESS','COMPLETE') and not effect['evidence']:
                raise ValueError('positive progress requires visual evidence')
        except Exception as exc:
            effect=dict(progress='UNKNOWN',task_progress='Progress assessment unavailable',evidence=[])
            attempt['effect_error']=f'{type(exc).__name__}: {exc}'
        attempt['effect']=effect;row['effect']=effect
        if matched is None:
            if audit['global_reentry']['result']!='NO_MATCH':
                row.update(status='RECOVERY_UNRESOLVED',reason='Recovery outcome matching is ambiguous')
                snapshot();return None
            matched=growth.observe_node(after,source=node,action=aid,task_progress=effect['task_progress'])
            audit['new_node_created']=True
            attempt.update(result_node=matched,new_node_created=True)
            row.update(matched_node=matched,new_node_created=True)
        state['nodes'][matched]['observation_count']+=1
        transition=growth.transition(attempt,matched,audit,effect,promote=False)
        save(folder/'transition.json',transition);row['transition_id']=transition['transition_id']
        if audit['new_node_created']:
            # A newly recognized state is an observed alternative, not proof that
            # the original edge's intended target was reached. Continue from it.
            row.update(status='REMATCHED',replacement='NOT_COMMITTED',reason='NEW_NODE_CREATED')
            snapshot();return after,matched
        if (matched==node or effect['progress'] not in ('PROGRESS','COMPLETE') or not effect['evidence']
                or (target and target!=node and matched!=target)):
            row.update(status='RECOVERY_UNRESOLVED',reason='Recovery did not verify progress to the intended existing target')
            snapshot();return None
        exact=command(source)
        if (len(exact['operations'])<2 or exact['operations'][0]['method']!='grasp'
                or any(op['method'] not in ('move_tcp','move_transfer') for op in exact['operations'][1:])
                or any(set(op['arguments'])!={'right'} for op in exact['operations'])):
            raise ValueError('Recovery has no reusable single-grasp motion')
        template=requests.ask('growth_action',dict(task=growth.base['goal'],exact_command=exact),
                             [('before',before.image),('after',after.image)])
        jsonschema.validate(template,SCHEMAS['growth_action'])
        edge=dict(edge_id=aid,source_node=node,target_node=matched,
                  seed_action=dict(exact_command=exact,source_observation=row['before']['path']),**template)
        revision=dict(status='COMMITTED',attempt_id=attempt['attempt_id'],
            transition_id=transition['transition_id'],trigger_transition_id=trigger['transition_id'],
            previous_edge=copy.deepcopy(action['edge']),edge=edge)
        # Commit after observation, matching, effect verification and template validation.
        action.setdefault('revisions',[]).append(revision);action['edge']=edge
        row.update(status='REMATCHED',replacement='COMMITTED',revision=len(action['revisions']))
        snapshot();return after,matched
    except Exception as exc:
        row.update(status='RECOVERY_UNRESOLVED',reason=f'{type(exc).__name__}: {exc}')
        snapshot();return None
