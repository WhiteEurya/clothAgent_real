"""One grasp-handle recovery, with audit only: no Graph nodes or policy changes."""
import copy
import logging
from pathlib import Path

from .discovery_data import digest, file_hash, save
from .graph_policy import command, evidence_file
from .graph_minimal_grounding import NoRobustAction

REASON = 'NO_ROBUST_ACTION_RECOVERY'


def observation_key(observation):
    """Content identity, not a new timestamp/path masquerading as new evidence."""
    cloud = Path(observation.image).parent/'world_xyz_mm.npy'
    return digest(dict(rgb=file_hash(observation.image),
                       xyz=file_hash(cloud) if cloud.exists() else None))


class RecoveryUnresolved(RuntimeError):
    pass


class HandleRecovery:
    def __init__(self, output, limit=1, on_update=None):
        if limit not in (0, 1):
            raise ValueError('max_recovery_attempts must be 0 or 1')
        self.output = Path(output)/'robustness_recovery'
        self.limit = limit
        self.attempts = []
        self.rejections = []
        self.rejected_observations = set()
        self.awaiting_match = False
        self.on_update = on_update

    def snapshot(self):
        save(self.output/'audit.json',dict(max_recovery_attempts=self.limit,
            recovery_attempts=len(self.attempts),graph_mutation_allowed=False,
            gate_rejections=self.rejections,attempts=self.attempts))
        # Read-only display events, separate from policy nodes and matching input.
        if self.on_update is not None:
            try:
                events=[]
                for index,row in enumerate(self.attempts,1):
                    rematch=row.get('rematch') or {}
                    events.append(dict(id=f'handle-recovery-{index}',
                        origin_node=row['origin_node'],origin_edge=row['origin_edge'],
                        status=row['status'],reasons=list(row['gate_rejection_reasons']),
                        before=copy.deepcopy(row.get('before')),after=copy.deepcopy(row.get('after')),
                        matched_node=(rematch.get('result') or {}).get('node_id') if rematch.get('reliable') else None,
                        redispatched=row['target_edge_reject_to_dispatch']))
                self.on_update(events)
            except Exception:
                logging.exception('Recovery display update failed; replay is unaffected')

    def check_grounding(self, edge, before):
        if (edge['edge_id'], observation_key(before)) in self.rejected_observations:
            raise NoRobustAction(['Same edge and unchanged RGB-D already rejected; grounding not repeated'],before)

    def recover(self, runtime, requests, graph, edge, node, before, rejection, runner, timeout_s):
        key = observation_key(before)
        self.rejected_observations.add((edge['edge_id'],key))
        self.rejections.append(dict(node_id=node,edge_id=edge['edge_id'],
            reasons=rejection.reasons,before=evidence_file(before.image),observation_key=key))
        self.snapshot()
        if len(self.attempts)>=self.limit:
            raise RecoveryUnresolved('Recovery budget exhausted; no second agentic recovery')
        folder=self.output/'attempt_001'
        row=dict(origin_node=node,origin_edge=edge['edge_id'],target_node=edge['target_node'],
                 gate_rejection_reasons=rejection.reasons,before=evidence_file(before.image),
                 observation=str(Path(before.image).parent.resolve()),observation_key=key,
                 status='PLANNING',recovery_action=None,recovery_execution=None,
                 after=None,rematch=None,target_edge_reject_to_dispatch=False,
                 subsequent_graph_outcomes=[],subsequent_matches=[])
        self.attempts.append(row)
        context=dict(reason=REASON,original_node=node,
            recommended_edge={k:copy.deepcopy(edge[k]) for k in
                              ('edge_id','source_node','target_node','action_family','action_template')},
            gate_rejection_reasons=rejection.reasons,current_observation=row['observation'],
            objective='One minimal low-risk manipulation to expose a reliable grasp handle for the recommended skill; not forced completion of its fold.',
            max_recovery_actions=1,graph_mutation_allowed=False)
        save(folder/'context.json',context);self.snapshot()
        try:
            result=runner(runtime,before,folder/'planner',graph,timeout_s=timeout_s,budget=requests,context=context)
            save(folder/'planner_result.json',result)
            source=result.get('source')
            row['recovery_source']=source
            outcome=result.get('outcome') or {}
            row['recovery_execution']=outcome
            if source:row['recovery_action']=command(source)
            if outcome.get('status')!='EXECUTED' or outcome.get('physical_execution') is not True:
                raise RecoveryUnresolved('Recovery unavailable or preflight/execution rejected: '
                    +str(outcome.get('status',result.get('result')))+': '+str(outcome.get('reason') or outcome.get('error') or ''))
            row['status']='AWAITING_OBSERVATION';self.snapshot()
            fresh=runtime.observe(folder/'after_observation')
            row['after']=evidence_file(fresh.image)
            row['after_observation']=str(Path(fresh.image).parent.resolve())
            row['observation_changed']=observation_key(fresh)!=key
            row['status']='AWAITING_REMATCH';self.awaiting_match=True
            self.snapshot()
            return fresh
        except Exception as exc:
            row.update(status='RECOVERY_UNRESOLVED',error=f'{type(exc).__name__}: {exc}')
            if row['after'] is None:
                row['after_unavailable_reason']='No successful recovery execution and safe recapture completed'
            self.snapshot()
            raise RecoveryUnresolved(row['error']) from exc

    def matched(self, match, reliable, before):
        if not self.attempts:return
        row=self.attempts[-1]
        record=dict(result=copy.deepcopy(match),reliable=reliable,current_rgb=evidence_file(before.image))
        if self.awaiting_match:
            row['rematch']=record
            row['status']='REMATCHED' if reliable else 'RECOVERY_UNRESOLVED'
            self.awaiting_match=False
            self.snapshot()
            if not reliable:
                raise RecoveryUnresolved('Fresh recovery observation did not reliably match an existing Graph node')
        else:
            row['subsequent_matches'].append(record)
            if row['subsequent_graph_outcomes']:
                event=row['subsequent_graph_outcomes'][-1]
                if event['after'] is None:
                    from .node_match_progress import node_match_progress
                    event['after']=evidence_file(before.image)
                    event['rematch']=record
                    event['progress_evaluation']=node_match_progress(event['source_node'],event['target_node'],
                                                                    match.get('node_id'),reliable=reliable)
            self.snapshot()

    def graph_outcome(self, edge, source, before, outcome):
        if not self.attempts:return
        row=self.attempts[-1]
        dispatched=outcome.get('physical_execution') is True
        row['subsequent_graph_outcomes'].append(dict(edge_id=edge['edge_id'],
            source_node=edge['source_node'],target_node=edge['target_node'],after=None,
            before=evidence_file(before.image),command=command(source),execution=copy.deepcopy(outcome),
            physical_cloth_outcome='UNKNOWN until fresh observation/matching; controller success is not cloth success'))
        if edge['edge_id']==row['origin_edge'] and dispatched:
            row['target_edge_reject_to_dispatch']=True
        self.snapshot()

    def finish(self, status, graph_unchanged):
        if not self.rejections:return
        for row in self.attempts:
            row['loop_status']=status
            row['graph_unchanged']=graph_unchanged
            if status=='RECOVERY_UNRESOLVED':row['status']=status
        self.snapshot()
