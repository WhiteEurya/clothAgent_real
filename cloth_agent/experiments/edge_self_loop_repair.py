"""Append-only edge failure evidence and verified repair template revisions."""
import copy
from pathlib import Path
from .discovery_data import save
from .graph_policy import command, evidence_file, frozen


class EdgeRepairs:
    def __init__(self, output, requests, graph_path=None):
        self.output=Path(output)/'edge_repairs';self.output.mkdir(parents=True)
        self.requests=requests;self.edges={};self.pending={}
        self.graph_path=Path(graph_path).resolve() if graph_path else None
        grounder=getattr(requests,'minimal_grounder',None)
        self.bundle=copy.deepcopy(grounder.bundle) if grounder else None
        if self.graph_path is not None and grounder is not None:
            from .graph_policy import file_hash
            import json
            registry=self.graph_path.parent/'edge_repair_registry'/file_hash(self.graph_path)/'latest.json'
            if registry.exists():
                manifest=json.loads(registry.read_text())
                # Explicitly choosing another bundle must not inherit unrelated revision history.
                if Path(manifest['host_templates']).resolve()==Path(grounder.path).resolve():
                    previous=json.loads(Path(manifest['edge_experience']).read_text())
                    self.edges=copy.deepcopy(previous['edges'])

    def snapshot(self):
        save(self.output/'edge_experience.json',dict(version=1,edges=self.edges,
            semantics='Attributes and command revisions of existing edge IDs; original seed preserved'))

    def begin(self, edge, transition, review, observation, folder, *, trigger='SELF_LOOP'):
        eid=edge['edge_id'];row=self.edges.setdefault(eid,dict(edge_id=eid,failures=[],revisions=[]))
        category={'UNCHANGED':'NO_VISIBLE_CHANGE_POSSIBLE_MISSED_GRASP',
                  'MOVED':'MOVED_WITHOUT_EXPECTED_STATE_TRANSITION'}.get(review['result'],'UNRESOLVED')
        if trigger=='IK_FAILED':category='IK_FAILED_BEFORE_DISPATCH'
        failure=dict(trigger=trigger,source_node=transition['source_node'],result_node=transition['actual_node'],
            category=category,z_miss='SUSPECTED_NOT_CONFIRMED' if review['result']=='UNCHANGED' else 'NOT_ESTABLISHED',
            visual_review=review,transition=copy.deepcopy(transition))
        row['failures'].append(failure)
        revision=dict(revision=len(row['revisions'])+1,status='PLANNING',replayable=False,
            before=evidence_file(observation.image),expected_target=edge['target_node'])
        row['revisions'].append(revision)
        self.pending[eid]=(edge,observation,Path(folder),revision)
        self.snapshot()
        return dict(reason='REPAIR_EXISTING_EDGE_AFTER_'+trigger,edge_id=eid,
            original_action_family=edge['action_family'],original_action_template=edge['action_template'],
            expected_target=edge['target_node'],failure=failure,
            instruction='Repair this edge for the current observation, preserving its manipulation role and intended target state. Do not pursue another task or repeat the failed command unchanged. Treat Z/contact cause as a hypothesis, not established truth. Host safety checks remain mandatory.')

    def executed(self,eid,result):
        revision=self.pending[eid][3]
        revision.update(status='AWAITING_FRESH_MATCH',command_source=result.get('source'),execution=result['outcome'])
        if result['outcome'].get('status')!='EXECUTED' or not result['outcome'].get('physical_execution'):
            revision['status']='NOT_EXECUTED_SUCCESSFULLY'
        self.snapshot()

    def finish(self,eid,node,after):
        edge,obs,folder,revision=self.pending.pop(eid)
        revision.update(result_node=node,after=evidence_file(after.image))
        if node!=edge['target_node']:
            revision['status']='SELF_LOOP' if node==edge['source_node'] else 'UNEXPECTED_OR_UNMATCHED_OUTCOME'
            self.snapshot();return
        revision['status']='EXPECTED_TARGET_OBSERVED'
        try:
            from .graph_template_compiler import seed_slots
            from .action_local_frame import compile_local
            from .graph_minimal_grounding import read,edge_hash
            round_folder=folder/'round_001'
            effective=(round_folder/'effective_motion.py').read_text()
            lowering=read(round_folder/'grasp_height_policy.json')
            exact=command(effective);revised=copy.deepcopy(edge)
            revised['seed_action']['exact_command']=exact
            step=dict(commanded_action=exact,intermediate_evidence={
                'grasp_height_policy.json':dict(content=lowering)})
            slots,_,errors,_=seed_slots(revised,step,obs)
            record=dict(seed_grasp_px=slots[0]['pixel_xy'],seed_target_px=slots[-1]['pixel_xy'],
                actual_grasp_lowering=lowering,grasp_semantic_relation=edge['action_template']['grasp_relation'],
                target_semantic_relation=edge['action_template']['target_relation'])
            template=compile_local(revised,step,obs,record,contact_z=True)
            # Binding remains to the original edge identity; new command provenance is explicit.
            template['edge_sha256']=edge_hash(edge)
            template['repair_provenance']=dict(revision=revision['revision'],exact_executed_command=exact,
                before=revision['before'],after=revision['after'],result_node=node,
                reconstruction_xy_error_mm=errors)
            if self.bundle is None:raise ValueError('no minimal template bundle available')
            self.bundle['edges'][eid]=template
            path=self.output/f'host_templates_revision_{sum(len(x["revisions"]) for x in self.edges.values()):03d}.json'
            frozen(path,self.bundle)
            revision.update(status='MINIMAL_SUPPORTED',replayable=True,host_templates=str(path.resolve()))
            self.edges[eid]['preferred_revision']=revision['revision']
            self.requests.minimal_grounder.bundle=copy.deepcopy(self.bundle)
            self.requests.minimal_grounder.path=path
            try:
                if self.graph_path is not None:
                    from .graph_policy import file_hash
                    import os, tempfile, json
                    registry=self.graph_path.parent/'edge_repair_registry'/file_hash(self.graph_path)
                    registry.mkdir(parents=True,exist_ok=True)
                    self.snapshot()
                    manifest=dict(graph_sha256=file_hash(self.graph_path),host_templates=str(path.resolve()),
                        templates_sha256=file_hash(path),edge_experience=str((self.output/'edge_experience.json').resolve()))
                    fd,tmp=tempfile.mkstemp(dir=registry,prefix='.latest_')
                    try:
                        with os.fdopen(fd,'w') as f:json.dump(manifest,f,indent=2)
                        os.replace(tmp,registry/'latest.json')
                    finally:
                        if Path(tmp).exists():Path(tmp).unlink()
                    revision['registry']=str(registry/'latest.json')
            except OSError as exc:
                revision["publication_error"]=str(exc)

        except (ValueError,KeyError,TypeError,OSError,IndexError) as exc:
            revision.update(status='MINIMAL_UNSUPPORTED',compile_error=str(exc))
        self.snapshot()


def repairable_ik_failure(outcome):
    """Only structured, pre-dispatch IK rejection; never collision/controller failures."""
    return (outcome.get('status')=='PLANNING_FAILED'
        and outcome.get('physical_execution') is False
        and outcome.get('failure_kind')=='IK_FAILED'
        and outcome.get('ik_failure',{}).get('kind')=='IK_NO_ACCEPTABLE_SOLUTION')


def latest_repair_templates(graph_path):
    from .graph_policy import load_frozen, file_hash
    from .graph_minimal_grounding import MinimalGrounder
    graph_path=Path(graph_path).resolve()
    manifest=graph_path.parent/'edge_repair_registry'/file_hash(graph_path)/'latest.json'
    if not manifest.exists():return None
    data=__import__('json').loads(manifest.read_text())
    if data['graph_sha256']!=file_hash(graph_path):raise ValueError('repair registry graph mismatch')
    path=Path(data['host_templates'])
    if file_hash(path)!=data['templates_sha256']:raise ValueError('repair registry template mismatch')
    grounder=MinimalGrounder(path)
    graph=load_frozen(graph_path)
    for edge in graph['edges']:
        if edge['edge_id'] in grounder.bundle['edges']:grounder.template(edge)
    return path
