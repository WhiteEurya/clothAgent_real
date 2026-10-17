"""Bounded grounding escalation; evidence-only substates, immutable base policy."""
import copy
from pathlib import Path
import jsonschema
from .graph_policy import save, evidence_file, command, checked_file, frozen, load_frozen, digest


def schema(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
TEXT=dict(type='string',minLength=1)
EVIDENCE=dict(type='array',items=TEXT,minItems=1)
SCHEMAS={
 'edge_applicability':schema(dict(applicable=dict(enum=['YES','NO','UNKNOWN']),confidence=dict(type='number',minimum=0,maximum=1),evidence=EVIDENCE)),
 'applicability_outcome':schema(dict(action_relation=dict(enum=['SAME_ROLE','DIFFERENT_ROLE','UNKNOWN']),evidence=EVIDENCE,decision_condition=TEXT))}
PROMPTS={
 'edge_applicability':'''Assess ONLY whether this selected edge's manipulation role is applicable to the CURRENT RGB compared to the successful seed reference with red grasp and blue target. Can an available cloth handle reasonably move the intended cloth region, and is the receiving region usable for the SAME manipulation? Similar overall state is insufficient. YES requires visible supporting evidence; NO requires a concrete incompatible decision condition. Local overlap, uncertain exact layer identity or grasping adjacent folded layers together alone must not cause NO or UNKNOWN; use UNKNOWN only when unresolved visual evidence prevents assessing the required handle/receiving relation. Do not plan an action or output coordinates. Short visible evidence only, no chain of thought.''',
 'applicability_outcome':'''Audit the actually executed command and before/after RGB against the original edge manipulation role. Different pixel coordinates, heights or yaw alone do NOT mean DIFFERENT_ROLE. Compare grasped semantic part/layer, receiving relation and intended manipulation. Report ONLY SAME_ROLE / DIFFERENT_ROLE / UNKNOWN. Do not evaluate progress or output a progress field; the Host derives progress exclusively from node matching. Describe a concise decision condition distinguishing this BEFORE state if supported. Never infer a permanent split or merge. No chain of thought.'''}


class ApplicabilityRecovery:
    def __init__(self,output,requests,live,graph):
        self.folder=Path(output)/'applicability_recovery'
        self.requests=requests;self.live=live;self.graph=graph
        self.rows=[];self.used=set();self.active=None
        self.nodes=[];self.actions=[]
        self.registry=Path(output).parent/'grounding_evidence_registry'/digest(graph)
        self.requests.grounding_reference_overrides={}
        if self.registry.exists():
            for path in sorted(self.registry.glob('*.json'),key=lambda p:p.stat().st_mtime):
                if path.name.endswith(('.sha256.json','.invalidated.json')) or Path(str(path)+'.invalidated.json').exists():continue
                entry=load_frozen(path)
                checked_file(entry['image'])
                self.requests.grounding_reference_overrides[entry['edge_id']]=entry['template']

    def snapshot(self):
        save(self.folder/'experience.json',dict(version=1,attempts=self.rows,provisional_nodes=self.nodes,
             observed_actions=self.actions,permanent_merge=False,base_policy_modified=False))

    def prepare(self,edge,before,folder):
        from .graph_minimal_grounding import marked_seed_reference,validate,GroundingUncertain
        from .graph_policy_replay import grounding_payload,validate_grounding
        eid=edge['edge_id']
        if eid in self.used:raise ValueError('applicability recovery budget exhausted for '+eid)
        self.used.add(eid);folder=Path(folder)/'applicability';folder.mkdir(parents=True)
        row=dict(id=len(self.rows),edge_id=eid,source_node=edge['source_node'],expected_target=edge['target_node'],
                 before=evidence_file(before.image),status='ASSESSING',replayable=False)
        self.rows.append(row);self.active=row
        self.snapshot()
        grounder=self.requests.minimal_grounder;t=grounder.template(edge)
        reference=marked_seed_reference(t,edge,folder)
        images=[('current',before.image),('successful_seed_reference',reference)]
        assessment=self.requests.ask('edge_applicability',dict(edge_id=eid,action_family=edge['action_family'],
            action_template=edge['action_template'],reference_instruction='Red=seed grasp; blue=seed target. Transfer roles, never absolute pixels.'),images)
        jsonschema.validate(assessment,SCHEMAS['edge_applicability'])
        row['assessment']=assessment
        branch=assessment['applicable'] if assessment['confidence']>=.8 else 'UNKNOWN'
        row['branch']=branch;row['status']='GROUNDING' if branch=='YES' else 'FRONTIER'
        self.snapshot()
        if branch!='YES':return None
        # Same frozen Host template; broaden visual context only.
        value=self.requests.ask('ground_edge_minimal',dict(**t['spec'],reference_instruction=
            'Red/blue are successful seed grasp/target. Locate corresponding manipulation roles in CURRENT RGB; do not copy reference pixels.'),images,
            validator=lambda v:validate(v,t['image_size']))
        save(folder/'reference_anchors.json',value)
        try:
            candidate=grounder.expand(value,edge,before,folder/'reference_grounding')
            row['grounding_mode']='REFERENCE_ASSISTED'
        except GroundingUncertain:
            payload=grounding_payload(self.graph,edge,before)
            payload['reference_instruction']='Ground ONLY this selected edge, preserving its primitive sequence and semantic role; reference pixels are not current coordinates.'
            candidate=self.requests.ask('ground_edge',payload,images,validator=validate_grounding)
            from cloth_agent.action_robustness import strip_grounding_evidence
            candidate=strip_grounding_evidence(candidate,folder/'full_grounding')
            row['grounding_mode']='FULL'
        row['candidate']=candidate
        row['status']='AWAITING_EXECUTION';self.snapshot()
        return candidate

    def context(self,edge):
        return dict(reason='GROUNDING_FAILED_APPLICABILITY_'+self.active['branch'],
            original_edge_id=edge['edge_id'],original_action_family=edge['action_family'],
            original_action_template=edge['action_template'],applicability=self.active['assessment'],
            instruction='Plan exactly ONE useful folding action for the current cloth. You may change manipulation role when required. Do not claim a node split or modify the graph. Existing Host safety validation is mandatory.')

    def executed(self,source,outcome):
        if self.active:
            self.active.update(command=command(source) if source else None,execution=outcome,
                               status='AWAITING_OBSERVATION' if outcome.get('status')=='EXECUTED' else 'EXECUTION_FAILED')
            self.snapshot()

    def finish(self,edge,node,after):
        row=self.active
        if row is None:return
        row.update(result_node=node,after=evidence_file(after.image))
        execution=row.get('execution',{})
        if execution.get('status')!='EXECUTED' or execution.get('physical_execution') is not True:
            row['status']='NO_VERIFIED_EXECUTION';self.active=None;self.snapshot();return
        from .node_match_progress import node_match_progress
        row['progress_evaluation']=node_match_progress(edge['source_node'],edge['target_node'],node)
        self.snapshot()
        images=[('before',Path(row['before']['path'])),('after',after.image)]
        try:
            effect=self.requests.ask('applicability_outcome',dict(original_action_family=edge['action_family'],
                original_action_template=edge['action_template'],actual_command=row.get('command'),matched_outcome=node),images)
            jsonschema.validate(effect,SCHEMAS['applicability_outcome']);row['role_evaluation']=effect
            useful=row['progress_evaluation']['progress']=='PROGRESS'
            if effect['action_relation']=='SAME_ROLE' and useful and node==edge['target_node']:
                row.update(status='EDGE_GROUNDING_EVIDENCE',conclusion='Same source-state hypothesis retained; add successful visual grounding evidence to this edge.')
                candidate=row.get('candidate')
                if candidate and candidate.get('status')=='READY':
                    from PIL import Image
                    t=copy.deepcopy(self.requests.minimal_grounder.template(edge))
                    anchors=candidate.get('anchors_px')
                    if anchors is None:
                        slots=candidate['slots'];anchors=dict(grasp_px=slots[0]['pixel_xy'],target_px=slots[-1]['pixel_xy'])
                    t.update(reference_image=row['before']['path'],reference_sha256=row['before']['sha256'],
                        image_size=list(Image.open(row['before']['path']).size),
                        seed_grasp_px=anchors['grasp_px'],seed_target_px=anchors['target_px'])
                    # Store a verified visual reference, never silently replace motion geometry.
                    entry=dict(edge_id=edge['edge_id'],image=row['before'],template=t,
                               outcome=row['after'],source_experience=str((self.folder/'experience.json').resolve()))
                    path=self.registry/(edge['edge_id']+'_'+digest(entry)[:16]+'.json')
                    if not path.exists():frozen(path,entry)
                    self.requests.grounding_reference_overrides[edge['edge_id']]=t
                    row['grounding_reference']=str(path.resolve())
            elif effect['action_relation']=='DIFFERENT_ROLE':
                nid='AP_'+str(row['id']).zfill(4)
                n=dict(id=nid,label=effect['decision_condition'],status='PROVISIONAL',parent_node=edge['source_node'],
                    representative_observation=row['before'],observation_count=1,decision_evidence=effect['evidence'])
                a=dict(id=nid+'_action',source=nid,targets=[node] if node else [],target=node,preferred=False,
                    exact_command=row.get('command'),outcome=row['progress_evaluation']['progress'],replayable=False)
                self.nodes.append(n);self.actions.append(a)
                row.update(status='PROVISIONAL_SUBSTATE',provisional_node=nid)
                self.live.update(self.graph,extra_nodes=self.nodes,extra_actions=self.actions)
            else:row['status']='UNRESOLVED'
        except Exception as exc:
            row.update(status='UNRESOLVED',audit_error=str(exc))
        self.active=None;self.snapshot()

    def invalidate_match(self,edge_id,conflict):
        # Retain the evidence, but withdraw a reference whose success match is now uncertain.
        if not self.rows or self.rows[-1]['edge_id']!=edge_id:return
        row=self.rows[-1]
        from .node_match_progress import node_match_progress
        row['match_conflict']=conflict
        row['progress_evaluation']=node_match_progress(row['source_node'],row.get('expected_target'),row.get('result_node'),conflict=True)
        row['status']='UNRESOLVED_MATCH_CONFLICT'
        path=row.get('grounding_reference')
        if path:
            save(Path(str(path)+'.invalidated.json'),dict(reason='MATCH_CONFLICT',evidence=conflict))
            self.requests.grounding_reference_overrides.pop(edge_id,None)
        self.snapshot()
