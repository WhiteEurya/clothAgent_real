"""Bounded one-step arm7 free planning plus append-only observation experience.

Recovery edges are observed experience, not automatically validated reusable policies.
The original preferred edges and their validation barriers are never changed.
"""
import copy
import json
from pathlib import Path
import jsonschema
from .discovery_data import digest, file_hash, read, save
from .graph_policy import checked_file, command, evidence_file, execute_recorded, frozen, load_frozen, planner_metrics
from .graph_policy_schema import NODE

DESCRIBE_PROMPT = '''Describe this unmatched CURRENT RGB as one candidate observation node for the
supplied task. No action is supplied: use visible cloth configuration, not a proposed strategy,
absolute pixel coordinates or presumed outcomes. Preserve the given node_id and observation_id.
Record uncertainty. This is one observed example, not a universal or validated state class.
Do not modify or merge any existing node. Return the NODE schema only.'''


class Recovery:
    def __init__(self, graph_path, graph, trajectory, output, requests):
        self.base_path = Path(graph_path).resolve()
        self.base = copy.deepcopy(graph)
        self.graph = copy.deepcopy(graph)
        self.trajectory = copy.deepcopy(trajectory)
        self.output = Path(output)
        self.requests = requests
        self.observations = {}
        self.nodes = []
        self.edges = []
        self.version = 0
        self.pending = None
        self.latest_path = None

    def observe_node(self, observation, folder):
        index = len(self.nodes)
        prefix = digest(str(self.output.resolve()))[:10]
        node_id = f'R_{prefix}_{index:03d}'
        obs_id = f'recovery_{prefix}_{index:03d}'
        def validate(node):
            jsonschema.validate(node, NODE)
            if node['node_id'] != node_id or node['representative_observations'] != [obs_id]:
                raise ValueError('new observation node IDs must match supplied IDs')
            if not node['observation_condition']['positive_evidence']:
                raise ValueError('new observed node requires visual evidence')
        node = self.requests.ask('recovery_node', dict(task=self.graph['goal'],
            node_id=node_id, observation_id=obs_id), [(obs_id, observation.image)], validator=validate)
        validate(node)
        save(Path(folder)/'new_observation_node.json', node)
        record = dict(rgb=evidence_file(observation.image))
        self.observations[obs_id] = record
        self.trajectory['observations'][obs_id] = record
        self.nodes.append(node)
        self.graph['nodes'].append(node)
        self.graph['match_nodes'].append(copy.deepcopy(node))
        self.snapshot()
        return node_id

    def transition(self, source_node, target_node, before, after, *, source, outcome, mode, edge_id=None):
        row = dict(edge_id=f"X_{digest(str(self.output.resolve()))[:10]}_{len(self.edges):03d}",
            source_node=source_node, target_node=target_node,
            mode=mode, reused_edge_id=edge_id,
            before=evidence_file(before), after=evidence_file(after),
            commanded_action=command(source), actual_execution=outcome,
            replayable=False, evidence_status='OBSERVED_TRANSITION_ONLY',
            limitation='Controller execution and a matched after image do not prove task progress or generalization.')
        self.edges.append(row)
        self.snapshot()

    def snapshot(self):
        self.version += 1
        graph = copy.deepcopy(self.graph)
        graph['recovery_extension'] = dict(base_graph=str(self.base_path), base_sha256=file_hash(self.base_path),
            nodes=self.nodes, observations=self.observations, transitions=self.edges,
            version=self.version, recovery_edges_replayable=False)
        self.latest_path = self.output / f'policy_extended_{self.version:03d}.json'
        frozen(self.latest_path, graph)
        # Separate experience transitions render alongside original policy edges, never replace them.
        lines=['# Recovery observation graph', '', '```mermaid', 'flowchart LR']
        aliases={n['node_id']:f'n{i}' for i,n in enumerate(graph['nodes'])}
        for n in graph['nodes']:
            lines.append(f'  {aliases[n["node_id"]]}["{n["node_id"]}"]')
        for e in graph['edges']:
            lines.append(f'  {aliases[e["source_node"]]} -->|{e["edge_id"]}| {aliases[e["target_node"]]}')
        for e in self.edges:
            lines.append(f'  {aliases[e["source_node"]]} -.->|{e["mode"]}| {aliases[e["target_node"]]}')
        lines += ['```', '', 'Dashed transitions are observed recovery experience, not preferred reusable actions.',
                  'New-node visual conditions and exact transition evidence are in the paired JSON.']
        self.latest_path.with_suffix('.md').write_text('\n'.join(lines)+'\n')
        return self.latest_path


def validate_extension(graph, validate_base):
    ext = graph['recovery_extension']
    base = load_frozen(ext['base_graph'])
    if file_hash(ext['base_graph']) != ext['base_sha256']:
        raise ValueError('recovery base hash mismatch')
    trajectory = copy.deepcopy(validate_base(base))
    if not ext['nodes'] and not ext['transitions']:
        raise ValueError('extension must contain observed nodes or transitions')
    expected=copy.deepcopy(base)
    expected['nodes'] += ext['nodes']
    expected['match_nodes'] += ext['nodes']
    expected['recovery_extension']=ext
    if graph != expected:
        raise ValueError('recovery extension altered original policy or edges')
    ids={n['node_id'] for n in base['nodes']}
    obs_ids=set(trajectory['observations'])
    if obs_ids & set(ext['observations']):
        raise ValueError('recovery observation ID collision')
    refs=[]
    for node in ext['nodes']:
        jsonschema.validate(node,NODE)
        if node['node_id'] in ids or not node['observation_condition']['positive_evidence']:
            raise ValueError('invalid recovery node')
        ids.add(node['node_id'])
        refs += node['representative_observations']
    if set(refs) != set(ext['observations']) or len(refs)!=len(set(refs)):
        raise ValueError('recovery observation coverage mismatch')
    for record in ext['observations'].values(): checked_file(record['rgb'])
    for edge in ext['transitions']:
        if edge['source_node'] not in ids or edge['target_node'] not in ids or edge['replayable'] is not False:
            raise ValueError('invalid audit-only recovery transition')
        checked_file(edge['before']);checked_file(edge['after'])
        if command(edge['commanded_action']['source']) != edge['commanded_action']:
            raise ValueError('recovery command mismatch')
    trajectory['observations'].update(ext['observations'])
    return trajectory


def free_plan_one(runtime, observation, folder, graph, *, timeout_s=900, model_call=None, budget=None, context=None):
    """Reuse the actual arm7 loop once, current observation, and its normal motion contract."""
    from cloth_agent.codex_loop import loop
    from cloth_agent.codex_astra_pipeline import run
    folder=Path(folder)
    record={}
    class OneObservation:
        def __getattr__(self,name):return getattr(runtime,name)
        def observe(self,directory):return observation
        def close(self):pass  # owner controls hardware lifecycle
        def execute(self,source,before,directory):
            if context is not None and context.get('reason')=='NO_ROBUST_ACTION_RECOVERY':
                from cloth_agent.codex_motion import parse_code
                calls=parse_code(source,allow_grasp=True)
                # A single acquisition plus held transport/release, never a multi-action program.
                if calls[0][0]!='grasp':
                    record['source']=source
                    record['outcome']=dict(status='PLANNING_FAILED',physical_execution=False,
                        failure_kind='INVALID_RECOVERY_ACTION',error='Recovery requires one grasp followed by held transport')
                    save(Path(directory)/'recovery_rejection.json',record['outcome'])
                    return record['outcome']
            if context is not None and context.get('reason','').startswith('REPAIR_EXISTING_EDGE_AFTER_'):
                previous=context['failure']['transition'].get('command_source')
                if previous and command(source)['operations']==command(previous)['operations']:
                    record['source']=source
                    record['outcome']=dict(status='PLANNING_FAILED',physical_execution=False,
                        failure_kind='UNCHANGED_REPAIR_COMMAND',error='Repair repeated the rejected/ineffective command unchanged')
                    save(Path(directory)/'repair_rejection.json',record['outcome'])
                    return record['outcome']
            collecting=context is not None and context.get('reason')=='COLLECT_ONLY_FULL_PLANNER'
            save(Path(directory)/'dispatch_intent.json',dict(event='COLLECT_FULL_PLAN' if collecting else 'FALLBACK_FREE_PLAN',
                before=evidence_file(before.image),command=command(source)))
            record['source']=source
            record['outcome']=execute_recorded(runtime,source,before,directory)
            return record['outcome']
    def call(*args,**kwargs):
        if budget is not None and budget.model_calls >= budget.max_calls:
            raise RuntimeError('shared model-call budget exhausted before fallback')
        if budget is not None: budget.model_calls += 1
        return (model_call or run)(*args,**kwargs)
    references=[(checked_file(x['artifact']),x['role']) for x in graph.get('goal_references',[])]
    task=('当前图策略未找到可用动作或执行后没有节点级进展。按正常arm7方式自由规划一个小步骤完成折叠目标；'
          '这一次不受某条图edge限制，不执行机器人，仅返回动作代码。'
          '仅当当前画面已经达到完整任务终态时才输出motion.complete()。\n'+str(graph['goal']))
    if context is not None and context.get('reason')=='COLLECT_ONLY_FULL_PLANNER':
        task=('根据当前 RGB-D、任务目标和任务参考图，独立规划本轮一个完整的布料操作。'
              '本轮采用完整 GPT planner 自主选择动作；Graph 仅记录经验，不提供选边、动作模板或目标节点。'
              '这不是恢复或已有动作修正请求。沿用当前 robustness policy、motion schema 和 Host 安全检查。'
              '只输出动作代码；只有当前观测已经达到完整任务终态时才输出 motion.complete()。\n'+str(graph['goal']))
    if context is not None and context.get('reason') in {'REPAIR_EXISTING_EDGE_AFTER_SELF_LOOP','REPAIR_EXISTING_EDGE_AFTER_IK_FAILED'}:
        task=('当前已知节点上的指定edge需要修正，具体原因见结构化failure（可能为执行前IK失败或执行后self-loop）。只修正这条edge：保持原action family、'
              'manipulation role与目标状态；根据当前RGB与失败证据重新规划一个动作。'
              '不要改做其他动作族，不要原样重复失败命令。动作必须通过现有Host验证。\n'+str(graph['goal']))
    if context is not None and context.get('reason')=='REPLAY_REVISED_EDGE':
        task=('Graph 已选择下方 selected_edge。根据当前 RGB-D 实例化这条已修正的 edge，保持其动作目标和已验证的抓取/搬运策略。'
              '历史 XYZ 仅作经验，按当前观察重新定位；自主选择实际 TCP 高度，不强制回到布面 Z。'
              '只输出本轮一个 grasp 和必要持布动作，沿用现有 Host 安全检查。\n'+str(graph['goal']))
    if context is not None and context.get('reason')=='NO_ROBUST_ACTION_RECOVERY':
        task=('Graph 推荐的 skill/edge 的 grasp gate 已拒绝，目标与原因见下方 context。'
              '本轮不是强行完成该 fold，也不是追求最大折叠进展；只选择一个最小、低风险 recovery manipulation，'
              '使下一次观测中推荐 skill 更容易获得清晰、可靠的 grasp handle。'
              '使用当前 RGB-D 和 observation，恢复抓点自身也遵循当前 robustness 拒绝条件；'
              '可合理带动目标区域的多层抓取允许提出动作，不能仅因局部重叠或层数不明而拒绝。'
              '只允许一次 grasp 加必要持布搬运及稳定放置，使用原 motion schema；不输出第二次抓取或连续恢复。'
              '沿用 workspace、IK、collision 和所有 Host safety 检查，不增加接口或绕过检查。'
              '若不存在可靠恢复，返回 motion.blocked(reason="NO_ROBUST_ACTION: 无可靠 recovery")；'
              '不要用 complete 代替恢复。宿主在本次恢复后重新观测、匹配已有节点；'
              '不得假定仍属原 state、指定恢复后 node 或修改 Graph。\n'+str(graph['goal']))
    if context is not None and context.get('reason')=='REPLAY_REVISED_EDGE':
        task+='\n所选 edge：'+json.dumps(context,ensure_ascii=False)
    elif context is not None and context.get('reason')!='COLLECT_ONLY_FULL_PLANNER':
        task += ('\n上一动作与最新匹配反馈：'+json.dumps(context,ensure_ascii=False)+
                 '\n请结合当前新观测重新规划修正，不原样重复上一失败方案。状态不变不等于抓空；'
                 '不得据此自动加深抓取或绕过Host安全检查。')
    result=loop(task,OneObservation(),folder,max_rounds=1,model_call=call,
                timeout_s=timeout_s,references=references)
    stats=planner_metrics(folder)
    if budget is not None:
        budget.elapsed += stats['reasoning_time_s']
        budget.tool_calls += stats['tool_calls']
        budget.visual_tool_calls += stats['visual_tool_calls']
        budget.completed_turns += stats['completed_model_turns']
    outcome=record.get('outcome') or read(folder/'round_001/outcome.json')
    source=record.get('source')
    if context is not None and context.get('reason')=='NO_ROBUST_ACTION_RECOVERY' and source is None:
        path=folder/'round_001/motion.py'
        if path.exists():source=path.read_text()
    return dict(result=result,outcome=outcome,source=source,metrics=stats)
