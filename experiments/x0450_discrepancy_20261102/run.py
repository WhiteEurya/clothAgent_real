"""Independent X0450-only analysis. Source bank and production files are read-only."""
import concurrent.futures,importlib.util,json,time,hashlib
from pathlib import Path
ROOT=Path('/mnt/newssd/sja/clothAgent_real');SRC=ROOT/'reports/intention_outcome_hypothesis_bank_v1';OUT=ROOT/'reports/x0450_discrepancy_20261102'
spec=importlib.util.spec_from_file_location('isolated_call',ROOT/'experiments/hypothesis_guided_prediction_v1/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT
obj,S,arr,enum=m.obj,m.S,m.arr,m.enum
INTENT=obj({'intended_regions':arr(S),'intended_changes':arr(S),'original_quotes':arr(S),
 'image_right_sleeve_expectation':enum('EXPLICIT_CHANGE','EXPLICIT_UNCHANGED','UNSPECIFIED','INDETERMINATE'),
 'right_sleeve_basis':S,'semantic_side_limits':S,'unsupported_inferences':arr(S)})
OBS=obj({'view_comparability':S,'whole_garment_changes':arr(S),'image_right_sleeve_before':S,'image_right_sleeve_after':S,
 'confirmed_visible_changes':arr(S),'provisional_changes':arr(S),'unconfirmed_changes':arr(S),
 'right_sleeve_corner_reveal':enum('CONFIRMED','PROVISIONAL','NO_CONFIRMED_REVEAL','UNKNOWN'),
 'right_sleeve_reveal_evidence':S,'alternative_visual_explanations':arr(S),'material_identity_and_side_limits':S})
FINAL=obj({'original_intention_summary':S,'observed_outcome_summary':S,
 'expectation_gap':enum('UNSPECIFIED_CHANGE','UNEXPECTED_CHANGE','EXPECTED_OBSERVED','INDETERMINATE'),
 'gap_evidence':S,'intended_goal_status':S,
 'hypothesis_opportunity':S,
 'competing_explanations':arr(obj({'label':S,'mechanism':S,'observational_basis':S,'testable_prediction':S,'falsification_condition':S,'unobserved_assumptions':S,'relationship_to_existing_bank':S})),
 'existing_hypothesis_audit':arr(obj({'hypothesis_id':S,'what_it_already_explains':S,'what_it_does_not_establish':S,'novelty_limit':S})),
 'existing_observation_vs_new_review':S,
 'worth_followup':enum('YES_PROVISIONALLY','NO','INSUFFICIENT'),
 'new_discovery_status':enum('ALREADY_RECORDED_EVENT','REFINEMENT_OF_EXISTING_EVENT','NEW_PROVISIONAL_EVENT','NO_VALID_EVENT'),
 'confirmed_corner_discovery':{'type':'boolean'},'final_answer':S,'limits':arr(S)})
BASE='你是独立Codex分析会话，仅使用本请求文字与图像，不调用任何工具，不读其他文件或旧会话。中文输出。只分析当前案例，不执行机器人、不写Skill、不更新假设库。'
IP=BASE+'''你只得到原始dispatch_intent.json。严格从原文提取执行前的意图，不知道Before/After或实际效果。判断是否明确预期画面右袖变化。原文未提及不等于预测不变，意图语义认知不等于实际材料身份。original_quotes必须逐字引用。'''
OP=BASE+'''图1为原始Before，图2原始After。没有原始意图、预测、历史模型评价或假设。独立观察整件衣服及画面右袖发生了什么。区分可直接确认的形状/可见范围变化、暂定变化及无法确认内容。普通袖面更清楚、袖口边缘更连续不等于同一语义袖口中点从自遮挡中显露。允许判断Before已可见，不能被Corner Reveal研究目标诱导。根据轮廓、缝线、印花、邻接关系给证据；同色布层无法辨认时保留不确定。不要构造动作规则；只列竞争性的视觉解释，不假装有过程帧。'''
FP=BASE+'''现在比较独立提取的原始意图与独立视觉观察；附原图供核对。没有此前离线预测或其评价结论。给出Intention–Outcome Discrepancy：画面右袖变化是否原意图未提及的任务相关效果？不能把未提及当作明确预测错误。再审计此案例已有Hypothesis Bank及已存观察，明确什么已经记录、什么只是更细化解释。
如果有操作关系值得研究，可保留至少两个竞争解释与反证条件；它们只是本报告中的待验证分析，不是新建假设库条目，不是Skill。优先解释原意图没有表述的关系；不得把已有H001/H002重写后计作新发现，也不得把外露面积/可辨性改善变成确认的Corner Reveal。新视觉审核与已有观察不同要原样并列，不用新结论覆盖旧记录。正确区分“相对于原始意图的新关系”和“相对于已有Bank的新发现”。最终明确回答：是否值得继续Hindsight研究？是否能计为新事件？是否为已确认的Corner Reveal？这三个问题分别回答。材料连接、受力、接触、翻转和覆盖层序未经观察不能当成事实。'''

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 idx=m.load(SRC/'experience_index.json');row=next(x for x in idx if x['experience_id']=='X0450');raw=Path(row['source']);dispatch=m.load(raw/'dispatch_intent.json');payload=m.load(row['payload'])
 bank=[h for h in m.load(SRC/'hypothesis_bank.json') if h['source_experience_id']=='X0450']
 oldobs=next(o for o in m.load(SRC/'experience_observations.json') if o['experience_id']=='X0450')['observation']
 sources=[SRC/'hypothesis_bank.json',SRC/'experience_observations.json',Path(row['payload']),raw/'dispatch_intent.json',raw/'execution.json',raw/'native_plan.json',Path(row['before']),Path(row['after'])]
 hashes={str(p):m.sha(p) for p in sources}
 assert m.sha(row['before'])==dispatch['observation_sha256']==row['before_sha256']
 assert m.sha(row['after'])==row['after_sha256']
 for p,h in row['raw_hashes'].items():assert m.sha(p)==h
 m.save(OUT/'source_record.json',row);m.save(OUT/'original_dispatch.json',dispatch);m.save(OUT/'original_action.json',{'indexed_action':payload['indexed_executed_action'],'native_command_sequence':payload['native_command_sequence'],'execution_information':payload['execution_information'],'intention_provenance':payload['intention_provenance']})
 m.save(OUT/'existing_hypotheses.json',bank);m.save(OUT/'existing_observation.json',oldobs)
 if not (OUT/'protocol.json').exists():m.save(OUT/'protocol.json',{'created_at':time.time(),'scope':'X0450 only','source_hashes':hashes,'runner_sha256':m.sha(__file__),'call_helper_sha256':m.sha(ROOT/'experiments/hypothesis_guided_prediction_v1/run.py'),'model':'gpt-6-astra','effort':'medium','profile':None,'isolation':'Intention extractor sees only original dispatch; observer sees only Before/After; synthesis sees their frozen outputs + original action + existing X0450 bank/observations, never previous prediction experiment or E353/human conclusions. Coordinator has prior context; analysis requests do not.','limits':'No independent timestamp certification. Old dispatch is producer-recorded pre-action intent; not reconstructed from After. Existing bank hashes checked, no writes.'})
 else:assert m.load(OUT/'protocol.json')['runner_sha256']==m.sha(__file__)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fi=pool.submit(m.call,'intent',IP,{'original_dispatch':dispatch},INTENT,[])
  fo=pool.submit(m.call,'observation',OP,{},OBS,[row['before'],row['after']])
  intent=fi.result();obs=fo.result()
 quotes=list(dispatch['plan'].values())
 for q in intent['original_quotes']:assert any(isinstance(v,str) and q in v for v in quotes)
 m.save(OUT/'independent_stages_frozen.json',{'frozen_at':time.time(),'intent':intent,'observation':obs,'intent_output_sha256':m.sha(Path(m.load(OUT/'calls/intent/result.json')['attempt_dir'])/'output.json'),'observation_output_sha256':m.sha(Path(m.load(OUT/'calls/observation/result.json')['attempt_dir'])/'output.json')})
 final=m.call('discrepancy_and_bank_audit',FP,{'original_dispatch':dispatch,'intention_extraction':intent,'independent_observation':obs,'actual_action_record':m.load(OUT/'original_action.json'),'existing_case_hypotheses':bank,'existing_case_observation':oldobs},FINAL,[row['before'],row['after']])
 assert {x['hypothesis_id'] for x in final['existing_hypothesis_audit']}=={x['hypothesis_id'] for x in bank}
 for p,h in hashes.items():assert m.sha(p)==h
 m.save(OUT/'result.json',{'source':row,'intent':intent,'observation':obs,'analysis':final,'existing_hypotheses':bank,'source_hashes_unchanged':True,'finished_at':time.time()});print('COMPLETE',final['new_discovery_status'],flush=True)
if __name__=='__main__':main()
