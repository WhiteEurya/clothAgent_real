"""Fresh native Codex context, read-only and no tools; hypotheses only."""
from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/offline_failure_mining_20261029';D=OUT/'native_analysis'
def obj(p):return {'type':'object','additionalProperties':False,'properties':p,'required':list(p)}
S={'type':'string'}
def main():
 D.mkdir(exist_ok=True)
 if (D/'invocation.json').exists():raise RuntimeError('Preserve the existing independent analysis; no automatic retry.')
 base='You are an independent experimental scientist analyzing real cloth-manipulation evidence. Parent conversation, historical skills and planner explanations are not supplied. Tools are disabled. Discover a research question from the raw evidence. Do not assume a prescribed geometric rule or invent feedback. Output the required JSON in Chinese. No policy code and no execution.'
 prompt='''从附件历史序列自主选择一个最值得研究的重复困难，不能直接把所有未 DONE 或面积不增长判成失败。每张图左 before、右 after，上下多行是相邻步骤；坐标为数据包原图像素，不是拼图坐标。只把可见证据写成事实，层序/受力/抓取成功等保留未知。数据包给出所有初筛序列，不预选抓取规则；你需要比较它们，并审查检索对照是否真的可比。明确说明目标区域长期未改善、重复动作、局部形变与整体移动；引用匿名 run、step 和图像区域。至少提出两个能产生不同预测的机制。选择一个优先真实验证实验，具体指定参考状态、可识别抓取区域和方向/距离、唯一改变变量、预期不同观察、必要的可重复重置或历史对照限制、判不清的情况及现有安全约束。只有建议，不执行、不写策略。如果当前证据不足以确认失败机制，要明确说不足。不要将人工规则作为既定答案，也不要复述没有证据的成功故事。输出独立分析及其上下文审计。'''
 packet=json.loads((OUT/'analysis_packet.json').read_text());prompt+='\n原始记录与固定评价：\n'+json.dumps(packet,ensure_ascii=False)
 schema=obj({'context_audit':S,'chosen_problem':S,'why_this_problem':S,'repeated_observations':{'type':'array','items':obj({'case_steps':S,'visible_evidence':S,'action_pattern':S,'uncertain_structure':S})},'control_assessment':S,'competing_explanations':{'type':'array','minItems':2,'items':obj({'id':S,'mechanism':S,'support':S,'counterevidence_or_unknown':S,'distinct_prediction':S})},'priority_experiment':obj({'reference_state':S,'action':S,'changed_variable':S,'reset_and_comparison':S,'predictions_by_mechanism':S,'measurements':S,'ambiguity_and_safety':S}),'other_patterns':S,'limitations':S})
 for n,v in [('base.txt',base),('prompt.txt',prompt),('schema.json',json.dumps(schema,ensure_ascii=False,indent=2))]:(D/n).write_text(v)
 cmd=['/home/sja/.local/bin/codex','exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','-s','read-only','-C','/tmp/oneshot_native_offline_empty','-m','gpt-6-astra','-c','model_provider="openai"','-c','forced_login_method="chatgpt"','-c','model_reasoning_effort="high"','-c','project_doc_max_bytes=0','-c','web_search="disabled"','-c','developer_instructions=""','-c','model_instructions_file='+json.dumps(str(D/'base.txt')),'--enable','skip_host_skill_discovery']
 for f in ['apps','plugins','multi_agent','shell_tool','unified_exec','memories','skill_search','image_generation','view_image','sleep_tool','browser_use','computer_use']:cmd+=['--disable',f]
 cmd+=['--json','--output-schema',str(D/'schema.json'),'-o',str(D/'answer.json')]
 for image in json.loads((OUT/'analysis_images.json').read_text()):cmd+=['-i',image]
 cmd+=['--','-'];env=os.environ.copy()
 for k in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_ORG_ID','OPENAI_PROJECT_ID']:env.pop(k,None)
 (D/'invocation.json').write_text(json.dumps({'argv':cmd,'parent_history_supplied':False,'old_skills_or_rationales_supplied':False},ensure_ascii=False,indent=2));start=time.time()
 with (D/'events.jsonl').open('w') as out,(D/'stderr.log').open('w') as err:r=subprocess.run(cmd,input=prompt,text=True,stdout=out,stderr=err,env=env,timeout=900)
 info={'exit_code':r.returncode,'elapsed_s':time.time()-start};(D/'status.json').write_text(json.dumps(info));print(info)
 if r.returncode:raise SystemExit(r.returncode)
if __name__=='__main__':main()
