"""Isolated native Codex inference; no configured model API and no robot tools."""
from pathlib import Path
import argparse, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/rsi_code_iteration_01_20261029'
BASE='''You are an independent experimental researcher for a single-view RGB deformable-cloth task. Discover and revise an executable policy from supplied real evidence. No human-designed manipulation method, old skill, planner rationale, previous experiment conclusion or parent conversation is supplied. Do not infer a prescribed answer. Your tools are disabled; reason from attached images and the explicit packet. Output only the required JSON. Be candid about uncertainty and do not fabricate physical outcomes.'''
COMMON='''任务：仅研究当前可见布料区域的局部折叠/皱褶，不处理完全隐藏材料。基本动作 grasp -> lift -> pull -> release。自主提出可证伪假设，自行选择可测量关系和算法；不要记忆图像坐标或固定世界方向。当前只做一次历史回放，不执行机器人。
评价已经冻结：现有紫色衣物分割、可见面积相对变化和刚体配准；面积不是布面面积/真实平整度。UNCALIBRATED 不等于已证实成功，UNKNOWN 不等于失败，配准失败不能编造局部运动。不同初始状态不是严格控制变量实验。机械命令完成不证明抓取成功。
请生成独立可执行 Python 模块，真实实现 policy(observation) 和 predict_delta(observation, recorded_action)。observation 只有 rgb:uint8 RGB HxWx3 与 mask:bool HxW（从 RGB 分割）；不能读取路径、ID、历史答案或硬编码示例像素。只可 import numpy, cv2, math，不得文件/网络/设备访问。无需实现 segmentation。
policy 输出字典 decision=ACTION/ABSTAIN/DONE、grasp_px=[整数x,y]、target_px=[整数x,y]、motion_type=lift_pull_place、grasp_angle_deg=None、reason=字符串；非 ACTION 坐标/motion_type 为 None。Pa 必须在 mask 中；Pb 必须在图内；不要把目标图像素复用为抓点。自行定义图像尺度下方向和距离，实际毫米与安全由既有 Host 核验（10–150mm），RGB 不应臆造标定。策略必须仅依赖本次输入特征，支持不同输入尺寸，能在新状态重新计算动作。函数 predict_delta 对任意给定 recorded_action（grasp_px,target_px）预测冻结可见面积 delta_ratio，返回有限 float 或 None 表示不知道。无需声称预测准确。
不得通过更改 evaluator 提升指标。必须保留不确定性，并把“行为对旋转/平移的一致性”与“真实操作成功”分开。新动作没有 after，不能拿附近历史动作的 after 当它的结果。代码限约180行，完整自包含，不用 markdown 代码块包装。'''
def obj(props):return {'type':'object','additionalProperties':False,'properties':props,'required':list(props)}
S={'type':'string'}
def setup(stage):
    folder=OUT/f'stage{stage}'
    if (folder/'invocation.json').exists():
        raise RuntimeError('This stage was already invoked; preserve the single-iteration record.')
    folder.mkdir(parents=True,exist_ok=True)
    if stage==1:
        extra='''本阶段只看到3条 seed 的真实 Before/After/已执行动作/固定反馈，以及3条 pool 的 Before 与真实已执行动作；pool 结果尚未揭示，heldout 未提供。先筛查所选实验能否研究可见布料，不支持则坦诚说明。自主提出至少两个竞争假设，对每个 pool 动作给出各假设的可检验预测（用面积变化数值/区间与可见状态描述）。从 P1/P2/P3 只选一条最能区分假设的历史操作，不只选最可能成功者。记录信息价值、选择理由、局限。生成尚未看到所选反馈时的初始 policy_v0 代码。不可要求重置、不能把这些不同状态当严格配对对照。context_audit 说明实际收到的上下文有无预设抓点/方向算法。'''
        schema=obj({'context_audit':S,'hypotheses':{'type':'array','minItems':2,'maxItems':3,'items':obj({'id':S,'mechanism':S,'measurable_features':S,'seed_evidence':S,'falsification':S})},'selected_replay':{'type':'string','enum':['P1','P2','P3']},'visible_scope_check':S,'predictions':S,'selection_reason':S,'information_gain_reason':S,'policy_design':S,'policy_code':S,'limitations':S})
    else:
        extra='''本阶段提供前一阶段已冻结的完整假设/预测/代码，和你自己选择的一条历史操作的真实 Before/After 与固定评价。没有其他 pool 结果或 heldout。请对预先预测逐项判断支持/反驳/未知，用真实反馈修正信念；一次观察不能建立普遍因果关系。仅做一次策略代码更新，写出完整 policy_v1，必须是计算逻辑的实际变更，不能只改 reason/prompt。指出更改位置、为何由新证据支持，以及仍只是猜测的部分。不得把不可观察量伪装测量。提出一个最有区分力的下一实验：具体可重复状态/动作对照、不同假设的不同预测、如何避免状态混淆；本轮不执行它。保留相同代码接口。'''
        schema=obj({'context_audit':S,'selected_feedback_interpretation':S,'supported':S,'refuted':S,'unknown':S,'code_changes':S,'policy_code':S,'next_experiment':S,'generalization_limits':S})
    (folder/'base.txt').write_text(BASE);(folder/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2))
    packet=json.loads((OUT/f'stage{stage}_packet.json').read_text())
    prompt=COMMON+'\n'+extra+'\n冻结数据包：\n'+json.dumps(packet,ensure_ascii=False)
    (folder/'prompt.txt').write_text(prompt)
    return folder,prompt
def call(stage):
    folder,prompt=setup(stage)
    cmd=['/home/sja/.local/bin/codex','exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','-s','read-only','-C','/tmp/oneshot_native_offline_empty','-m','gpt-6-astra','-c','model_provider="openai"','-c','forced_login_method="chatgpt"','-c','model_reasoning_effort="high"','-c','project_doc_max_bytes=0','-c','web_search="disabled"','-c','developer_instructions=""','-c','model_instructions_file='+json.dumps(str(folder/'base.txt')),'--enable','skip_host_skill_discovery']
    for f in ['apps','plugins','multi_agent','shell_tool','unified_exec','memories','skill_search','image_generation','view_image','sleep_tool','browser_use','computer_use']:cmd+=['--disable',f]
    cmd+=['--json','--output-schema',str(folder/'schema.json'),'-o',str(folder/'answer.json')]
    for img in json.loads((OUT/f'stage{stage}_images.json').read_text()):cmd+=['-i',img]
    cmd+=['--','-'];env=os.environ.copy()
    for k in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_ORG_ID','OPENAI_PROJECT_ID']:env.pop(k,None)
    (folder/'invocation.json').write_text(json.dumps({'argv':cmd,'parent_history_supplied':False,'stage':stage},ensure_ascii=False,indent=2))
    start=time.time()
    with (folder/'events.jsonl').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
        result=subprocess.run(cmd,input=prompt,text=True,stdout=stdout,stderr=stderr,env=env,timeout=900)
    info={'exit_code':result.returncode,'elapsed_s':time.time()-start};(folder/'status.json').write_text(json.dumps(info));print(json.dumps(info),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int,choices=[1,2]);a=p.parse_args();call(a.stage)
