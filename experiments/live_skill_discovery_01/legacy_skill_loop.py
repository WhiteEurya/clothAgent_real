"""Live Skill Discovery 01. Nothing connects or moves without --execute."""
from pathlib import Path
import argparse,base64,copy,hashlib,json,sys,time,signal
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
MAX_ITERATIONS=6
MIN_PULL_MM=20.0
MAX_PULL_MM=40.0
BASE='''根据当前RGB，为T-shirt上一处可见局部卷折选择一次lift_pull_place操作，优先研究袖部。
说明原始目标区域、当前可见卷折及预期可见变化。Pa为抓点grasp_px，Pb为释放目标target_px，使用原图整数像素，左上为原点，x向右y向下。grasp_angle_deg必须为null，沿用Home夹爪方向。
没有明确局部卷折或无法选择动作则ABSTAIN，坐标和motion_type为null。只输出指定JSON。'''
TEACHER='''你是独立的离线Teacher。只依据本轮真实Before、已执行Action、After、原目标、当前Skill及自动视觉评价；自动评价可能错误，必须核对图像。
每轮重新判断是否有值得学习或修订的操作规律。明确改善时可归纳有效操作；效果变差时考虑不同方案，不继续强化造成退化的方案；没有可见运动可能是抓空、滑移或其他原因，不得仅凭前后图确定抓空，更不能直接当作布料机制失败。UNKNOWN不能当作成功或失败，不得用无法对应的旧目标证明新规则；仅有未知效果时允许NO_NEW_SKILL。未知项明确保留。
最多输出一条Skill，包含trigger、grasp_selection、direction_selection、expected_effect，合计尽量不超过150个汉字。Trigger只依赖当前观测，不引用固定坐标。说明图像证据、物理假设及相对原方案的区别；坐标或方向微调不能声称新机制。
只允许单次lift_pull_place接口，不增加动作阶段、硬件或控制算法。证据不足或没有可靠新规律时输出NO_NEW_SKILL；不要强行给出答案。只输出指定JSON。'''

S={'type':'string'}
def obj(fields):return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
def enum(*vs):return {'type':'string','enum':list(vs)}
POINT={'anyOf':[{'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2},{'type':'null'}]}
PLAN=obj({'decision':enum('ACTION','ABSTAIN'),'grasp_px':POINT,'target_px':POINT,'motion_type':{'type':['string','null'],'enum':['lift_pull_place',None]},'grasp_angle_deg':{'type':'null'},'target_region':S,'visible_fold':S,'expected_visible_change':S,'short_reason':S})
SK=obj({k:S for k in ['trigger','grasp_selection','direction_selection','expected_effect']})
TEACHER_SCHEMA=obj({'decision':enum('NEW_SKILL','NO_NEW_SKILL'),'observed_gap':S,'physical_hypothesis':S,'mechanism_difference':S,'skill':SK,'unknowns':S})
def save(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
class StopExperiment(RuntimeError):pass

def execution_authorization(folder,binding):
    # Session-level authorization; never fabricate a per-action human approval.
    save(folder/'execution_authorization.json',{'mode':'automatic',
        'source':'user requested no confirmations; execute after existing checks',
        'per_action_human_review':False,'binding':binding,'at':datetime.now().isoformat()})

def visual_review(folder,name,instructions,task,images,choices,binding):
    value=request(folder/(name+'_model'),instructions,task,images,
        obj({'choice':enum(*choices),'note':S,'unknowns':S}))
    if value['choice'] not in choices or not value['note'].strip():
        raise StopExperiment('视觉评价输出无效')
    result={**value,'source':'model_visual_assessment','human_verified':False,'binding':binding}
    save(folder/(name+'.json'),result)
    print(name+': '+result['choice']+' — '+result['note'],flush=True)
    return result

def review_mechanism(root,folder,before,plan,intent,base_plan):
    frozen=read(root/'skill_frozen.json')
    context=read(root/'skill_freeze.json')
    teacher_dir=root/context['source_round']/'teacher'
    teacher=read(teacher_dir/'parsed.json')
    # Use the actual recorded image path; filenames are runtime-owned.
    attachments=read(teacher_dir/'attachments.json')
    base_before=Path(attachments[0]['path'])
    return visual_review(folder,'mechanism_review',
        '比较两次操作的作用机制。第一张是原动作Before，第二张是当前Before。只依据可见结构、实际动作参数及待检验假设判断；模型声称不同不是证据。仅换方向、微调坐标或重复抓取为NOT_DISTINCT；无法从图像及动作区分为UNKNOWN。不要生成动作或Skill；不推断物理成功。',
        json.dumps({'base_plan':base_plan,'new_plan':plan,'new_intent':intent,'skill':frozen,
            'unverified_hypothesis':teacher['physical_hypothesis'],'claimed_difference':teacher['mechanism_difference']},ensure_ascii=False),
        [base_before,before.image],['DISTINCT','NOT_DISTINCT','UNKNOWN'],
        {'base_plan':digest(base_plan),'new_plan':digest(plan)})

def request(folder,instructions,task,images,schema,*,transport=None):
    from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
    from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
    if (folder/'attempt.json').exists():raise StopExperiment('请求已尝试；本实验不自动重试')
    content=[{'type':'input_text','text':task}]
    manifest=[]
    for p in images:
        content.append({'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode(),'detail':'high'})
        manifest.append({'path':str(p),'sha256':sha(p)})
    body={'model':'gpt-6-astra','instructions':instructions,'input':[{'role':'user','content':content}],'tools':[],'tool_choice':'none','parallel_tool_calls':False,'reasoning':{'effort':'high'},'text':{'format':{'type':'json_schema','name':'live_skill_discovery_01','strict':True,'schema':schema}},'store':False,'stream':True}
    # Audit the actual serialized request, including schema and attachments, before transport.
    save(folder/'audited_payload.json',body);save(folder/'attachments.json',manifest)
    save(folder/'context_audit.json',{'instructions_sha256':digest(instructions),'payload_sha256':digest(body),'tools':[],'history':False,'agent_cli':False,'project_skill_injection':False,'provider_internal_context':'not inspectable'})
    value,metrics=(transport if transport is not None else direct_once)(folder,body,wire.auth_headers(),600)
    if read(folder/'request_payload.json')!=body:raise StopExperiment('发送请求与审计请求不一致')
    save(folder/'parsed.json',value);return value

def planner(folder,before,skill=None,original_goal=None):
    from PIL import Image
    from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
    task=f'当前图像尺寸：{Image.open(before.image).size}。本实验Pa到Pb的桌面XY牵拉距离限定为{MIN_PULL_MM:g}–{MAX_PULL_MM:g}毫米（不是像素或抬升高度），执行前由标定深度核验。'
    if skill is not None:
        task+='\n仅本次验证使用的冻结Skill：'+json.dumps(skill,ensure_ascii=False)
        if original_goal is not None:task+='\n仍需改善的原目标（不改变目标）：'+json.dumps(original_goal,ensure_ascii=False)
    value=request(folder/'model',BASE,task,[before.image],PLAN)
    plan={k:value[k] for k in ['decision','grasp_px','target_px','motion_type','grasp_angle_deg','short_reason']}
    wire.validate_plan(plan,list(Image.open(before.image).size),require_grasp_rotation=True,require_single_sentence=False)
    save(folder/'plan.json',plan);save(folder/'intent.json',value)
    wire.overlay(before.image,plan,folder/'action_overlay.png',folder.name)
    return plan,value

def runtime_policy(runtime):
    return {'path_validation_enabled':runtime.config.path_validation_enabled,
        'collision_envelopes_disabled_arms':sorted(runtime.config.collision_envelopes_disabled_arms),
        'direct_joint_moves':getattr(runtime,'direct_joint_moves',False),
        'home_recover_stopped':getattr(runtime.config,'home_recover_stopped',False)}

def make_runtime(config):
    """Reuse the existing Arm6 full-episode runtime without configuration overrides."""
    from cloth_agent.experiments.trajectory_memory_full_episode_v1 import EpisodeCollectionRuntime
    runtime=EpisodeCollectionRuntime(config)
    runtime._live_skill_policy=runtime_policy(runtime)
    return runtime

def connect_with_diagnostics(runtime,root):
    from cloth_agent.dual_arm.execution import XArmConnection
    connections={};audit={'status':'CONNECTING','arms':[]}
    try:
        for key,config in runtime.config.arms.items():
            if key=='right':
                from cloth_agent.dual_arm.static_arm import StaticArmObstacle
                record=runtime.config.static_arm_obstacles['right']
                connections[key]=StaticArmObstacle(config,runtime.models[key],record)
                audit['arms'].append({'arm':key,'status':'STATIC_OBSTACLE_NOT_CONNECTED',
                    'live_feedback':False,'reference':record})
                save(root/'connection_audit.json',audit)
                continue
            if key!='left':raise StopExperiment('只允许连接Arm6/left')
            row={'arm':key,'ip':config.ip,'status':'CONNECTING'}
            audit['arms'].append(row);save(root/'connection_audit.json',audit)
            print(f'CONNECT {key} ({config.ip}): identity and live feedback; no motion',flush=True)
            try:
                connections[key]=XArmConnection(config,runtime.cancel,recover_stopped=getattr(runtime.config,'home_recover_stopped',False))
            except BaseException as exc:
                row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
                audit['status']='FAILED';save(root/'connection_audit.json',audit)
                if isinstance(exc,KeyboardInterrupt):raise
                raise StopExperiment(f'连接 {key} ({config.ip}) 失败：{type(exc).__name__}: {exc}；尚未reset或规划') from exc
            row['status']='CONNECTED';save(root/'connection_audit.json',audit)
        audit['status']='CONNECTED';save(root/'connection_audit.json',audit)
        return connections
    except BaseException:
        for connection in connections.values():
            try:connection.disconnect()
            except Exception:pass
        raise


def connect_stationary(runtime,root):
    import fcntl,numpy as np
    from cloth_agent.dual_arm.execution import DualArmCoordinator
    runtime._lease=open('/tmp/clothagent_codex_dual_arm.lock','w')
    fcntl.flock(runtime._lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    connections=connect_with_diagnostics(runtime,root)
    try:
        runtime.coordinator=DualArmCoordinator(runtime.config,runtime.models,connections,runtime.cancel,root/'controller',realtime=True)
    except BaseException:
        for connection in connections.values():
            try:connection.disconnect()
            except Exception:pass
        raise
    expected={k:np.asarray(v['joints']) for k,v in runtime.coordinator.snapshots().items()}
    runtime.arrival(expected) # Connection itself never moves; reset preparation is approved separately.

def safety_assert(runtime):
    if runtime_policy(runtime)!=runtime._live_skill_policy:
        raise StopExperiment('既有Arm6运行配置发生变化')
    if getattr(runtime,'auto_grasp_branch',None):raise StopExperiment('不允许额外自动恢复分支')

def check_pull_distance(folder):
    import math
    points=read(folder/'points.json')
    a,b=(p['world_xyz_mm'] for p in points)
    distance=math.hypot(b[0]-a[0],b[1]-a[1])
    passed=math.isfinite(distance) and MIN_PULL_MM<=distance<=MAX_PULL_MM
    record={'distance_xy_mm':distance if math.isfinite(distance) else None,
        'min_translation_mm':MIN_PULL_MM,'max_translation_mm':MAX_PULL_MM,
        'passed':passed,'scope':'live experiment cloth manipulation only; reset unchanged'}
    save(folder/'pull_distance_check.json',record)
    if not passed:raise StopExperiment(f'牵拉距离 {distance:.2f} mm 超出本实验 {MIN_PULL_MM:g}–{MAX_PULL_MM:g} mm 范围')
    profile=read(folder/'motion_profile.json')
    profile.update(min_translation_mm=MIN_PULL_MM,max_translation_mm=MAX_PULL_MM)
    save(folder/'motion_profile.json',profile)

def prepare(runtime,before,plan,folder):
    import numpy as np
    from cloth_agent.experiments.one_shot_goal_full_episode_v1 import prepare_action
    safety_assert(runtime)
    operations,initial=prepare_action(runtime,before,plan,folder)
    check_pull_distance(folder)
    # Prepare the existing return primitive now for action-specific approval.
    last_kind,last=operations[-1]
    final=last.end if last_kind=='move' else {k:np.asarray(v) for k,v in last['joints'].items()}
    owner,target=runtime.perception_target()
    old_open=getattr(runtime,'open_at_home_on_return',False)
    runtime.open_at_home_on_return=True
    try:_,source=runtime._preparation_program(final,owner,target)
    finally:runtime.open_at_home_on_return=old_open
    (folder/'return_motion.py').write_text(source)
    return_ops=runtime._plan_preparation(source,final)
    operations=operations+return_ops
    serialized=[{'kind':kind,'data':data.to_dict() if kind=='move' else data} for kind,data in operations]
    save(folder/'approved_native_plan.json',serialized)
    save(folder/'host_validation.json',{'status':'PASS','runtime_policy':runtime_policy(runtime),'full_path_validation':runtime.config.path_validation_enabled,'grasp_depth_spread_check':False,'return_path_included':True,'inactive_arm_live_feedback':False,'inactive_arm_geometry':'existing Arm6 runtime saved static pose','manipulation_commands':len(operations)-len(return_ops),'return_commands':len(return_ops)})
    return operations,initial,digest(serialized)

def show_review(folder,before,plan,intent):
    import html
    page='<meta charset="utf-8"><title>动作记录</title><h1>'+folder.name+'</h1><p>自动执行记录；保留具体计划和轨迹哈希，无逐动作人工确认。</p><pre>'+html.escape(json.dumps({'plan':plan,'intent':intent,'host':read(folder/'host_validation.json')},ensure_ascii=False,indent=2))+'</pre><img style="max-width:100%" src="action_overlay.png"><p>本轮包含已规划的释放、撤离及返回观察位；不包含下一次抓取。</p>'
    (folder/'review.html').write_text(page)
    print('动作记录：'+str(folder/'review.html'),flush=True)

def dispatch_once(runtime,root,state,folder,before,plan,intent,*,base_plan=None):
    from cloth_agent import codex_recovery
    from cloth_agent.preference_one_shot import verify_scene
    from cloth_agent.gamepad_collect import TrajectoryRecorder
    if state['dispatches']>=MAX_ITERATIONS:raise StopExperiment('六轮动作预算耗尽')
    try:
        operations,initial,native_hash=prepare(runtime,before,plan,folder)
    except BaseException as exc:
        save(folder/'host_validation.json',{'status':'REJECT','error':f'{type(exc).__name__}: {exc}'})
        raise
    show_review(folder,before,plan,intent)
    if base_plan is not None:
        # Mechanism comparison is an observation, not a safety gate or proof of novelty.
        review_mechanism(root,folder,before,plan,intent,base_plan)
    binding={'before_sha256':sha(before.image),'plan_sha256':digest(plan),'native_sha256':native_hash}
    execution_authorization(folder,binding)
    # Model latency cannot silently invalidate the observed scene.
    fresh=runtime.capture_current(folder/'pre_dispatch')
    verify_scene(before,fresh,plan,folder,automatic=False)
    if digest(read(folder/'plan.json'))!=binding['plan_sha256'] or sha(before.image)!=binding['before_sha256']:raise StopExperiment('规划后输入被修改')
    if digest(read(folder/'approved_native_plan.json'))!=native_hash:raise StopExperiment('规划后轨迹文件被修改')
    codex_recovery.check_stationary(runtime,before.meta['joints_deg'],folder/'stationary_check.json')
    runtime.arrival(initial);safety_assert(runtime)
    recorder=TrajectoryRecorder(runtime,folder)
    try:
        recorder.start();recorder.stage('automatic_single_manipulation_and_return');recorder.check()
        state['dispatches']+=1;save(root/'status.json',state)
        save(folder/'dispatch_intent.json',dict(binding,physical_execution=True,includes_planned_return=True))
        try:
            result=runtime._execute_operations(operations,folder)
        except BaseException as exc:
            save(folder/'execution.json',{'status':'EXECUTION_EXCEPTION','physical_execution':True,'error':f'{type(exc).__name__}: {exc}'})
            raise
        save(folder/'execution.json',result)
        if result['status']!='EXECUTED':raise StopExperiment('执行异常；不得当作布料失败')
        recorder.check()
    finally:recorder.close()
    recorder.check()
    after=runtime.capture_current(folder/'after')
    return after

def observe_outcome(folder,before,after,intent):
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(1280,390),'white');d=ImageDraw.Draw(canvas)
    for i,(label,p) in enumerate([('Before',before.image),('After',after.image)]):
        im=Image.open(p).convert('RGB');im.thumbnail((630,355));canvas.paste(im,(i*640,30));d.text((i*640+5,5),label,fill='black')
    canvas.save(folder/'outcome_comparison.png')
    print('目标：'+intent['target_region']+'；预期：'+intent['expected_visible_change']+'\n查看：'+str(folder/'outcome_comparison.png'),flush=True)
    return visual_review(folder,'outcome_review',
        '只评价原目标卷折的可见变化。第一张Before，第二张After。明确折叠或遮挡减少才选IMPROVED；目标清晰可比且无明确改善选NO_CLEAR_IMPROVEMENT；目标折叠、遮挡或卷曲明确增加选WORSENED；衣物整体和目标均无可见运动选NO_VISIBLE_CHANGE（只能说可能抓空或滑移，不能确认原因）；遮挡、视角变化、整体移动或证据不足无法判断时选UNKNOWN。不得把整体平移旋转当作展开，不推断传力机制，不生成动作或Skill。说明具体图像证据与未知项。',
        json.dumps({'original_goal':intent},ensure_ascii=False),[before.image,after.image],
        ['IMPROVED','NO_CLEAR_IMPROVEMENT','WORSENED','NO_VISIBLE_CHANGE','UNKNOWN'],
        {'before':sha(before.image),'after':sha(after.image),'intent':digest(intent)})


def startup_reset(runtime,root,state):
    """Reuse normal initialize(); record every reset/positioning dispatch."""
    import numpy as np
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
    from cloth_agent.dual_arm.observation import Observation
    from cloth_agent.preference_one_shot import verify_scene
    folder=root/'initialization'
    original=runtime._execute_operations
    original_request=episode.direct_once
    def reset_request(directory,body,headers,timeout):
        if (body['instructions']!=episode.reset.PROMPT+' Return only schema-valid JSON.'
                or body.get('tools')!=[] or len(body['input'])!=1
                or len(body['input'][0]['content'])!=2 or 'previous_response_id' in body):
            raise StopExperiment('Reset请求审计不一致；不发送')
        save(Path(directory)/'context_audit.json',{'reset_prompt_exact':True,'no_episode_history':True})
        return original_request(directory,body,headers,timeout)
    def approved(operations,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        safety_assert(runtime)
        serialized=[{'kind':kind,'data':data.to_dict() if kind=='move' else data} for kind,data in operations]
        path=directory/'approved_native_plan.json';save(path,serialized);bound=digest(serialized)
        is_reset=directory.name=='lift_and_drop'
        if is_reset and state['reset_manipulations']>=1:raise StopExperiment('开场reset预算耗尽')
        print('Reset/观察位准备轨迹：'+str(path),flush=True)
        if is_reset:print('Reset抓取预览：'+str(folder/'grasp_preview.png'),flush=True)
        execution_authorization(directory,{'native_sha256':bound,'reset_manipulation':is_reset})
        if digest(read(path))!=bound:raise StopExperiment('Reset规划后轨迹文件变化')
        if is_reset:
            before=Observation.load(folder/'before');fresh=runtime.capture_current(folder/'pre_reset_dispatch')
            point=read(folder/'grasp.json')
            verify_scene(before,fresh,{'grasp_px':point['pixel_xy']},directory,automatic=False)
        if operations:
            kind,data=operations[0]
            initial=data.start if kind=='move' else {k:np.asarray(v) for k,v in data['joints'].items()}
            runtime.arrival(initial)
        safety_assert(runtime)
        state['reset_motion_dispatches']+=1
        if is_reset:state['reset_manipulations']+=1
        save(root/'status.json',state)
        result=original(operations,directory)
        save(directory/'execution.json',result)
        if result['status']!='EXECUTED':raise StopExperiment('Reset/准备执行异常；停止')
        return result
    runtime._execute_operations=approved
    episode.direct_once=reset_request
    try:
        result=episode.initialize(runtime,folder,episode.planner.auth_headers(),600,press_after_drop=False)
        body=read(folder/'region_request/request_payload.json')
        audited=(body['instructions']==episode.reset.PROMPT+' Return only schema-valid JSON.'
            and body['tools']==[] and len(body['input'])==1 and len(body['input'][0]['content'])==2
            and 'previous_response_id' not in body)
        save(folder/'context_audit.json',{'existing_reset_prompt_exact':audited,'reset_request_separate_from_planner':True})
        if not audited:raise StopExperiment('Reset请求审计不一致')
        if result['status']!='EXECUTED':raise StopExperiment('Reset未完成')
    finally:
        runtime._execute_operations=original
        episode.direct_once=original_request


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    state={'experiment':'live_skill_discovery_01','dispatches':0,'iterations':0,'teacher_calls':0,
           'reset_manipulations':0,'reset_motion_dispatches':0,'termination':'RUNNING'};save(root/'status.json',state)
    (root/'entrypoint_snapshot.py').write_bytes(Path(__file__).read_bytes())
    save(root/'protocol.json',{'max_manipulations':MAX_ITERATIONS,'max_iterations':MAX_ITERATIONS,
        'pull_distance_xy_mm':[MIN_PULL_MM,MAX_PULL_MM],'max_teacher_calls':MAX_ITERATIONS,'startup_reset':True,'reset_manipulations_max':1,'reset_counts_against_iterations':False,
        'automatic_retry':False,'base_prompt':BASE,'teacher_prompt':TEACHER,'model':'gpt-6-astra','effort':'high',
        'approval':'session-authorized automatic execution; no per-action human confirmation',
        'evaluation':'independent model visual assessment; UNKNOWN remains unknown and continues with fresh target; not human verified',
        'evaluation_calls_max':{'outcome':6,'mechanism':5},
        'safety':'Unmodified EpisodeCollectionRuntime and one_shot_goal_full_episode_v1.prepare_action; effective policy saved separately',
        'loop':'review each completed action; one Teacher per round; NO_NEW_SKILL continues; worsening retires current skill; UNKNOWN continues with fresh target; faults stop',
        'production_prompt_modified':False})
    runtime=None;teacher=None;failure=None;skill=None;skill_hash=None;pending_goal=None;last_after=None;history=[];skill_source_plan=None
    oldsig={}
    try:
        runtime=make_runtime(args.config)
        save(root/'runtime_policy.json',runtime_policy(runtime))
        def interrupted(signum,frame):runtime.cancel.set();raise KeyboardInterrupt('现场停止')
        for sig in [signal.SIGINT,signal.SIGTERM]:oldsig[sig]=signal.signal(sig,interrupted)
        state['phase']='CONNECTING';save(root/'status.json',state)
        connect_stationary(runtime,root)
        state['phase']='STARTUP_RESET';save(root/'status.json',state)
        startup_reset(runtime,root,state)
        for index in range(MAX_ITERATIONS):
            state['phase']=f'ROUND_{index:03d}';save(root/'status.json',state)
            folder=root/f'round_{index:03d}';folder.mkdir()
            before=runtime.capture_current(folder/'before')
            if last_after is not None:
                from cloth_agent.experiments.trajectory_memory_full_episode_v1 import unchanged
                if not unchanged(last_after.image,before.image)[0]:raise StopExperiment('轮间场景改变；停止')
            if skill is not None and sha(root/'skill_frozen.json')!=skill_hash:raise StopExperiment('冻结Skill变化')
            state['iterations']+=1;save(root/'status.json',state)
            plan,intent=planner(folder,before,skill,original_goal=pending_goal)
            if plan['decision']!='ACTION':state['termination']='PLANNER_ABSTAIN';break
            after=dispatch_once(runtime,root,state,folder,before,plan,intent,
                base_plan=skill_source_plan if skill is not None else None)
            # Hold the failed target fixed until improved; then allow a new local target.
            evaluated_intent=pending_goal or intent
            outcome=observe_outcome(folder,before,after,evaluated_intent)
            history.append({'round':folder.name,'mode':'LEARNED' if skill else 'BASE','intent':intent,
                'evaluated_target':evaluated_intent,'outcome':outcome,'skill_used_sha256':skill_hash if skill else None})
            save(root/'rounds.json',history);last_after=after
            choice=outcome['choice']
            pending_goal=None if choice in ['IMPROVED','UNKNOWN'] else {k:evaluated_intent[k] for k in ['target_region','visible_fold','expected_visible_change']}
            if skill is None and choice in ['NO_CLEAR_IMPROVEMENT','WORSENED']:
                failure={'round':folder.name,'plan':plan,'intent':evaluated_intent,'outcome':outcome}
            if choice=='UNKNOWN':
                print('效果无法确定：不计作成功或失败；下一轮依据最新RGB重新定位目标。',flush=True)
            active_skill=skill
            if choice=='WORSENED':
                skill=None;skill_hash=None;skill_source_plan=None
                print('本轮效果变差：停用当前Skill，重新评估替代方案；无可靠替代时回到Base。',flush=True)
            if choice=='NO_VISIBLE_CHANGE':
                print('未见明显运动：可能抓空或滑移，未确认；下一轮重新观察和选点。',flush=True)
            if state['teacher_calls']>=MAX_ITERATIONS:raise StopExperiment('Teacher调用预算耗尽')
            state['teacher_calls']+=1;save(root/'status.json',state)
            teacher_dir=folder/'teacher'
            teacher_input={'original_intent':evaluated_intent,'actual_plan':plan,
                'execution':read(folder/'execution.json'),'outcome_assessment':outcome,
                'active_skill':active_skill,'measured_points':read(folder/'points.json'),
                'motion_profile':read(folder/'motion_profile.json')}
            teacher=request(teacher_dir,TEACHER,'第一张Before，第二张After。本轮记录：'+json.dumps(teacher_input,ensure_ascii=False),[before.image,after.image],TEACHER_SCHEMA)
            candidate=teacher['skill']
            accepted=teacher['decision']=='NEW_SKILL' and all(candidate.values()) and sum('\u4e00'<=c<='\u9fff' for v in candidate.values() for c in v)<=150
            transition={'outcome':choice,'teacher_decision':teacher['decision'],'candidate_accepted':bool(accepted),
                'previous_skill_sha256':history[-1]['skill_used_sha256'],
                'possible_missed_grasp':choice=='NO_VISIBLE_CHANGE','missed_grasp_confirmed':False,
                'retired_due_to_worsening':choice=='WORSENED' and active_skill is not None}
            if accepted:
                skill=candidate
                save(folder/'skill_candidate.json',skill)
                save(root/'skill_frozen.json',skill);skill_hash=sha(root/'skill_frozen.json');skill_source_plan=plan
                freeze={'sha256':skill_hash,'source_round':folder.name,'teacher_request':sha(teacher_dir/'request_payload.json')}
                save(folder/'skill_freeze.json',freeze);save(root/'skill_freeze.json',freeze)
                print('本轮Skill已保存，下一轮重新观察后使用；机制假设尚未证实。',flush=True)
            else:
                print('本轮没有可靠新Skill，继续下一轮。',flush=True)
            transition['next_mode']='LEARNED' if skill else 'BASE'
            transition['has_next_round']=index+1<MAX_ITERATIONS
            save(root/'active_skill.json',{'enabled':skill is not None,'sha256':skill_hash,
                'skill':skill,'updated_after_round':folder.name})
            save(folder/'skill_transition.json',transition)
            history[-1]['skill_transition']=transition;save(root/'rounds.json',history)
        else:state['termination']='ITERATION_LIMIT'
    except BaseException as exc:
        state['termination']='STOPPED';state['error']=f'{type(exc).__name__}: {exc}'
        if runtime is not None and runtime.coordinator is not None:
            try:runtime.stop()
            except Exception as stop_exc:state['stop_error']=str(stop_exc)
    finally:
        if runtime is not None:
            try:runtime.close()
            except Exception as exc:state['close_error']=str(exc)
        for sig,handler in oldsig.items():signal.signal(sig,handler)
        save(root/'status.json',state)
        if state['termination']=='STOPPED':
            print('\nSTOPPED ['+state.get('phase','INITIALIZATION')+']: '+state.get('error','unknown error'),flush=True)
            print(f"已下发展开动作 {state['dispatches']} 次，reset抓取 {state['reset_manipulations']} 次；本次没有自动重启。",flush=True)
        learned=[x for x in history if x['mode']=='LEARNED']
        answers=[
            '1. Base为什么没有达到目标？\n'+('观察：'+failure['outcome']['note']+'；原因未知。' if failure else '未获得可归纳的明确Base失败；各轮结果见rounds.json，安全或执行错误不等于布料失败。'),
            '2. Teacher自主提出了什么新操作假设？（以下为最近一轮；全部记录见各轮teacher）\n'+('假设（未证实）：'+teacher['physical_hypothesis']+'；未知：'+teacher['unknowns'] if teacher else '未调用Teacher。'),
            '3. 新动作与原动作有什么本质区别？\n'+('Teacher主张：'+teacher['mechanism_difference']+'；模型视觉评估见各轮mechanism_review.json（无人核验）；该主张不是物理证明。' if teacher else '没有新动作。'),
            '4. 新动作实际造成了什么布料变化？\n'+('；'.join(x['round']+'：'+x['outcome']['note'] for x in learned) or '未知或未执行；不得以提案代替实际效果。'),
            '5. 是否有证据表明原目标卷折得到改善？\n'+('；'.join(x['round']+' '+x['evaluated_target']['target_region']+'：'+x['outcome']['choice'] for x in history) or '未知')]
        (root/'report.md').write_text('\n\n'.join(answers)+'\n\n终止：'+state['termination']+'\n'+state.get('error','')+'\n')
        print('\n\n'.join(answers)+'\n记录：'+str(root),flush=True)
        if state['termination']=='ITERATION_LIMIT':print('6轮已结束，不会自动reset或开始下一场。',flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json');args=p.parse_args(argv)
    if args.execute and args.dry_run:p.error('choose execute OR dry-run')
    if not args.execute:
        print(json.dumps({'base_prompt':BASE,'teacher_prompt':TEACHER,'max_manipulations':MAX_ITERATIONS,'max_teacher_calls':MAX_ITERATIONS,'requires_specific_live_approval':False,'automatic_visual_evaluation':True,'max_iterations':MAX_ITERATIONS,'startup_reset':True,'reset_counts_against_iterations':False,'connected':False},ensure_ascii=False,indent=2));return 0
    args.output=args.output or ROOT/'reports'/('live_skill_discovery_01_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run(args)
    return 0 if read(args.output/'status.json')['termination']!='STOPPED' else 1
if __name__=='__main__':raise SystemExit(main())
