"""Observe -> Astra motion code -> native planner/executor -> next round."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import count
import json
from pathlib import Path
import signal
import shutil
import time

from .run_timing import RoundTiming
from .run_storage import runs_root

from .codex_astra_pipeline import ROOT, DEFAULT_TASK, run
from .codex_motion import CONTRACT, parse_code

DEFAULT_REFERENCES = (
    (ROOT/'data/reference/fold_states/active/state_01_left_sleeve.png', '步骤 1：左袖向内折后的状态'),
    (ROOT/'data/reference/fold_states/active/state_02_right_sleeve.png', '步骤 2：右袖向内折后的状态'),
    (ROOT/'data/reference/fold_states/active/state_03_left_side.png', '步骤 3：左侧衣身向内折后的状态'),
    (ROOT/'data/reference/fold_states/active/state_04_right_side.png', '步骤 4：右侧衣身向内折后的状态'),
    (ROOT/'data/reference/fold_states/active/state_05_bottom_hem.png', '步骤 5：下摆向上折后的终态'),
)

COOPERATION_POLICY = (
    '自主选择单臂或双臂整理并折叠衣服，动作按代码顺序执行，一次移动一臂。'
    '允许任意一臂独立抓取、另一臂保持不动，也允许两臂选择不同位置配合，不强制双爪抓取。'
    '夹爪闭合后可以继续尝试折叠，效果在本轮结束拍照后判断，无需中途持布确认。'
    '抓点使用 world_xyz_mm.npy（毫米，已包含配置的点云修正），各自取坐标，不再重复修正。\n'
)


def stage_references(output, references):
    rows=[];images=[]
    for i,(source,role) in enumerate(references):
        source=Path(source).resolve()
        if not source.is_file():raise ValueError(f'reference image is missing: {source}')
        destination=Path(output)/'references'/f'{i+1:02d}_{source.name}'
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,destination)
        images.append(destination)
        rows.append(dict(path=str(destination.resolve()),source=str(source),role=role,
                         sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    write(Path(output)/'references.json',rows)
    return images,rows


def write(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def loop(task, runtime, output, *, max_rounds=None, memory=None, model_call=run, timeout_s=900,
         references=(), unfold=False, remember_iterations=False):
    if max_rounds is not None and max_rounds<1:
        raise ValueError('max_rounds must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    reference_images,reference_rows=stage_references(output,references)
    reference_context=('\n其余附件按折叠顺序排列，是各步完成后的参考图：'+json.dumps(reference_rows,ensure_ascii=False)+
                       '\n按照参考图顺序折叠，自行根据现场判断当前阶段和下一步。左右以衣领在上、下摆在下且不镜像的衣物方向为准，不是机械臂编号。'+
                       '\n参考图可能来自不同衣物、相机和桌面，只借鉴结构与折叠形状，不照搬像素、位置、尺寸或动作坐标。') if reference_rows else ''
    policy=getattr(runtime,'planning_policy',COOPERATION_POLICY)
    contract=getattr(runtime,'motion_contract',CONTRACT)
    completion='如果当前观测表明衣物已整齐叠好，只输出 motion.complete()。'
    if unfold:
        if len(reference_rows)!=1:
            raise ValueError('unfold requires exactly one initial demonstration image')
        active=getattr(runtime,'active_arms',())
        if active not in (('right',),('left',)):
            raise ValueError('unfold requires a single-arm runtime')
        initial_offset_mm=15. if active==('right',) else 5.
        runtime.grasp_offset_mm=initial_offset_mm
        contract=contract.replace('15 mm',f'{initial_offset_mm:g} mm')
        policy=(f'只使用 {active[0]} 单臂将揉乱的衣服逐步展开复原，另一臂保持原位。'
                '辨认衣领、两袖、衣身和下摆，逐步解开堆叠与翻折，恢复参考图的展开轮廓和相对结构。'
                '选择单夹爪可完成的动作，尽可能和human demonstration最开始的样子相似。'
                '参考图只提供目标形状，不能直接拿它的像素或坐标指挥机器人。'
                '抓点从当前 world_xyz_mm.npy 读取毫米坐标，使用观测原始抓取高度，不要自行添加下降偏移。\n')
        reference_context=('\n目标附件是 human demonstration 最开始的展开状态：'+json.dumps(reference_rows,ensure_ascii=False)+
                           '\n这是展开复原的唯一目标，不是折叠动作序列。允许衣物、视角与尺寸不同，比较衣物本身的结构。')
        contract=contract.replace('已叠好时','已恢复参考图的展开状态时').replace('折叠关键','展开关键')
        completion='只有当前图像显示两袖、衣身和下摆已展开，且整体形状接近目标参考图时，才输出 motion.complete()。'
    # Legacy memory argument is accepted but deliberately never read or written.
    session_started=time.perf_counter()
    result={'status':'ROUND_LIMIT','physical_execution':runtime.physical,'rounds':0}
    try:
        capability_text=''
        if callable(getattr(runtime,'capability_report',None)):
            from .robot_capabilities import render_capabilities
            capabilities=runtime.capability_report()
            capability_text=render_capabilities(capabilities)
            if unfold:
                capability_text=capability_text.replace('15 mm',f'{initial_offset_mm:g} mm').replace('--grasp-offset-mm 15',f'--grasp-offset-mm {initial_offset_mm:g}')
            write(output/'robot_capabilities.json',capabilities)
            (output/'robot_capabilities.md').write_text(capability_text)
        previous_outcome=None;before=None;previous_source=None
        memory_record=None;previous_image=None
        for index in count(1):
            if max_rounds is not None and index>max_rounds:break
            folder=output/f'round_{index:03d}';folder.mkdir()
            round_contract=contract
            round_capabilities=capability_text
            if unfold:
                offset=runtime.grasp_offset_mm
                round_contract=contract.replace(f'{initial_offset_mm:g} mm',f'{offset:g} mm')
                round_contract+=f'\n本轮宿主额外下降 {offset:g} mm。闭合完成后继续原定搬运动作，不根据夹爪位置或机械抓取信号判空、中断或自动加深；效果由后续图像判断。输出原始高度，不重复扣减。'
                round_capabilities=capability_text.replace(f'{initial_offset_mm:g} mm',f'{offset:g} mm').replace(f'--grasp-offset-mm {initial_offset_mm:g}',f'--grasp-offset-mm {offset:g}')
            with RoundTiming(folder,index) as timing:
                retry_ik=(previous_outcome is not None and before is not None
                    and previous_outcome.get('failure_kind')=='IK_FAILED'
                    and previous_outcome.get('status')=='PLANNING_FAILED'
                    and previous_outcome.get('physical_execution') is False)
                if retry_ik:
                    (folder/'before').symlink_to(before.directory.resolve(),target_is_directory=True)
                    write(folder/'observation_reuse.json',dict(reason='IK_FAILED_WITHOUT_EXECUTION',
                        source=str(before.directory.resolve())))
                    print(f'Round {index}: 将 IK 失败反馈给 GPT，仅修正已有动作代码，不重新规划任务。',flush=True)
                else:
                    with timing.measure('observation_s'):
                        before=runtime.observe(folder/'before')
                prompt=(f'{task}\n只规划本轮的一小步，不执行动作。观测和初始状态在 {before.directory}。\n'
                        +policy+
                        getattr(runtime,'preparation_description','每轮结束后宿主会先回 Home，再到 perception 位姿拍照；请在本轮中先将衣物放稳并松开夹爪。\n')+
                        '第一张附件是当前 RGB。先查看 RGB 和 world_xyz_mm.npy，再决定抓点；所有坐标必须有观测依据。\n'+reference_context+'\n'+round_contract+
                        '\n'+completion+'完成或 blocked 必须单独输出，不与移动混用。')
                if previous_outcome is not None:
                    prompt+='\n上一轮完整结果：'+json.dumps(previous_outcome,ensure_ascii=False)
                    if retry_ik:
                        # A code repair request replaces the task-planning
                        # prompt: do not ask the model to inspect images and
                        # choose grasp points/folding strategy again.
                        prompt=('这是已有动作代码的修正请求，不是新的任务规划。'
                            '上一次代码因 IK 失败被拒绝，未执行任何动作。'
                            '失败报告给出了指令序号（从 1 开始）、机械臂、具体求解阶段、目标和误差，以及求解次数、迭代预算和拒绝原因。数值求解预算内失败不等于几何上绝对不可达；依据报告修正，不原样重复。'
                            '保留未受失败影响的抓点、折叠顺序和动作意图，不重新决定折叠策略。'
                            '如果报告明确为真实抓点 IK 失败，允许依据同一保存观测局部修正失败抓点、朝向或双臂分配及关联动作；不要凭空把点移到能力边界上。'
                            '其他情况下保留抓点，只修复失败的中间动作。'
                            '只修正导致失败的代码及必要的相关指令，未受影响的动作保持不变。'
                            '如果失败的是接近/撤离等辅助点，请调整辅助点或路线，不必重复原来的高度。'
                            '不要把辅助点 IK 失败直接当作最终抓点不可达。'
                            '报告的 rejected_motion_source 是宿主处理后的实际规划代码；以 GPT 原始代码为修改基础，不重复施加本次动作契约中规定的宿主抓取偏移。'
                            '返回修正后的完整动作代码（不是补丁），保留未执行的前序动作，不输出解释或反思。\n'
                            +round_contract+
                            '\n完整失败报告：'+json.dumps(previous_outcome,ensure_ascii=False)+
                            '\n上一次 GPT 原始动作代码：\n'+previous_source)
                        if previous_outcome.get('ik_failure',{}).get('phase')=='grasp_goal_ik':
                            prompt+='\n真实抓点修正需要的原观测目录（只读，不重新采集）：'+str(before.directory)
                        write(folder/'replanning_feedback.json',dict(mode='code_repair',outcome=previous_outcome,original_source=previous_source))
                    else:
                        prompt+='\n若上一轮 blocked 或规划失败，请结合新观测调整方案；blocked 仅跳过当轮，不结束任务。'
                if capability_text:
                    prompt+='\n\n以下是当前模型生成的能力参考，选点或修改代码前请阅读：\n'+round_capabilities
                images=[] if retry_ik else [before.image,*reference_images]
                if unfold and retry_ik:
                    prompt=prompt.replace('折叠顺序','展开顺序').replace('折叠策略','展开策略')
                if remember_iterations:
                    from .codex_iteration_memory import memory_prompt,parse_answer
                    write(folder/'memory_input.json',memory_record)
                    if previous_image is not None and not retry_ik:
                        images.append(previous_image)
                        prompt+='\n最后一张附件是上一轮动作前的 RGB，只用于与当前第一张图比较效果，不是目标参考图。'
                    prompt+=memory_prompt(memory_record)
                with timing.measure('gpt_code_repair_s' if retry_ik else 'gpt_planning_s'):
                    call=model_call(prompt,output=folder/'planner',read_only=True,raw_prompt=True,
                                    images=images,timeout_s=timeout_s)
                if call['status']!='AGENT_FINISHED':
                    timing.status='MODEL_FAILED'
                    result.update(status='MODEL_FAILED',model_result=call);break
                source=(folder/'planner/answer.txt').read_text()
                if remember_iterations:
                    source,model_memory=parse_answer(source)
                (folder/'motion.py').write_text(source)
                calls=parse_code(source,allow_complete=True,allow_grasp=True)
                if remember_iterations:
                    write(folder/'model_memory.json',model_memory)
                terminal=calls[0][0]
                if terminal in ('complete','blocked'):
                    outcome=dict(status='MODEL_REPORTED_COMPLETE' if terminal=='complete' else 'BLOCKED',
                                 physical_execution=False,**calls[0][1])
                else:
                    with timing.measure('motion_pipeline_s'):
                        outcome=runtime.execute(source,before,folder)
                    timing.seconds.update(outcome.get('timings_s',{}))
                timing.status=outcome['status']
                write(folder/'outcome.json',outcome)
                if remember_iterations:
                    memory_record=dict(iteration=index,memory=model_memory,code=source,outcome=outcome,
                                       observation=str(before.directory.resolve()))
                    write(output/'iteration_memory.json',memory_record)
                    with (output/'iteration_history.jsonl').open('a') as history:
                        history.write(json.dumps(memory_record,ensure_ascii=False)+'\n')
                    previous_image=before.image
                print(f'Round {index}: {outcome["status"]}',flush=True)
                if outcome.get('error'):
                    print(f'失败原因：{outcome["error"]}',flush=True)
                    if outcome.get('ik_failure'):
                        failure=outcome['ik_failure']
                        phase={'tcp_direct_ik':'TCP 插值点 IK','rrt_goal_ik':'RRT 开始搜索前的终点 IK',
                               'grasp_goal_ik':'真实抓点 IK','host_approach_ik':'宿主生成的接近点 IK','transfer_goal_ik':'自由转移终点 IK'}.get(failure['phase'],failure['phase'])
                        print(f'IK 失败步骤：第 {failure["command_index"]} 条指令，{failure["robot"]}，{phase}。',flush=True)
                    if outcome.get('failure_context'):
                        print('失败位置：'+json.dumps(outcome['failure_context'],ensure_ascii=False),flush=True)
                    if outcome.get('failed_command'):
                        print('执行指令：'+json.dumps(outcome['failed_command'],ensure_ascii=False),flush=True)
                    print(f'详细记录：{(folder/"outcome.json").resolve()}',flush=True)
                previous_outcome=outcome;previous_source=source
                result['rounds']=index
                if outcome['status'] in ('EXECUTION_FAILED','MODEL_REPORTED_COMPLETE'):
                    result.update(status=outcome['status'],outcome=outcome);break
                # IK-only rejection reuses the stationary observation; after
                # execution or other outcomes the next attempt captures anew.
    except KeyboardInterrupt:
        runtime.stop();result['status']='INTERRUPTED'
    except Exception as exc:
        runtime.stop();result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        runtime.close()
        result['elapsed_s']=time.perf_counter()-session_started
        write(output/'result.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task')
    parser.add_argument('--unfold',action='store_true',help='单臂将揉乱衣物恢复到 demonstration 初始展开图')
    parser.add_argument('--remember-iterations',action='store_true',help='保存 GPT 记忆并提供给下一轮；每次 run 独立')
    arms=parser.add_mutually_exclusive_group()
    arms.add_argument('--arm7-only',action='store_true',help='仅连接 arm7/right；arm6 使用保存的固定位置参与碰撞检查')
    arms.add_argument('--arm6-only',action='store_true',help='仅 arm6/left 运动；arm7 保持原位并参与碰撞检查')
    parser.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-rounds',type=int,default=None,help='可选轮数上限；默认持续运行直到 complete')
    parser.add_argument('--timeout-s',type=float,default=900)
    parser.add_argument('--viser-port',type=int,default=8088)
    execution=parser.add_mutually_exclusive_group()
    execution.add_argument('--auto-execute',dest='auto_execute',action='store_true',help='验证通过后显示预览并自动执行')
    execution.add_argument('--manual-execute',dest='auto_execute',action='store_false',help='验证通过后等待 Viser 点击')
    parser.set_defaults(auto_execute=None)
    parser.add_argument('--memory',type=Path,help='兼容旧启动命令；不再读取或写入经验文件')
    parser.add_argument('--reference',type=Path,action='append',help='按折叠顺序指定参考图，可重复；替换默认五张参考图')
    args=parser.parse_args(argv)
    if args.auto_execute is None:args.auto_execute=True
    if (args.max_rounds is not None and args.max_rounds<1) or args.timeout_s<=0:parser.error('rounds/timeout must be positive')
    if args.unfold and not (args.arm6_only or args.arm7_only):parser.error('--unfold requires --arm6-only or --arm7-only')
    references=[(p,f'用户指定的第 {i+1} 张顺序参考图') for i,p in enumerate(args.reference)] if args.reference else DEFAULT_REFERENCES
    if args.unfold:
        if args.reference and len(args.reference)!=1:parser.error('--unfold requires exactly one --reference')
        references=[(args.reference[0] if args.reference else ROOT/'data/reference/fold_states/active/state_00_unfolded.png',
                     '复原目标：human demonstration 最初的展开状态')]
    for path,_ in references:
        if not path.is_file():parser.error(f'reference image is missing: {path}')
    prefix='loop_arm6_' if args.arm6_only else ('loop_arm7_' if args.arm7_only else 'loop_')
    if args.unfold:prefix=prefix.replace('loop_','unfold_')
    output=args.output or runs_root(ROOT)/'codex_astra'/datetime.now(timezone.utc).strftime(prefix+'%Y%m%dT%H%M%S%fZ')
    if args.arm6_only:
        from .codex_arm6_runtime import Arm6Runtime
        runtime=Arm6Runtime(args.config)
    elif args.arm7_only:
        from .codex_arm7_runtime import Arm7Runtime
        runtime=Arm7Runtime(args.config)
    else:
        from .codex_dual_runtime import DualArmRuntime
        runtime=DualArmRuntime(args.config)
    from .trajectory_preview import TrajectoryPreview
    runtime.preview=TrajectoryPreview(runtime.config,port=args.viser_port,wait=not args.auto_execute)
    print('执行方式：验证通过后自动执行' if args.auto_execute else '执行方式：验证通过后等待 Viser 点击',flush=True)
    previous=signal.getsignal(signal.SIGTERM)
    def terminate(*_):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,terminate)
    try:
        task=args.task or ('只使用 arm6 单臂整理并按顺序整齐折叠桌上的衣服。' if args.arm6_only else '只使用 arm7 单臂整理并按顺序整齐折叠桌上的衣服。' if args.arm7_only else DEFAULT_TASK)
        if args.unfold and args.task is None:task='将被揉乱的衣服展开，恢复到 human demonstration 最开始图片的样子。'
        result=loop(task,runtime,output,max_rounds=args.max_rounds,memory=args.memory,timeout_s=args.timeout_s,
                    references=references,unfold=args.unfold,remember_iterations=args.remember_iterations)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['status'] in ('ROUND_LIMIT','MODEL_REPORTED_COMPLETE') else 1
    finally:
        runtime.close();runtime.preview.close();signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':raise SystemExit(main())
