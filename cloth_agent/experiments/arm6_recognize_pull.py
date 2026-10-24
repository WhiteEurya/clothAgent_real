"""Standalone cloth unfolding: explore visible corners when unrecognizable, unfold when recognizable."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal
import numpy as np
from .discovery_schema import obj, array, S, N, enum
from .discovery_data import save
from .graph_policy import Requests
from .discovery_model import CodexMultimodal
from .arm6_replay_media import ReplayMedia
from .arm6_corner_pull import SCHEMA as PULL_SCHEMA, PROMPT as PULL_PROMPT, UNFOLD_PROMPT, run_once
from .replay_failures import capture_cli_failure,stop_after_failure,close_cli_runtime,cli_failed

RECOGNITION=obj(state=enum('RECOGNIZABLE','UNRECOGNIZABLE'),confidence=N,
    state_description=S,visible_parts=array(S),occluded_or_uncertain_parts=array(S),reason=S,
    unfolded={'type':'boolean'},unfolded_evidence=array(S))
RECOGNITION_PROMPT='''这是一个只看当前照片的衣物认识实验。先判断：你是否能可靠辨认当前衣服的整体布局，
主要部位和可见折叠/遮挡关系，并能具体说明它目前是怎样摆放或折叠的？
RECOGNIZABLE 表示有可见依据支持上述判断；不要求已展开或平整。仅知道“这是衣服/T恤”、
仅找到一个外露布角、或根据常识猜测被挡部位，都不算认识整体状态。
若主要布局因堆叠遮挡而无法辨认，返回 UNRECOGNIZABLE。不要强行猜领口、袖口或下摆。
描述能看到的部位和无法辨认的部位，并简述判断依据。confidence 是对本次分类的把握，
不是展开完成度。同时判断是否已展开：主要衣身伸展、袖部不再明显内折压住衣身、下摆未回折覆盖上身。
允许普通小褶皱，不要求熨平。只有当前图像有明确证据时 unfolded=true，并列出 unfolded_evidence。
认不清时 unfolded=false。这里只判断，不规划动作，也不引用任何参考图、状态库或历史示范。'''


def make_requests(output,args,backend=None):
    return Requests(output/'model',backend or CodexMultimodal(args.model,args.timeout_s),
        max_calls=args.max_calls,schemas={'recognize':RECOGNITION,'unknown_pull':PULL_SCHEMA,'unfold':PULL_SCHEMA},
        prompts={'recognize':RECOGNITION_PROMPT,'unknown_pull':PULL_PROMPT,'unfold':UNFOLD_PROMPT},
        common='根据给定照片回答独立衣物实验的问题。只输出符合 schema 的 JSON，不输出代码。',
        protocol='arm6-standalone-recognize-pull-v1')


def decision_status(recognition,threshold):
    if recognition['confidence']<threshold:return 'RECOGNITION_UNCERTAIN'
    if recognition['state']=='RECOGNIZABLE':
        if not recognition['visible_parts'] or not recognition['state_description'].strip():
            return 'RECOGNITION_UNCERTAIN'
        if recognition['unfolded'] and recognition['unfolded_evidence']:
            return 'MODEL_REPORTED_UNFOLDED'
        return 'UNFOLD'
    return 'PULL'


def loop(runtime,requests,output,media,result,*,max_steps=12,max_exploration_pulls=3,threshold=.8,max_distance_mm=150.,grasp_offset_mm=-15.):
    before=runtime.observe(output/'initial_observation');previous=None
    for index in range(max_steps+1):
        if runtime.cancel.is_set():raise KeyboardInterrupt('recognition experiment cancelled')
        step=output/f'step_{index:04d}';step.mkdir()
        media.begin(index,before.image);media.phase('recognition')
        recognition=requests.ask('recognize',{},[('current',before.image)])
        save(step/'recognition.json',recognition)
        result['recognitions'].append(dict(step=index,**recognition))
        result['last_observation']=str(before.image)
        print('ARM6 RECOGNITION '+str(recognition),flush=True)
        if runtime.cancel.is_set():raise KeyboardInterrupt('recognition experiment cancelled after model call')
        status=decision_status(recognition,threshold)
        if status not in ('PULL','UNFOLD'):
            result['status']=status;media.snapshot('after',before.image);break
        if index==max_steps:
            result['status']='STEP_LIMIT';media.snapshot('after',before.image);break
        kind='unknown_pull' if status=='PULL' else 'unfold'
        if kind=='unknown_pull' and result.get('exploration_pulls',0)>=max_exploration_pulls:
            result['status']='EXPLORATION_LIMIT';media.snapshot('after',before.image);break
        status,after,history=run_once(runtime,before,requests,step/'corner_pull',media,result,
            threshold=threshold,max_distance_mm=min(80.,max_distance_mm) if kind=='unknown_pull' else max_distance_mm,
            previous=previous,grasp_offset_mm=grasp_offset_mm,kind=kind,recognition=recognition)
        if status!='EXECUTED':
            result['status']=status;media.snapshot('after',before.image);break
        previous=history;before=after;result['last_observation']=str(before.image)
        save(output/'result.json',dict(result,status='RECOGNIZING_AFTER_PULL'))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true',help='run the standalone experiment on Arm6')
    mode.add_argument('--image',type=Path,help='recognition only on a saved RGB; no robot connection or movement')
    parser.add_argument('--config',type=Path,default=Path('config/dual_arm.local.json'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-steps',type=int,default=12)
    parser.add_argument('--max-exploration-pulls',type=int,default=3)
    parser.add_argument('--max-pull-mm',type=float,default=150.)
    parser.add_argument('--grasp-offset-mm',type=float,default=-15.)
    parser.add_argument('--confidence-threshold',type=float,default=.8)
    parser.add_argument('--max-calls',type=int,default=48)
    parser.add_argument('--model',default='gpt-6-astra')
    parser.add_argument('--timeout-s',type=float,default=180.)
    args=parser.parse_args(argv)
    if (not 1<=args.max_steps<=30 or not 0<=args.max_exploration_pulls<=10 or not 10<=args.max_pull_mm<=150 or not -30<=args.grasp_offset_mm<=10
            or not 0<args.confidence_threshold<=1 or args.max_calls<1 or not np.isfinite(args.timeout_s) or args.timeout_s<=0):
        parser.error('invalid pull count, distance, grasp offset, confidence, call limit or timeout')
    if args.image and not args.image.is_file():parser.error('image does not exist')
    output=args.output or Path('reports')/('arm6_recognize_pull_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    result=dict(status='PREPARING',experiment='standalone_recognize_and_pull',robot='Arm6',
        physical_actions=0,pull_actions=0,exploration_pulls=0,unfold_actions=0,steps=[],recognitions=[],parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        recognition_definition='visible overall garment layout and folding/occlusion relations; not task completion')
    save(output/'result.json',result)
    runtime=None;requests=None;media=ReplayMedia(output,title='Arm6 recognition and corner pull')
    prior=signal.getsignal(signal.SIGTERM)
    def interrupted(*_):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        requests=make_requests(output,args)
        if args.image:
            destination=output/'input.png'
            from PIL import Image
            with Image.open(args.image) as im:im.convert('RGB').save(destination)
            recognition=requests.ask('recognize',{},[('current',destination)])
            save(output/'recognition.json',recognition)
            result.update(status='OFFLINE_RECOGNITION_ONLY',recognitions=[recognition],robot_calls=0,
                          proposed_next=decision_status(recognition,args.confidence_threshold))
        else:
            from cloth_agent.gamepad_collect_runtime import CollectionRuntime
            runtime=CollectionRuntime(args.config);runtime.config.home_recover_stopped=True
            loop(runtime,requests,output,media,result,max_steps=args.max_steps,max_exploration_pulls=args.max_exploration_pulls,threshold=args.confidence_threshold,
                 max_distance_mm=args.max_pull_mm,grasp_offset_mm=args.grasp_offset_mm)
    except (Exception,KeyboardInterrupt) as exc:
        result.update(capture_cli_failure(output,exc))
        if runtime is not None:
            runtime.cancel.set();stop_after_failure(runtime,result)
    finally:
        signal.signal(signal.SIGTERM,prior)
        close_cli_runtime(runtime,result)
        if requests is not None:result['model_metrics']=requests.metrics()
        try:
            video=media.finish(result['status']);result['timing_file']=str(output/'timing.json')
            if video:
                destination=output/'experiment_summary.mp4';Path(video).rename(destination);result['video']=str(destination)
        except Exception as exc:result['media_error']=str(exc)
        save(output/'result.json',result)
        print(f'实验结束：{result["status"]}，拉动 {result["pull_actions"]} 次。记录：{output}',flush=True)
    return 1 if cli_failed(result) else 0


if __name__=='__main__':raise SystemExit(main())
