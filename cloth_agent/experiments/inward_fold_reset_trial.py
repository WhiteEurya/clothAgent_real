"""One inward pick/place followed by one existing reset; capture and stop.

Operator supplies pixel selections from each fresh fixed-ZED image over stdin.
Uses existing runtime, action compiler and complete path checks unchanged.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw
from .partitioned_unfold import ROOT, configure_runtime, prepare_action, prepare_reset
from .reset_parking import find_parking, execute_parking
from .discovery_data import save
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime
from cloth_agent.perception import PerceptionConfig
from cloth_agent.unfold_video import UnfoldVideo
from cloth_agent.random_fold_reset import sample_grasp


def selection(observation, stage):
    print(json.dumps(dict(stage=stage,image=str(observation.image)),ensure_ascii=False),flush=True)
    return json.loads(input())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--seed',type=int,default=2026111201)
    args=parser.parse_args()
    if not args.execute:parser.error('--execute required for this single physical trial')
    name='inward_fold_reset_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    root=ROOT/'reports'/name;root.mkdir()
    runtime=configure_runtime(PartitionedReplayRuntime(ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json'))
    result=dict(status='STARTING',physical_actions=0,output=str(root),seed=args.seed)
    try:
        config=PerceptionConfig.load(ROOT,ROOT/'config/perception.zed_dual_xarm7_20261103.json')
        runtime.video_recorder=UnfoldVideo(config,ROOT/'videos'/name,root)
        runtime.video_recorder.start()
        before=runtime.observe(root/'before',reset_zed=True)
        selected=selection(before,'SELECT_INWARD_FOLD')
        plan=dict(decision='ACTION',grasp_px=selected['grasp_px'],target_px=selected['target_px'],
            grasp_angle_deg=None,motion_type='lift_pull_place',is_smoothing=False,
            action_purpose='向内折叠一个外露端部，构造初始化状态',expected_stop_state='端部折入衣身形成覆盖',
            short_reason=selected['reason'])
        folder=root/'inward_fold';folder.mkdir();save(folder/'plan.json',plan)
        source=prepare_action(runtime,before,plan,folder)
        with Image.open(before.image) as im:
            im=im.convert('RGB');draw=ImageDraw.Draw(im)
            draw.line([tuple(plan['grasp_px']),tuple(plan['target_px'])],fill='red',width=5)
            x,y=plan['grasp_px'];draw.ellipse((x-8,y-8,x+8,y+8),outline='yellow',width=3)
            im.save(folder/'action.png')
        execution=runtime.execute(source,before,folder/'execution');save(folder/'execution.json',execution)
        if execution['status']!='EXECUTED':
            result.update(status='FOLD_FAILED',execution=execution);return
        result['physical_actions']=1
        time.sleep(2)
        folded=runtime.observe(root/'after_fold',reset_zed=True)
        region=selection(folded,'SELECT_RESET_CENTER')
        folder=root/'reset';folder.mkdir();save(folder/'region.json',region)
        repairs=[]
        point=sample_grasp(folded,region,None,60.,args.seed,minimum_radius_mm=40.,depth_repairs=repairs)
        save(folder/'grasp.json',point);save(folder/'depth_repairs.json',repairs)
        source=prepare_reset(runtime,folded,point,dict(release_world_z_mm=400.,yaw_delta=0.,drop_offset_xy=[0.,0.]),folder,
            grasp_yaw_offset_deg=90.,release_world_z_mm=400.)
        parking=find_parking(runtime,folded,source,folder);source=parking.get('reset_source',source)
        if parking.get('parking_required',True):folded=execute_parking(runtime,folded,parking,folder)
        execution=runtime.execute(source,folded,folder/'execution',post_release_descent_mm=50.)
        save(folder/'execution.json',execution)
        if execution['status']!='EXECUTED':
            result.update(status='RESET_FAILED',execution=execution);return
        result['physical_actions']=2
        time.sleep(2)
        after=runtime.observe(root/'after_reset',reset_zed=True)
        result.update(status='COMPLETED',after_image=str(after.image))
    except BaseException as exc:
        result.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(root/'summary.json',result)
        runtime.close()
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
