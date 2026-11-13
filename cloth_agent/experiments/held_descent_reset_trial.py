"""One random pickup, high lift and held descent before release; capture and stop.

Operator supplies pixel selections from each fresh fixed-ZED image over stdin.
Uses existing runtime, action compiler and complete path checks unchanged.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw
from .partitioned_unfold import ROOT, configure_runtime, prepare_reset
from cloth_agent.codex_motion import parse_code
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
    parser.add_argument('--seed',type=int,default=2026111202)
    parser.add_argument('--selected-point',action='store_true',
                        help='Use one visually selected pixel from the fresh image instead of random center sampling')
    parser.add_argument('--release-world-z-mm',type=float,default=250.)
    parser.add_argument('--high-transfer-mm',type=float,choices=(400.,450.,500.,550.,600.),
                        help='Trial only: lift vertically at grasp before transferring at this world Z')
    args=parser.parse_args()
    if not args.execute:parser.error('--execute required for this single physical trial')
    if not 200. <= args.release_world_z_mm < 400.:parser.error('trial release Z must be 200..399 mm')
    name='held_descent_reset_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    root=ROOT/'reports'/name;root.mkdir()
    runtime=configure_runtime(PartitionedReplayRuntime(ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json'))
    result=dict(status='STARTING',physical_actions=0,output=str(root),seed=args.seed)
    try:
        config=PerceptionConfig.load(ROOT,ROOT/'config/perception.zed_dual_xarm7_20261103.json')
        runtime.video_recorder=UnfoldVideo(config,ROOT/'videos'/name,root)
        runtime.video_recorder.start()
        before=runtime.observe(root/'before',reset_zed=True)
        folded=before
        region=selection(folded,'SELECT_GRASP_PIXEL' if args.selected_point else 'SELECT_RESET_CENTER')
        folder=root/'reset';folder.mkdir();save(folder/'region.json',region)
        repairs=[]
        if args.selected_point:
            pixel=region['pixel']
            with Image.open(folded.image) as image:
                width,height=image.size
            if (not isinstance(pixel,list) or len(pixel)!=2
                    or any(type(v) is not int for v in pixel)
                    or not 5<=pixel[0]<width-5 or not 5<=pixel[1]<height-5):
                raise ValueError('selected pixel must be inside the current image')
            xyz,resolved=folded.sample_nearby_measured(pixel,pixel,5,repairs)
            point=dict(pixel_xy=resolved,world_xyz_mm=xyz.tolist(),
                       selection=region,sampling='VISUALLY_SELECTED_SINGLE_POINT')
        else:
            point=sample_grasp(folded,region,None,60.,args.seed,minimum_radius_mm=40.,depth_repairs=repairs)
        save(folder/'grasp.json',point);save(folder/'depth_repairs.json',repairs)
        source=prepare_reset(runtime,folded,point,dict(release_world_z_mm=400.,yaw_delta=0.,drop_offset_xy=[0.,0.]),folder,
            grasp_yaw_offset_deg=90.,release_world_z_mm=400.)
        # The existing compiler opens only after the last move_tcp, so adding
        # this waypoint keeps the cloth held throughout the descent.
        final=list(parse_code(source,allow_grasp=True)[-1][1]['right'])
        final[2]=args.release_world_z_mm/1000.
        source+=f'motion.move_tcp(right={final!r})\n'
        peak=args.high_transfer_mm or 400.
        if args.high_transfer_mm is not None:
            grasp=list(parse_code(source,allow_grasp=True)[0][1]['right'])
            lift=grasp.copy();lift[2]=peak/1000.
            transfer=final.copy();transfer[2]=peak/1000.
            source=(f'motion.grasp(right={grasp!r})\n'
                    f'motion.move_tcp(right={lift!r})\n'
                    f'motion.move_tcp(right={transfer!r})\n'
                    f'motion.move_tcp(right={final!r})\n')
        save(folder/'descent.json',dict(peak_world_z_mm=peak,release_world_z_mm=args.release_world_z_mm,
             high_transfer=args.high_transfer_mm is not None,
             hold_closed_during_descent=True,post_release_descent_mm=50.))
        parking=find_parking(runtime,folded,source,folder);source=parking.get('reset_source',source)
        geometry=json.loads((folder/'geometry.json').read_text())
        geometry.update(peak_world_z_mm=peak,release_world_z_mm=args.release_world_z_mm,
                        release_policy=('VERTICAL_LIFT_HIGH_TRANSFER_HELD_DESCENT' if args.high_transfer_mm
                                        else 'MIDLINE_PEAK_THEN_HELD_DESCENT'))
        geometry['release']['release_world_z_mm']=args.release_world_z_mm
        save(folder/'geometry.json',geometry)
        if parking.get('parking_required',True):folded=execute_parking(runtime,folded,parking,folder)
        execution=runtime.execute(source,folded,folder/'execution',post_release_descent_mm=50.)
        save(folder/'execution.json',execution)
        if execution['status']!='EXECUTED':
            result.update(status='RESET_FAILED',execution=execution);return
        result['physical_actions']=1
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
