"""One visually selected pickup/drop, then photograph and exit for manual reset."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from .discovery_data import save
from .partitioned_unfold import (ROOT, configure_runtime, prepare_reset,
    RESET_RELEASE_WORLD_Z_MM, RESET_RETURN_DESCENT_MM)
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime
from cloth_agent.perception import PerceptionConfig
from cloth_agent.unfold_video import UnfoldVideo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--grasp-yaw-offset-deg', type=float, choices=(-90.,0.,90.), default=0.)
    parser.add_argument('--retreat-left', action='store_true')
    parser.add_argument('--release-world-z-mm', type=float, default=RESET_RELEASE_WORLD_Z_MM)
    parser.add_argument('--midline-shift-mm', type=float, choices=(0.,100.), default=0.,
                        help='shift release along the partition line toward positive world Y')
    parser.add_argument('--far-translation-mm', type=float, choices=(-100.,-50.,0.,50.,100.,125.), default=0.,
                        help='positive toward image top; negative toward image bottom')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required; this command opens tools and returns arms Home')
    name = 'reset_point_trial_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root = ROOT/'reports'/name
    root.mkdir()
    runtime = configure_runtime(PartitionedReplayRuntime(
        ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json'))
    result = dict(status='STARTING', physical_actions=0, output=str(root))
    try:
        config = PerceptionConfig.load(ROOT, ROOT/'config/perception.zed_dual_xarm7_20261103.json')
        runtime.video_recorder = UnfoldVideo(config, ROOT/'videos'/name, root)
        runtime.video_recorder.start()
        before = runtime.observe(root/'before', reset_zed=True)
        print('BEFORE_IMAGE: ' + str(before.image), flush=True)
        print('Enter JSON {"pixel": [x,y], "label": "..."}; one drop only:', flush=True)
        selected = json.loads(input())
        pixel = selected['pixel']
        with Image.open(before.image) as image:
            width, height = image.size
        if (len(pixel) != 2 or any(type(v) is not int for v in pixel)
                or not 5 <= pixel[0] < width-5 or not 5 <= pixel[1] < height-5):
            raise ValueError('selected pixel must be inside the current image')
        before.validate_for(runtime.config, live=True, check_age=False)
        repairs = []
        xyz, resolved = before.sample_nearby_measured(pixel, pixel, 5, repairs)
        point = dict(pixel_xy=resolved, world_xyz_mm=xyz.tolist(),
                     selection=selected, sampling='VISUALLY_SELECTED_SINGLE_POINT')
        save(root/'point.json', point)
        save(root/'local_depth_repairs.json', dict(repairs=repairs, raw_cloud_modified=False))
        with Image.open(before.image) as image:
            image = image.convert('RGB')
            draw = ImageDraw.Draw(image)
            x, y = resolved
            draw.ellipse((x-9,y-9,x+9,y+9),outline='red',width=3)
            image.save(root/'selected_point.png')
        release = dict(release_world_z_mm=args.release_world_z_mm,
                       yaw_delta=0., drop_offset_xy=[0.,0.])
        source = prepare_reset(runtime, before, point, release, root,
                               grasp_yaw_offset_deg=args.grasp_yaw_offset_deg,
                               release_world_z_mm=args.release_world_z_mm)
        if args.midline_shift_mm:
            import numpy as np
            from cloth_agent.codex_motion import parse_code
            geometry = json.loads((root/'geometry.json').read_text())
            axis = np.asarray(geometry['assignment']['right_axis_xy'])
            along = np.array([axis[1], -axis[0]])
            if along[1] < 0:
                along = -along
            pose = list(parse_code(source, allow_grasp=True)[-1][1]['right'])
            pose[:2] = (np.asarray(pose[:2]) + along*args.midline_shift_mm/1000.).tolist()
            lines = source.splitlines()
            lines[-1] = f'motion.move_tcp(right={pose!r})'
            source = '\n'.join(lines)+'\n'
            _, preview, _ = runtime.compile(source,before,post_release_descent_mm=RESET_RETURN_DESCENT_MM)
            (root/'motion.py').write_text(source,encoding='utf-8')
            (root/'robotapi_preview.py').write_text(preview,encoding='utf-8')
            geometry.update(midline_shift_mm=args.midline_shift_mm,
                            drop_world_mm=[v*1000 for v in pose[:3]])
            save(root/'geometry.json',geometry)
        if args.far_translation_mm:
            from cloth_agent.codex_motion import parse_code
            pose = list(parse_code(source, allow_grasp=True)[-1][1]['right'])
            pose[1] -= args.far_translation_mm/1000.
            source += f'motion.move_tcp(right={pose!r})\n'
            _, preview, _ = runtime.compile(source,before,post_release_descent_mm=RESET_RETURN_DESCENT_MM)
            (root/'motion.py').write_text(source,encoding='utf-8')
            (root/'robotapi_preview.py').write_text(preview,encoding='utf-8')
            geometry = json.loads((root/'geometry.json').read_text())
            geometry.update(far_translation_mm=args.far_translation_mm,
                            midline_before_translation_world_mm=geometry['drop_world_mm'],
                            drop_world_mm=[v*1000 for v in pose[:3]])
            save(root/'geometry.json',geometry)
        if args.retreat_left:
            from .reset_parking import find_parking, execute_parking
            parking = find_parking(runtime, before, source, root)
            source = parking.get('reset_source', source)
            if parking.get('parking_required', True):
                before = execute_parking(runtime, before, parking, root)
        execution = runtime.execute(source, before, root/'execution',
                                   post_release_descent_mm=RESET_RETURN_DESCENT_MM)
        result.update(status=execution['status'], execution=execution)
        if execution['status'] == 'EXECUTED':
            result['physical_actions'] = 1
            time.sleep(2)
            after = runtime.observe(root/'after', reset_zed=True)
            result.update(status='COMPLETED_WAITING_MANUAL_UNFOLD', after_image=str(after.image))
        else:
            result['error'] = execution.get('error')
    except BaseException as exc:
        result.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(root/'summary.json', result)
        runtime.close()
        print(json.dumps({k:v for k,v in result.items() if k != 'execution'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
