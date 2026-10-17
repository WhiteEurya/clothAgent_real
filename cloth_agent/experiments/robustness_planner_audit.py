"""Offline paired action generation on saved real RGB-D; no robot runtime/dispatch.

Uses each saved pre-change request as the control, fresh sessions for both arms
of the comparison, and changes only the robustness prompt. Outputs are proposals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cloth_agent.action_robustness import CODE_PROMPT, strip_code_evidence
from cloth_agent.codex_astra_pipeline import run, MODEL, EFFORT
from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.observation import Observation


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_mapping(obs):
    """Recover the recorded pinhole mapping, including the nonrigid Z correction.

    Fitting against the saved dense cloud avoids applying today's calibration to
    historical data. Check reprojection on a separate pixel grid before drawing.
    """
    affine = np.asarray(obs.meta.get('corrected_world_from_camera_affine_mm',
                                     obs.meta['world_from_camera_mm']), float)
    inverse = np.linalg.inv(affine)

    def samples(offset):
        y, x = np.mgrid[offset:obs.xyz.shape[0]:13, offset:obs.xyz.shape[1]:13]
        world = obs.xyz[y, x].reshape(-1, 3)
        camera = (np.c_[world, np.ones(len(world))] @ inverse.T)[:, :3]
        good = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
        return camera[good], np.c_[x.ravel(), y.ravel()][good]

    camera, uv = samples(0)
    if len(camera) < 100:
        raise ValueError('insufficient saved geometry for projection')
    k = np.eye(3)
    for axis in range(2):
        a = np.c_[camera[:, axis] / camera[:, 2], np.ones(len(camera))]
        k[axis, axis], k[axis, 2] = np.linalg.lstsq(a, uv[:, axis], rcond=None)[0]
    camera, uv = samples(6)
    q = camera @ k.T
    error = np.linalg.norm(q[:, :2] / q[:, 2, None] - uv, axis=1)
    if len(error) < 100 or np.max(error) > 1:
        raise ValueError('saved geometry cannot be projected within 1 pixel')
    return dict(intrinsics=k.tolist(), corrected_world_from_camera_affine_mm=affine.tolist(),
                held_out_max_reprojection_error_px=float(np.max(error)),
                source='pinhole fitted to recorded cloud; independent grid check, not physical calibration validation')


def project(xyz_m, mapping):
    world = np.asarray(xyz_m) * 1000
    inv = np.linalg.inv(mapping['corrected_world_from_camera_affine_mm'])
    camera = (np.c_[world, np.ones(len(world))] @ inv.T)[:, :3]
    if not np.isfinite(camera).all() or (camera[:, 2] <= 0).any():
        raise ValueError('unprojectable waypoint')
    uv = camera @ np.asarray(mapping['intrinsics']).T
    return uv[:, :2] / uv[:, 2, None]


def overlay(obs, source, mapping, destination, label):
    calls = parse_code(source, allow_grasp=True, allow_complete=True)
    im = Image.open(obs.image).convert('RGB')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('DejaVuSans.ttf', 19)
    draw.rectangle((0, 0, im.width, 68), fill='black')
    draw.text((8, 4), label + ' | proposed, NOT executed', fill='white', font=font)
    points = [dict(operation=name, pose=kw['right']) for name, kw in calls if 'right' in kw]
    result = dict(terminal=calls[0][0], waypoints=points)
    if not points:
        draw.text((8, 34), calls[0][0] + ': ' + str(calls[0][1])[:95], fill='orange', font=font)
    else:
        uv = project([p['pose'][:3] for p in points], mapping)
        result.update(Pa_world_m=points[0]['pose'][:3], Pb_world_m=points[-1]['pose'][:3],
                      Pa_pixel_xy=uv[0].tolist(), Pb_pixel_xy=uv[-1].tolist())
        draw.text((8, 34), 'Pa cyan / Pb magenta | yellow TCP path; dashed transfer is endpoint link only', fill='white', font=font)
        for i in range(1, len(points)):
            a, b = uv[i-1], uv[i]
            if points[i]['operation'] == 'move_transfer':
                for t in np.arange(0, 1, .1):
                    draw.line([tuple(a+(b-a)*t), tuple(a+(b-a)*min(t+.05, 1))], fill='orange', width=3)
            else:
                draw.line([tuple(a), tuple(b)], fill='yellow', width=3)
        used_labels = []
        for i, (p, xy) in enumerate(zip(points, uv)):
            p['pixel_xy'] = xy.tolist()
            x, y = xy
            color = 'cyan' if i == 0 else 'magenta' if i == len(points)-1 else 'yellow'
            tag = 'Pa' if i == 0 else 'Pb' if i == len(points)-1 else str(i)
            draw.ellipse((x-6, y-6, x+6, y+6), outline=color, width=3)
            for dx,dy in [(10,-24), (10,10), (-38,-24), (-38,12), (10,-50), (-38,38)]:
                box = (x+dx-3,y+dy-3,x+dx+34,y+dy+24)
                if not any(box[0]<b[2] and box[2]>b[0] and box[1]<b[3] and box[3]>b[1] for b in used_labels):
                    break
            used_labels.append(box)
            draw.text((x+dx,y+dy), tag, fill=color, font=font, stroke_width=2, stroke_fill='black')
        # Keep heights readable without covering the grasp neighborhood.
        heights = ' -> '.join(f'{p["pose"][2]*1000:.1f}' for p in points)
        draw.rectangle((0,im.height-35,im.width,im.height),fill='black')
        draw.text((8,im.height-29), 'Commanded world Z (mm), Pa -> waypoints -> Pb: '+heights,
                  fill='white',font=font)
        x0,y0 = np.maximum(np.floor(uv.min(axis=0)-75),[0,68]).astype(int)
        x1,y1 = np.minimum(np.ceil(uv.max(axis=0)+75),[im.width,im.height-35]).astype(int)
        if x1>x0 and y1>y0:
            im.crop((x0,y0,x1,y1)).resize((2*(x1-x0),2*(y1-y0))).save(destination.with_name('detail.png'))
    im.save(destination)
    return result


def compare(rounds, output, workers=2, timeout=900):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    manifest = dict(model=MODEL, reasoning_effort=EFFORT, physical_execution=False,
                    design='paired fresh sessions, same saved request/images; only robustness block differs',
                    limitations=['One stochastic sample per variant; no physical execution or success measurement.',
                                 'Saved observations keep their own calibration; old captures are historical controls.'],
                    samples=[])
    for index, round_path in enumerate(rounds, 1):
        round_path = Path(round_path).resolve()
        obs = Observation.load(round_path / 'before')
        if obs.meta.get('synthetic') is not False:
            raise ValueError('only real saved observations are allowed')
        request_path = round_path / 'planner/request.json'
        request = json.loads(request_path.read_text())
        if 'ACTION_ROBUSTNESS' in request['task']:
            raise ValueError('baseline already contains new robustness policy')
        cmd = request['command']
        images = [Path(cmd[i+1]) for i, arg in enumerate(cmd) if arg == '--image']
        if not images or images[0].resolve() != obs.image.resolve():
            raise ValueError('saved request does not attach current RGB first')
        sample = output / f'sample_{index:02d}'
        sample.mkdir()
        mapping = camera_mapping(obs)
        save(sample/'projection.json', mapping)
        common = request['task'] + (
            '\n这是离线单轮对照，无会话历史。仅根据本次提供的观察与参考生成动作，不执行、不采集、不连接机器人。'
            '\n只读本请求列出的 observation 文件、附件及坐标所需文件；不要读取其他实验、planner 答案、Graph 或新版规划代码。'
            '\n仅返回本请求指定的动作代码。不要调用机器人接口、网络、相机、采集脚本或更改文件。')
        (sample/'baseline_prompt.txt').write_text(common)
        (sample/'robust_prompt.txt').write_text(common+'\n'+CODE_PROMPT)
        row = dict(id=sample.name, observation=str(obs.directory), captured_at=obs.meta['captured_at'],
                   calibration_id=obs.meta['calibration_id'], source_request=str(request_path),
                   source_request_sha256=sha(request_path), inputs={str(p):sha(p) for p in [*images, obs.directory/'world_xyz_mm.npy']})
        manifest['samples'].append(row)
        for variant in ('baseline', 'robust'):
            jobs.append((sample, variant, obs, images, mapping))
    save(output/'manifest.json', manifest)

    def generate(job):
        sample, variant, obs, images, mapping = job
        folder = sample/variant
        result = run((sample/f'{variant}_prompt.txt').read_text(), output=folder,
                     read_only=True, raw_prompt=True, images=images, timeout_s=timeout)
        if result['status'] != 'AGENT_FINISHED':
            return dict(sample=sample.name, variant=variant, status=result['status'])
        source = strip_code_evidence((folder/'answer.txt').read_text(), folder)
        (folder/'motion.py').write_text(source)
        action = overlay(obs, source, mapping, folder/'overlay.png', sample.name+' / '+variant)
        save(folder/'action.json', action)
        return dict(sample=sample.name, variant=variant, status='PROPOSAL_GENERATED', **action)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(generate, job):job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(sample=job[0].name, variant=job[1], status='FAILED', error=str(exc))
            results.append(result)
            save(output/'results.json', results)
            print(json.dumps({k:result[k] for k in ('sample','variant','status')}, ensure_ascii=False), flush=True)
    pairs = []
    for row in manifest['samples']:
        sample = output/row['id']
        files = [sample/v/'overlay.png' for v in ('baseline','robust')]
        if all(p.exists() for p in files):
            ims = [Image.open(p).convert('RGB') for p in files]
            pair = Image.new('RGB', (sum(im.width for im in ims), max(im.height for im in ims)))
            pair.paste(ims[0], (0,0));pair.paste(ims[1], (ims[0].width,0))
            pair.save(sample/'comparison.png');pairs.append(pair)
    if pairs:
        sheet = Image.new('RGB', (pairs[0].width, sum(im.height for im in pairs)))
        y = 0
        for pair in pairs:
            sheet.paste(pair,(0,y));y += pair.height
        sheet.save(output/'comparison.png')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', action='append', required=True, dest='rounds')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    compare(args.rounds, args.output, args.workers, args.timeout)


if __name__ == '__main__':
    main()
