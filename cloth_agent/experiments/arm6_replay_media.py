"""Persist replay keyframes and monotonic timings; render an offline recap."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import time


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


class ReplayMedia:
    def __init__(self, output, *, title="Arm6 replay", sequence_video=False):
        self.output = Path(output)
        self.started = time.monotonic()
        self.data = dict(started_at=datetime.now(timezone.utc).isoformat(), steps=[], title=title)
        self.step = None
        self.phase_name = None
        self.phase_started = self.started
        self.sequence_video = sequence_video

    def phase(self, name):
        now = time.monotonic()
        if self.step is not None and self.phase_name is not None:
            phases = self.step['phases_s']
            phases[self.phase_name] = phases.get(self.phase_name, 0) + now - self.phase_started
        self.phase_name, self.phase_started = name, now
        self.flush()

    def begin(self, index, image):
        self.phase(None)
        self.step = dict(index=index, started_s=time.monotonic()-self.started,
                         started_at=datetime.now(timezone.utc).isoformat(), phases_s={})
        self.data['steps'].append(self.step)
        self.snapshot('before', image)
        self.phase('matching')

    def snapshot(self, name, image):
        destination = self.output / f"step_{self.step['index']:04d}" / f'{name}.png'
        shutil.copyfile(image, destination)
        self.step[name] = str(destination.relative_to(self.output))
        self.step[name + '_source'] = str(image)
        self.step[name + '_saved_at'] = datetime.now(timezone.utc).isoformat()
        self.flush()

    def edge(self, edge_id):
        self.step['edge_id'] = edge_id
        self.flush()

    def flush(self):
        if self.step is not None:
            self.step['elapsed_s'] = sum(self.step['phases_s'].values())
            write_json(self.output / f"step_{self.step['index']:04d}" / 'timing.json', self.step)
        write_json(self.output / 'timing.json', self.data)

    def finish(self, status):
        self.phase(None)
        self.data.update(status=status, total_s=time.monotonic()-self.started,
                         finished_at=datetime.now(timezone.utc).isoformat())
        self.flush()
        if not self.data['steps']:
            return None
        video = render_video(self.output, self.data)
        if self.sequence_video:
            try:self.data['sequence_video'] = render_sequence_video(self.output, self.data)
            except Exception as exc:self.data['sequence_video_error'] = f'{type(exc).__name__}: {exc}'
            self.flush()
        return video


def render_sequence_video(output, data):
    """Chronological captured RGBs, 0.5 s per image; never interpolate robot motion."""
    import hashlib
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    output=Path(output);frames=output/'sequence_frames';frames.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    previous=None;rows=[]
    for step in data['steps']:
        for stage in ('before','action_before','after'):
            if not step.get(stage):continue
            path=output/step[stage]
            with Image.open(path) as im:
                im=im.convert('RGB');digest=hashlib.sha256(im.tobytes()).hexdigest()
                if digest==previous:continue
                previous=digest;canvas=Image.new('RGB',(1280,800),'#101827')
                picture=ImageOps.contain(im,(1280,720));canvas.paste(picture,((1280-picture.width)//2,60+(720-picture.height)//2))
            draw=ImageDraw.Draw(canvas)
            draw.text((20,8),f"{data.get('title','Arm6 replay')} | step {step['index']:02d} | {step.get('edge_id','state check')} | {stage}",font=font,fill='white')
            elapsed=sum(step['phases_s'].values())
            draw.text((20,34),f"Step {elapsed:.1f}s | Run {data['total_s']:.1f}s | {data['status']} | RGB sequence: 0.5s/frame",font=font,fill='#b9c8dd')
            canvas.save(frames/f'{len(rows):04d}.png')
            rows.append(dict(step=step['index'],stage=stage,source=str(path),display_duration_s=.5))
    if not rows:return None
    write_json(output/'sequence_frames.json',dict(kind='captured_rgb_sequence',frames=rows))
    destination=output/'replay_sequence.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','2','-i',str(frames/'%04d.png'),
                    '-frames:v',str(len(rows)*5),'-vf','fps=10','-c:v','libx264','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(destination)],check=True,capture_output=True,timeout=120)
    return str(destination)


def render_video(output, data):
    """Each still lasts three seconds; this is not continuous camera footage."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    output = Path(output)
    frames = output / 'video_frames'
    frames.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    for number, step in enumerate(data['steps']):
        canvas = Image.new('RGB', (1280, 900), '#101827')
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 22), f"{data.get('title', 'Arm6 replay')} | Step {step['index']:02d} | {step.get('edge_id', 'state check')}", font=font, fill='white')
        draw.text((30, 65), 'BEFORE', font=font, fill='#81d4fa')
        draw.text((655, 65), 'AFTER (returned to perception)', font=font, fill='#81d4fa')
        for key, x in [('before', 30), ('after', 655)]:
            if step.get(key):
                with Image.open(output / step[key]) as image:
                    image = ImageOps.contain(image.convert('RGB'), (595, 530))
                    canvas.paste(image, (x+(595-image.width)//2, 110+(530-image.height)//2))
            else:
                draw.text((x, 330), 'No end frame captured', font=font, fill='#fbbf24')
        phases = step['phases_s']
        elapsed = sum(phases.values())
        draw.text((30, 665), f"Step elapsed: {elapsed:.2f}s | Started at run +{step['started_s']:.2f}s", font=font, fill='white')
        labels = [f'{name}: {value:.2f}s' for name, value in phases.items()]
        timing_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
        for row, label in enumerate(labels):
            draw.text((30+625*(row % 2), 710+32*(row//2)), label, font=timing_font, fill='#cbd5e1')
        draw.text((30, 820), f"Run: {data['status']} | Total: {data['total_s']:.2f}s", font=font, fill='white')
        draw.text((30, 858), 'Keyframe recap - 3 seconds per step, not real-time video', font=font, fill='#94a3b8')
        canvas.save(frames / f'{number:04d}.png')
    destination = output / 'replay_summary.mp4'
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '1/3', '-i',
               str(frames/'%04d.png'), '-frames:v', str(len(data['steps'])*30),
               '-c:v', 'libx264', '-r', '10', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return str(destination)
