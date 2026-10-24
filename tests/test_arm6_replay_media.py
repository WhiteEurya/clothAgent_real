import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from cloth_agent.experiments.arm6_replay_media import ReplayMedia


def test_timing_and_keyframes_survive_interrupted_step(tmp_path, monkeypatch):
    from cloth_agent.experiments import arm6_replay_media as module
    now = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(module, 'render_video', lambda *args: 'video.mp4')
    image = tmp_path/'source.png'
    Image.new('RGB', (32, 32), 'red').save(image)
    (tmp_path/'step_0000').mkdir()
    media = ReplayMedia(tmp_path)
    media.begin(0, image)
    now[0] += 2
    media.edge('e001')
    media.phase('execution')
    now[0] += 3
    media.snapshot('after', image)
    media.phase(None)
    (tmp_path/'step_0001').mkdir()
    media.begin(1, image)
    now[0] += 4
    media.finish('ERROR')
    data = json.loads((tmp_path/'timing.json').read_text())
    assert data['total_s'] == 9
    assert data['steps'][0]['phases_s'] == {'matching': 2, 'execution': 3}
    assert data['steps'][1]['phases_s'] == {'matching': 4}
    assert 'after' not in data['steps'][1]
    assert (tmp_path/data['steps'][0]['before']).read_bytes() == image.read_bytes()
    assert json.loads((tmp_path/'step_0001/timing.json').read_text()) == data['steps'][1]


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg unavailable')
def test_actual_video_encoding(tmp_path):
    image = tmp_path/'source.png'
    Image.new('RGB', (64, 32), 'blue').save(image)
    media = ReplayMedia(tmp_path)
    for index in range(2):
        (tmp_path/f'step_{index:04d}').mkdir()
        media.begin(index, image)
        if not index:
            media.snapshot('after', image)
    video = media.finish('ERROR')
    assert Path(video).stat().st_size > 1000
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'json', video], check=True, capture_output=True, text=True)
    assert float(json.loads(probe.stdout)['format']['duration']) == pytest.approx(6, abs=.2)


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg unavailable')
def test_sequence_video_uses_half_second_frames_and_deduplicates_recapture(tmp_path):
    before=tmp_path/'source.png';after=tmp_path/'end.png'
    Image.new('RGB',(64,32),'red').save(before);Image.new('RGB',(64,32),'blue').save(after)
    media=ReplayMedia(tmp_path,title='Arm6 operator-assisted graph route',sequence_video=True)
    (tmp_path/'step_0000').mkdir();media.begin(0,before);media.snapshot('after',after)
    (tmp_path/'step_0001').mkdir();media.begin(1,after)
    media.finish('GOAL_REACHED')
    sequence=json.loads((tmp_path/'sequence_frames.json').read_text())
    assert len(sequence['frames'])==2
    assert [r['stage'] for r in sequence['frames']]==['before','after']
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','json',
                          str(tmp_path/'replay_sequence.mp4')],check=True,capture_output=True,text=True)
    assert float(json.loads(probe.stdout)['format']['duration'])==pytest.approx(1,abs=.1)
