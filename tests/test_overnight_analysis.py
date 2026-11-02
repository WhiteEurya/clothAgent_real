import csv,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from cloth_agent.experiments import overnight_analysis as analysis,automatic_top5_batch as batch


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_model_done_not_ground_truth_and_review_labels_survive_refresh(tmp_path):
    for i,reason in enumerate(['DONE','ROUND_LIMIT','SAFETY_STOP'],1):
        ep=tmp_path/f'episode_{i:04d}'
        save(ep/'episode_summary.json',dict(finished_at='test',termination_reason=reason,physical_actions=2))
        save(ep/'protocol.json',dict(initialization_center_radius_mm=20))
    s=analysis.analyze(tmp_path)
    assert s['eligible_completed']==2 and s['model_done_rate']==.5
    assert s['human_review']['labeled']==0 and len(s['excluded_from_done_rate'])==1
    f=tmp_path/'analysis/human_review.csv'
    with f.open() as stream:rows=list(csv.DictReader(stream))
    rows[0]['human_done']='NOT_DONE';rows[0]['notes']='模型误判'
    analysis.csv_write(f,rows,['episode','human_done','notes'])
    s=analysis.analyze(tmp_path)
    assert s['human_review']['labeled']==1 and s['human_review']['rate']==0
    assert '模型误判' in f.read_text()
    assert analysis.wilson(0,0) is None


@pytest.mark.parametrize('low_disk',[False,True])
def test_overnight_finishes_episode_then_respects_deadline_or_disk(tmp_path,monkeypatch,low_disk):
    clock=[100.];commands=[]
    class Server:
        def shutdown(self):pass
        def server_close(self):pass
    monkeypatch.setattr(batch,'serve',lambda *a:Server())
    monkeypatch.setattr(batch.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(batch.shutil,'disk_usage',lambda p:SimpleNamespace(free=(1 if low_disk else 100)*1024**3))
    def launch(command,log,control):
        commands.append(command);folder=Path(command[command.index('--output')+1])
        assert '--initialize-seed' in command
        assert 0<=float(command[command.index('--initialize-center-radius-mm')+1])<=40
        save(folder/'episode_summary.json',dict(termination_reason='DONE',finished_at='test',physical_actions=1))
        clock[0]+=3601
        return 0
    monkeypatch.setattr(batch.base,'launch_episode',launch)
    goal=tmp_path/'goal.png';Image.new('RGB',(100,100)).save(goal)
    config=tmp_path/'config.json';config.write_text('{}')
    batch.run(SimpleNamespace(output=tmp_path/'batch',episodes=1000,max_rounds=13,max_actions=12,
        port=8770,goal_image=goal,config=config,auto_next=True,max_hours=1,min_free_gb=20,
        initialize_center_radius_range_mm=[0,40],schedule_seed=123,systematic_report=True))
    assert len(commands)==(0 if low_disk else 1)
    summary=json.loads((tmp_path/'batch/batch_summary.json').read_text())
    assert summary['last_termination']==('DISK_RESERVE_STOP' if low_disk else 'DURATION_LIMIT')
    assert (tmp_path/'batch/analysis/summary.json').exists()
    schedule=json.loads((tmp_path/'batch/reset_schedule.json').read_text())
    assert len(schedule['schedule'])==1000 and schedule['seed']==123
